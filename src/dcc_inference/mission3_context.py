"""Text-only M3 residual models: contrastive memory and adjacent-turn evidence.

The memory contains Training representations, never validation/test annotations.
All models retain sigmoid >= .5, including empty predictions. This module is also
copied beside mission3_bert.py when creating an offline submission package.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import mission3_bert as baseline
except ImportError:  # standalone candidate package
    import mission3_bert as baseline

FORMAT = "dcc-m3-context-v1"
LABELS = baseline.LABELS
METHODS = ("control", "memory", "evidence", "memory_removed", "answers_removed")
DEFAULTS = dict(dimension=128, dropout=.1, positive_k=8, negative_k=8,
                overall_k=32, temperature=.1, max_length=256)


def text_and_spans(doc: Any) -> tuple[str, list[tuple[int, int]]]:
    """Exactly the baseline text; recover boundaries without adding any tokens."""
    utterances = doc.get("utterances", []) if isinstance(doc, dict) else []
    if not isinstance(utterances, list):
        utterances = []
    parts, spans, length = [], [], 0
    for utterance in utterances:
        text = utterance.get("text", "") if isinstance(utterance, dict) else ""
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            start = length + bool(parts)
            spans.append((start, start + len(text)))
            parts.append(text)
            length = start + len(text)
    return " ".join(parts), spans


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_key(name: str) -> str:
    """Paired file stem, without the date suffix; identity is never a feature."""
    stem = Path(unicodedata.normalize("NFC", name)).stem
    return re.sub(r"_\d{8}$", "", stem)


def tokenize_documents(tokenizer, documents: Sequence[dict], max_length: int = 256):
    if not tokenizer.is_fast:
        raise ValueError("M3 context requires a fast tokenizer with character offsets")
    texts_spans = [text_and_spans(doc) for doc in documents]
    encoded = tokenizer([text for text, _ in texts_spans], padding=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True,
                        return_tensors="pt")
    offsets = encoded.pop("offset_mapping")
    turn_ids = torch.full(encoded["input_ids"].shape, -1, dtype=torch.long)
    for row, (_, spans) in enumerate(texts_spans):
        cursor = 0
        for token, (start, end) in enumerate(offsets[row].tolist()):
            if end <= start:
                continue  # CLS/SEP/padding, including the empty transcript
            while cursor < len(spans) and spans[cursor][1] <= start:
                cursor += 1
            if cursor < len(spans) and start < spans[cursor][1] and end > spans[cursor][0]:
                turn_ids[row, token] = cursor
    encoded["turn_ids"] = turn_ids
    return encoded, [text_hash(text) for text, _ in texts_spans]


def pool_turns(hidden: torch.Tensor, turn_ids: torch.Tensor):
    """Mean contextual token representations for each visible utterance."""
    count = max(1, int(turn_ids.max().item()) + 1)
    index = turn_ids.clamp_min(0)
    visible = turn_ids >= 0
    values = hidden.float() * visible.unsqueeze(-1)
    sums = torch.zeros(hidden.shape[0], count, hidden.shape[-1], device=hidden.device)
    sums.scatter_add_(1, index.unsqueeze(-1).expand_as(hidden), values)
    counts = torch.zeros(hidden.shape[0], count, device=hidden.device)
    counts.scatter_add_(1, index, visible.float())
    return sums / counts.clamp_min(1).unsqueeze(-1), counts > 0


class MemoryBank(nn.Module):
    def __init__(self, vectors, labels, calls, hashes, *, split="training"):
        super().__init__()
        if split != "training":
            raise ValueError("Only Training may populate the retrieval bank")
        vectors = torch.as_tensor(vectors).float()
        labels = torch.as_tensor(labels).float()
        if vectors.ndim != 2 or labels.shape != (len(vectors), 9):
            raise ValueError("Invalid memory vector/label shapes")
        if len(calls) != len(vectors) or len(hashes) != len(vectors):
            raise ValueError("Memory identities must align with vectors")
        if not torch.isfinite(vectors).all() or not torch.isfinite(labels).all():
            raise ValueError("Non-finite memory")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Memory labels must be binary")
        self.register_buffer("vectors", vectors)
        self.register_buffer("keys", F.normalize(vectors, dim=-1))
        self.register_buffer("labels", labels)
        self.calls, self.hashes = list(calls), list(hashes)
        self.exclusions: dict[str, list[int]] = {}
        for i, (call, digest) in enumerate(zip(calls, hashes)):
            self.exclusions.setdefault("c:" + call, []).append(i)
            self.exclusions.setdefault("h:" + digest, []).append(i)

    def similarities(self, queries, calls, hashes):
        if len(calls) != len(queries) or len(hashes) != len(queries):
            raise ValueError("Every query needs its call and transcript identity")
        scores = F.normalize(queries.float(), dim=-1) @ self.keys.T
        for i, (call, digest) in enumerate(zip(calls, hashes)):
            excluded = self.exclusions.get("c:" + call, []) + self.exclusions.get("h:" + digest, [])
            if excluded:
                scores[i, excluded] = -torch.inf
        return scores

    def _neighbors(self, scores, k, temperature):
        if scores.shape[1] == 0:
            shape = (len(scores), 0)
            return scores.new_zeros(shape), torch.empty(shape, dtype=torch.long, device=scores.device)
        values, indices = scores.topk(min(k, scores.shape[1]), dim=-1)
        valid = torch.isfinite(values)
        # All-excluded neighborhoods must have zero weight, not NaN softmax.
        weights = (values.masked_fill(~valid, -1e4) / temperature).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return weights, indices

    def retrieve(self, queries, calls, hashes, config):
        scores = self.similarities(queries, calls, hashes)
        weight, index = self._neighbors(scores, config["overall_k"], config["temperature"])
        vote = (self.labels[index] * weight.unsqueeze(-1)).sum(1)
        entropy = -(vote.clamp(1e-6, 1-1e-6) * vote.clamp(1e-6, 1-1e-6).log()
                    + (1-vote).clamp(1e-6, 1-1e-6) * (1-vote).clamp(1e-6, 1-1e-6).log()) / math.log(2)
        prototypes, statistics = [], []
        for label in range(9):
            pair, means, availability = [], [], []
            for positive, k in ((True, config["positive_k"]), (False, config["negative_k"])):
                eligible = self.labels[:, label].bool() == positive
                subset = scores.masked_fill(~eligible.unsqueeze(0), -torch.inf)
                w, idx = self._neighbors(subset, k, config["temperature"])
                pair.append((self.vectors[idx] * w.unsqueeze(-1)).sum(1))
                means.append((subset.gather(1, idx).masked_fill(w == 0, 0) * w).sum(1))
                availability.append((w.sum(1) > 0).float())
            prototypes.append(torch.stack(pair, 1))
            statistics.append(torch.stack([means[0], means[1], means[0]-means[1],
                                           vote[:, label], entropy[:, label],
                                           availability[0]*availability[1]], -1))
        return torch.stack(prototypes, 1), torch.stack(statistics, 1)


class ResidualHead(nn.Module):
    def __init__(self, hidden_size, method, config):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown M3 method: {method}")
        self.method, self.config = method, dict(config)
        dimension = config["dimension"]
        self.project = nn.Linear(hidden_size, dimension)
        self.dropout = nn.Dropout(config["dropout"])
        self.label_queries = nn.Parameter(torch.randn(9, dimension) / math.sqrt(dimension))
        self.window = nn.Linear(3 * hidden_size, dimension)
        width = 4 * dimension + 6 if method.startswith("memory") else 2 * dimension
        self.residual = nn.Sequential(nn.Linear(width, dimension), nn.ReLU(),
                                      nn.Dropout(config["dropout"]), nn.Linear(dimension, 1))
        self.gate = nn.Linear(6 if method.startswith("memory") else dimension, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, cls, base_logits, turns, turn_mask, *, bank=None, calls=(), hashes=()):
        query = self.dropout(torch.tanh(self.project(cls.float())))
        query = query[:, None, :].expand(-1, 9, -1)
        if self.method.startswith("memory"):
            if bank is None:
                raise ValueError("Memory method requires a Training bank")
            if self.method == "memory_removed":
                prototypes = cls.new_zeros(len(cls),9,2,cls.shape[-1])
                stats = cls.new_zeros(len(cls),9,6)
            else:
                prototypes, stats = bank.retrieve(cls, calls, hashes, self.config)
            projected = torch.tanh(self.project(prototypes))
            labels = self.label_queries[None].expand(len(cls), -1, -1)
            inputs = torch.cat([query, projected[:, :, 0], projected[:, :, 1], labels, stats], -1)
            gate = self.gate(stats).sigmoid()
            supported = (stats[:, :, 5] > 0).unsqueeze(-1) if self.method == "memory" else 1
        elif self.method in ("evidence", "answers_removed"):
            padded = F.pad(turns.float()*turn_mask.unsqueeze(-1), (0, 0, 0, 2))
            following = [padded[:, i:i+turns.shape[1]] for i in range(3)]
            if self.method == "answers_removed":
                following[1:] = [torch.zeros_like(following[0]), torch.zeros_like(following[0])]
            windows = torch.tanh(self.window(torch.cat(following, -1)))
            attention = torch.einsum("btd,ld->blt", windows, self.label_queries) / math.sqrt(query.shape[-1])
            attention = attention.masked_fill(~turn_mask[:, None, :], -1e4).softmax(-1)
            attention = attention * turn_mask[:, None, :]
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            evidence = torch.einsum("blt,btd->bld", attention, windows)
            inputs = torch.cat([query, evidence], -1)
            gate = self.gate(evidence).sigmoid()
            supported = turn_mask.any(-1)[:, None, None]
        else:
            inputs = torch.cat([query, self.label_queries[None].expand(len(cls), -1, -1)], -1)
            gate, supported = self.gate(query).sigmoid(), 1
        return base_logits.float() + (self.residual(inputs) * gate * supported).squeeze(-1)


class ContextModel(nn.Module):
    requires_documents = True

    def __init__(self, base, method, config=None, bank=None):
        super().__init__()
        self.base = base
        self.config = dict(DEFAULTS, **(config or {}))
        self.head = ResidualHead(base.classifier.in_features, method, self.config)
        self.bank = bank

    def extract(self, ids, mask, turn_ids):
        hidden = self.base.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        cls = hidden[:, 0].float()
        turns, turn_mask = pool_turns(hidden, turn_ids)
        logits = self.base.classifier(self.base.dropout(hidden[:, 0])).float()
        return cls, logits, turns, turn_mask

    def forward(self, ids, mask, turn_ids, calls=(), hashes=()):
        cls, logits, turns, turn_mask = self.extract(ids, mask, turn_ids)
        # Keep the residual module in fp32 under the encoder's AMP context.
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self.head(cls, logits, turns, turn_mask, bank=self.bank, calls=calls, hashes=hashes)


def load_payload(payload, device=None):
    from transformers import AutoTokenizer
    if payload.get("format") != FORMAT or tuple(payload.get("labels", ())) != LABELS:
        raise ValueError("Invalid M3 context checkpoint format/labels")
    config = dict(payload["context_config"])
    if config["max_length"] != 256 or payload.get("threshold", .5) != .5:
        raise ValueError("M3 context requires max_length=256 and threshold=0.5")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder) / Path(name).name).write_text(text, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
    base = baseline._build(files, payload)
    memory = payload.get("memory")
    if payload["method"].startswith("memory") and memory is None:
        raise ValueError("A memory checkpoint must include its Training bank")
    bank = MemoryBank(**memory) if memory is not None else None
    model = ContextModel(base, payload["method"], config, bank)
    # The bank is serialized separately, so its derived keys are not duplicated.
    state = model.state_dict()
    state.update(payload["model"])
    expected = {key for key in state if not key.startswith("bank.")}
    if set(payload["model"]) != expected:
        raise ValueError("Context checkpoint model keys do not match architecture")
    model.load_state_dict(state, strict=True)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model, tokenizer, list(LABELS), 256


def export_payload(model, base_payload, source):
    memory = None
    if model.bank is not None:
        memory = dict(vectors=model.bank.vectors.detach().cpu(), labels=model.bank.labels.detach().cpu(),
                      calls=model.bank.calls, hashes=model.bank.hashes, split="training")
    return dict(format=FORMAT, files=base_payload["files"], labels=list(LABELS), max_length=256,
                dropout=base_payload.get("dropout", .1), threshold=.5,
                method=model.head.method, context_config=model.config, source=source,
                memory=memory, model={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()
                                     if not k.startswith("bank.")})


def probabilities(model, tokenizer, documents, names, max_length=256, batch_size=32):
    if len(names) != len(documents):
        raise ValueError("File names and documents must align")
    device = next(model.parameters()).device
    out = np.zeros((len(documents), 9), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            docs, chunk_names = documents[start:start+batch_size], names[start:start+batch_size]
            encoded, hashes = tokenize_documents(tokenizer, docs, max_length)
            tensors = {k:v.to(device) for k,v in encoded.items()}
            logits = model(tensors["input_ids"], tensors["attention_mask"], tensors["turn_ids"],
                           [call_key(name) for name in chunk_names], hashes)
            out[start:start+len(docs)] = logits.sigmoid().cpu().numpy()
    return out
