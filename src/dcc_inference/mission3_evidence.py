"""Offline symptom evidence model with bounded utterance windows and state head."""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
try:
    from . import mission3_bert as baseline
except ImportError:
    import mission3_bert as baseline
try:
    from dcc_data.mission3_evidence_data import tokenize_views
except ImportError:
    from mission3_evidence_data import tokenize_views

FORMAT = "dcc-m3-evidence-v2"
DEFAULTS = dict(dimension=128, dropout=.1, max_length=256, max_windows=6)


class EvidenceHead(nn.Module):
    def __init__(self, hidden, mode, config):
        super().__init__()
        self.mode = mode
        dimension = config["dimension"]
        self.project = nn.Sequential(nn.Linear(hidden, dimension), nn.Tanh(), nn.Dropout(config["dropout"]))
        self.queries = nn.Parameter(torch.randn(9, dimension)/math.sqrt(dimension))
        self.states = nn.Linear(dimension, 9*3)
        self.residual = nn.Linear(dimension+3, 1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, cls, logits, mask):
        hidden = self.project(cls.float())
        state_logits = self.states(hidden).reshape(*hidden.shape[:2], 9, 3)
        attention = torch.einsum("bwd,ld->blw", hidden, self.queries)/math.sqrt(hidden.shape[-1])
        attention = attention.masked_fill(~mask[:, None], -1e4).softmax(-1)
        pooled = torch.einsum("blw,bwd->bld", attention, hidden)
        state_probs = torch.einsum("blw,bwls->bls", attention, state_logits.softmax(-1))
        if self.mode not in ("states", "states_removed"):
            state_probs = state_probs*0
        elif self.mode == "states_removed":
            state_probs = state_probs*0
        correction = self.residual(torch.cat([pooled, state_probs], -1)).squeeze(-1)
        return logits[:, 0].float()+correction, state_logits


class EvidenceModel(nn.Module):
    requires_documents = True
    checkpoint_format = FORMAT

    def __init__(self, base, mode, config=None):
        super().__init__()
        self.base, self.mode = base, mode
        self.config = dict(DEFAULTS, **(config or {}))
        self.head = EvidenceHead(base.classifier.in_features, mode, self.config)

    def extract(self, ids, attention_mask, owners, window_numbers, count, width):
        hidden = self.base.bert(input_ids=ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.base.classifier(self.base.dropout(hidden))
        cls = hidden.new_zeros(count, width, hidden.shape[-1], dtype=torch.float32)
        scores = logits.new_zeros(count, width, 9, dtype=torch.float32)
        mask = torch.zeros(count, width, dtype=torch.bool, device=ids.device)
        cls[owners, window_numbers] = hidden.float()
        scores[owners, window_numbers] = logits.float()
        mask[owners, window_numbers] = True
        return cls, scores, mask

    def forward(self, packed):
        cls, logits, mask = self.extract(**packed)
        with torch.autocast(device_type=cls.device.type, enabled=False):
            return self.head(cls, logits, mask)


def pack(tokenizer, documents, mode, config):
    views = [tokenize_views(tokenizer, doc, mode, config["max_length"], config["max_windows"]) for doc in documents]
    flat = [dict(input_ids=v["ids"], attention_mask=[1]*len(v["ids"])) for windows in views for v in windows]
    encoded = tokenizer.pad(flat, padding=True, return_tensors="pt")
    packed = dict(ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                  owners=torch.tensor([i for i, windows in enumerate(views) for _ in windows]),
                  window_numbers=torch.tensor([j for windows in views for j in range(len(windows))]),
                  count=len(documents), width=max(map(len, views)))
    return packed, views


def to_device(packed, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in packed.items()}


def export_payload(model, original, source):
    return dict(format=FORMAT, files=original["files"], labels=list(baseline.LABELS),
                max_length=model.config["max_length"], config=model.config, mode=model.mode,
                model={key: value.detach().cpu() for key, value in model.state_dict().items()}, source=source)


def load_payload(payload, device=None):
    from transformers import AutoTokenizer
    if payload.get("format") != FORMAT or tuple(payload.get("labels", [])) != baseline.LABELS:
        raise ValueError("Invalid evidence checkpoint")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, value in files.items():
            if Path(name).name != name:
                raise ValueError("Unsafe tokenizer file name")
            (Path(folder)/name).write_text(value, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
    base = baseline._build(files, payload)
    model = EvidenceModel(base, payload["mode"], payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model, tokenizer, list(baseline.LABELS), model.config["max_length"]


def probabilities(model, tokenizer, documents, names=None, max_length=256, batch_size=2):
    if max_length != model.config["max_length"]:
        raise ValueError("Inference length must match the checkpoint")
    device = next(model.parameters()).device
    out = np.zeros((len(documents), 9), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            packed, _ = pack(tokenizer, documents[start:start+batch_size], model.mode, model.config)
            logits, _ = model(to_device(packed, device))
            out[start:start+batch_size] = logits.float().sigmoid().cpu().numpy()
    return out
