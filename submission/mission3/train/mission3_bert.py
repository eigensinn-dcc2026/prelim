"""Mission 3 BERT-family classifier: self-contained checkpoint, export and probabilities.

The lab experiments (``scripts/lab/m3-kcbert-005.py`` and relatives) save
``best.ckpt`` = ``{"config", "seed", "epoch", "model": state_dict, ...}`` where the
state dict belongs to ``Classifier`` below (``bert`` encoder + dropout + Linear on
the first token). Those files need the Hugging Face model folder next to them.
:func:`export_checkpoint` folds the folder's config and tokenizer files into one
``.pt`` so inference runs offline from that single file; :func:`load_checkpoint`
and :func:`probabilities` use it.

Input text is :func:`transcript`, the same rule the lab scripts train on:
``utterances[].text`` only, whitespace collapsed, joined with one space.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FORMAT = "dcc-m3-bert-v1"
LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
MAX_LENGTH = 256
# Weights are stored in the state dict; these files are not copied into the export.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")
_SKIP_FILES = {"dcc_revision.json"}


def transcript(doc: Any) -> str:
    """Training text rule of the lab scripts; other JSON fields are never read."""

    if not isinstance(doc, dict):
        return ""
    utterances = doc.get("utterances", [])
    if not isinstance(utterances, list):
        return ""
    texts = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        text = utterance.get("text", "")
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            texts.append(text)
    return " ".join(texts)


def read_text(json_path: str | Path) -> str:
    with Path(json_path).open("r", encoding="utf-8-sig") as handle:
        return transcript(json.load(handle))


def _classifier(config: Any, n_labels: int, dropout: float = 0.1):
    import torch
    from transformers import AutoModel

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bert = AutoModel.from_config(config)
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(config.hidden_size, n_labels)

        def forward(self, ids, mask):
            hidden = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            return self.classifier(self.dropout(hidden))

    return Classifier()


def _text_files(model_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.name in _SKIP_FILES or path.suffix in _WEIGHT_SUFFIXES:
            continue
        try:
            files[path.name] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"tokenizer/config file is not text, cannot embed: {path}") from error
    if "config.json" not in files:
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    return files


def export_checkpoint(lab_ckpt: str | Path, model_dir: str | Path, output: str | Path) -> dict:
    """Write a self-contained inference checkpoint; returns its metadata (no tensors)."""

    import torch

    lab_ckpt, model_dir, output = Path(lab_ckpt), Path(model_dir), Path(output)
    source = torch.load(lab_ckpt, map_location="cpu", weights_only=False)
    config = source.get("config", {})
    labels = tuple(config.get("labels", ()))
    if labels != LABELS:
        raise ValueError(f"checkpoint labels {labels} != Mission 3 labels {LABELS}")
    if int(config.get("max_length", -1)) != MAX_LENGTH:
        raise ValueError(f"checkpoint max_length {config.get('max_length')} != {MAX_LENGTH}")
    files = _text_files(model_dir)
    digest = hashlib.sha256()
    with lab_ckpt.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    meta = dict(
        format=FORMAT,
        labels=list(LABELS),
        max_length=MAX_LENGTH,
        dropout=float(config.get("dropout", 0.1)),
        source=dict(exp_id=config.get("exp_id"), model=config.get("model"),
                    revision=config.get("revision"), seed=source.get("seed"),
                    epoch=source.get("epoch"), lab_ckpt_sha256=digest.hexdigest()),
    )
    state = {key: value.contiguous() for key, value in source["model"].items()}
    # Fail now, not at submission time, if the state dict does not fit the classifier.
    _build(files, meta).load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta, files=files, model=state), output)
    return meta


def _build(files: dict[str, str], meta: dict):
    from transformers import AutoConfig

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "config.json").write_text(files["config.json"], encoding="utf-8")
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
    return _classifier(config, len(meta["labels"]), meta.get("dropout", 0.1))


def is_bert_checkpoint(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}


def load_checkpoint(path: str | Path, device: str | None = None):
    """``(model, tokenizer, labels, max_length)`` from an exported checkpoint, offline."""

    import torch
    from transformers import AutoTokenizer

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} checkpoint; export the lab best.ckpt with "
            "scripts/mission3_export_bert_ckpt.py"
        )
    labels = [str(label) for label in payload["labels"]]
    if tuple(labels) != LABELS:
        raise ValueError(f"checkpoint labels {labels} != Mission 3 labels {list(LABELS)}")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder) / Path(name).name).write_text(text, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = _build(files, payload)
    model.load_state_dict(payload["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, labels, int(payload["max_length"])


def probabilities(model, tokenizer, texts: Sequence[str], max_length: int = MAX_LENGTH,
                  batch_size: int = 32) -> np.ndarray:
    """Sigmoid probabilities ``(len(texts), n_labels)`` in input order (fp32)."""

    import torch

    device = next(model.parameters()).device
    # Sort by length so each padded batch is short; results go back to input order.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    out = np.zeros((len(texts), model.classifier.out_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            encoded = tokenizer([texts[index] for index in chunk], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            out[chunk] = logits.float().sigmoid().cpu().numpy()
    return out
