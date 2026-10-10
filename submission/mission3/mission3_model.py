"""Mission 3 model code: KcBERT multilabel classifier loaded from one offline checkpoint.

The checkpoint (.pt) holds the trained weights, the label list and the text of
config.json / tokenizer files, so nothing is downloaded at inference time.
Input is the call transcript only (``utterances[].text``).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FORMAT = "dcc-m3-bert-v1"
# The 9 evaluated symptoms, in the order the output cell is written.
LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
MAX_LENGTH = 256
THRESHOLD = 0.5  # fixed by the organizers; no per-class tuning or post-processing


def transcript(doc: Any) -> str:
    """Join ``utterances[].text`` (whitespace collapsed) with one space."""

    if not isinstance(doc, dict) or not isinstance(doc.get("utterances"), list):
        return ""
    texts = []
    for utterance in doc["utterances"]:
        text = utterance.get("text", "") if isinstance(utterance, dict) else ""
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            texts.append(text)
    return " ".join(texts)


def read_text(json_path: str | Path) -> str:
    with Path(json_path).open("r", encoding="utf-8-sig") as handle:
        return transcript(json.load(handle))


def _build(files: dict[str, str], dropout: float, n_labels: int):
    """BERT encoder + dropout + linear head on the first token."""

    import torch
    from transformers import AutoConfig, AutoModel

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "config.json").write_text(files["config.json"], encoding="utf-8")
        config = AutoConfig.from_pretrained(folder, local_files_only=True)

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


def load_checkpoint(path: str | Path, device: str | None = None):
    """``(model, tokenizer, labels, max_length)``, fully offline."""

    import torch
    from transformers import AutoTokenizer

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    labels = tuple(str(label) for label in payload["labels"])
    if labels != LABELS:
        raise ValueError(f"checkpoint labels {labels} != {LABELS}")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder) / Path(name).name).write_text(text, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = _build(files, float(payload.get("dropout", 0.1)), len(labels))
    model.load_state_dict(payload["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, list(labels), int(payload["max_length"])


def probabilities(model, tokenizer, texts: Sequence[str], max_length: int = MAX_LENGTH,
                  batch_size: int = 32) -> np.ndarray:
    """Sigmoid probabilities ``(len(texts), 9)`` in input order."""

    import torch

    device = next(model.parameters()).device
    # Sort by length so padded batches stay short; results are put back in input order.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            encoded = tokenizer([texts[index] for index in chunk], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            out[chunk] = logits.float().sigmoid().cpu().numpy()
    return out


def symptom_cell(binary_row: Sequence[int]) -> str:
    """``str(list)`` of the positive labels in LABELS order, e.g. ``['두통', '복통']``."""

    return str([label for label, flag in zip(LABELS, binary_row) if int(flag) == 1])
