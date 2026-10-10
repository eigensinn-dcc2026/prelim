"""Mission 3 module: 9-way multilabel symptoms from the call transcript (owner: DCC M3).

Two checkpoint kinds, chosen by file suffix:

- ``.pt`` / ``.pth`` / ``.ckpt``: a BERT-family classifier exported by
  ``scripts/mission3_export_bert_ckpt.py`` (:mod:`dcc_inference.mission3_bert`,
  lab training text rule, runs offline from that one file).
- anything else (``.joblib``): the P0 TF-IDF baseline below.

P0 wraps the frozen ``scripts/mission3_baseline 최종 수정본.py`` (do not edit it).
Text is that script's ``read_one_json`` (utf-8-sig, ``transcript_from_json``);
probabilities are ``vectorizer.transform`` + ``classifier.predict_proba`` on the
joblib dict that script saves. ``audio_dir`` is accepted for the shared
signature and is not read.

Decisions are ``probability >= 0.5`` per label (:func:`binarize`) and nothing
else: a row with no label at or above 0.5 is written as ``[]``. The organizers
ruled out the earlier min-1 rule (argmax on empty rows) on 2026-10-09.
The symptom cell is :func:`format_symptom_cell`: Python ``str(list)`` with
ASCII single quotes, labels in ``TARGET_SYMPTOMS`` order
(``docs/submission_format.md``, #58). The quote style will be re-confirmed
with the organizers at the 10-10 mentoring.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dcc_inference import mission3_bert

COLUMNS = ("label file name", "symptom")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "mission3_baseline 최종 수정본.py"

# Final representative seed; self-contained offline export (not committed).
CKPT_FILENAME = "m3_roberta003_seed42.pt"
DEFAULT_CKPT = REPOSITORY_ROOT / "checkpoints" / "mission3" / CKPT_FILENAME

# Fixed by the competition rules. Not the checkpoint's stored threshold.
THRESHOLD = 0.5

# Symptom cell format: docs/submission_format.md (#58, team decision 2026-10-06).
# Python ``str(list)`` -- ASCII single quotes, one space after each comma -- with labels
# in ``TARGET_SYMPTOMS`` order; no labels is ``[]``.
# ``inference.py`` writes the CSV with pandas' default QUOTE_MINIMAL, so a multi-label
# cell appears in the file as ``"['두통', '복통']"``. The official PDF p.9 example uses
# curly quotes; the quote style will be re-confirmed with the organizers at the 10-10 mentoring.
SYMPTOM_CELL_EXAMPLE = "['두통', '복통']"


def _script() -> ModuleType:
    """The frozen P0 baseline, imported once."""

    module = sys.modules.get("mission3_baseline_p0")
    if module is not None and Path(getattr(module, "__file__", "")).resolve() == SCRIPT_PATH.resolve():
        return module
    spec = importlib.util.spec_from_file_location("mission3_baseline_p0", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mission3_baseline_p0"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("mission3_baseline_p0", None)
        raise
    return module


def binarize(probabilities: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Binary labels at ``>= threshold``; ``threshold`` itself is positive.

    No post-processing: a row that is all below ``threshold`` stays all 0
    (organizer ruling 2026-10-09, min-1 not allowed). ``probabilities`` is not
    modified.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError(
            f"probabilities must have shape (n_rows, n_labels), got {probabilities.shape}"
        )
    return (probabilities >= threshold).astype(np.int8)


def format_symptom_cell(labels: Sequence[str]) -> str:
    """Symptom cell text: ``str(list)`` of the labels in ``TARGET_SYMPTOMS`` order.

    ``["복통", "두통"]`` -> ``"['두통', '복통']"``; no labels -> ``"[]"``.
    ``ast.literal_eval`` of the cell gives the ordered list back. Labels outside
    the nine P0 ``TARGET_SYMPTOMS`` and repeated labels raise ``ValueError``.
    Format: ``docs/submission_format.md`` (#58). The quote style will be
    re-confirmed with the organizers at the 10-10 mentoring.
    """

    order = [str(label) for label in _script().TARGET_SYMPTOMS]
    names = [str(label) for label in labels]
    unknown = sorted(set(names) - set(order))
    if unknown:
        raise ValueError(f"not a Mission 3 target symptom: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError(f"repeated symptom labels: {names}")
    chosen = set(names)
    return str([label for label in order if label in chosen])


def positive_labels(binary_row: Sequence[int], symptom_names: Sequence[str]) -> list[str]:
    """Symptom names whose flag is 1, in ``symptom_names`` order."""

    if len(binary_row) != len(symptom_names):
        raise ValueError(
            f"binary row length {len(binary_row)} != {len(symptom_names)} symptoms"
        )
    return [symptom_names[index] for index, flag in enumerate(binary_row) if int(flag) == 1]


def load_checkpoint(ckpt_path: str | Path) -> tuple[Any, Any, list[str]]:
    """Load the P0 joblib dict (vectorizer, classifier, target_symptoms)."""

    import joblib

    payload = joblib.load(ckpt_path)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(payload).__name__}")
    required = {"vectorizer", "classifier", "target_symptoms"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"checkpoint missing {sorted(missing)}")
    symptoms = [str(label) for label in payload["target_symptoms"]]
    expected = [str(label) for label in _script().TARGET_SYMPTOMS]
    if symptoms != expected:
        raise ValueError(
            "checkpoint target_symptoms do not match P0 TARGET_SYMPTOMS "
            f"(got {symptoms})"
        )
    return payload["vectorizer"], payload["classifier"], symptoms


def _probabilities(vectorizer: Any, classifier: Any, texts: list[str], n_labels: int) -> np.ndarray:
    features = vectorizer.transform(texts)
    probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
    expected = (len(texts), n_labels)
    if probabilities.shape != expected:
        raise RuntimeError(
            f"predict_proba shape {probabilities.shape} != {expected} "
            "(rows, P0 target symptoms)"
        )
    return probabilities


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path | None = None) -> pd.DataFrame:
    """One official Mission 3 row per label JSON, in ``sorted(rglob('*.json'))`` order."""

    del audio_dir  # transcript only; kept so the signature matches registry.load_predict
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir}")
    json_paths = sorted(label_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no JSON files under {label_dir}")

    if mission3_bert.is_bert_checkpoint(ckpt_path):
        # BERT-family export (scripts/mission3_export_bert_ckpt.py): lab training text rule.
        names = [json_path.name for json_path in json_paths]
        model, tokenizer, symptoms, max_length = mission3_bert.load_checkpoint(ckpt_path)
        if getattr(model, "requires_documents", False):
            from .mission3_context import probabilities as context_probabilities
            documents = [json.loads(path.read_text(encoding="utf-8-sig")) for path in json_paths]
            probabilities = context_probabilities(model, tokenizer, documents, names, max_length)
        else:
            texts = [mission3_bert.read_text(json_path) for json_path in json_paths]
            probabilities = mission3_bert.probabilities(model, tokenizer, texts, max_length)
    else:
        script = _script()
        names = []
        texts = []
        for json_path in json_paths:
            record = script.read_one_json(json_path)
            names.append(str(record["label_file_name"]))
            texts.append(str(record["text"]))
        vectorizer, classifier, symptoms = load_checkpoint(ckpt_path)
        probabilities = _probabilities(vectorizer, classifier, texts, len(symptoms))
    binary = binarize(probabilities, threshold=THRESHOLD)
    cells = [
        format_symptom_cell(positive_labels(row, symptoms))
        for row in binary
    ]
    print(f"[mission3] rows={len(names)} ckpt={ckpt_path.name}", file=sys.stderr)
    return pd.DataFrame(list(zip(names, cells)), columns=list(COLUMNS), dtype=object)
