"""Score a Mission 3 submission CSV against the label JSONs (Macro F1 over 9 symptoms).

    python scripts/mission3_score_csv.py --csv outputs/mission3.csv --label_dir <json folder> \
        --json_output reports/mission3/<name>.json

Gold labels are each JSON's ``symptom`` list restricted to the nine targets. Every
JSON under ``--label_dir`` must have exactly one CSV row and vice versa. An empty
prediction ``[]`` is a valid row. Scoring only; it never changes predictions.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dcc_inference.mission3_bert import LABELS  # noqa: E402


def gold_labels(label_dir: Path) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    for path in sorted(label_dir.rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        symptoms = doc.get("symptom", [])
        symptoms = [symptoms] if isinstance(symptoms, str) else symptoms
        names = {unicodedata.normalize("NFC", s.strip()) for s in symptoms if isinstance(s, str)}
        if path.name in gold:
            raise ValueError(f"duplicate label file name: {path.name}")
        gold[path.name] = names & set(LABELS)
    return gold


def score(csv_path: Path, label_dir: Path) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    frame = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
    if list(frame.columns) != ["label file name", "symptom"]:
        raise ValueError(f"unexpected columns {list(frame.columns)}")
    gold = gold_labels(label_dir)
    predicted = {}
    for name, cell in zip(frame["label file name"], frame["symptom"]):
        labels = ast.literal_eval(cell)
        if not isinstance(labels, list) or set(labels) - set(LABELS):
            raise ValueError(f"bad symptom cell for {name}: {cell!r}")
        if name in predicted:
            raise ValueError(f"duplicate CSV row: {name}")
        predicted[name] = set(labels)
    if set(predicted) != set(gold):
        raise ValueError(f"CSV rows and label JSONs differ: {len(set(predicted) ^ set(gold))} names")
    names = sorted(gold)
    truth = np.array([[label in gold[n] for label in LABELS] for n in names], dtype=int)
    pred = np.array([[label in predicted[n] for label in LABELS] for n in names], dtype=int)
    per = f1_score(truth, pred, average=None, zero_division=0)
    return dict(
        rows=len(names),
        macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
        micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
        macro_precision=float(precision_score(truth, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(truth, pred, average="macro", zero_division=0)),
        empty_predictions=int((pred.sum(axis=1) == 0).sum()),
        per_symptom_f1={label: float(value) for label, value in zip(LABELS, per)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--json_output", type=Path)
    args = parser.parse_args(argv)
    result = score(args.csv, args.label_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
