#!/usr/bin/env python3
"""Evaluate the saved L1 model checkpoint against the official Validation set.

The model was trained/early-stopped entirely on Training data (see
scripts/train_mission2_baseline.py) -- this is the first time it sees
official Validation, purely for a final, no-peeking accuracy check. Nothing
here re-trains or re-tunes the model.

Usage:
    python scripts/evaluate_mission2_l1_on_official_validation.py \\
        --model checkpoints/mission2/m2_baseline_lgbm_tuned.pkl \\
        --features data/local_cache/m2_features_validation_official.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402

from dcc_data.mission2_train import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/mission2/m2_baseline_lgbm_tuned.pkl"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/local_cache/m2_features_validation_official.parquet")
    )
    parser.add_argument(
        "--log-output", type=Path, default=Path("reports/mission2/m2_l1_official_validation_result.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, meta = load_model(args.model)
    feature_columns = meta["feature_columns"]

    df = pd.read_parquet(args.features)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation features missing columns the model expects: {missing}")

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df["speaker"].to_numpy(dtype=int)

    predictions = model.predict(X)
    accuracy = float(accuracy_score(y, predictions))
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    recall_daewon = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else float("nan")
    recall_singoja = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else float("nan")

    result = {
        "n_utterances": len(df),
        "n_calls": int(df["call_id"].nunique()),
        "accuracy": accuracy,
        "recall_daewon_0": recall_daewon,
        "recall_singoja_1": recall_singoja,
        "recall_gap_singoja_minus_daewon": recall_singoja - recall_daewon,
        "confusion_matrix": cm.tolist(),
        "model_checkpoint": str(args.model),
        "model_seed": meta.get("seed"),
    }

    print(f"Official Validation: {result['n_calls']} calls / {result['n_utterances']} utterances")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"대원(0) recall:  {recall_daewon:.4f}")
    print(f"신고자(1) recall: {recall_singoja:.4f}")
    print(f"격차:            {recall_singoja - recall_daewon:+.4f}")
    print(f"Confusion matrix: {cm.tolist()}")

    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
