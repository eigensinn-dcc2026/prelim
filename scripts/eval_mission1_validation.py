#!/usr/bin/env python3
"""Evaluate Mission 1 checkpoints on the official Validation feature cache.

Fits nothing: scaler statistics and model weights stay frozen. This script is
the repeatable comparison entry point for model/hyperparameter selection.

Usage:
    python scripts/eval_mission1_validation.py \\
        --features data/m1_features_validation.parquet \\
        --gbm-ckpt checkpoints/mission1/m1_baseline_lgbm.pkl \\
        --lr-ckpt checkpoints/mission1/m1_baseline_lr.pkl \\
        --log-output reports/mission1/m1_val_eval.json
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

import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from dcc_data.guards import VALIDATION  # noqa: E402
from dcc_data.mission1_features import ACOUSTIC_FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission1_train import (  # noqa: E402
    INTERNAL_TARGET_ACCURACY,
    MAJORITY_BASELINE_ACCURACY,
    Mission1Fit,
    evaluate_frame,
    load_mission1_gbm,
    load_mission1_lr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/m1_features_validation.parquet"),
    )
    parser.add_argument(
        "--gbm-ckpt",
        type=Path,
        default=Path("checkpoints/mission1/m1_baseline_lgbm.pkl"),
    )
    parser.add_argument(
        "--lr-ckpt",
        type=Path,
        default=Path("checkpoints/mission1/m1_baseline_lr.pkl"),
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=Path("reports/mission1/m1_val_eval.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features)
    if "split" in df.columns:
        foreign = df.loc[df["split"] != VALIDATION, "split"]
        if len(foreign):
            raise ValueError(
                f"validation eval expected only Validation rows, found {foreign.unique().tolist()}"
            )

    gbm, gbm_meta = load_mission1_gbm(args.gbm_ckpt)
    lr_pipeline, lr_meta = load_mission1_lr(args.lr_ckpt)
    if tuple(gbm_meta["feature_columns"]) != ACOUSTIC_FEATURE_COLUMNS:
        raise ValueError("GBM checkpoint feature_columns do not match ACOUSTIC_FEATURE_COLUMNS")
    if tuple(lr_meta["feature_columns"]) != ACOUSTIC_FEATURE_COLUMNS:
        raise ValueError("LR checkpoint feature_columns do not match ACOUSTIC_FEATURE_COLUMNS")
    if not isinstance(lr_pipeline, Pipeline):
        raise TypeError("LR checkpoint is not a Pipeline; refusing to evaluate")

    fitted = Mission1Fit(
        gbm=gbm,
        lr_pipeline=lr_pipeline,
        feature_columns=ACOUSTIC_FEATURE_COLUMNS,
        majority_class=str(gbm_meta.get("majority_class", "F")),
        majority_class_accuracy=float(gbm_meta.get("majority_class_accuracy", 0.0)),
        n_train=int(gbm_meta.get("n_train", 0)),
    )
    measured = evaluate_frame(df, fitted)

    payload = {
        "실험 ID": "M1-EXP-001-val",
        "majority_baseline": MAJORITY_BASELINE_ACCURACY,
        "internal_target": INTERNAL_TARGET_ACCURACY,
        "n_validation": len(df),
        "n_train_checkpoint": fitted.n_train,
        "measured": measured,
    }
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Documented majority baseline (Seoul Training F): {MAJORITY_BASELINE_ACCURACY:.4f}")
    print(f"Validation majority ({measured['majority_class']}): {measured['majority_class_accuracy']:.4f}")
    print(f"Official Validation GBM accuracy: {measured['gbm']['accuracy']:.4f}")
    print(f"Official Validation LR accuracy:  {measured['lr']['accuracy']:.4f}")
    print(f"GBM recall F/M: {measured['gbm']['recall_F']:.4f}/{measured['gbm']['recall_M']:.4f}")
    print(f"LR recall F/M:  {measured['lr']['recall_F']:.4f}/{measured['lr']['recall_M']:.4f}")
    print(f"GBM confusion matrix: {measured['gbm']['confusion_matrix']}")
    print(f"LR confusion matrix:  {measured['lr']['confusion_matrix']}")
    print(f"Eval log: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
