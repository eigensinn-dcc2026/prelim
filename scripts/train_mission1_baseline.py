#!/usr/bin/env python3
"""Train the Mission 1 baseline: GBM and Logistic Regression over 52-dim acoustics.

Reads the Training feature parquet produced by scripts/build_mission1_dataset.py.
Runs an internal gender-stratified 8:2 sanity split, then refits both models on
all Training rows. Official Validation is not used here.

Usage:
    python scripts/train_mission1_baseline.py \\
        --features data/m1_features_train.parquet \\
        --gbm-output checkpoints/mission1/m1_baseline_lgbm.pkl \\
        --lr-output checkpoints/mission1/m1_baseline_lr.pkl \\
        --log-output reports/mission1/m1_exp_log.json
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

from dcc_data.mission1_train import (  # noqa: E402
    MAJORITY_BASELINE_ACCURACY,
    assert_training_features,
    build_experiment_log,
    feature_distribution_summary,
    fit_on_training,
    majority_from_frame,
    run_internal_sanity,
    save_mission1_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", type=Path, default=Path("data/m1_features_train.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--gbm-output",
        type=Path,
        default=Path("checkpoints/mission1/m1_baseline_lgbm.pkl"),
    )
    parser.add_argument(
        "--lr-output",
        type=Path,
        default=Path("checkpoints/mission1/m1_baseline_lr.pkl"),
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=Path("reports/mission1/m1_exp_log.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = assert_training_features(pd.read_parquet(args.features))

    sample_majority_class, sample_majority_accuracy = majority_from_frame(df)
    sanity = run_internal_sanity(df, seed=args.seed, test_size=args.test_size)
    fitted = fit_on_training(df, seed=args.seed)
    save_mission1_models(
        fitted,
        gbm_path=args.gbm_output,
        lr_path=args.lr_output,
        seed=args.seed,
    )

    log = build_experiment_log(
        seed=args.seed,
        n_train_calls=len(df),
        sample_majority_class=sample_majority_class,
        sample_majority_accuracy=sample_majority_accuracy,
        sanity=sanity,
        feature_distribution=feature_distribution_summary(df),
    )
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    holdout = sanity["holdout"]
    print(f"Documented majority baseline (Seoul Training F): {MAJORITY_BASELINE_ACCURACY:.4f}")
    print(f"Training-sample majority ({sample_majority_class}): {sample_majority_accuracy:.4f}")
    print(f"Internal sanity GBM accuracy: {holdout['gbm']['accuracy']:.4f}")
    print(f"Internal sanity LR accuracy:  {holdout['lr']['accuracy']:.4f}")
    print(f"Internal sanity GBM recall F/M: {holdout['gbm']['recall_F']:.4f}/{holdout['gbm']['recall_M']:.4f}")
    print(f"Internal sanity LR recall F/M:  {holdout['lr']['recall_F']:.4f}/{holdout['lr']['recall_M']:.4f}")
    print(f"GBM confusion matrix: {holdout['gbm']['confusion_matrix']}")
    print(f"LR confusion matrix:  {holdout['lr']['confusion_matrix']}")
    print(f"GBM saved: {args.gbm_output}")
    print(f"LR saved:  {args.lr_output}")
    print(f"Experiment log: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
