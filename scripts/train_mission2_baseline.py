#!/usr/bin/env python3
"""Train the Mission 2 baseline (L1): GBM over 53-dim acoustic statistics.

Reads the feature parquet produced by scripts/build_mission2_dataset.py,
fits a call_id-grouped train/val split (the split that matters -- see
dcc_data.mission2_train for why a plain random split leaks) plus a plain
random split (diagnostic only, logged for comparison), and saves the fitted
model + metadata.

Usage:
    python scripts/train_mission2_baseline.py \\
        --features data/m2_features_train.parquet \\
        --model-output checkpoints/mission2/m2_baseline_lgbm.pkl \\
        --log-output reports/mission2/m2_exp_log.json

Note on checkpoint format: the contest's submission format requires
.pt/.pth/.ckpt, and a pickled GBM is not that. That packaging decision is
deferred (see dcc_data.mission2_train's module docstring); this script only
guarantees save_model/load_model stay swappable.
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

from dcc_data.mission2_train import build_experiment_log, fit_baseline, save_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("data/m2_features_train.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-output", type=Path, default=Path("checkpoints/mission2/m2_baseline_lgbm.pkl"))
    parser.add_argument("--log-output", type=Path, default=Path("reports/mission2/m2_exp_log.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features)

    result = fit_baseline(df, seed=args.seed, test_size=args.test_size)

    save_model(
        args.model_output,
        result.model,
        meta={
            "feature_columns": list(result.feature_columns),
            "seed": args.seed,
            "majority_class": result.majority_class,
        },
    )

    n_sampled_calls = int(df["call_id"].nunique())
    log = build_experiment_log(result, seed=args.seed, n_sampled_calls=n_sampled_calls)
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Majority-class accuracy (this sample): {result.majority_class_accuracy:.4f}")
    print(f"Random-split accuracy:                 {result.random_split_accuracy:.4f}")
    print(f"GroupShuffleSplit accuracy (call_id):   {result.group_split_accuracy:.4f}")
    print(f"Leakage gap (random - group):           {result.random_split_accuracy - result.group_split_accuracy:.4f}")
    print(f"Confusion matrix (val, group split):    {result.confusion_matrix}")
    print("Top feature importances:")
    for name, importance in result.feature_importance_top20[:10]:
        print(f"  {name}: {importance:.4f}")
    print(f"Model saved: {args.model_output}")
    print(f"Experiment log: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
