#!/usr/bin/env python3
"""Push 대원(class 0) recall up on the tuned LightGBM L1 model.

Two independent levers, both cheap to test against the same fixed
GroupShuffleSplit (seed=42, identical to every other L1 comparison run):

1. **Decision threshold sweep** (free -- one model fit, many thresholds):
   shift the predict_proba(class=1) cutoff away from 0.5 toward 신고자
   needing more confidence before winning. Only moves the decision boundary,
   doesn't change what the model learned.
2. **class_weight sweep** (costs one refit per weight): upweight class 0
   in the training loss itself, so the model is trained to care more about
   getting 대원 right, not just scored differently afterward.

Both report accuracy + per-class recall so the accuracy/fairness trade-off
is visible, not just whichever direction "looks better" in isolation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier, early_stopping  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402

from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402

TUNED_PARAMS = dict(num_leaves=255, learning_rate=0.02, n_estimators=1150)


def recalls(y_true, y_pred) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    r0 = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else float("nan")
    r1 = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else float("nan")
    return r0, r1


def threshold_sweep(model, X_val, y_val, thresholds) -> list[dict]:
    proba_1 = model.predict_proba(X_val)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba_1 >= t).astype(int)
        acc = float(accuracy_score(y_val, pred))
        r0, r1 = recalls(y_val, pred)
        rows.append(
            {"threshold": t, "accuracy": acc, "recall_daewon_0": r0, "recall_singoja_1": r1, "gap": r1 - r0}
        )
        print(f"  threshold={t:.2f}  acc={acc:.4f}  대원recall={r0:.4f}  신고자recall={r1:.4f}  격차={r1 - r0:+.4f}")
    return rows


def class_weight_sweep(X_train, y_train, X_val, y_val, seed, weights) -> list[dict]:
    rows = []
    for w in weights:
        class_weight = {0: w, 1: 1.0}
        start = time.monotonic()
        model = LGBMClassifier(random_state=seed, verbosity=-1, class_weight=class_weight, **TUNED_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        fit_seconds = time.monotonic() - start
        pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, pred))
        r0, r1 = recalls(y_val, pred)
        rows.append(
            {
                "daewon_weight": w,
                "accuracy": acc,
                "recall_daewon_0": r0,
                "recall_singoja_1": r1,
                "gap": r1 - r0,
                "best_iteration": model.best_iteration_,
                "fit_seconds": fit_seconds,
            }
        )
        print(
            f"  weight(대원)={w:.2f}  acc={acc:.4f}  대원recall={r0:.4f}  신고자recall={r1:.4f}  "
            f"격차={r1 - r0:+.4f}  ({fit_seconds:.1f}s)"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", type=Path, default=Path("data/local_cache/m2_features_train_local_full.parquet")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--log-output", type=Path, default=Path("reports/mission2/m2_l1_daewon_recall_push.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features)

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df["speaker"].to_numpy(dtype=int)
    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    cache_path = Path("checkpoints/mission2/m2_l1_tuned_base_model.pkl")
    if cache_path.exists():
        print(f"[1/2] Loading cached base tuned LightGBM from {cache_path}...")
        with cache_path.open("rb") as stream:
            base_model = pickle.load(stream)
    else:
        print("[1/2] Fitting base tuned LightGBM (no class weight) for threshold sweep...")
        base_model = LGBMClassifier(random_state=args.seed, verbosity=-1, **TUNED_PARAMS)
        base_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as stream:
            pickle.dump(base_model, stream)

    # NOTE on a bug in the first run of this script: thresholds *below* 0.5
    # were swept, i.e. "predict class 1 more easily" -- that pushes 신고자
    # recall up and 대원 recall DOWN, the opposite of this script's stated
    # goal. To raise 대원 recall via the threshold, class 1 needs to be
    # HARDER to predict, so thresholds above 0.5 are what's needed instead.
    print("Threshold sweep (same model, cutoff moved from 0.5 upward to favor 대원):")
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    threshold_results = threshold_sweep(base_model, X_val, y_val, thresholds)

    print("\n[2/2] class_weight sweep (retrains each time):")
    # Finer-grained near w=1.0-1.5, since the first run showed w=1.5 already
    # overshoots past equal recall (gap flips from +9.12%p at w=1.0 to
    # -7.68%p at w=1.5) -- the balanced point is somewhere in between.
    weights = [1.1, 1.2, 1.3]
    weight_results = class_weight_sweep(X_train, y_train, X_val, y_val, args.seed, weights)

    # Merge into the existing log (from the first run) instead of clobbering.
    existing: dict[str, list[dict]] = {"threshold_sweep": [], "class_weight_sweep": []}
    if args.log_output.exists():
        existing = json.loads(args.log_output.read_text(encoding="utf-8"))

    thresholds_by_key = {row["threshold"]: row for row in existing.get("threshold_sweep", [])}
    for row in threshold_results:
        thresholds_by_key[row["threshold"]] = row
    weights_by_key = {row["daewon_weight"]: row for row in existing.get("class_weight_sweep", [])}
    for row in weight_results:
        weights_by_key[row["daewon_weight"]] = row

    output = {
        "threshold_sweep": sorted(thresholds_by_key.values(), key=lambda r: r["threshold"]),
        "class_weight_sweep": sorted(weights_by_key.values(), key=lambda r: r["daewon_weight"]),
    }
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
