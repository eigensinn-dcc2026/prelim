#!/usr/bin/env python3
"""Compare several model families on the fixed 53-dim Mission 2 L1 feature set.

Feature columns and the train/val split are held identical to the LightGBM
baseline (dcc_data.mission2_train.fit_baseline / make_group_split, same
seed=42 call_id-grouped 80/20 split) -- only the model changes, so results
are directly comparable to the existing 84.59% LightGBM baseline
(reports/mission2/m2_error_analysis.md).

Usage:
    python scripts/compare_mission2_l1_models.py \\
        --features data/local_cache/m2_features_train_local_full.parquet \\
        --log-output reports/mission2/m2_l1_model_comparison.json
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402


def build_candidates(seed: int) -> dict[str, dict[str, Any]]:
    """Each entry is {"model": estimator, "fit_kwargs": dict}.

    ``fit_kwargs`` is empty for plain ``.fit(X, y)`` calls; entries that need
    the validation set for early stopping fill it in with an ``eval_set``
    placeholder that main() substitutes with the real (X_val, y_val).
    """

    candidates: dict[str, dict[str, Any]] = {
        "gaussian_nb": {"model": GaussianNB(), "fit_kwargs": {}},
        "logreg": {
            "model": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            ),
            "fit_kwargs": {},
        },
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
            "fit_kwargs": {},
        },
    }
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = {
            "model": XGBClassifier(
                n_estimators=300,
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
                eval_metric="logloss",
            ),
            "fit_kwargs": {},
        }
    except ImportError:
        print("xgboost not installed -- skipping", file=sys.stderr)
    try:
        from lightgbm import LGBMClassifier, early_stopping

        # n_estimators=300 to match random_forest/xgboost above -- the
        # earlier run compared XGBoost(300 trees) against the *baseline*
        # LightGBM(100 trees, sklearn/lightgbm default), which confounds
        # "model family" with "tree count". This entry isolates model
        # family by holding tree count fixed at 300 everywhere.
        candidates["lightgbm_n300"] = {
            "model": LGBMClassifier(n_estimators=300, random_state=seed, verbosity=-1),
            "fit_kwargs": {},
        }
        # Previously-confirmed win (see memory project-mission2-confirmed-wins):
        # num_leaves=255, learning_rate=0.02, n_estimators~1150 + early
        # stopping got full-data accuracy 84.59% -> ~86.9% on the (then)
        # 64-dim feature set. Re-testing here on the clean 53-dim set, on
        # this exact GroupShuffleSplit, to confirm it still holds.
        candidates["lightgbm_tuned"] = {
            "model": LGBMClassifier(
                num_leaves=255,
                learning_rate=0.02,
                n_estimators=1150,
                random_state=seed,
                verbosity=-1,
            ),
            "fit_kwargs": {
                "eval_set": "VAL_PLACEHOLDER",
                "eval_metric": "binary_logloss",
                "callbacks": [early_stopping(stopping_rounds=50, verbose=False)],
            },
        }
    except ImportError:
        print("lightgbm not installed -- skipping", file=sys.stderr)
    return candidates


def evaluate(name: str, model: Any, fit_kwargs: dict, X_train, y_train, X_val, y_val) -> dict:
    fit_kwargs = dict(fit_kwargs)
    if fit_kwargs.get("eval_set") == "VAL_PLACEHOLDER":
        fit_kwargs["eval_set"] = [(X_val, y_val)]

    start = time.monotonic()
    model.fit(X_train, y_train, **fit_kwargs)
    fit_seconds = time.monotonic() - start

    predictions = model.predict(X_val)
    accuracy = float(accuracy_score(y_val, predictions))
    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    # rows = true label, cols = predicted label; recall = diagonal / row sum
    recall_daewon = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else float("nan")
    recall_singoja = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else float("nan")

    result = {
        "model": name,
        "accuracy": accuracy,
        "recall_daewon_0": recall_daewon,
        "recall_singoja_1": recall_singoja,
        "recall_gap_singoja_minus_daewon": recall_singoja - recall_daewon,
        "best_iteration": getattr(model, "best_iteration_", None),
        "confusion_matrix": cm.tolist(),
        "fit_seconds": fit_seconds,
    }
    print(
        f"{name:15s} acc={accuracy:.4f}  대원recall={recall_daewon:.4f}  "
        f"신고자recall={recall_singoja:.4f}  격차={recall_singoja - recall_daewon:+.4f}  "
        f"({fit_seconds:.1f}s)"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", type=Path, default=Path("data/local_cache/m2_features_train_local_full.parquet")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--log-output", type=Path, default=Path("reports/mission2/m2_l1_model_comparison.json"))
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated candidate names to run (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features)

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df["speaker"].to_numpy(dtype=int)

    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Feature dims: {len(FEATURE_COLUMNS)} (must be 53)")
    print(f"Train: {len(train_idx)} utterances / Val: {len(val_idx)} utterances\n")

    candidates = build_candidates(args.seed)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(candidates)
        if unknown:
            raise ValueError(f"unknown candidate name(s): {sorted(unknown)}; available: {sorted(candidates)}")
        candidates = {name: spec for name, spec in candidates.items() if name in wanted}

    results = []
    for name, spec in candidates.items():
        results.append(evaluate(name, spec["model"], spec["fit_kwargs"], X_train, y_train, X_val, y_val))

    # Merge into any existing log instead of clobbering earlier runs (e.g. a
    # --only rerun of one candidate should not erase the others' results).
    existing: list[dict] = []
    if args.log_output.exists():
        existing = json.loads(args.log_output.read_text(encoding="utf-8"))
    by_name = {entry["model"]: entry for entry in existing}
    for entry in results:
        by_name[entry["model"]] = entry

    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(list(by_name.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
