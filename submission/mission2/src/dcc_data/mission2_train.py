"""Mission 2 baseline (L1) model: GBM over acoustic statistics.

Two splits are always computed and logged, never just one:

- A call_id-grouped split (``GroupShuffleSplit``), which is the split that
  actually matters: utterances from the same call never appear on both sides,
  so the model cannot "memorize this call's voice = this speaker".
- A plain random split, logged purely as a diagnostic. The gap between the
  two accuracies is itself evidence of how much speaker-memorization leakage
  would occur if a naive split were used -- this number is meant to go
  straight into the team's slide deck, not to pick a model.

Model persistence is deliberately isolated behind ``save_model``/``load_model``
so the on-disk format (currently pickle) can be swapped later without
touching the training or inference code -- the contest's submission format
requires ``.pt``/``.pth``/``.ckpt``, which a pickled GBM is not, and that
mismatch is a packaging decision for later, not something to solve here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .mission2_features import FEATURE_COLUMNS


def _build_model(seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbosity=-1)
    except ImportError:
        return HistGradientBoostingClassifier(random_state=seed)


@dataclass
class BaselineResult:
    model: Any
    feature_columns: tuple[str, ...]
    group_split_accuracy: float
    random_split_accuracy: float
    majority_class_accuracy: float
    majority_class: int
    confusion_matrix: list[list[int]]
    feature_importance_top20: list[tuple[str, float]]
    n_train_group: int
    n_val_group: int
    train_call_ids: int
    val_call_ids: int


def make_group_split(
    df: pd.DataFrame, *, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for call_id-grouped 80/20 split.

    Deterministic given the same ``df`` row order, ``seed`` and
    ``test_size`` -- callers outside training (e.g. an error-analysis
    script re-loading the same features parquet) can call this directly
    to reproduce exactly which rows were held out, instead of duplicating
    the GroupShuffleSplit call and risking it drifting out of sync with
    ``fit_baseline``.
    """

    if df["call_id"].nunique() < 2:
        raise ValueError("need at least 2 distinct call_id values for GroupShuffleSplit")
    groups = df["call_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # GroupShuffleSplit only uses len(X) and groups, never X's/y's values, so a
    # cheap placeholder array is exactly equivalent to passing the real X, y.
    placeholder = np.zeros(len(df))
    train_idx, val_idx = next(splitter.split(placeholder, placeholder, groups))
    return train_idx, val_idx


def fit_baseline(df: pd.DataFrame, *, seed: int = 42, test_size: float = 0.2) -> BaselineResult:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df["speaker"].to_numpy(dtype=int)
    groups = df["call_id"].to_numpy()

    majority_class = int(round(y.mean()))
    majority_class_accuracy = float(max(y.mean(), 1 - y.mean()))

    train_idx, val_idx = make_group_split(df, seed=seed, test_size=test_size)

    train_ids = set(groups[train_idx])
    val_ids = set(groups[val_idx])
    overlap = train_ids & val_ids
    if overlap:
        # This must never happen -- GroupShuffleSplit guarantees disjoint
        # groups. Asserted here (not just in tests) because a leak at this
        # exact point is the single most misleading failure mode: validation
        # accuracy would look great while being worthless.
        raise AssertionError(f"GroupShuffleSplit leaked {len(overlap)} call_id(s) across train/val")

    group_model = _build_model(seed)
    group_model.fit(X[train_idx], y[train_idx])
    group_predictions = group_model.predict(X[val_idx])
    group_split_accuracy = float(accuracy_score(y[val_idx], group_predictions))
    cm = confusion_matrix(y[val_idx], group_predictions, labels=[0, 1]).tolist()

    # Diagnostic only: an ordinary (non-grouped) split, to measure the
    # leakage gap. This model and split are never used for anything else.
    rx_train, rx_val, ry_train, ry_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    random_model = _build_model(seed)
    random_model.fit(rx_train, ry_train)
    random_split_accuracy = float(accuracy_score(ry_val, random_model.predict(rx_val)))

    importances = getattr(group_model, "feature_importances_", None)
    if importances is None:
        top20: list[tuple[str, float]] = []
    else:
        order = np.argsort(importances)[::-1][:20]
        top20 = [(FEATURE_COLUMNS[i], float(importances[i])) for i in order]

    return BaselineResult(
        model=group_model,
        feature_columns=FEATURE_COLUMNS,
        group_split_accuracy=group_split_accuracy,
        random_split_accuracy=random_split_accuracy,
        majority_class_accuracy=majority_class_accuracy,
        majority_class=majority_class,
        confusion_matrix=cm,
        feature_importance_top20=top20,
        n_train_group=len(train_idx),
        n_val_group=len(val_idx),
        train_call_ids=len(train_ids),
        val_call_ids=len(val_ids),
    )


def save_model(path: str | Path, model: Any, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump({"model": model, "meta": meta}, stream)


def load_model(path: str | Path) -> tuple[Any, dict]:
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    return payload["model"], payload["meta"]


def build_experiment_log(result: BaselineResult, *, seed: int, n_sampled_calls: int) -> dict:
    """Matches the M2-EXP-001 template. ``verdict`` is left for the real run."""

    return {
        "실험 ID": "M2-EXP-001",
        "가설": "발화 조각의 음향 통계 feature와 duration만으로 speaker를 유의미하게 분류할 수 있다",
        "baseline": 0.520,
        "변경 변수": "없음 (첫 baseline 수립)",
        "고정 변수": f"통화 {n_sampled_calls}건 샘플, seed={seed}, LightGBM(또는 HistGradientBoosting) 기본 파라미터",
        "평가 지표": "Accuracy",
        "validation": "call_id GroupShuffleSplit 8:2",
        "채택 기준": "GroupSplit accuracy >= 0.70",
        "기각 기준": "GroupSplit accuracy < 0.60 (-> L2 딥러닝 경로로 조기 전환)",
        "verdict": None,
        "measured": {
            "group_split_accuracy": result.group_split_accuracy,
            "random_split_accuracy": result.random_split_accuracy,
            "leakage_gap_random_minus_group": result.random_split_accuracy - result.group_split_accuracy,
            "majority_class_accuracy_this_sample": result.majority_class_accuracy,
            "n_train_utterances": result.n_train_group,
            "n_val_utterances": result.n_val_group,
            "n_train_calls": result.train_call_ids,
            "n_val_calls": result.val_call_ids,
            "confusion_matrix_val_group_split": result.confusion_matrix,
            "feature_importance_top20": result.feature_importance_top20,
        },
    }
