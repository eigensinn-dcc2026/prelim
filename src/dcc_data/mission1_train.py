"""Mission 1 baseline models: Logistic Regression and GBM over 52-dim acoustics.

Training rows are the only rows allowed to ``fit``. Official Validation may
be used to compare models, but scaler statistics and classifier weights are
never learned from it.

The documented Seoul Training majority rate (F, 15,542 / 29,200 = 53.23%) is
recorded as a reference floor. 70% is an internal team target, not a
pass/fail gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .guards import TRAINING, SplitContaminationError
from .mission1_features import ACOUSTIC_FEATURE_COLUMNS
from .mission2_train import load_model, save_model

GENDER_CLASSES: tuple[str, str] = ("F", "M")
GENDER_TO_LABEL: dict[str, int] = {"F": 0, "M": 1}
LABEL_TO_GENDER: dict[int, str] = {0: "F", 1: "M"}
MAJORITY_BASELINE_ACCURACY = 15542 / 29200  # Seoul Training F rate, 53.23%
INTERNAL_TARGET_ACCURACY = 0.70


@dataclass
class Mission1Fit:
    gbm: Any
    lr_pipeline: Pipeline
    feature_columns: tuple[str, ...]
    majority_class: str
    majority_class_accuracy: float
    n_train: int


def assert_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fail if a feature table contains any non-Training split provenance."""

    if "split" not in df.columns:
        raise SplitContaminationError("feature table has no split provenance")
    foreign = df.loc[df["split"] != TRAINING, "split"]
    if len(foreign):
        examples = foreign.astype(str).head(10).tolist()
        raise SplitContaminationError(
            f"Training fit/split received non-Training split(s): {examples}"
        )
    return df


def encode_gender(values: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    encoded: list[int] = []
    for gender in values:
        if gender not in GENDER_TO_LABEL:
            raise ValueError(f"gender must be 'F' or 'M', got {gender!r}")
        encoded.append(GENDER_TO_LABEL[gender])
    return np.asarray(encoded, dtype=int)


def features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in ACOUSTIC_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    forbidden = {"startAt", "endAt", "start_at", "end_at", "duration_sec", "text", "symptom"}
    overlap = forbidden.intersection(df.columns)
    if overlap:
        raise ValueError(f"disallowed columns present in feature table: {sorted(overlap)}")
    X = df[list(ACOUSTIC_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = encode_gender(df["gender"])
    return X, y


def majority_from_frame(df: pd.DataFrame) -> tuple[str, float]:
    counts = df["gender"].value_counts()
    majority_class = str(counts.idxmax())
    majority_accuracy = float(counts.max() / len(df))
    return majority_class, majority_accuracy


def stratified_sanity_split(
    df: pd.DataFrame, *, seed: int = 42, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training-internal 8:2 split. Not the official Validation set."""

    assert_training_features(df)
    if df["gender"].nunique() < 2:
        raise ValueError("need both F and M labels for a stratified sanity split")
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["gender"],
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def _build_gbm(seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed)


def _build_lr_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = [GENDER_TO_LABEL["F"], GENDER_TO_LABEL["M"]]
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix_labels": list(GENDER_CLASSES),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "recall_F": float(recalls[0]),
        "recall_M": float(recalls[1]),
        "n_samples": int(len(y_true)),
        "n_F": int(np.sum(y_true == GENDER_TO_LABEL["F"])),
        "n_M": int(np.sum(y_true == GENDER_TO_LABEL["M"])),
    }


def feature_distribution_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in ACOUSTIC_FEATURE_COLUMNS:
        values = df[column].to_numpy(dtype=np.float64)
        summary[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def fit_on_training(df: pd.DataFrame, *, seed: int = 42) -> Mission1Fit:
    """Fit GBM and StandardScaler+LR on Training rows only."""

    assert_training_features(df)
    X, y = features_and_labels(df)
    majority_class, majority_accuracy = majority_from_frame(df)

    gbm = _build_gbm(seed)
    gbm.fit(X, y)
    lr_pipeline = _build_lr_pipeline(seed)
    lr_pipeline.fit(X, y)

    return Mission1Fit(
        gbm=gbm,
        lr_pipeline=lr_pipeline,
        feature_columns=ACOUSTIC_FEATURE_COLUMNS,
        majority_class=majority_class,
        majority_class_accuracy=majority_accuracy,
        n_train=len(df),
    )


def evaluate_frame(df: pd.DataFrame, fitted: Mission1Fit) -> dict[str, object]:
    """Predict only. Never fits the scaler or either classifier."""

    X, y = features_and_labels(df)
    majority_class, majority_accuracy = majority_from_frame(df)
    return {
        "majority_class": majority_class,
        "majority_class_accuracy": majority_accuracy,
        "gbm": compute_metrics(y, fitted.gbm.predict(X)),
        "lr": compute_metrics(y, fitted.lr_pipeline.predict(X)),
    }


def run_internal_sanity(
    df: pd.DataFrame, *, seed: int = 42, test_size: float = 0.2
) -> dict[str, object]:
    """Fit copies on an internal Training split. Saved models are not these copies."""

    train_df, holdout_df = stratified_sanity_split(df, seed=seed, test_size=test_size)
    fitted = fit_on_training(train_df, seed=seed)
    return {
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "holdout": evaluate_frame(holdout_df, fitted),
    }


def checkpoint_meta(fitted: Mission1Fit, *, seed: int) -> dict[str, object]:
    return {
        "feature_columns": list(fitted.feature_columns),
        "gender_to_label": dict(GENDER_TO_LABEL),
        "seed": seed,
        "majority_class": fitted.majority_class,
        "majority_class_accuracy": fitted.majority_class_accuracy,
        "n_train": fitted.n_train,
    }


def save_mission1_models(
    fitted: Mission1Fit,
    *,
    gbm_path: str | Path,
    lr_path: str | Path,
    seed: int,
) -> None:
    meta = checkpoint_meta(fitted, seed=seed)
    save_model(gbm_path, fitted.gbm, meta)
    save_model(lr_path, fitted.lr_pipeline, meta)


def load_mission1_gbm(path: str | Path) -> tuple[Any, dict]:
    return load_model(path)


def load_mission1_lr(path: str | Path) -> tuple[Pipeline, dict]:
    model, meta = load_model(path)
    if not isinstance(model, Pipeline):
        raise TypeError("expected a scikit-learn Pipeline for the LR checkpoint")
    return model, meta


def build_experiment_log(
    *,
    seed: int,
    n_train_calls: int,
    sample_majority_class: str,
    sample_majority_accuracy: float,
    sanity: dict[str, object],
    feature_distribution: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "실험 ID": "M1-EXP-001",
        "가설": "통화 단위 caller waveform의 52차원 음향 통계만으로 gender(F/M)를 분류할 수 있다",
        "majority_baseline": MAJORITY_BASELINE_ACCURACY,
        "majority_baseline_note": "Seoul Training F 15,542 / 29,200 = 53.23%. 반드시 기록하는 바닥값.",
        "internal_target": INTERNAL_TARGET_ACCURACY,
        "internal_target_note": (
            "팀 내부 1차 목표. 미달 시 즉시 파이프라인 오류로 단정하지 않고 "
            "confusion matrix, F/M recall, feature 분포를 확인한 뒤 다음 실험으로 간다."
        ),
        "변경 변수": "없음 (첫 baseline 수립)",
        "고정 변수": (
            f"Seoul Training {n_train_calls}건, seed={seed}, "
            "52-d acoustic_feature_vector, duration_sec 제외, "
            "LightGBM(또는 HistGradientBoosting) vs StandardScaler+LogisticRegression"
        ),
        "평가 지표": "Accuracy (보조: confusion matrix, F/M recall)",
        "validation": (
            "Training 내부 gender stratified 8:2는 sanity check. "
            "공식 Validation은 모델 비교·선택에 반복 사용 가능. "
            "Validation으로 fit하거나 scaler 통계를 학습하지 않음."
        ),
        "verdict": None,
        "measured": {
            "training_sample_majority_class": sample_majority_class,
            "training_sample_majority_accuracy": sample_majority_accuracy,
            "internal_sanity": sanity,
            "feature_distribution_training": feature_distribution,
        },
    }
