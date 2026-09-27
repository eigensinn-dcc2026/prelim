"""Rule-aware experiment helpers for Yeonwoo's MFCC40 + LR reference.

This module is deliberately separate from ``mission2_features`` and
``mission2_train``.  Those modules implement the team's other 53-feature GBM
pipeline; importing their feature/model contract here would invalidate a
single-factor comparison against the verified MFCC40 Logistic Regression.

The functions below do not start any large run on import.  Loading caches,
fitting models, and evaluating predictions only happen when a caller invokes
the corresponding function explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


ANNOTATION_TIME_UNIT = "milliseconds"
SPEAKER_LABEL_MAPPING = {0: "119_dispatcher", 1: "caller"}
BASELINE_CACHE_SCHEMA_VERSION = "mission2_mfcc_mean_std_v1"
BASELINE_CACHE_TAG = "mfcc20_nfft512_hop160_v1"
EXPECTED_BASELINE_CODE_SHA256 = (
    "8a628a44801a3925360f91adf4b03b3f3e7dbdd9eb7ec20ccf0670340d6ecd13"
)
MFCC_CONFIG = {
    "sample_rate": 8_000,
    "n_mfcc": 20,
    "n_fft": 512,
    "hop_length": 160,
}
BASELINE_FEATURE_NAMES = tuple(
    [f"mfcc_{index:02d}_mean" for index in range(1, 21)]
    + [f"mfcc_{index:02d}_std" for index in range(1, 21)]
)
REFERENCE_MODEL_CONFIG = {
    "solver": "lbfgs",
    "max_iter": 1_000,
    "class_weight": None,
    "random_state": 42,
}
EXPECTED_CALL_COUNTS = {"Training": 29_200, "Validation": 3_640}
EXPECTED_UTTERANCE_COUNTS = {"Training": 889_358, "Validation": 111_947}
EXPECTED_CACHE_METADATA = {
    split: {
        "cache_schema_version": BASELINE_CACHE_SCHEMA_VERSION,
        "source_split": split,
        "source_call_count": EXPECTED_CALL_COUNTS[split],
        "mfcc_config": MFCC_CONFIG,
        "feature_dimension": 40,
        "utterance_count": EXPECTED_UTTERANCE_COUNTS[split],
    }
    for split in ("Training", "Validation")
}
REFERENCE_CONFUSION_MATRIX = np.array(
    [[41_190, 12_655], [8_715, 49_387]], dtype=np.int64
)
REFERENCE_ACCURACY = 90_577 / 111_947
REFERENCE_RECALL_0 = 41_190 / 53_845
REFERENCE_RECALL_1 = 49_387 / 58_102
HIGH_CONFIDENCE_THRESHOLD = 0.90

FORBIDDEN_MODEL_FEATURES = frozenset(
    {
        "text",
        "speaker",
        "label",
        "target",
        "call_id",
        "record_key",
        "utterance_id",
        "utt_idx",
        "utterance_index",
        "annotation_row",
        "startat",
        "endat",
        "startat_ms",
        "endat_ms",
        "start_at",
        "end_at",
        "start_at_ms",
        "end_at_ms",
        "start_ms",
        "end_ms",
        "timestamp",
        "absolute_timestamp",
        "crop_start_frame",
        "crop_end_frame",
        "source_position",
        "id",
        "filename",
        "file_path",
        "audio_path",
        "json_path",
        "stem",
        "split",
        "overlap",
        "overlap_status",
        "overlap_partner_count",
        "maximum_overlap_ratio",
        "quality_status",
        "quality_flags",
    }
)

EXPERIMENT_TABLE_COLUMNS = [
    "experiment_id",
    "baseline_or_candidate",
    "added_feature",
    "train_rows",
    "validation_rows",
    "feature_dim",
    "accuracy",
    "accuracy_delta",
    "recall_0",
    "recall_1",
    "recall_0_delta",
    "recall_1_delta",
    "high_conf_error_count",
    "overlap_accuracy",
    "short_accuracy",
    "rule_compliance",
    "notes",
]


@dataclass(frozen=True)
class AlignedSplit:
    """MFCC matrix and tracking metadata aligned by stable record key."""

    split: str
    X_mfcc: np.ndarray
    y: np.ndarray
    record_keys: np.ndarray
    tracking: pd.DataFrame


@dataclass
class ExperimentResult:
    """Observed outputs from one explicit Logistic Regression run."""

    experiment_id: str
    feature_names: tuple[str, ...]
    train_shape: tuple[int, int]
    evaluation_shape: tuple[int, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    scaler: StandardScaler
    model: LogisticRegression
    metrics: dict[str, Any]
    fit_audit: dict[str, Any]
    contract_passed: bool = False


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], frame_name: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"{frame_name} missing columns: {missing}")


def make_record_key(split: object, call_id: object, annotation_row: object) -> str:
    """Return ``split|call_id|zero-based JSON array position``."""

    split_text = str(split)
    call_text = str(call_id)
    if "|" in split_text or "|" in call_text:
        raise AssertionError("record_key components must not contain '|'")
    if not isinstance(annotation_row, Integral) or isinstance(
        annotation_row, (bool, np.bool_)
    ):
        raise AssertionError("annotation_row must be an integer")
    position = int(annotation_row)
    if position < 0:
        raise AssertionError("annotation_row must be non-negative")
    return f"{split_text}|{call_text}|{position}"


def duration_ms_from_bounds(
    start_at_ms: Sequence[float] | np.ndarray,
    end_at_ms: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Calculate duration in the source annotation's confirmed ms unit."""

    start = np.asarray(start_at_ms, dtype=np.float64)
    end = np.asarray(end_at_ms, dtype=np.float64)
    if start.shape != end.shape:
        raise ValueError(f"startAt_ms/endAt_ms shape mismatch: {start.shape} vs {end.shape}")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("startAt_ms/endAt_ms contains NaN or inf")
    if np.any(start < 0):
        raise ValueError("startAt_ms must be non-negative")
    if np.any(end <= start):
        raise ValueError("endAt_ms must be greater than startAt_ms")
    return end - start


def validate_model_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Reject labels, identifiers, absolute timestamps, order, and slice metadata."""

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("at least one model feature is required")
    if len(set(names)) != len(names):
        raise ValueError("model feature names must be unique")
    for name in names:
        if name.casefold() in FORBIDDEN_MODEL_FEATURES:
            raise ValueError(f"forbidden model feature: {name}")
    return names


def validate_cache_metadata(metadata: Mapping[str, Any], split: str) -> None:
    if split not in EXPECTED_CACHE_METADATA:
        raise ValueError(f"unknown split: {split}")
    expected = EXPECTED_CACHE_METADATA[split]
    if dict(metadata) != expected:
        raise AssertionError(
            f"cache metadata mismatch for {split}: actual={dict(metadata)!r}, expected={expected!r}"
        )


def notebook_code_sha256(path: Path) -> str:
    """Hash code-cell source only, matching the frozen Error Analysis gate."""

    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    code_sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    payload = "\n\n# %%\n\n".join(code_sources).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def frozen_cache_paths(cache_root: Path, split: str) -> tuple[Path, Path]:
    prefix = Path(cache_root) / f"{split.lower()}_{BASELINE_CACHE_TAG}"
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def load_frozen_mfcc_cache(cache_root: Path, split: str) -> dict[str, np.ndarray]:
    """Load only an exact compatible cache; never trigger extraction."""

    npz_path, metadata_path = frozen_cache_paths(cache_root, split)
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing frozen cache for {split}; automatic extraction is disabled: "
            f"{npz_path} / {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_cache_metadata(metadata, split)
    with np.load(npz_path, allow_pickle=False) as cache:
        bundle = {name: cache[name] for name in cache.files}
    expected_arrays = {"X", "y", "stem", "start_at", "end_at", "duration"}
    if set(bundle) != expected_arrays:
        raise AssertionError(
            f"{split} cache arrays mismatch: {sorted(bundle)} vs {sorted(expected_arrays)}"
        )
    expected_rows = EXPECTED_UTTERANCE_COUNTS[split]
    if bundle["X"].shape != (expected_rows, len(BASELINE_FEATURE_NAMES)):
        raise AssertionError(f"invalid {split} MFCC shape: {bundle['X'].shape}")
    if any(len(bundle[name]) != expected_rows for name in expected_arrays - {"X"}):
        raise AssertionError(f"{split} cache array length mismatch")
    if not np.isfinite(bundle["X"]).all():
        raise AssertionError(f"{split} MFCC cache contains NaN or inf")
    if set(np.unique(bundle["y"]).tolist()) != {0, 1}:
        raise AssertionError(f"{split} label mapping must be exactly {{0, 1}}")
    return bundle


def align_frozen_cache(
    bundle: Mapping[str, np.ndarray],
    manifest: pd.DataFrame,
    split: str,
    *,
    expected_rows: int | None = None,
) -> AlignedSplit:
    """Align a frozen cache to the Raw EDA manifest and verify every row field."""

    expected_rows = (
        EXPECTED_UTTERANCE_COUNTS[split] if expected_rows is None else expected_rows
    )
    required_arrays = {"X", "y", "stem", "start_at", "end_at", "duration"}
    missing_arrays = sorted(required_arrays - set(bundle))
    if missing_arrays:
        raise AssertionError(f"{split} cache missing arrays: {missing_arrays}")
    require_columns(
        manifest,
        [
            "record_key",
            "split",
            "call_id",
            "annotation_row",
            "speaker",
            "startAt_ms",
            "endAt_ms",
            "duration_sec",
        ],
        "utterance manifest",
    )
    split_manifest = manifest.loc[manifest["split"] == split].copy()
    if len(split_manifest) != expected_rows:
        raise AssertionError(
            f"{split} manifest row mismatch: {len(split_manifest)} vs {expected_rows}"
        )
    if not split_manifest["record_key"].is_unique:
        raise AssertionError(f"{split} manifest record_key is not unique")

    cache_rows = pd.DataFrame(
        {
            "cache_position": np.arange(expected_rows, dtype=np.int64),
            "call_id_cache": np.asarray(bundle["stem"]).astype(str),
            "startAt_ms_cache": np.asarray(bundle["start_at"], dtype=np.float64),
            "endAt_ms_cache": np.asarray(bundle["end_at"], dtype=np.float64),
            "duration_ms_cache": np.asarray(bundle["duration"], dtype=np.float64),
            "speaker_cache": np.asarray(bundle["y"], dtype=np.int64),
        }
    )
    if len(cache_rows) != expected_rows or np.asarray(bundle["X"]).shape != (
        expected_rows,
        len(BASELINE_FEATURE_NAMES),
    ):
        raise AssertionError(f"{split} cache shape/row mismatch")
    cache_rows["annotation_row_cache"] = cache_rows.groupby(
        "call_id_cache", sort=False
    ).cumcount()
    cache_rows["record_key"] = [
        make_record_key(split, call_id, annotation_row)
        for call_id, annotation_row in cache_rows[
            ["call_id_cache", "annotation_row_cache"]
        ].itertuples(index=False, name=None)
    ]
    if not cache_rows["record_key"].is_unique:
        raise AssertionError(f"{split} cache-derived record_key is not unique")

    joined = cache_rows.merge(
        split_manifest,
        on="record_key",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(joined) != expected_rows or not (joined["_merge"] == "both").all():
        missing = int((joined["_merge"] != "both").sum())
        raise AssertionError(
            f"{split} cache/manifest alignment failed: rows={len(joined)}, missing={missing}"
        )
    joined = joined.sort_values("cache_position", kind="stable").reset_index(drop=True)
    if not (joined["call_id_cache"] == joined["call_id"].astype(str)).all():
        raise AssertionError(f"{split} call_id mismatch")
    np.testing.assert_array_equal(
        joined["annotation_row_cache"].to_numpy(dtype=np.int64),
        joined["annotation_row"].to_numpy(dtype=np.int64),
    )
    np.testing.assert_array_equal(
        joined["speaker_cache"].to_numpy(dtype=np.int64),
        joined["speaker"].to_numpy(dtype=np.int64),
    )
    np.testing.assert_allclose(
        joined["startAt_ms_cache"], joined["startAt_ms"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        joined["endAt_ms_cache"], joined["endAt_ms"], rtol=0, atol=0
    )
    duration_ms = duration_ms_from_bounds(
        joined["startAt_ms"], joined["endAt_ms"]
    )
    np.testing.assert_allclose(
        joined["duration_ms_cache"], duration_ms, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        joined["duration_sec"].to_numpy(dtype=float),
        duration_ms / 1_000.0,
        rtol=0,
        atol=1e-12,
    )

    tracking_columns = [
        "record_key",
        "split",
        "call_id",
        "annotation_row",
        "startAt_ms",
        "endAt_ms",
        "duration_sec",
    ]
    optional_tracking = [
        "audio_path",
        "json_path",
        "crop_start_frame",
        "crop_end_frame",
        "crop_ok",
        "boundary_status",
    ]
    tracking = joined[
        tracking_columns
        + [column for column in optional_tracking if column in joined.columns]
    ].copy()
    tracking["duration_ms"] = duration_ms
    tracking = tracking[
        tracking_columns[:6]
        + ["duration_ms", "duration_sec"]
        + [column for column in optional_tracking if column in tracking.columns]
    ]
    return AlignedSplit(
        split=split,
        X_mfcc=np.asarray(bundle["X"], dtype=np.float32),
        y=np.asarray(bundle["y"], dtype=np.int64),
        record_keys=joined["record_key"].astype(str).to_numpy(),
        tracking=tracking.reset_index(drop=True),
    )


def build_duration_feature_frame(tracking: pd.DataFrame) -> pd.DataFrame:
    """Return only ``record_key`` and direct ``endAt_ms - startAt_ms`` duration."""

    require_columns(
        tracking, ["record_key", "startAt_ms", "endAt_ms"], "tracking"
    )
    if not tracking["record_key"].is_unique:
        raise AssertionError("tracking record_key is not unique")
    duration_ms = duration_ms_from_bounds(
        tracking["startAt_ms"], tracking["endAt_ms"]
    )
    if "duration_ms" in tracking.columns:
        np.testing.assert_allclose(
            tracking["duration_ms"].to_numpy(dtype=float),
            duration_ms,
            rtol=0,
            atol=0,
        )
    return pd.DataFrame(
        {"record_key": tracking["record_key"].astype(str), "duration_ms": duration_ms}
    )


def assemble_experiment_matrix(
    aligned: AlignedSplit,
    extra_features: pd.DataFrame | None,
    extra_feature_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Join extra values by record key and append only named legal columns."""

    extra_names = tuple(str(name) for name in extra_feature_names)
    feature_names = validate_model_feature_names((*BASELINE_FEATURE_NAMES, *extra_names))
    base = np.asarray(aligned.X_mfcc, dtype=np.float32)
    if base.shape != (len(aligned.y), len(BASELINE_FEATURE_NAMES)):
        raise AssertionError(f"invalid MFCC40 matrix shape: {base.shape}")
    if not extra_names:
        if extra_features is not None:
            raise ValueError("extra_features must be None when no extra feature is selected")
        return base.copy(), feature_names
    if extra_features is None:
        raise ValueError("extra_features is required for selected extra feature names")
    extra_matrix = align_extra_feature_matrix(
        aligned, extra_features, extra_names
    )
    matrix = np.column_stack([base, extra_matrix]).astype(np.float32, copy=False)
    if matrix.shape != (len(aligned.y), len(feature_names)):
        raise AssertionError(f"invalid experiment matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise AssertionError("experiment matrix contains NaN or inf")
    return matrix, feature_names


def align_extra_feature_matrix(
    aligned: AlignedSplit,
    extra_features: pd.DataFrame,
    extra_feature_names: Sequence[str],
) -> np.ndarray:
    """Align selected extra columns once without copying the MFCC40 matrix."""

    extra_names = validate_model_feature_names(extra_feature_names)
    require_columns(extra_features, ["record_key", *extra_names], "extra features")
    if not extra_features["record_key"].is_unique:
        raise AssertionError("extra feature record_key is not unique")

    key_frame = pd.DataFrame(
        {
            "row_position": np.arange(len(aligned.record_keys), dtype=np.int64),
            "record_key": np.asarray(aligned.record_keys).astype(str),
        }
    )
    joined = key_frame.merge(
        extra_features[["record_key", *extra_names]].assign(
            record_key=lambda frame: frame["record_key"].astype(str)
        ),
        on="record_key",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(joined) != len(key_frame) or not (joined["_merge"] == "both").all():
        missing = int((joined["_merge"] != "both").sum())
        raise AssertionError(
            f"baseline/candidate row alignment failed: rows={len(joined)}, missing={missing}"
        )
    joined = joined.sort_values("row_position", kind="stable")
    extra_matrix = joined[list(extra_names)].to_numpy(dtype=np.float32)
    if extra_matrix.shape != (len(aligned.y), len(extra_names)):
        raise AssertionError(f"invalid aligned extra feature shape: {extra_matrix.shape}")
    if not np.isfinite(extra_matrix).all():
        raise AssertionError("aligned extra features contain NaN or inf")
    return extra_matrix


def assert_single_factor_comparison(
    baseline_X: np.ndarray,
    candidate_X: np.ndarray,
    baseline_y: np.ndarray,
    candidate_y: np.ndarray,
    baseline_feature_names: Sequence[str],
    candidate_feature_names: Sequence[str],
    expected_added_features: Sequence[str],
) -> None:
    """Prove that the candidate changes only the declared appended columns."""

    baseline_names = tuple(baseline_feature_names)
    candidate_names = tuple(candidate_feature_names)
    added_names = tuple(expected_added_features)
    if baseline_names != BASELINE_FEATURE_NAMES:
        raise AssertionError("baseline feature names do not match frozen MFCC40")
    if candidate_names != (*baseline_names, *added_names):
        raise AssertionError("candidate feature names are not baseline + declared features")
    baseline_X = np.asarray(baseline_X)
    candidate_X = np.asarray(candidate_X)
    if candidate_X.shape != (baseline_X.shape[0], baseline_X.shape[1] + len(added_names)):
        raise AssertionError("candidate shape is not baseline + declared features")
    try:
        np.testing.assert_array_equal(candidate_X[:, : baseline_X.shape[1]], baseline_X)
    except AssertionError as exc:
        raise AssertionError("MFCC40 values changed in candidate comparison") from exc
    np.testing.assert_array_equal(np.asarray(candidate_y), np.asarray(baseline_y))
    validate_model_feature_names(candidate_names)


def group_split_indices(
    call_ids: Sequence[object] | np.ndarray,
    *,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create the repository's existing call-level GroupShuffleSplit."""

    groups = np.asarray(call_ids).astype(str)
    if len(groups) == 0 or len(np.unique(groups)) < 2:
        raise ValueError("at least two distinct call_id values are required")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    positions = np.arange(len(groups), dtype=np.int64)
    train_idx, validation_idx = next(splitter.split(positions, groups=groups))
    train_calls = set(groups[train_idx])
    validation_calls = set(groups[validation_idx])
    overlap = train_calls & validation_calls
    if overlap:
        raise AssertionError(f"GroupShuffleSplit leaked {len(overlap)} call_id values")
    audit = {
        "seed": seed,
        "test_size": test_size,
        "train_rows": len(train_idx),
        "internal_validation_rows": len(validation_idx),
        "train_calls": len(train_calls),
        "internal_validation_calls": len(validation_calls),
        "call_id_overlap": len(overlap),
    }
    return train_idx, validation_idx, audit


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise AssertionError(f"prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not set(np.unique(y_true)).issubset({0, 1}) or not set(
        np.unique(y_pred)
    ).issubset({0, 1}):
        raise AssertionError("labels/predictions must be 0 or 1")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    recalls = recall_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "recall_0": float(recalls[0]),
        "recall_1": float(recalls[1]),
    }


def fit_reference_logistic_regression(
    *,
    experiment_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    feature_names: Sequence[str],
    training_scope: str,
    evaluation_scope: str,
) -> ExperimentResult:
    """Fit the frozen scaler/LR policy with no evaluation-side fit path."""

    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    if not training_scope.startswith("Training"):
        raise AssertionError("scaler/model fit scope must be Training")
    if evaluation_scope == training_scope:
        raise AssertionError("evaluation scope must be distinct from fit scope")
    names = validate_model_feature_names(feature_names)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_evaluation = np.asarray(X_evaluation, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_evaluation = np.asarray(y_evaluation, dtype=np.int64)
    if X_train.shape != (len(y_train), len(names)):
        raise AssertionError(f"invalid Training matrix shape: {X_train.shape}")
    if X_evaluation.shape != (len(y_evaluation), len(names)):
        raise AssertionError(f"invalid evaluation matrix shape: {X_evaluation.shape}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_evaluation).all():
        raise AssertionError("Training/evaluation matrix contains NaN or inf")
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise AssertionError("Training labels must contain exactly classes 0 and 1")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_evaluation_scaled = scaler.transform(X_evaluation)
    if X_train_scaled.dtype != np.float32 or X_evaluation_scaled.dtype != np.float32:
        raise AssertionError("reference scaler/model input must remain float32")
    model = LogisticRegression(**REFERENCE_MODEL_CONFIG)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_evaluation_scaled)
    probabilities = model.predict_proba(X_evaluation_scaled)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1], dtype=np.int64))
    if probabilities.shape != (len(y_evaluation), 2):
        raise AssertionError(f"invalid probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise AssertionError("probabilities contain NaN or inf")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-7)
    metrics = prediction_metrics(y_evaluation, predictions)
    fit_audit = {
        "scaler_fit_scope": training_scope,
        "model_fit_scope": training_scope,
        "evaluation_scope": evaluation_scope,
        "scaler_fit_rows": len(y_train),
        "model_fit_rows": len(y_train),
        "evaluation_rows": len(y_evaluation),
    }
    return ExperimentResult(
        experiment_id=experiment_id,
        feature_names=names,
        train_shape=tuple(X_train.shape),
        evaluation_shape=tuple(X_evaluation.shape),
        y_true=y_evaluation,
        y_pred=np.asarray(predictions, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        scaler=scaler,
        model=model,
        metrics=metrics,
        fit_audit=fit_audit,
    )


def run_baseline_reproduction(
    training: AlignedSplit,
    validation: AlignedSplit,
    *,
    experiment_id: str = "M2-REFERENCE-MFCC40-LR",
) -> ExperimentResult:
    """Re-fit the reference and open the downstream gate only on exact reproduction."""

    if training.split != "Training" or validation.split != "Validation":
        raise AssertionError("baseline reproduction requires Training -> Validation")
    if training.X_mfcc.shape != (
        EXPECTED_UTTERANCE_COUNTS["Training"],
        40,
    ) or validation.X_mfcc.shape != (
        EXPECTED_UTTERANCE_COUNTS["Validation"],
        40,
    ):
        raise AssertionError("baseline reproduction row/dimension contract mismatch")
    if not all(str(key).startswith("Training|") for key in training.record_keys):
        raise AssertionError("Training record_key source prefix mismatch")
    if not all(str(key).startswith("Validation|") for key in validation.record_keys):
        raise AssertionError("Validation record_key source prefix mismatch")
    result = fit_reference_logistic_regression(
        experiment_id=experiment_id,
        X_train=training.X_mfcc,
        y_train=training.y,
        X_evaluation=validation.X_mfcc,
        y_evaluation=validation.y,
        feature_names=BASELINE_FEATURE_NAMES,
        training_scope="Training",
        evaluation_scope="Validation",
    )
    result.fit_audit["training_record_keys"] = training.record_keys.copy()
    result.fit_audit["validation_record_keys"] = validation.record_keys.copy()
    expected_fit_audit = {
        "scaler_fit_scope": "Training",
        "model_fit_scope": "Training",
        "evaluation_scope": "Validation",
        "scaler_fit_rows": EXPECTED_UTTERANCE_COUNTS["Training"],
        "model_fit_rows": EXPECTED_UTTERANCE_COUNTS["Training"],
        "evaluation_rows": EXPECTED_UTTERANCE_COUNTS["Validation"],
    }
    for audit_name, expected_value in expected_fit_audit.items():
        if result.fit_audit[audit_name] != expected_value:
            raise AssertionError(
                f"baseline fit audit mismatch for {audit_name}: "
                f"{result.fit_audit[audit_name]!r} vs {expected_value!r}"
            )
    model_params = result.model.get_params()
    for parameter, expected_value in REFERENCE_MODEL_CONFIG.items():
        if model_params[parameter] != expected_value:
            raise AssertionError(
                f"baseline Logistic Regression parameter changed: {parameter}"
            )
    np.testing.assert_array_equal(
        np.asarray(result.metrics["confusion_matrix"], dtype=np.int64),
        REFERENCE_CONFUSION_MATRIX,
    )
    expected_rates = {
        "accuracy": REFERENCE_ACCURACY,
        "recall_0": REFERENCE_RECALL_0,
        "recall_1": REFERENCE_RECALL_1,
    }
    for metric_name, expected_value in expected_rates.items():
        if not np.isclose(result.metrics[metric_name], expected_value):
            raise AssertionError(
                f"baseline {metric_name} mismatch: "
                f"actual={result.metrics[metric_name]!r}, expected={expected_value!r}"
            )
    return replace(result, contract_passed=True)


def add_overlap_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Add evaluation-only overlap labels from half-open annotation intervals."""

    require_columns(
        frame,
        ["record_key", "call_id", "startAt_ms", "endAt_ms"],
        "overlap frame",
    )
    result = frame.reset_index(drop=True).copy()
    partner_count = np.zeros(len(result), dtype=np.int64)
    for _, group in result.groupby("call_id", sort=False):
        positions = sorted(
            group.index,
            key=lambda position: (
                float(result.at[position, "startAt_ms"]),
                float(result.at[position, "endAt_ms"]),
                str(result.at[position, "record_key"]),
            ),
        )
        for left_offset, left_position in enumerate(positions):
            left_start = float(result.at[left_position, "startAt_ms"])
            left_end = float(result.at[left_position, "endAt_ms"])
            for right_position in positions[left_offset + 1 :]:
                right_start = float(result.at[right_position, "startAt_ms"])
                if right_start >= left_end:
                    break
                right_end = float(result.at[right_position, "endAt_ms"])
                if min(left_end, right_end) > max(left_start, right_start):
                    partner_count[left_position] += 1
                    partner_count[right_position] += 1
    result["overlap_partner_count"] = partner_count
    result["overlap_status"] = np.where(
        partner_count > 0, "overlap", "no_overlap"
    )
    return result


def _slice_metrics(frame: pd.DataFrame, slice_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_value, group in frame.groupby(
        slice_column, observed=True, dropna=False
    ):
        metrics = prediction_metrics(group["y_true"], group["y_pred"])
        rows.append({"slice": str(slice_value), **metrics})
    return pd.DataFrame(rows)


def build_experiment_report(
    result: ExperimentResult,
    *,
    training_tracking: pd.DataFrame,
    evaluation_tracking: pd.DataFrame,
    baseline_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build global and slice metrics from this run's observed predictions."""

    require_columns(
        training_tracking, ["record_key", "duration_ms"], "Training tracking"
    )
    require_columns(
        evaluation_tracking,
        ["record_key", "call_id", "startAt_ms", "endAt_ms", "duration_ms"],
        "evaluation tracking",
    )
    if len(evaluation_tracking) != len(result.y_true):
        raise AssertionError("evaluation tracking/prediction row mismatch")
    if "training_record_keys" in result.fit_audit:
        np.testing.assert_array_equal(
            training_tracking["record_key"].astype(str).to_numpy(),
            np.asarray(result.fit_audit["training_record_keys"]).astype(str),
        )
    if "validation_record_keys" in result.fit_audit:
        np.testing.assert_array_equal(
            evaluation_tracking["record_key"].astype(str).to_numpy(),
            np.asarray(result.fit_audit["validation_record_keys"]).astype(str),
        )
    probabilities = np.asarray(result.probabilities, dtype=float)
    evaluation = evaluation_tracking.reset_index(drop=True).copy()
    evaluation["y_true"] = result.y_true
    evaluation["y_pred"] = result.y_pred
    evaluation["confidence"] = probabilities.max(axis=1)
    evaluation["correct"] = result.y_true == result.y_pred
    evaluation = add_overlap_metadata(evaluation)

    duration_quantiles = training_tracking["duration_ms"].astype(float).quantile(
        [0.05, 0.25, 0.75, 0.95]
    )
    evaluation["duration_slice"] = pd.cut(
        evaluation["duration_ms"],
        bins=[-np.inf, *duration_quantiles.tolist(), np.inf],
        labels=["very_short", "short", "medium", "long", "very_long"],
        include_lowest=True,
    )
    evaluation["confidence_slice"] = pd.cut(
        evaluation["confidence"],
        bins=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, np.inf],
        labels=[
            "0.50-0.60",
            "0.60-0.70",
            "0.70-0.80",
            "0.80-0.90",
            "0.90-0.95",
            "0.95-0.99",
            "0.99-1.00",
        ],
        include_lowest=True,
        right=False,
    )
    duration_summary = _slice_metrics(evaluation, "duration_slice")
    overlap_summary = _slice_metrics(evaluation, "overlap_status")
    confidence_summary = _slice_metrics(evaluation, "confidence_slice")
    global_metrics = result.metrics
    high_confidence_error_count = int(
        ((~evaluation["correct"]) & (evaluation["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)).sum()
    )

    if baseline_report is None:
        accuracy_delta = 0.0 if result.feature_names == BASELINE_FEATURE_NAMES else np.nan
        recall_0_delta = 0.0 if result.feature_names == BASELINE_FEATURE_NAMES else np.nan
        recall_1_delta = 0.0 if result.feature_names == BASELINE_FEATURE_NAMES else np.nan
        high_confidence_error_delta = (
            0 if result.feature_names == BASELINE_FEATURE_NAMES else np.nan
        )
    else:
        baseline_global = baseline_report["global_metrics"]
        accuracy_delta = global_metrics["accuracy"] - baseline_global["accuracy"]
        recall_0_delta = global_metrics["recall_0"] - baseline_global["recall_0"]
        recall_1_delta = global_metrics["recall_1"] - baseline_global["recall_1"]
        high_confidence_error_delta = (
            high_confidence_error_count
            - int(baseline_report["high_confidence_error_count"])
        )

    added_features = (
        result.feature_names[len(BASELINE_FEATURE_NAMES) :]
        if result.feature_names[: len(BASELINE_FEATURE_NAMES)]
        == BASELINE_FEATURE_NAMES
        else result.feature_names
    )
    return {
        "experiment_id": result.experiment_id,
        "feature_names": result.feature_names,
        "added_features": added_features,
        "train_shape": result.train_shape,
        "validation_shape": result.evaluation_shape,
        "global_metrics": global_metrics,
        "accuracy_delta": accuracy_delta,
        "recall_0_delta": recall_0_delta,
        "recall_1_delta": recall_1_delta,
        "high_confidence_error_count": high_confidence_error_count,
        "high_confidence_error_delta": high_confidence_error_delta,
        "duration_summary": duration_summary,
        "overlap_summary": overlap_summary,
        "confidence_summary": confidence_summary,
        "duration_cutoffs_ms": duration_quantiles,
        "fit_audit": result.fit_audit,
        "rule_compliance": "validated_by_runtime_guards",
        "notes": "",
    }


def run_feature_experiment(
    *,
    experiment_id: str,
    training: AlignedSplit,
    validation: AlignedSplit,
    extra_training: pd.DataFrame,
    extra_validation: pd.DataFrame,
    extra_feature_names: Sequence[str],
    baseline_result: ExperimentResult,
) -> ExperimentResult:
    """Run one selected feature/group against the exact open baseline gate."""

    if not baseline_result.contract_passed:
        raise RuntimeError("reference baseline reproduction gate did not pass")
    if baseline_result.feature_names != BASELINE_FEATURE_NAMES:
        raise AssertionError("baseline gate was not produced by the MFCC40 contract")
    if baseline_result.train_shape != training.X_mfcc.shape:
        raise AssertionError("candidate Training rows differ from the reproduced baseline")
    if baseline_result.evaluation_shape != validation.X_mfcc.shape:
        raise AssertionError("candidate Validation rows differ from the reproduced baseline")
    if training.split != "Training" or validation.split != "Validation":
        raise AssertionError("official experiment requires Training -> Validation")
    X_train, feature_names = assemble_experiment_matrix(
        training, extra_training, extra_feature_names
    )
    X_validation, validation_feature_names = assemble_experiment_matrix(
        validation, extra_validation, extra_feature_names
    )
    if feature_names != validation_feature_names:
        raise AssertionError("Training/Validation feature names differ")
    assert_single_factor_comparison(
        training.X_mfcc,
        X_train,
        training.y,
        training.y,
        BASELINE_FEATURE_NAMES,
        feature_names,
        extra_feature_names,
    )
    assert_single_factor_comparison(
        validation.X_mfcc,
        X_validation,
        validation.y,
        validation.y,
        BASELINE_FEATURE_NAMES,
        feature_names,
        extra_feature_names,
    )
    np.testing.assert_array_equal(
        training.record_keys,
        baseline_result.fit_audit["training_record_keys"],
    )
    np.testing.assert_array_equal(
        validation.record_keys,
        baseline_result.fit_audit["validation_record_keys"],
    )
    result = fit_reference_logistic_regression(
        experiment_id=experiment_id,
        X_train=X_train,
        y_train=training.y,
        X_evaluation=X_validation,
        y_evaluation=validation.y,
        feature_names=feature_names,
        training_scope="Training",
        evaluation_scope="Validation",
    )
    result.fit_audit["training_record_keys"] = training.record_keys.copy()
    result.fit_audit["validation_record_keys"] = validation.record_keys.copy()
    return result


def print_experiment_report(report: Mapping[str, Any]) -> None:
    """Print the required comparison outputs without deciding experiment success."""

    metrics = report["global_metrics"]
    print("Experiment ID:", report["experiment_id"])
    print("Feature names:", list(report["feature_names"]))
    print("Feature dimension:", len(report["feature_names"]))
    print("Training shape:", report["train_shape"])
    print("Validation shape:", report["validation_shape"])
    print("Accuracy:", metrics["accuracy"])
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("speaker=0 Recall:", metrics["recall_0"])
    print("speaker=1 Recall:", metrics["recall_1"])
    print("Accuracy delta vs baseline:", report["accuracy_delta"])
    print("Recall 0 delta vs baseline:", report["recall_0_delta"])
    print("Recall 1 delta vs baseline:", report["recall_1_delta"])
    print("High-confidence errors:", report["high_confidence_error_count"])
    print("High-confidence error delta:", report["high_confidence_error_delta"])


def empty_experiment_table() -> pd.DataFrame:
    """Return an unpopulated schema; no unexecuted metric is fabricated."""

    return pd.DataFrame(columns=EXPERIMENT_TABLE_COLUMNS)


def _slice_accuracy(summary: pd.DataFrame, slice_name: str) -> float:
    matched = summary.loc[summary["slice"] == slice_name, "accuracy"]
    return float(matched.iloc[0]) if len(matched) == 1 else np.nan


def append_comparison_row(
    table: pd.DataFrame,
    report: Mapping[str, Any],
    *,
    notes: str = "",
) -> pd.DataFrame:
    """Append one actually observed report to the fixed comparison schema."""

    if list(table.columns) != EXPERIMENT_TABLE_COLUMNS:
        raise AssertionError("experiment comparison table schema changed")
    experiment_id = str(report["experiment_id"])
    if experiment_id in set(table["experiment_id"].astype(str)):
        raise AssertionError(f"duplicate experiment_id: {experiment_id}")
    metrics = report["global_metrics"]
    added = tuple(report["added_features"])
    is_reference = tuple(report["feature_names"]) == BASELINE_FEATURE_NAMES
    row = {
        "experiment_id": experiment_id,
        "baseline_or_candidate": "baseline" if is_reference else "candidate",
        "added_feature": ", ".join(added),
        "train_rows": int(report["train_shape"][0]),
        "validation_rows": int(report["validation_shape"][0]),
        "feature_dim": int(report["validation_shape"][1]),
        "accuracy": float(metrics["accuracy"]),
        "accuracy_delta": report["accuracy_delta"],
        "recall_0": float(metrics["recall_0"]),
        "recall_1": float(metrics["recall_1"]),
        "recall_0_delta": report["recall_0_delta"],
        "recall_1_delta": report["recall_1_delta"],
        "high_conf_error_count": int(report["high_confidence_error_count"]),
        "overlap_accuracy": _slice_accuracy(report["overlap_summary"], "overlap"),
        "short_accuracy": _slice_accuracy(report["duration_summary"], "short"),
        "rule_compliance": report["rule_compliance"],
        "notes": notes,
    }
    return pd.concat([table, pd.DataFrame([row])], ignore_index=True)
