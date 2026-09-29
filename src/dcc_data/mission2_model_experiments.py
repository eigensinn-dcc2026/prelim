"""Fair M00 versus M01 model-comparison contracts for DCC Mission 2.

This module can build the aligned 42D E05 matrix directly from the immutable
MFCC40, candidate-feature, and manifest caches already present in the
repository workspace.  It does not import the feature-experiment branch and
never falls back to waveform extraction.  That keeps Checkpoints A--H runnable
without PR #16 and prevents the earlier 53D GBM pipeline from becoming an
accidental dependency.

No function in this module starts feature extraction.  Model fitting also
requires an explicit runtime gate so importing the module is always cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


MFCC40_FEATURE_NAMES = tuple(
    [f"mfcc_{index:02d}_mean" for index in range(1, 21)]
    + [f"mfcc_{index:02d}_std" for index in range(1, 21)]
)
E05_EXTRA_FEATURE_NAMES = ("spectral_rolloff", "zcr")
E05_FEATURE_NAMES = MFCC40_FEATURE_NAMES + E05_EXTRA_FEATURE_NAMES
CACHE_BACKEND = "standalone_adapter"

MFCC_CACHE_SCHEMA = "mission2_mfcc_mean_std_v1"
CANDIDATE_CACHE_SCHEMA = "mission2_candidate_audio_features_v1"
MANIFEST_CACHE_SCHEMA = "mission2_raw_eda_v1"
MFCC_CONFIG: Mapping[str, int] = MappingProxyType({
    "sample_rate": 8_000,
    "n_mfcc": 20,
    "n_fft": 512,
    "hop_length": 160,
})

M00_LOGISTIC_REGRESSION_CONFIG: Mapping[str, Any] = MappingProxyType({
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": 42,
})
M01_HGB_CONFIG: Mapping[str, Any] = MappingProxyType({
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
})
M02_EXTRATREES_CONFIG: Mapping[str, Any] = MappingProxyType({
    "n_estimators": 128,
    "criterion": "gini",
    "max_depth": 18,
    "min_samples_split": 2,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "bootstrap": False,
    "class_weight": None,
    "n_jobs": -1,
    "random_state": 42,
})
M03_QDA_CONFIG: Mapping[str, Any] = MappingProxyType({
    "priors": None,
    "reg_param": 0.0,
    "store_covariance": False,
    "tol": 0.0001,
})
M04_MLP_CONFIG: Mapping[str, Any] = MappingProxyType({
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 1024,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_iter": 30,
    "shuffle": True,
    "random_state": 42,
    "tol": 0.0001,
    "early_stopping": False,
    "n_iter_no_change": 10,
})
M04_CONVERGENCE_CHECK_ID = "M04_ITER60"
M04_CONVERGENCE_CHECK_CONFIG: Mapping[str, Any] = MappingProxyType({
    **dict(M04_MLP_CONFIG),
    "max_iter": 60,
})
SCREENING_MODEL_IDS = ("M02", "M03", "M04")
SCREENING_MODEL_NAMES: Mapping[str, str] = MappingProxyType({
    "M02": "ExtraTreesClassifier",
    "M03": "QuadraticDiscriminantAnalysis",
    "M04": "MLPClassifier",
})
SCREENING_MODEL_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "M02": M02_EXTRATREES_CONFIG,
    "M03": M03_QDA_CONFIG,
    "M04": M04_MLP_CONFIG,
})

EXPECTED_INTERNAL_SPLIT: Mapping[str, int] = MappingProxyType({
    "total_rows": 889_358,
    "train_rows": 711_467,
    "validation_rows": 177_891,
    "train_calls": 23_360,
    "validation_calls": 5_840,
    "call_overlap": 0,
})
EXPECTED_OFFICIAL_ROWS: Mapping[str, int] = MappingProxyType({
    "Training": 889_358,
    "Validation": 111_947,
})
EXPECTED_OFFICIAL_CALLS: Mapping[str, int] = MappingProxyType({
    "Training": 29_200,
    "Validation": 3_640,
})

FROZEN_E05_OFFICIAL_REFERENCE: Mapping[str, Any] = MappingProxyType({
    "experiment_id": "E05",
    "training_rows": 889_358,
    "accuracy": 0.8131169214003054,
    "confusion_matrix": ((41_697, 12_148), (8_773, 49_329)),
    "recall_speaker_0": 0.7743894512025258,
    "recall_speaker_1": 0.8490069188668204,
    "correct_count": 91_026,
    "high_confidence_error_count": 2_158,
    "validation_rows": 111_947,
})

DECISION_TABLE_COLUMNS = (
    "experiment_id",
    "model_name",
    "model_config",
    "feature_dimension",
    "train_rows",
    "evaluation_rows",
    "accuracy",
    "confusion_matrix",
    "correct_count",
    "recall_speaker_0",
    "recall_speaker_1",
    "high_confidence_error_count",
    "duration_summary",
    "overlap_accuracy",
    "no_overlap_accuracy",
    "confidence_summary",
    "runtime_seconds",
    "peak_memory_bytes",
    "notes",
)


@dataclass(frozen=True)
class ValidatedE05Data:
    """An immutable view of a row-aligned E05 matrix."""

    X: np.ndarray
    y: np.ndarray
    record_keys: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class StandaloneCachePaths:
    """Exact read-only cache inputs required by the standalone adapter."""

    mfcc_npz: Path
    mfcc_metadata: Path
    candidate_parquet: Path | tuple[Path, ...]
    candidate_metadata: Path | tuple[Path, ...]
    manifest_parquet: Path
    manifest_metadata: Path


@dataclass(frozen=True)
class StandaloneE05Cache:
    """Validated E05 matrix plus non-model tracking fields and cache audit."""

    validated: ValidatedE05Data
    tracking: pd.DataFrame
    X_mfcc: np.ndarray
    aligned_extra_features: np.ndarray
    cache_audit: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedComparison:
    """One shared scaler/split payload consumed by both M00 and M01."""

    feature_names: tuple[str, ...]
    X_train_scaled: np.ndarray
    X_evaluation_scaled: np.ndarray
    y_train: np.ndarray
    y_evaluation: np.ndarray
    training_record_keys: np.ndarray
    evaluation_record_keys: np.ndarray
    scaler_fit_record_keys: np.ndarray
    train_idx: np.ndarray
    validation_idx: np.ndarray
    scaler: StandardScaler
    scaler_fit_scope: str
    training_scope: str
    evaluation_scope: str
    split_audit: Mapping[str, Any]


@dataclass(frozen=True)
class ModelExperimentResult:
    """Observed predictions plus the complete model/scaler audit trail."""

    experiment_id: str
    model_name: str
    model_config: Mapping[str, Any]
    feature_names: tuple[str, ...]
    train_shape: tuple[int, int]
    evaluation_shape: tuple[int, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    metrics: Mapping[str, Any]
    scaler: StandardScaler
    model: Any
    fit_audit: Mapping[str, Any]


@dataclass(frozen=True)
class PromotionSelection:
    """User-approved candidate IDs frozen before any promoted Official run."""

    model_ids: tuple[str, ...]
    internal_result_fingerprints: Mapping[str, str]
    selection_fingerprint: str


SplitFunction = Callable[..., tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]


def standalone_training_cache_paths(repo_root: str | Path) -> StandaloneCachePaths:
    """Resolve the frozen Training cache names without discovering alternatives."""

    root = Path(repo_root)
    return StandaloneCachePaths(
        mfcc_npz=root
        / "data/processed/mission2/training_mfcc20_nfft512_hop160_v1.npz",
        mfcc_metadata=root
        / "data/processed/mission2/training_mfcc20_nfft512_hop160_v1.json",
        candidate_parquet=root
        / "data/processed/mission2_candidate_features/training_candidate_0c0d679a84c9.parquet",
        candidate_metadata=root
        / "data/processed/mission2_candidate_features/training_candidate_0c0d679a84c9.json",
        manifest_parquet=root / "tmp/mission2_eda/utterance_manifest_v1.parquet",
        manifest_metadata=root / "tmp/mission2_eda/utterance_manifest_meta_v1.json",
    )


def standalone_validation_cache_paths(repo_root: str | Path) -> StandaloneCachePaths:
    """Resolve the confirmed two-cache Validation feature contract."""

    root = Path(repo_root)
    candidate_root = root / "data/processed/mission2_candidate_features"
    return StandaloneCachePaths(
        mfcc_npz=root
        / "data/processed/mission2/validation_mfcc20_nfft512_hop160_v1.npz",
        mfcc_metadata=root
        / "data/processed/mission2/validation_mfcc20_nfft512_hop160_v1.json",
        candidate_parquet=(
            candidate_root / "validation_candidate_d6e692b7dcda.parquet",
            candidate_root / "validation_candidate_eaff31c61239.parquet",
        ),
        candidate_metadata=(
            candidate_root / "validation_candidate_d6e692b7dcda.json",
            candidate_root / "validation_candidate_eaff31c61239.json",
        ),
        manifest_parquet=root / "tmp/mission2_eda/utterance_manifest_v1.parquet",
        manifest_metadata=root / "tmp/mission2_eda/utterance_manifest_meta_v1.json",
    )


def _path_tuple(value: Path | tuple[Path, ...], *, label: str) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        return (Path(value),)
    resolved = tuple(Path(path) for path in value)
    if not resolved:
        raise ValueError(f"{label} paths cannot be empty")
    return resolved


def record_key_signature(record_keys: Sequence[str]) -> str:
    """Match the candidate-cache SHA-256 signature contract from PR #16."""

    digest = hashlib.sha256()
    for key in sorted(str(value) for value in record_keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} metadata is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} metadata must be a JSON object")
    return value


def _require_metadata_value(
    metadata: Mapping[str, Any], name: str, expected: Any, *, label: str
) -> None:
    observed = metadata.get(name)
    if observed != expected:
        raise ValueError(
            f"{label} metadata mismatch for {name}: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _validate_key_column(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    if "record_key" not in frame.columns:
        raise ValueError(f"{label} cache is missing record_key")
    if frame["record_key"].isna().any():
        raise ValueError(f"{label} record_key contains null")
    keys = np.asarray(frame["record_key"].astype(str).tolist(), dtype=str)
    if np.any(np.char.str_len(keys) == 0):
        raise ValueError(f"{label} record_key contains an empty value")
    if frame["record_key"].duplicated().any():
        raise ValueError(f"{label} record_key must be one-to-one")
    return keys


def _require_exact_key_set(
    observed_keys: Sequence[str], expected_keys: Sequence[str], *, label: str
) -> None:
    observed = set(map(str, observed_keys))
    expected = set(map(str, expected_keys))
    if observed != expected:
        raise ValueError(
            f"{label} key set mismatch: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )


def load_standalone_e05_cache(
    paths: StandaloneCachePaths,
    *,
    split: str,
    expected_rows: int,
    expected_calls: int,
) -> StandaloneE05Cache:
    """Load and align frozen caches; fail closed instead of extracting features.

    The NPZ row order is canonical.  Its stable key is reconstructed with the
    repository contract ``{split}|{call_id}|{zero_based_annotation_row}``, where
    annotation rows are counted in the existing per-call NPZ order.  Manifest
    and candidate rows are then aligned one-to-one by that key.
    """

    candidate_paths = _path_tuple(
        paths.candidate_parquet, label="candidate parquet"
    )
    candidate_metadata_paths = _path_tuple(
        paths.candidate_metadata, label="candidate metadata"
    )
    if len(candidate_paths) != len(candidate_metadata_paths):
        raise ValueError("candidate parquet/metadata path count mismatch")
    require_existing_cache_files(
        (
            paths.mfcc_npz,
            paths.mfcc_metadata,
            *candidate_paths,
            *candidate_metadata_paths,
            paths.manifest_parquet,
            paths.manifest_metadata,
        )
    )
    mfcc_npz_path = Path(paths.mfcc_npz)
    mfcc_metadata_path = Path(paths.mfcc_metadata)
    manifest_path = Path(paths.manifest_parquet)
    manifest_metadata_path = Path(paths.manifest_metadata)

    mfcc_meta = _read_json_object(mfcc_metadata_path, label="MFCC")
    candidate_metas = [
        _read_json_object(
            path,
            label="candidate" if len(candidate_metadata_paths) == 1 else f"candidate[{index}]",
        )
        for index, path in enumerate(candidate_metadata_paths)
    ]
    manifest_meta = _read_json_object(manifest_metadata_path, label="manifest")
    _require_metadata_value(
        mfcc_meta, "cache_schema_version", MFCC_CACHE_SCHEMA, label="MFCC"
    )
    _require_metadata_value(mfcc_meta, "feature_dimension", 40, label="MFCC")
    _require_metadata_value(mfcc_meta, "mfcc_config", dict(MFCC_CONFIG), label="MFCC")
    _require_metadata_value(mfcc_meta, "source_split", split, label="MFCC")
    _require_metadata_value(mfcc_meta, "utterance_count", expected_rows, label="MFCC")
    _require_metadata_value(mfcc_meta, "source_call_count", expected_calls, label="MFCC")

    selected_feature_sources: dict[str, int] = {}
    candidate_feature_names: list[tuple[str, ...]] = []
    for index, candidate_meta in enumerate(candidate_metas):
        label = "candidate" if len(candidate_metas) == 1 else f"candidate[{index}]"
        _require_metadata_value(
            candidate_meta,
            "cache_schema_version",
            CANDIDATE_CACHE_SCHEMA,
            label=label,
        )
        _require_metadata_value(candidate_meta, "source_split", split, label=label)
        _require_metadata_value(candidate_meta, "row_count", expected_rows, label=label)
        _require_metadata_value(candidate_meta, "waveform_only", True, label=label)
        extraction_config = candidate_meta.get("extraction_config")
        if not isinstance(extraction_config, Mapping):
            raise ValueError(f"{label} extraction_config metadata must be an object")
        for name in ("sample_rate", "n_fft", "hop_length"):
            _require_metadata_value(
                extraction_config,
                name,
                MFCC_CONFIG[name],
                label=f"{label} extraction",
            )
        names_value = candidate_meta.get("feature_names")
        if not isinstance(names_value, list):
            raise ValueError(f"{label} feature_names metadata must be a list")
        names = tuple(str(name) for name in names_value)
        _require_metadata_value(
            candidate_meta, "feature_count", len(names), label=label
        )
        candidate_feature_names.append(names)
        for name in E05_EXTRA_FEATURE_NAMES:
            if name not in names:
                continue
            if name in selected_feature_sources:
                raise ValueError(f"candidate feature {name} appears in multiple caches")
            selected_feature_sources[name] = index
    missing_selected = [
        name for name in E05_EXTRA_FEATURE_NAMES if name not in selected_feature_sources
    ]
    if missing_selected:
        raise ValueError(f"candidate metadata is missing {missing_selected}")

    _require_metadata_value(
        manifest_meta, "schema_version", MANIFEST_CACHE_SCHEMA, label="manifest"
    )
    _require_metadata_value(
        manifest_meta,
        "call_id_definition",
        "file stem; use (split, call_id) as group key",
        label="manifest",
    )
    _require_metadata_value(
        manifest_meta,
        "allowed_annotation_fields",
        ["id", "startAt", "endAt", "speaker"],
        label="manifest",
    )
    for index, candidate_meta in enumerate(candidate_metas):
        if candidate_meta.get("source_signature") != manifest_meta.get(
            "source_signature"
        ):
            raise ValueError(
                f"candidate[{index}]/manifest source_signature mismatch"
            )

    with np.load(mfcc_npz_path, allow_pickle=False) as bundle:
        required_members = {"X", "y", "stem", "start_at", "end_at", "duration"}
        missing_members = sorted(required_members - set(bundle.files))
        if missing_members:
            raise ValueError(f"MFCC cache is missing members: {missing_members}")
        X_mfcc = np.asarray(bundle["X"])
        raw_y = np.asarray(bundle["y"])
        if raw_y.dtype.kind not in "iu":
            raise ValueError(f"MFCC y must use an integer dtype; observed {raw_y.dtype}")
        y = raw_y.astype(np.int64, copy=False)
        call_ids = np.asarray(bundle["stem"]).astype(str)
        start_at = np.asarray(bundle["start_at"], dtype=np.float64)
        end_at = np.asarray(bundle["end_at"], dtype=np.float64)
        duration_ms = np.asarray(bundle["duration"], dtype=np.float64)

    if X_mfcc.shape != (expected_rows, 40):
        raise ValueError(
            f"MFCC cache must be ({expected_rows}, 40); observed {X_mfcc.shape}"
        )
    if X_mfcc.dtype != np.float32:
        raise ValueError(f"MFCC cache must be float32; observed {X_mfcc.dtype}")
    for name, values in {
        "y": y,
        "call_id": call_ids,
        "start_at": start_at,
        "end_at": end_at,
        "duration": duration_ms,
    }.items():
        if values.ndim != 1 or len(values) != expected_rows:
            raise ValueError(f"MFCC {name} row alignment mismatch")
    if np.any(np.char.str_len(call_ids) == 0):
        raise ValueError("MFCC call_id contains an empty value")
    if len(np.unique(call_ids)) != expected_calls:
        raise ValueError(
            f"MFCC call count mismatch: expected {expected_calls}, "
            f"observed {len(np.unique(call_ids))}"
        )
    if not np.isfinite(X_mfcc).all():
        raise ValueError("MFCC cache contains NaN or inf")
    if not all(np.isfinite(values).all() for values in (start_at, end_at, duration_ms)):
        raise ValueError("MFCC tracking values contain NaN or inf")

    canonical = pd.DataFrame({"call_id": call_ids})
    canonical["annotation_row"] = canonical.groupby("call_id", sort=False).cumcount()
    canonical["record_key"] = (
        split
        + "|"
        + canonical["call_id"].astype(str)
        + "|"
        + canonical["annotation_row"].astype(str)
    )
    canonical["_canonical_row"] = np.arange(expected_rows, dtype=np.int64)
    canonical_keys = _validate_key_column(canonical, label="MFCC-derived")

    manifest_columns = (
        "split",
        "call_id",
        "record_key",
        "annotation_row",
        "speaker",
        "startAt_ms",
        "endAt_ms",
        "duration_sec",
    )
    try:
        manifest = pd.read_parquet(
            manifest_path,
            columns=list(manifest_columns),
            filters=[("split", "==", split)],
        )
    except Exception as exc:
        raise ValueError(f"manifest cache is missing required columns: {exc}") from exc
    missing_manifest_columns = sorted(set(manifest_columns) - set(manifest.columns))
    if missing_manifest_columns:
        raise ValueError(f"manifest cache is missing columns: {missing_manifest_columns}")
    manifest = manifest.loc[manifest["split"].astype(str) == split].copy()
    if len(manifest) != expected_rows:
        raise ValueError(
            f"manifest row mismatch: expected {expected_rows}, observed {len(manifest)}"
        )
    manifest_keys = _validate_key_column(manifest, label="manifest")
    _require_exact_key_set(manifest_keys, canonical_keys, label="manifest")
    aligned_manifest = canonical.merge(
        manifest,
        on="record_key",
        how="left",
        validate="one_to_one",
        suffixes=("_mfcc", "_manifest"),
        sort=False,
    ).sort_values("_canonical_row", kind="stable")
    if aligned_manifest[
        ["call_id_manifest", "annotation_row_manifest", "speaker"]
    ].isna().any().any():
        raise ValueError("manifest alignment produced null values")
    if not np.array_equal(
        aligned_manifest["call_id_mfcc"].astype(str).to_numpy(), call_ids
    ) or not np.array_equal(
        aligned_manifest["annotation_row_mfcc"].to_numpy(dtype=np.int64),
        aligned_manifest["annotation_row_manifest"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("manifest call_id/annotation_row alignment mismatch")
    manifest_y = aligned_manifest["speaker"].to_numpy(dtype=np.int64)
    if not np.array_equal(manifest_y, y):
        raise ValueError("manifest/MFCC y alignment mismatch")
    if not np.array_equal(
        aligned_manifest["startAt_ms"].to_numpy(dtype=np.float64), start_at
    ):
        raise ValueError("manifest/MFCC startAt_ms alignment mismatch")
    if not np.array_equal(
        aligned_manifest["endAt_ms"].to_numpy(dtype=np.float64), end_at
    ):
        raise ValueError("manifest/MFCC endAt_ms alignment mismatch")
    derived_duration_ms = end_at - start_at
    if not np.array_equal(duration_ms, derived_duration_ms):
        raise ValueError("MFCC duration does not equal end_at - start_at")
    if not np.allclose(
        aligned_manifest["duration_sec"].to_numpy(dtype=np.float64),
        derived_duration_ms / 1_000.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("manifest/MFCC duration alignment mismatch")

    aligned_feature_values: dict[str, np.ndarray] = {}
    candidate_source_dtypes: set[str] = set()
    for index, (candidate_path, candidate_meta, names) in enumerate(
        zip(candidate_paths, candidate_metas, candidate_feature_names, strict=True)
    ):
        label = "candidate" if len(candidate_metas) == 1 else f"candidate[{index}]"
        try:
            candidate = pd.read_parquet(candidate_path)
        except Exception as exc:
            raise ValueError(f"{label} cache cannot be read: {exc}") from exc
        for name in ("record_key", "call_id", *names):
            if name not in candidate.columns:
                raise ValueError(f"{label} cache is missing {name}")
        if len(candidate) != expected_rows:
            raise ValueError(
                f"{label} row mismatch: expected {expected_rows}, "
                f"observed {len(candidate)}"
            )
        candidate_keys = _validate_key_column(candidate, label=label)
        _require_exact_key_set(candidate_keys, canonical_keys, label=label)
        if record_key_signature(candidate_keys) != candidate_meta.get(
            "record_key_signature"
        ):
            raise ValueError(f"{label} record_key_signature mismatch")
        selected_here = [
            name
            for name in E05_EXTRA_FEATURE_NAMES
            if selected_feature_sources[name] == index
        ]
        aligned_candidate = canonical[
            ["record_key", "call_id", "_canonical_row"]
        ].merge(
            candidate[["record_key", "call_id", *selected_here]],
            on="record_key",
            how="left",
            validate="one_to_one",
            suffixes=("_mfcc", "_candidate"),
            sort=False,
        ).sort_values("_canonical_row", kind="stable")
        if aligned_candidate["call_id_candidate"].isna().any():
            raise ValueError(f"{label} alignment produced null values")
        if not np.array_equal(
            aligned_candidate["call_id_mfcc"].astype(str).to_numpy(),
            aligned_candidate["call_id_candidate"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{label} call_id alignment mismatch")
        for name in selected_here:
            candidate_source_dtypes.add(str(candidate[name].dtype))
            values = aligned_candidate[name].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"candidate selected features contain NaN or inf: {name}")
            aligned_feature_values[name] = values.astype(np.float32)
    if len(candidate_source_dtypes) != 1:
        raise ValueError("candidate selected feature dtypes differ")
    aligned_extra = np.column_stack(
        [aligned_feature_values[name] for name in E05_EXTRA_FEATURE_NAMES]
    ).astype(np.float32, copy=False)

    X42 = np.empty((expected_rows, 42), dtype=np.float32)
    X42[:, :40] = X_mfcc
    X42[:, 40:] = aligned_extra
    validated = validate_e05_contract(
        X42,
        y,
        canonical_keys,
        E05_FEATURE_NAMES,
        expected_rows=expected_rows,
        frozen_mfcc40=X_mfcc,
        expected_extra_features=aligned_extra,
    )
    tracking = pd.DataFrame(
        {
            "record_key": canonical_keys,
            "call_id": call_ids,
            "annotation_row": canonical["annotation_row"].to_numpy(dtype=np.int64),
            "speaker": y,
            "startAt_ms": start_at,
            "endAt_ms": end_at,
            "duration_ms": duration_ms,
            "duration_sec": duration_ms / 1_000.0,
        }
    )
    return StandaloneE05Cache(
        validated=validated,
        tracking=tracking,
        X_mfcc=X_mfcc,
        aligned_extra_features=aligned_extra,
        cache_audit={
            "backend": CACHE_BACKEND,
            "split": split,
            "rows": expected_rows,
            "calls": expected_calls,
            "canonical_order": "frozen_mfcc_npz",
            "record_key_signature": record_key_signature(canonical_keys),
            "candidate_source_dtype": next(iter(candidate_source_dtypes)),
            "final_dtype": str(X42.dtype),
            "automatic_extraction": False,
        },
    )


def standalone_group_split_indices(
    call_ids: Sequence[str], *, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create the fixed one-shot GroupShuffleSplit used by both models."""

    groups = np.asarray(call_ids).astype(str)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("call_id groups must be a non-empty one-dimensional array")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, validation_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    train_idx = np.asarray(train_idx, dtype=np.int64)
    validation_idx = np.asarray(validation_idx, dtype=np.int64)
    train_calls = set(groups[train_idx].tolist())
    validation_calls = set(groups[validation_idx].tolist())
    return train_idx, validation_idx, {
        "seed": seed,
        "test_size": test_size,
        "total_rows": int(len(groups)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "train_calls": int(len(train_calls)),
        "validation_calls": int(len(validation_calls)),
        "call_overlap": int(len(train_calls & validation_calls)),
    }


def validate_e05_contract(
    X: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    feature_names: Sequence[str],
    *,
    expected_rows: int | None = None,
    frozen_mfcc40: np.ndarray | None = None,
    expected_extra_features: np.ndarray | None = None,
) -> ValidatedE05Data:
    """Validate exact E05 columns, dtype, finiteness, labels, and row keys."""

    names = tuple(str(name) for name in feature_names)
    if names != E05_FEATURE_NAMES:
        raise ValueError(
            "E05 42D feature contract mismatch: expected exact MFCC40 followed by "
            "spectral_rolloff and zcr"
        )

    matrix = np.asarray(X)
    labels = np.asarray(y, dtype=np.int64)
    keys = np.asarray(record_keys).astype(str)
    if matrix.ndim != 2 or matrix.shape[1] != 42:
        raise ValueError(f"E05 matrix must be exactly 42D; received {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"E05 matrix must preserve the float32 contract; got {matrix.dtype}")
    if matrix.shape[0] != len(labels) or matrix.shape[0] != len(keys):
        raise ValueError("E05 row alignment mismatch among X, y, and record_key")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"E05 row contract mismatch: expected {expected_rows}, got {matrix.shape[0]}"
        )
    if len(np.unique(keys)) != len(keys) or np.any(np.char.str_len(keys) == 0):
        raise ValueError("record_key must be non-empty and one-to-one")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("labels must contain exactly the binary classes 0 and 1")

    if frozen_mfcc40 is not None:
        expected_mfcc = np.asarray(frozen_mfcc40)
        if expected_mfcc.shape != (matrix.shape[0], 40) or not np.array_equal(
            matrix[:, :40], expected_mfcc
        ):
            raise ValueError("first 40 E05 values do not equal the frozen MFCC40 cache")
    if expected_extra_features is not None:
        expected_extra = np.asarray(expected_extra_features)
        if expected_extra.shape != (matrix.shape[0], 2) or not np.array_equal(
            matrix[:, 40:], expected_extra
        ):
            raise ValueError("final E05 values do not equal aligned rolloff/zcr features")

    if not np.isfinite(matrix).all():
        raise ValueError("E05 matrix contains NaN or inf")
    return ValidatedE05Data(
        X=matrix,
        y=labels,
        record_keys=keys,
        feature_names=names,
    )


def create_fixed_internal_split(
    call_ids: Sequence[str],
    *,
    split_function: SplitFunction,
    expected_contract: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Call a GroupSplit implementation and enforce the frozen split audit.

    Checkpoints A--H pass :func:`standalone_group_split_indices`; dependency
    injection remains available for contract tests and approved future adapters.
    """

    groups = np.asarray(call_ids).astype(str)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("call_id groups must be a non-empty one-dimensional array")
    if np.any(np.char.str_len(groups) == 0):
        raise ValueError("call_id groups contain an empty value")

    train_idx, validation_idx, helper_audit = split_function(
        groups, seed=42, test_size=0.2
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    validation_idx = np.asarray(validation_idx, dtype=np.int64)
    all_idx = np.concatenate([train_idx, validation_idx])
    if (
        train_idx.ndim != 1
        or validation_idx.ndim != 1
        or len(np.unique(all_idx)) != len(groups)
        or set(all_idx.tolist()) != set(range(len(groups)))
    ):
        raise ValueError("fixed split indices must partition every row exactly once")

    train_calls = set(groups[train_idx].tolist())
    validation_calls = set(groups[validation_idx].tolist())
    overlap = train_calls & validation_calls
    audit: dict[str, Any] = {
        "seed": 42,
        "test_size": 0.2,
        "total_rows": int(len(groups)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "train_calls": int(len(train_calls)),
        "validation_calls": int(len(validation_calls)),
        "call_overlap": int(len(overlap)),
    }
    if overlap:
        raise ValueError("call_id overlap must be zero")

    supplied_audit = dict(helper_audit)
    for name, actual in audit.items():
        if name in supplied_audit and supplied_audit[name] != actual:
            raise ValueError(
                f"split helper audit mismatch for {name}: "
                f"reported {supplied_audit[name]!r}, observed {actual!r}"
            )

    if expected_contract is not None:
        mismatches = {
            name: {"expected": expected, "actual": audit.get(name)}
            for name, expected in expected_contract.items()
            if audit.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"fixed internal split contract mismatch: {mismatches}")
    return train_idx, validation_idx, audit


def prepare_internal_comparison(
    *,
    X: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    feature_names: Sequence[str],
    split_function: SplitFunction,
    expected_rows: int | None = None,
    expected_split_contract: Mapping[str, int] | None,
) -> PreparedComparison:
    """Build the one shared split and internal-train-only StandardScaler."""

    validated = validate_e05_contract(
        X, y, record_keys, feature_names, expected_rows=expected_rows
    )
    groups = np.asarray(call_ids).astype(str)
    if len(groups) != len(validated.y):
        raise ValueError("call_id row alignment mismatch")
    train_idx, validation_idx, audit = create_fixed_internal_split(
        groups,
        split_function=split_function,
        expected_contract=expected_split_contract,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(validated.X[train_idx]).astype(
        np.float32, copy=False
    )
    X_validation_scaled = scaler.transform(validated.X[validation_idx]).astype(
        np.float32, copy=False
    )
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(
        X_validation_scaled
    ).all():
        raise ValueError("standardized internal matrix contains NaN or inf")

    return PreparedComparison(
        feature_names=validated.feature_names,
        X_train_scaled=X_train_scaled,
        X_evaluation_scaled=X_validation_scaled,
        y_train=validated.y[train_idx],
        y_evaluation=validated.y[validation_idx],
        training_record_keys=validated.record_keys[train_idx],
        evaluation_record_keys=validated.record_keys[validation_idx],
        scaler_fit_record_keys=validated.record_keys[train_idx].copy(),
        train_idx=train_idx,
        validation_idx=validation_idx,
        scaler=scaler,
        scaler_fit_scope="internal_train_only",
        training_scope="Training/internal_train",
        evaluation_scope="Training/internal_validation",
        split_audit=audit,
    )


def prepare_official_comparison(
    *,
    training_X: np.ndarray,
    training_y: Sequence[int],
    training_record_keys: Sequence[str],
    validation_X: np.ndarray,
    validation_y: Sequence[int],
    validation_record_keys: Sequence[str],
    feature_names: Sequence[str],
    user_approved: bool = False,
    expected_training_rows: int | None = EXPECTED_OFFICIAL_ROWS["Training"],
    expected_validation_rows: int | None = EXPECTED_OFFICIAL_ROWS["Validation"],
) -> PreparedComparison:
    """Fit one scaler on full Training for a separately approved official run."""

    if not user_approved:
        raise RuntimeError(
            "user_approved=True is required before preparing Official Validation"
        )
    training = validate_e05_contract(
        training_X,
        training_y,
        training_record_keys,
        feature_names,
        expected_rows=expected_training_rows,
    )
    validation = validate_e05_contract(
        validation_X,
        validation_y,
        validation_record_keys,
        feature_names,
        expected_rows=expected_validation_rows,
    )
    key_overlap = set(training.record_keys.tolist()) & set(
        validation.record_keys.tolist()
    )
    if key_overlap:
        raise ValueError(
            f"Training/Validation record_key overlap must be zero; "
            f"observed {len(key_overlap)}"
        )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(training.X).astype(np.float32, copy=False)
    X_validation_scaled = scaler.transform(validation.X).astype(np.float32, copy=False)
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(
        X_validation_scaled
    ).all():
        raise ValueError("standardized official matrix contains NaN or inf")

    train_idx = np.arange(len(training.y), dtype=np.int64)
    validation_idx = np.arange(len(validation.y), dtype=np.int64)
    return PreparedComparison(
        feature_names=training.feature_names,
        X_train_scaled=X_train_scaled,
        X_evaluation_scaled=X_validation_scaled,
        y_train=training.y,
        y_evaluation=validation.y,
        training_record_keys=training.record_keys,
        evaluation_record_keys=validation.record_keys,
        scaler_fit_record_keys=training.record_keys.copy(),
        train_idx=train_idx,
        validation_idx=validation_idx,
        scaler=scaler,
        scaler_fit_scope="full_training_only",
        training_scope="Training/full",
        evaluation_scope="Official Validation",
        split_audit={
            "training_rows": int(len(training.y)),
            "validation_rows": int(len(validation.y)),
            "scaler_fit_scope": "Training/full",
        },
    )


def build_m00_model() -> LogisticRegression:
    """Construct the frozen linear M00 reference model."""

    return LogisticRegression(**M00_LOGISTIC_REGRESSION_CONFIG)


def build_m01_model() -> HistGradientBoostingClassifier:
    """Construct the single precommitted nonlinear M01 model."""

    return HistGradientBoostingClassifier(**M01_HGB_CONFIG)


def build_screening_model(model_id: str) -> Any:
    """Construct one precommitted M02--M04 estimator without search."""

    if model_id == "M02":
        return ExtraTreesClassifier(**M02_EXTRATREES_CONFIG)
    if model_id == "M03":
        return QuadraticDiscriminantAnalysis(**M03_QDA_CONFIG)
    if model_id == "M04":
        return MLPClassifier(**M04_MLP_CONFIG)
    raise ValueError(f"unknown screening model: {model_id!r}")


def build_m04_convergence_check_model() -> MLPClassifier:
    """Build the max_iter-only M04 follow-up without changing other settings."""

    changed = {
        name
        for name in M04_MLP_CONFIG
        if M04_MLP_CONFIG[name] != M04_CONVERGENCE_CHECK_CONFIG[name]
    }
    if changed != {"max_iter"}:
        raise ValueError("M04 convergence check may change only max_iter")
    if M04_MLP_CONFIG["max_iter"] != 30 or M04_CONVERGENCE_CHECK_CONFIG[
        "max_iter"
    ] != 60:
        raise ValueError("M04 convergence check requires max_iter 30 -> 60")
    return MLPClassifier(**M04_CONVERGENCE_CHECK_CONFIG)


def _canonical_config_value(value: Any) -> Any:
    """Normalize supported config values without guessing at object semantics."""

    if isinstance(value, np.generic):
        return _canonical_config_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("config fingerprint requires finite numeric values")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("config fingerprint requires string mapping keys")
        return {key: _canonical_config_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported config fingerprint value: {type(value).__name__}")


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash one canonical JSON representation of a supported config mapping."""

    if not isinstance(config, Mapping):
        raise TypeError("config fingerprint requires a mapping")
    payload = json.dumps(
        _canonical_config_value(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calculate_prediction_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, Any]:
    """Calculate the frozen primary and class-wise metrics."""

    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("prediction/label shapes must be identical one-dimensional arrays")
    if not set(np.unique(truth).tolist()) <= {0, 1}:
        raise ValueError("y_true contains a non-binary label")
    if not set(np.unique(prediction).tolist()) <= {0, 1}:
        raise ValueError("y_pred contains a non-binary class")
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    recalls = recall_score(truth, prediction, labels=[0, 1], average=None, zero_division=0)
    return {
        "sample_count": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "confusion_matrix": matrix.tolist(),
        "recall_0": float(recalls[0]),
        "recall_1": float(recalls[1]),
        "recall_speaker_0": float(recalls[0]),
        "recall_speaker_1": float(recalls[1]),
        "correct_count": int(np.sum(truth == prediction)),
    }


def validate_binary_probabilities(
    probabilities: Any,
    *,
    expected_rows: int,
    classes: Sequence[int] | None = None,
) -> np.ndarray:
    """Validate the shared [speaker=0, speaker=1] predict_proba contract."""

    if classes is not None and not np.array_equal(np.asarray(classes), [0, 1]):
        raise ValueError(f"model classes must be [0, 1], received {classes!r}")
    values = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (expected_rows, 2)
    if values.shape != expected_shape:
        raise ValueError(
            f"predict_proba shape mismatch: expected {expected_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("predict_proba values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("predict_proba values must be within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("predict_proba rows must sum to one")
    return values


def _rate_matches_count(rate: Any, numerator: int, denominator: int) -> bool:
    """Accept a float32 rate representation only when its discrete count agrees."""

    try:
        observed = float(rate)
    except (TypeError, ValueError):
        return False
    if denominator < 0 or numerator < 0 or numerator > denominator:
        return False
    exact = numerator / denominator if denominator else 0.0
    if not np.isfinite(observed) or not 0.0 <= observed <= 1.0:
        return False
    # Exact row/count checks remain authoritative; one float32 ULP covers a
    # stored rate rounded through float32 without accepting a changed count.
    return abs(observed - exact) <= float(np.spacing(np.float32(exact)))


ARRAY_FINGERPRINT_VERSION = "sha256-dtype-jsonshape-c-v1"


def _array_fingerprint(values: np.ndarray) -> str:
    """Fingerprint transformed values together with their shape and dtype."""

    original = np.asarray(values)
    if original.dtype.hasobject:
        raise TypeError("array fingerprint does not support object dtype")
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(str(original.dtype).encode("ascii"))
    digest.update(json.dumps(original.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _run_model(
    prepared: PreparedComparison,
    *,
    experiment_id: str,
    model_name: str,
    model_config: Mapping[str, Any],
    model: Any,
    allow_fit: bool,
) -> ModelExperimentResult:
    if not allow_fit:
        raise RuntimeError("allow_fit=True is required for any model fit")
    if prepared.feature_names != E05_FEATURE_NAMES:
        raise ValueError("prepared data no longer satisfies the E05 feature contract")
    if prepared.scaler_fit_scope not in {"internal_train_only", "full_training_only"}:
        raise ValueError("invalid scaler fit scope")

    fit_started = perf_counter()
    model.fit(prepared.X_train_scaled, prepared.y_train)
    fit_seconds = perf_counter() - fit_started
    prediction_started = perf_counter()
    prediction = np.asarray(model.predict(prepared.X_evaluation_scaled), dtype=np.int64)
    probabilities = validate_binary_probabilities(
        model.predict_proba(prepared.X_evaluation_scaled),
        expected_rows=len(prepared.y_evaluation),
        classes=model.classes_,
    )
    prediction_seconds = perf_counter() - prediction_started

    metrics = calculate_prediction_metrics(prepared.y_evaluation, prediction)
    fit_audit = {
        "training_scope": prepared.training_scope,
        "evaluation_scope": prepared.evaluation_scope,
        "scaler_fit_scope": prepared.scaler_fit_scope,
        "training_record_keys": prepared.training_record_keys.copy(),
        "validation_record_keys": prepared.evaluation_record_keys.copy(),
        "scaler_fit_record_keys": prepared.scaler_fit_record_keys.copy(),
        "train_idx": prepared.train_idx.copy(),
        "validation_idx": prepared.validation_idx.copy(),
        "split_audit": dict(prepared.split_audit),
        "model_config_fingerprint": config_fingerprint(model_config),
        "X_train_fingerprint": _array_fingerprint(prepared.X_train_scaled),
        "X_evaluation_fingerprint": _array_fingerprint(
            prepared.X_evaluation_scaled
        ),
        "y_train_fingerprint": _array_fingerprint(prepared.y_train),
        "y_evaluation_fingerprint": _array_fingerprint(prepared.y_evaluation),
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "runtime_seconds": float(fit_seconds + prediction_seconds),
        "peak_memory_bytes": None,
        "peak_memory_note": (
            "not measured; portable Python allocation tracking would omit "
            "native estimator memory"
        ),
    }
    return ModelExperimentResult(
        experiment_id=experiment_id,
        model_name=model_name,
        model_config=dict(model_config),
        feature_names=prepared.feature_names,
        train_shape=prepared.X_train_scaled.shape,
        evaluation_shape=prepared.X_evaluation_scaled.shape,
        y_true=prepared.y_evaluation.copy(),
        y_pred=prediction,
        probabilities=probabilities,
        metrics=metrics,
        scaler=prepared.scaler,
        model=model,
        fit_audit=fit_audit,
    )


def run_internal_m00(
    prepared: PreparedComparison, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Fit M00 only after the caller's explicit internal-run switch."""

    if prepared.evaluation_scope != "Training/internal_validation":
        raise ValueError("M00 internal run requires Training/internal_validation")
    return _run_model(
        prepared,
        experiment_id="M00",
        model_name="LogisticRegression",
        model_config=M00_LOGISTIC_REGRESSION_CONFIG,
        model=build_m00_model(),
        allow_fit=allow_fit,
    )


def run_internal_m01(
    prepared: PreparedComparison, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Fit the precommitted M01 only after the caller's internal-run switch."""

    if prepared.evaluation_scope != "Training/internal_validation":
        raise ValueError("M01 internal run requires Training/internal_validation")
    return _run_model(
        prepared,
        experiment_id="M01",
        model_name="HistGradientBoostingClassifier",
        model_config=M01_HGB_CONFIG,
        model=build_m01_model(),
        allow_fit=allow_fit,
    )


def run_internal_screening(
    prepared: PreparedComparison, *, allow_fit: bool = False
) -> tuple[ModelExperimentResult, ...]:
    """Fit exactly the three frozen model-family candidates on one payload."""

    if prepared.evaluation_scope != "Training/internal_validation":
        raise ValueError("model-family screening requires Training/internal_validation")
    results: list[ModelExperimentResult] = []
    for model_id in SCREENING_MODEL_IDS:
        results.append(
            _run_model(
                prepared,
                experiment_id=model_id,
                model_name=SCREENING_MODEL_NAMES[model_id],
                model_config=SCREENING_MODEL_CONFIGS[model_id],
                model=build_screening_model(model_id),
                allow_fit=allow_fit,
            )
        )
    return tuple(results)


def run_internal_m04_reference(
    prepared: PreparedComparison, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Recreate only frozen M04 on the caller's current internal payload."""

    if prepared.evaluation_scope != "Training/internal_validation":
        raise ValueError("M04 reference requires Training/internal_validation")
    return _run_model(
        prepared,
        experiment_id="M04",
        model_name=SCREENING_MODEL_NAMES["M04"],
        model_config=M04_MLP_CONFIG,
        model=build_screening_model("M04"),
        allow_fit=allow_fit,
    )


def summarize_mlp_convergence(
    model: Any, *, convergence_warning_messages: Sequence[str]
) -> dict[str, Any]:
    """Summarize observed optimizer stopping and training-loss direction."""

    n_iter = int(model.n_iter_)
    loss_curve = np.asarray(model.loss_curve_, dtype=np.float64)
    if loss_curve.ndim != 1 or len(loss_curve) == 0:
        raise ValueError("MLP loss_curve_ must be a non-empty one-dimensional array")
    if not np.isfinite(loss_curve).all():
        raise ValueError("MLP loss_curve_ contains NaN or inf")
    if len(loss_curve) != n_iter:
        raise ValueError("MLP loss_curve_ length must equal n_iter_")
    max_iter = int(model.get_params()["max_iter"])
    tail = loss_curve[-min(10, len(loss_curve)) :]
    tail_delta = float(tail[-1] - tail[0])
    tail_slope = (
        float(np.polyfit(np.arange(len(tail), dtype=np.float64), tail, 1)[0])
        if len(tail) > 1
        else 0.0
    )
    messages = tuple(str(message) for message in convergence_warning_messages)
    return {
        "convergence_warning_raised": bool(messages),
        "convergence_warning_messages": messages,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "reached_max_iter": n_iter >= max_iter,
        "loss_curve_length": int(len(loss_curve)),
        "initial_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "tail_loss_delta": tail_delta,
        "tail_loss_slope": tail_slope,
        "loss_curve": tuple(float(value) for value in loss_curve),
    }


def run_internal_m04_convergence_check(
    prepared: PreparedComparison, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Run the max_iter=60 M04 check on the fixed internal split only."""

    if prepared.evaluation_scope != "Training/internal_validation":
        raise ValueError(
            "M04 convergence check requires Training/internal_validation"
        )
    if not allow_fit:
        raise RuntimeError("allow_fit=True is required for any model fit")
    model = build_m04_convergence_check_model()
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        result = _run_model(
            prepared,
            experiment_id=M04_CONVERGENCE_CHECK_ID,
            model_name="MLPClassifier",
            model_config=M04_CONVERGENCE_CHECK_CONFIG,
            model=model,
            allow_fit=True,
        )
    convergence_messages = tuple(
        str(observed.message)
        for observed in observed_warnings
        if issubclass(observed.category, ConvergenceWarning)
    )
    convergence = summarize_mlp_convergence(
        model, convergence_warning_messages=convergence_messages
    )
    return replace(
        result,
        fit_audit={**dict(result.fit_audit), "mlp_convergence": convergence},
    )


def run_official_m01(
    prepared: PreparedComparison,
    *,
    allow_fit: bool = False,
    user_approved: bool = False,
) -> ModelExperimentResult:
    """Fit M01 on full Training only after both official gates are open."""

    if not user_approved:
        raise RuntimeError("user_approved=True is required for Official Validation")
    if prepared.evaluation_scope != "Official Validation":
        raise ValueError("official M01 run requires Official Validation prepared data")
    return _run_model(
        prepared,
        experiment_id="M01_OFFICIAL",
        model_name="HistGradientBoostingClassifier",
        model_config=M01_HGB_CONFIG,
        model=build_m01_model(),
        allow_fit=allow_fit,
    )


def run_official_m00(
    prepared: PreparedComparison,
    *,
    allow_fit: bool = False,
    user_approved: bool = False,
) -> ModelExperimentResult:
    """Fit same-run M00 on full Training after both official gates are open."""

    if not user_approved:
        raise RuntimeError("user_approved=True is required for Official Validation")
    if prepared.evaluation_scope != "Official Validation":
        raise ValueError("official M00 run requires Official Validation prepared data")
    return _run_model(
        prepared,
        experiment_id="M00_OFFICIAL",
        model_name="LogisticRegression",
        model_config=M00_LOGISTIC_REGRESSION_CONFIG,
        model=build_m00_model(),
        allow_fit=allow_fit,
    )


def calculate_paired_prediction_changes(
    *,
    y_true: Sequence[int],
    m00_pred: Sequence[int],
    m01_pred: Sequence[int],
    m00_record_keys: Sequence[str],
    m01_record_keys: Sequence[str],
) -> dict[str, Any]:
    """Compare paired row predictions only after exact record-key alignment."""

    truth = np.asarray(y_true, dtype=np.int64)
    linear = np.asarray(m00_pred, dtype=np.int64)
    nonlinear = np.asarray(m01_pred, dtype=np.int64)
    linear_keys = np.asarray(m00_record_keys).astype(str)
    nonlinear_keys = np.asarray(m01_record_keys).astype(str)
    if not np.array_equal(linear_keys, nonlinear_keys):
        raise ValueError("record_key alignment mismatch in paired comparison")
    if truth.shape != linear.shape or truth.shape != nonlinear.shape:
        raise ValueError("paired predictions and y_true must have identical shapes")
    if len(np.unique(linear_keys)) != len(linear_keys):
        raise ValueError("record_key must remain one-to-one in paired comparison")

    m00_correct = linear == truth
    m01_correct = nonlinear == truth
    disagreement = linear != nonlinear
    return {
        "both_correct": int(np.sum(m00_correct & m01_correct)),
        "both_wrong": int(np.sum((~m00_correct) & (~m01_correct))),
        "m00_wrong_to_m01_correct": int(np.sum((~m00_correct) & m01_correct)),
        "m00_correct_to_m01_wrong": int(np.sum(m00_correct & (~m01_correct))),
        "prediction_disagreement_count": int(np.sum(disagreement)),
        "prediction_disagreement_rate": float(np.mean(disagreement)),
    }


def compare_model_results(
    m00: ModelExperimentResult, m01: ModelExperimentResult
) -> dict[str, Any]:
    """Calculate frozen internal deltas without declaring a winner."""

    return _compare_model_results_with_ids(
        m00, m01, expected_ids=("M00", "M01")
    )


def _compare_model_results_with_ids(
    m00: ModelExperimentResult,
    m01: ModelExperimentResult,
    *,
    expected_ids: tuple[str, str],
) -> dict[str, Any]:
    """Validate shared comparison inputs and calculate paired deltas."""

    if (m00.experiment_id, m01.experiment_id) != expected_ids:
        raise ValueError(
            f"comparison requires experiment IDs {expected_ids}; received "
            f"{(m00.experiment_id, m01.experiment_id)}"
        )
    if m00.feature_names != E05_FEATURE_NAMES or m01.feature_names != E05_FEATURE_NAMES:
        raise ValueError("M00/M01 feature contracts must both equal E05 42D")
    if m00.train_shape != m01.train_shape or m00.evaluation_shape != m01.evaluation_shape:
        raise ValueError("M00/M01 train/evaluation shapes differ")
    if dict(m00.model_config) != dict(M00_LOGISTIC_REGRESSION_CONFIG):
        raise ValueError("M00 LogisticRegression config changed")
    if dict(m01.model_config) != dict(M01_HGB_CONFIG):
        raise ValueError("M01 HGB precommit config changed")
    if m00.scaler is not m01.scaler:
        raise ValueError("M00/M01 must use the identical shared scaler object")
    for audit_name in (
        "train_idx",
        "validation_idx",
        "training_record_keys",
        "validation_record_keys",
        "scaler_fit_record_keys",
    ):
        if not np.array_equal(m00.fit_audit[audit_name], m01.fit_audit[audit_name]):
            raise ValueError(f"M00/M01 {audit_name} alignment mismatch")
    for audit_name in (
        "X_train_fingerprint",
        "X_evaluation_fingerprint",
        "y_train_fingerprint",
        "y_evaluation_fingerprint",
    ):
        if m00.fit_audit.get(audit_name) != m01.fit_audit.get(audit_name):
            raise ValueError(f"M00/M01 {audit_name} mismatch")
    for audit_name in ("training_scope", "evaluation_scope", "scaler_fit_scope"):
        if m00.fit_audit[audit_name] != m01.fit_audit[audit_name]:
            raise ValueError(f"M00/M01 {audit_name} mismatch")
    if not np.array_equal(m00.y_true, m01.y_true):
        raise ValueError("M00/M01 y alignment mismatch")
    paired = calculate_paired_prediction_changes(
        y_true=m00.y_true,
        m00_pred=m00.y_pred,
        m01_pred=m01.y_pred,
        m00_record_keys=m00.fit_audit["validation_record_keys"],
        m01_record_keys=m01.fit_audit["validation_record_keys"],
    )
    return {
        "evaluation_rows": int(len(m00.y_true)),
        "m00_accuracy": float(m00.metrics["accuracy"]),
        "m01_accuracy": float(m01.metrics["accuracy"]),
        "m00_confusion_matrix": m00.metrics["confusion_matrix"],
        "m01_confusion_matrix": m01.metrics["confusion_matrix"],
        "accuracy_delta": float(m01.metrics["accuracy"] - m00.metrics["accuracy"]),
        "correct_count_delta": int(
            m01.metrics["correct_count"] - m00.metrics["correct_count"]
        ),
        "recall_speaker_0_delta": float(
            m01.metrics["recall_speaker_0"] - m00.metrics["recall_speaker_0"]
        ),
        "recall_speaker_1_delta": float(
            m01.metrics["recall_speaker_1"] - m00.metrics["recall_speaker_1"]
        ),
        **paired,
    }


def compare_official_model_results(
    m00: ModelExperimentResult, m01: ModelExperimentResult
) -> dict[str, Any]:
    """Compare same-run Official M00/M01 results without declaring a winner."""

    if m00.fit_audit.get("evaluation_scope") != "Official Validation" or m01.fit_audit.get(
        "evaluation_scope"
    ) != "Official Validation":
        raise ValueError("official comparison requires Official Validation results")
    return _compare_model_results_with_ids(
        m00, m01, expected_ids=("M00_OFFICIAL", "M01_OFFICIAL")
    )


def _base_experiment_id(experiment_id: str) -> str:
    suffix = "_OFFICIAL"
    return experiment_id[: -len(suffix)] if experiment_id.endswith(suffix) else experiment_id


def _validate_shared_result_payload(
    reference: ModelExperimentResult, candidate: ModelExperimentResult
) -> None:
    if reference.feature_names != E05_FEATURE_NAMES or candidate.feature_names != E05_FEATURE_NAMES:
        raise ValueError("screening results must both satisfy the exact E05 42D contract")
    if reference.train_shape != candidate.train_shape:
        raise ValueError("screening results use different training rows")
    if reference.evaluation_shape != candidate.evaluation_shape:
        raise ValueError("screening results use different evaluation rows")
    if reference.scaler is not candidate.scaler:
        raise ValueError("screening results must use the identical shared scaler object")
    for audit_name in (
        "train_idx",
        "validation_idx",
        "training_record_keys",
        "validation_record_keys",
        "scaler_fit_record_keys",
    ):
        if not np.array_equal(
            reference.fit_audit[audit_name], candidate.fit_audit[audit_name]
        ):
            raise ValueError(f"screening {audit_name} alignment mismatch")
    for audit_name in (
        "X_train_fingerprint",
        "X_evaluation_fingerprint",
        "y_train_fingerprint",
        "y_evaluation_fingerprint",
    ):
        if reference.fit_audit.get(audit_name) != candidate.fit_audit.get(audit_name):
            raise ValueError(f"screening {audit_name} mismatch")
    for audit_name in ("training_scope", "evaluation_scope", "scaler_fit_scope"):
        if reference.fit_audit.get(audit_name) != candidate.fit_audit.get(audit_name):
            raise ValueError(f"screening {audit_name} mismatch")
    if not np.array_equal(reference.y_true, candidate.y_true):
        raise ValueError("screening y alignment mismatch")


def compare_model_candidate_to_reference(
    reference: ModelExperimentResult, candidate: ModelExperimentResult
) -> dict[str, Any]:
    """Compare one aligned M02--M04 result against current-best M01."""

    reference_is_official = reference.experiment_id == "M01_OFFICIAL"
    if reference.experiment_id not in {"M01", "M01_OFFICIAL"}:
        raise ValueError("screening reference must be M01 or M01_OFFICIAL")
    candidate_id = _base_experiment_id(candidate.experiment_id)
    if candidate_id not in SCREENING_MODEL_IDS:
        raise ValueError("candidate must be one of the frozen M02--M04 models")
    if candidate.experiment_id.endswith("_OFFICIAL") != reference_is_official:
        raise ValueError("reference and candidate evaluation scopes differ")
    if dict(candidate.model_config) != dict(SCREENING_MODEL_CONFIGS[candidate_id]):
        raise ValueError(f"{candidate_id} precommit config changed")
    _validate_shared_result_payload(reference, candidate)
    paired = calculate_paired_prediction_changes(
        y_true=reference.y_true,
        m00_pred=reference.y_pred,
        m01_pred=candidate.y_pred,
        m00_record_keys=reference.fit_audit["validation_record_keys"],
        m01_record_keys=candidate.fit_audit["validation_record_keys"],
    )
    return {
        "reference_id": reference.experiment_id,
        "candidate_id": candidate.experiment_id,
        "evaluation_rows": int(len(reference.y_true)),
        "reference_accuracy": float(reference.metrics["accuracy"]),
        "candidate_accuracy": float(candidate.metrics["accuracy"]),
        "accuracy_delta": float(
            candidate.metrics["accuracy"] - reference.metrics["accuracy"]
        ),
        "correct_count_delta": int(
            candidate.metrics["correct_count"] - reference.metrics["correct_count"]
        ),
        "recall_speaker_0_delta": float(
            candidate.metrics["recall_speaker_0"]
            - reference.metrics["recall_speaker_0"]
        ),
        "recall_speaker_1_delta": float(
            candidate.metrics["recall_speaker_1"]
            - reference.metrics["recall_speaker_1"]
        ),
        "both_correct": paired["both_correct"],
        "both_wrong": paired["both_wrong"],
        "reference_wrong_to_candidate_correct": paired[
            "m00_wrong_to_m01_correct"
        ],
        "reference_correct_to_candidate_wrong": paired[
            "m00_correct_to_m01_wrong"
        ],
        "prediction_disagreement_count": paired[
            "prediction_disagreement_count"
        ],
        "prediction_disagreement_rate": paired[
            "prediction_disagreement_rate"
        ],
        "runtime_seconds": candidate.fit_audit.get("runtime_seconds"),
        "peak_memory_bytes": candidate.fit_audit.get("peak_memory_bytes"),
    }


def compare_m04_convergence_check(
    m04: ModelExperimentResult, candidate: ModelExperimentResult
) -> dict[str, Any]:
    """Compare max_iter=60 with M04 on exactly paired internal rows."""

    if m04.experiment_id != "M04" or candidate.experiment_id != (
        M04_CONVERGENCE_CHECK_ID
    ):
        raise ValueError("comparison requires M04 and M04_ITER60")
    if m04.fit_audit.get("evaluation_scope") != "Training/internal_validation":
        raise ValueError("M04 comparison requires Training/internal_validation")
    if candidate.fit_audit.get("evaluation_scope") != (
        "Training/internal_validation"
    ):
        raise ValueError("M04_ITER60 comparison requires Training/internal_validation")
    if dict(m04.model_config) != dict(M04_MLP_CONFIG):
        raise ValueError("original M04 config changed")
    if dict(candidate.model_config) != dict(M04_CONVERGENCE_CHECK_CONFIG):
        raise ValueError("M04_ITER60 precommit config changed")
    changed = {
        name
        for name in M04_MLP_CONFIG
        if M04_MLP_CONFIG[name] != M04_CONVERGENCE_CHECK_CONFIG[name]
    }
    if changed != {"max_iter"}:
        raise ValueError("M04 convergence comparison may change only max_iter")
    _validate_shared_result_payload(m04, candidate)
    convergence = candidate.fit_audit.get("mlp_convergence")
    if not isinstance(convergence, Mapping):
        raise ValueError("M04_ITER60 convergence diagnostics are missing")
    paired = calculate_paired_prediction_changes(
        y_true=m04.y_true,
        m00_pred=m04.y_pred,
        m01_pred=candidate.y_pred,
        m00_record_keys=m04.fit_audit["validation_record_keys"],
        m01_record_keys=candidate.fit_audit["validation_record_keys"],
    )
    return {
        "reference_id": "M04",
        "candidate_id": M04_CONVERGENCE_CHECK_ID,
        "evaluation_rows": int(len(m04.y_true)),
        "m04_accuracy": float(m04.metrics["accuracy"]),
        "iter60_accuracy": float(candidate.metrics["accuracy"]),
        "accuracy_delta": float(
            candidate.metrics["accuracy"] - m04.metrics["accuracy"]
        ),
        "m04_confusion_matrix": m04.metrics["confusion_matrix"],
        "iter60_confusion_matrix": candidate.metrics["confusion_matrix"],
        "m04_recall_speaker_0": float(m04.metrics["recall_speaker_0"]),
        "iter60_recall_speaker_0": float(
            candidate.metrics["recall_speaker_0"]
        ),
        "recall_speaker_0_delta": float(
            candidate.metrics["recall_speaker_0"]
            - m04.metrics["recall_speaker_0"]
        ),
        "m04_recall_speaker_1": float(m04.metrics["recall_speaker_1"]),
        "iter60_recall_speaker_1": float(
            candidate.metrics["recall_speaker_1"]
        ),
        "recall_speaker_1_delta": float(
            candidate.metrics["recall_speaker_1"]
            - m04.metrics["recall_speaker_1"]
        ),
        "correct_count_delta": int(
            candidate.metrics["correct_count"] - m04.metrics["correct_count"]
        ),
        "both_correct": paired["both_correct"],
        "both_wrong": paired["both_wrong"],
        "m04_wrong_to_iter60_correct": paired["m00_wrong_to_m01_correct"],
        "m04_correct_to_iter60_wrong": paired["m00_correct_to_m01_wrong"],
        "prediction_disagreement_count": paired[
            "prediction_disagreement_count"
        ],
        "prediction_disagreement_rate": paired[
            "prediction_disagreement_rate"
        ],
        "convergence_warning_raised": bool(
            convergence["convergence_warning_raised"]
        ),
        "n_iter": int(convergence["n_iter"]),
        "max_iter": int(convergence["max_iter"]),
        "final_loss": float(convergence["final_loss"]),
        "tail_loss_delta": float(convergence["tail_loss_delta"]),
        "tail_loss_slope": float(convergence["tail_loss_slope"]),
    }


def build_screening_comparisons(
    reference: ModelExperimentResult,
    candidates: Sequence[ModelExperimentResult],
) -> list[dict[str, Any]]:
    """Build M01-relative rows only for a complete M02--M04 internal run."""

    observed_ids = tuple(result.experiment_id for result in candidates)
    expected_ids = tuple(
        f"{model_id}_OFFICIAL" if reference.experiment_id.endswith("_OFFICIAL") else model_id
        for model_id in SCREENING_MODEL_IDS
    )
    if observed_ids != expected_ids:
        raise ValueError(
            f"screening comparison requires {expected_ids}; observed {observed_ids}"
        )
    return [compare_model_candidate_to_reference(reference, result) for result in candidates]


def _internal_result_fingerprint(result: ModelExperimentResult) -> str:
    payload = {
        "experiment_id": result.experiment_id,
        "model_config_fingerprint": result.fit_audit["model_config_fingerprint"],
        "X_train_fingerprint": result.fit_audit["X_train_fingerprint"],
        "X_evaluation_fingerprint": result.fit_audit["X_evaluation_fingerprint"],
        "y_train_fingerprint": result.fit_audit["y_train_fingerprint"],
        "y_evaluation_fingerprint": result.fit_audit["y_evaluation_fingerprint"],
        "validation_record_key_signature": record_key_signature(
            result.fit_audit["validation_record_keys"]
        ),
    }
    return config_fingerprint(payload)


def _promotion_selection_fingerprint(
    model_ids: Sequence[str], result_fingerprints: Mapping[str, str]
) -> str:
    return config_fingerprint(
        {
            "model_ids": list(model_ids),
            "internal_result_fingerprints": dict(result_fingerprints),
        }
    )


def freeze_promotion_selection(
    model_ids: Sequence[str],
    *,
    internal_results: Sequence[ModelExperimentResult],
    user_approved: bool = False,
) -> PromotionSelection:
    """Freeze one or two observed candidates before any Official evaluation."""

    if not user_approved:
        raise RuntimeError("user_approved=True is required to freeze promotion")
    selected = tuple(model_ids)
    if len(selected) not in {1, 2} or len(set(selected)) != len(selected):
        raise ValueError("promotion must contain one or two unique model IDs")
    if any(model_id not in SCREENING_MODEL_IDS for model_id in selected):
        raise ValueError("promotion contains a model outside frozen M02--M04")
    result_by_id = {result.experiment_id: result for result in internal_results}
    fingerprints: dict[str, str] = {}
    for model_id in selected:
        result = result_by_id.get(model_id)
        if result is None:
            raise ValueError(f"{model_id} has no executed internal result")
        if result.fit_audit.get("evaluation_scope") != "Training/internal_validation":
            raise ValueError(f"{model_id} is not an internal screening result")
        if dict(result.model_config) != dict(SCREENING_MODEL_CONFIGS[model_id]):
            raise ValueError(f"{model_id} precommit config changed")
        fingerprints[model_id] = _internal_result_fingerprint(result)
    frozen_fingerprints = MappingProxyType(dict(fingerprints))
    return PromotionSelection(
        model_ids=selected,
        internal_result_fingerprints=frozen_fingerprints,
        selection_fingerprint=_promotion_selection_fingerprint(
            selected, frozen_fingerprints
        ),
    )


def run_promoted_official_models(
    prepared: PreparedComparison,
    *,
    promotion: PromotionSelection,
    allow_fit: bool = False,
    user_approved: bool = False,
) -> tuple[ModelExperimentResult, ...]:
    """Run only the pre-frozen one or two candidates on Official Validation."""

    if not user_approved:
        raise RuntimeError("user_approved=True is required for promoted Official run")
    if not allow_fit:
        raise RuntimeError("allow_fit=True is required for promoted Official run")
    if prepared.evaluation_scope != "Official Validation":
        raise ValueError("promoted models require Official Validation prepared data")
    if (
        len(promotion.model_ids) not in {1, 2}
        or len(set(promotion.model_ids)) != len(promotion.model_ids)
        or any(model_id not in SCREENING_MODEL_IDS for model_id in promotion.model_ids)
        or set(promotion.internal_result_fingerprints) != set(promotion.model_ids)
    ):
        raise ValueError(
            "promotion must contain one or two frozen M02--M04 model IDs"
        )
    expected_fingerprint = _promotion_selection_fingerprint(
        promotion.model_ids, promotion.internal_result_fingerprints
    )
    if promotion.selection_fingerprint != expected_fingerprint:
        raise ValueError("promotion selection fingerprint mismatch")
    results: list[ModelExperimentResult] = []
    for model_id in promotion.model_ids:
        result = _run_model(
            prepared,
            experiment_id=f"{model_id}_OFFICIAL",
            model_name=SCREENING_MODEL_NAMES[model_id],
            model_config=SCREENING_MODEL_CONFIGS[model_id],
            model=build_screening_model(model_id),
            allow_fit=True,
        )
        results.append(
            replace(
                result,
                fit_audit={
                    **dict(result.fit_audit),
                    "promotion_fingerprint": promotion.selection_fingerprint,
                },
            )
        )
    return tuple(results)


def _slice_accuracy(summary: Any, label: str) -> float | None:
    if isinstance(summary, pd.DataFrame):
        if summary.empty or not {"slice", "accuracy"} <= set(summary.columns):
            return None
        match = summary.loc[summary["slice"].astype(str) == label, "accuracy"]
        return None if match.empty else float(match.iloc[0])
    if isinstance(summary, Mapping):
        value = summary.get(label)
        if isinstance(value, Mapping) and "accuracy" in value:
            return float(value["accuracy"])
    return None


def _overlap_mask(tracking: pd.DataFrame) -> np.ndarray:
    """Return whether each half-open utterance interval overlaps a peer call row."""

    required = {"call_id", "startAt_ms", "endAt_ms"}
    missing = sorted(required - set(tracking.columns))
    if missing:
        raise ValueError(f"tracking is missing overlap columns: {missing}")
    frame = tracking.loc[:, ["call_id", "startAt_ms", "endAt_ms"]].copy()
    frame["_row"] = np.arange(len(frame), dtype=np.int64)
    frame["startAt_ms"] = pd.to_numeric(frame["startAt_ms"], errors="raise")
    frame["endAt_ms"] = pd.to_numeric(frame["endAt_ms"], errors="raise")
    if not np.isfinite(frame[["startAt_ms", "endAt_ms"]].to_numpy()).all():
        raise ValueError("tracking intervals contain NaN or inf")
    if (frame["endAt_ms"] < frame["startAt_ms"]).any():
        raise ValueError("tracking contains a negative interval")
    ordered = frame.sort_values(
        ["call_id", "startAt_ms", "endAt_ms", "_row"], kind="stable"
    )
    previous_max_end = ordered.groupby("call_id", sort=False)["endAt_ms"].transform(
        lambda values: values.cummax().shift(1)
    )
    next_start = ordered.groupby("call_id", sort=False)["startAt_ms"].shift(-1)
    ordered["_overlap"] = (
        (ordered["startAt_ms"] < previous_max_end)
        | (next_start < ordered["endAt_ms"])
    ).fillna(False)
    return (
        ordered.sort_values("_row", kind="stable")["_overlap"]
        .to_numpy(dtype=bool)
    )


def _accuracy_rows(
    labels: Sequence[str], truth: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    label_array = np.asarray(labels).astype(str)
    rows: list[dict[str, Any]] = []
    for label in pd.unique(label_array):
        selected = label_array == label
        rows.append(
            {
                "slice": str(label),
                "count": int(np.sum(selected)),
                "accuracy": float(np.mean(truth[selected] == prediction[selected])),
            }
        )
    return pd.DataFrame(rows, columns=("slice", "count", "accuracy"))


def build_standalone_diagnostic_report(
    result: ModelExperimentResult,
    *,
    training_tracking: pd.DataFrame,
    evaluation_tracking: pd.DataFrame,
    high_confidence_threshold: float = 0.90,
) -> dict[str, Any]:
    """Build duration, overlap, and confidence diagnostics without PR #16."""

    required = {"record_key", "call_id", "startAt_ms", "endAt_ms"}
    for label, tracking in (
        ("training", training_tracking),
        ("evaluation", evaluation_tracking),
    ):
        missing = sorted(required - set(tracking.columns))
        if missing:
            raise ValueError(f"{label} tracking is missing columns: {missing}")
        keys = tracking["record_key"].astype(str).to_numpy()
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"{label} tracking record_key must be one-to-one")
    training_keys = training_tracking["record_key"].astype(str).to_numpy()
    evaluation_keys = evaluation_tracking["record_key"].astype(str).to_numpy()
    if not np.array_equal(
        training_keys, np.asarray(result.fit_audit["training_record_keys"]).astype(str)
    ):
        raise ValueError("training tracking record_key alignment mismatch")
    if not np.array_equal(
        evaluation_keys,
        np.asarray(result.fit_audit["validation_record_keys"]).astype(str),
    ):
        raise ValueError("evaluation tracking record_key alignment mismatch")

    truth = np.asarray(result.y_true, dtype=np.int64)
    prediction = np.asarray(result.y_pred, dtype=np.int64)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    if len(evaluation_tracking) != len(truth):
        raise ValueError("evaluation tracking/result row mismatch")
    if "duration_ms" in training_tracking.columns:
        training_duration = pd.to_numeric(
            training_tracking["duration_ms"], errors="raise"
        ).to_numpy(dtype=np.float64)
        evaluation_duration = pd.to_numeric(
            evaluation_tracking["duration_ms"], errors="raise"
        ).to_numpy(dtype=np.float64)
    elif "duration_sec" in training_tracking.columns:
        training_duration = (
            pd.to_numeric(training_tracking["duration_sec"], errors="raise")
            .to_numpy(dtype=np.float64)
            * 1_000.0
        )
        evaluation_duration = (
            pd.to_numeric(evaluation_tracking["duration_sec"], errors="raise")
            .to_numpy(dtype=np.float64)
            * 1_000.0
        )
    else:
        training_duration = (
            training_tracking["endAt_ms"].to_numpy(dtype=np.float64)
            - training_tracking["startAt_ms"].to_numpy(dtype=np.float64)
        )
        evaluation_duration = (
            evaluation_tracking["endAt_ms"].to_numpy(dtype=np.float64)
            - evaluation_tracking["startAt_ms"].to_numpy(dtype=np.float64)
        )
    if not np.isfinite(training_duration).all() or not np.isfinite(
        evaluation_duration
    ).all():
        raise ValueError("duration values contain NaN or inf")
    boundaries = np.quantile(training_duration, [0.05, 0.25, 0.75, 0.95])
    duration_names = np.asarray(
        ["q00_q05", "q05_q25", "q25_q75", "q75_q95", "q95_q100"]
    )
    duration_labels = duration_names[
        np.searchsorted(boundaries, evaluation_duration, side="right")
    ]
    duration_summary = _accuracy_rows(duration_labels, truth, prediction)

    overlap = _overlap_mask(evaluation_tracking)
    overlap_summary = _accuracy_rows(
        np.where(overlap, "overlap", "no_overlap"), truth, prediction
    )
    confidence = probabilities.max(axis=1)
    confidence_labels = pd.cut(
        confidence,
        bins=[0.0, 0.60, 0.70, 0.80, 0.90, 1.0],
        labels=["<0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", ">=0.90"],
        include_lowest=True,
    ).astype(str)
    confidence_summary = _accuracy_rows(confidence_labels, truth, prediction)
    high_confidence_error_mask = (prediction != truth) & (
        confidence >= high_confidence_threshold
    )
    high_confidence_errors = evaluation_tracking.loc[
        high_confidence_error_mask
    ].copy()
    high_confidence_errors["y_true"] = truth[high_confidence_error_mask]
    high_confidence_errors["y_pred"] = prediction[high_confidence_error_mask]
    high_confidence_errors["confidence"] = confidence[high_confidence_error_mask]
    return {
        "global_metrics": dict(result.metrics),
        "duration_boundaries_ms": {
            name: float(value)
            for name, value in zip(("q05", "q25", "q75", "q95"), boundaries)
        },
        "duration_summary": duration_summary,
        "overlap_summary": overlap_summary,
        "confidence_summary": confidence_summary,
        "high_confidence_errors": high_confidence_errors,
        "high_confidence_error_count": int(np.sum(high_confidence_error_mask)),
        "high_confidence_threshold": float(high_confidence_threshold),
    }


def augment_model_report(
    result: ModelExperimentResult, base_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Add model fields to a validated diagnostic report.

    Checkpoints A--H supply :func:`build_standalone_diagnostic_report` output.
    The adapter remains compatible with the earlier report mapping contract.
    """

    report = dict(base_report)
    high_confidence_errors = report.get("high_confidence_errors")
    if "high_confidence_error_count" in report:
        high_confidence_count = int(report["high_confidence_error_count"])
    elif high_confidence_errors is not None:
        high_confidence_count = int(len(high_confidence_errors))
    else:
        high_confidence_count = 0
    overlap_summary = report.get("overlap_summary", {})
    report.update(
        {
            "experiment_id": result.experiment_id,
            "model_name": result.model_name,
            "model_config": dict(result.model_config),
            "model_config_fingerprint": result.fit_audit[
                "model_config_fingerprint"
            ],
            "feature_names": result.feature_names,
            "feature_dimension": len(result.feature_names),
            "train_shape": result.train_shape,
            "validation_shape": result.evaluation_shape,
            "global_metrics": dict(result.metrics),
            "correct_count": int(result.metrics["correct_count"]),
            "high_confidence_error_count": high_confidence_count,
            "overlap_accuracy": _slice_accuracy(overlap_summary, "overlap"),
            "no_overlap_accuracy": _slice_accuracy(overlap_summary, "no_overlap"),
            "confidence_interpretation": (
                "diagnostic_only; probability magnitudes are not a calibration "
                "comparison between model classes"
            ),
            "runtime_seconds": result.fit_audit.get("runtime_seconds"),
            "peak_memory_bytes": result.fit_audit.get("peak_memory_bytes"),
        }
    )
    return report


def build_decision_table(
    reports: Sequence[Mapping[str, Any] | None],
) -> pd.DataFrame:
    """Create rows only for observed reports and never infer a winner."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        if report is None:
            continue
        global_metrics = dict(report.get("global_metrics", {}))
        rows.append(
            {
                "experiment_id": report["experiment_id"],
                "model_name": report["model_name"],
                "model_config": report.get("model_config"),
                "feature_dimension": report["feature_dimension"],
                "train_rows": int(report["train_shape"][0]),
                "evaluation_rows": int(report["validation_shape"][0]),
                "accuracy": global_metrics.get("accuracy"),
                "confusion_matrix": global_metrics.get("confusion_matrix"),
                "correct_count": report.get(
                    "correct_count", global_metrics.get("correct_count")
                ),
                "recall_speaker_0": global_metrics.get(
                    "recall_speaker_0", global_metrics.get("recall_0")
                ),
                "recall_speaker_1": global_metrics.get(
                    "recall_speaker_1", global_metrics.get("recall_1")
                ),
                "high_confidence_error_count": report.get(
                    "high_confidence_error_count"
                ),
                "duration_summary": report.get("duration_summary"),
                "overlap_accuracy": report.get("overlap_accuracy"),
                "no_overlap_accuracy": report.get("no_overlap_accuracy"),
                "confidence_summary": report.get("confidence_summary"),
                "runtime_seconds": report.get("runtime_seconds"),
                "peak_memory_bytes": report.get("peak_memory_bytes"),
                "notes": report.get("notes", ""),
            }
        )
    return pd.DataFrame(rows, columns=DECISION_TABLE_COLUMNS)


def validate_frozen_e05_row_level_result(
    result: Any,
    *,
    reference: Mapping[str, Any] = FROZEN_E05_OFFICIAL_REFERENCE,
) -> dict[str, Any]:
    """Verify a row-level E05 result against the immutable aggregate ledger."""

    if tuple(result.feature_names) != E05_FEATURE_NAMES:
        raise ValueError("frozen E05 row-level result has a different feature contract")
    reference_values = dict(reference)
    expected_train_shape = (int(reference_values["training_rows"]), 42)
    expected_validation_shape = (int(reference_values["validation_rows"]), 42)
    if tuple(result.train_shape) != expected_train_shape:
        raise ValueError(
            f"frozen E05 Training shape mismatch: expected {expected_train_shape}, "
            f"observed {tuple(result.train_shape)}"
        )
    if tuple(result.evaluation_shape) != expected_validation_shape:
        raise ValueError(
            "frozen E05 Validation shape mismatch: expected "
            f"{expected_validation_shape}, observed {tuple(result.evaluation_shape)}"
        )
    if type(result.model) is not LogisticRegression:
        raise ValueError("frozen E05 row-level result must use LogisticRegression")
    model_params = result.model.get_params()
    if any(
        model_params[name] != expected
        for name, expected in M00_LOGISTIC_REGRESSION_CONFIG.items()
    ):
        raise ValueError("frozen E05 LogisticRegression config mismatch")

    expected_fit_audit = {
        "scaler_fit_scope": "Training",
        "model_fit_scope": "Training",
        "evaluation_scope": "Validation",
        "scaler_fit_rows": expected_train_shape[0],
        "model_fit_rows": expected_train_shape[0],
        "evaluation_rows": expected_validation_shape[0],
    }
    for name, expected in expected_fit_audit.items():
        if result.fit_audit.get(name) != expected:
            raise ValueError(
                f"frozen E05 {name} mismatch: expected {expected!r}, "
                f"observed {result.fit_audit.get(name)!r}"
            )
    keys = np.asarray(result.fit_audit["validation_record_keys"]).astype(str)
    truth = np.asarray(result.y_true, dtype=np.int64)
    prediction = np.asarray(result.y_pred, dtype=np.int64)
    probabilities = validate_binary_probabilities(
        result.probabilities, expected_rows=len(truth)
    )
    if len(keys) != len(truth) or len(np.unique(keys)) != len(keys):
        raise ValueError("frozen E05 record_key ledger is not one-to-one")

    observed = calculate_prediction_metrics(truth, prediction)
    confidence = probabilities.max(axis=1)
    observed["high_confidence_error_count"] = int(
        np.sum((prediction != truth) & (confidence >= 0.90))
    )
    exact_fields = ("sample_count", "correct_count", "high_confidence_error_count")
    reference_values["sample_count"] = reference_values["validation_rows"]
    for name in exact_fields:
        if observed[name] != reference_values[name]:
            raise ValueError(
                f"frozen E05 {name} mismatch: "
                f"expected {reference_values[name]!r}, observed {observed[name]!r}"
            )
    if not np.array_equal(
        np.asarray(observed["confusion_matrix"], dtype=np.int64),
        np.asarray(reference_values["confusion_matrix"], dtype=np.int64),
    ):
        raise ValueError(
            "frozen E05 confusion_matrix mismatch: "
            f"expected {reference_values['confusion_matrix']!r}, "
            f"observed {observed['confusion_matrix']!r}"
        )
    reference_matrix = np.asarray(reference_values["confusion_matrix"], dtype=np.int64)
    rate_counts = {
        "accuracy": (int(reference_values["correct_count"]), len(truth)),
        "recall_speaker_0": (int(reference_matrix[0, 0]), int(reference_matrix[0].sum())),
        "recall_speaker_1": (int(reference_matrix[1, 1]), int(reference_matrix[1].sum())),
    }
    for name, (numerator, denominator) in rate_counts.items():
        if not _rate_matches_count(reference_values[name], numerator, denominator):
            raise ValueError(
                f"frozen E05 {name} reference mismatch: stored "
                f"{reference_values[name]!r} disagrees with exact count "
                f"{numerator}/{denominator}; observed {observed[name]!r}"
            )
    return observed


def compare_to_frozen_e05_official(
    m01: ModelExperimentResult,
    *,
    frozen_e05_result: ModelExperimentResult,
) -> dict[str, Any]:
    """Compare M01 to a verified row-level E05 result after exact alignment."""

    validate_frozen_e05_row_level_result(frozen_e05_result)
    if not np.array_equal(
        np.asarray(frozen_e05_result.y_true, dtype=np.int64), m01.y_true
    ):
        raise ValueError("frozen E05/M01 y alignment mismatch")
    paired = calculate_paired_prediction_changes(
        y_true=m01.y_true,
        m00_pred=frozen_e05_result.y_pred,
        m01_pred=m01.y_pred,
        m00_record_keys=frozen_e05_result.fit_audit["validation_record_keys"],
        m01_record_keys=m01.fit_audit["validation_record_keys"],
    )
    reference = FROZEN_E05_OFFICIAL_REFERENCE
    return {
        "accuracy_delta": float(m01.metrics["accuracy"] - reference["accuracy"]),
        "correct_count_delta": int(
            m01.metrics["correct_count"] - reference["correct_count"]
        ),
        **paired,
        "statistical_significance_claimed": False,
        "winner_declared": False,
    }


def validate_m00_official_reproducibility(
    result: ModelExperimentResult,
    *,
    reference: Mapping[str, Any] = FROZEN_E05_OFFICIAL_REFERENCE,
) -> dict[str, Any]:
    """Gate M01 interpretation on same-run M00 aggregate reproducibility.

    This validates observed aggregate metrics only.  It neither requires nor
    fabricates a historical row-level prediction artifact.
    """

    expected = dict(reference)
    if result.experiment_id != "M00_OFFICIAL":
        raise ValueError("M00 reproducibility requires M00_OFFICIAL result")
    if result.feature_names != E05_FEATURE_NAMES:
        raise ValueError("M00 Official feature contract differs from E05 42D")
    if type(result.model) is not LogisticRegression:
        raise ValueError("M00 Official model must be LogisticRegression")
    if dict(result.model_config) != dict(M00_LOGISTIC_REGRESSION_CONFIG):
        raise ValueError("M00 Official LogisticRegression config changed")
    expected_train_shape = (int(expected["training_rows"]), 42)
    expected_validation_shape = (int(expected["validation_rows"]), 42)
    if result.train_shape != expected_train_shape:
        raise ValueError(
            f"M00 Official Training shape mismatch: expected {expected_train_shape}, "
            f"observed {result.train_shape}"
        )
    if result.evaluation_shape != expected_validation_shape:
        raise ValueError(
            "M00 Official Validation shape mismatch: expected "
            f"{expected_validation_shape}, observed {result.evaluation_shape}"
        )
    if result.fit_audit.get("scaler_fit_scope") != "full_training_only":
        raise ValueError("M00 Official scaler was not fit on full Training only")
    if result.fit_audit.get("evaluation_scope") != "Official Validation":
        raise ValueError("M00 Official evaluation scope mismatch")

    observed = calculate_prediction_metrics(result.y_true, result.y_pred)
    probabilities = validate_binary_probabilities(
        result.probabilities,
        expected_rows=len(result.y_true),
        classes=result.model.classes_,
    )
    confidence = probabilities.max(axis=1)
    observed["high_confidence_error_count"] = int(
        np.sum((result.y_pred != result.y_true) & (confidence >= 0.90))
    )
    observed_provenance = {
        name: result.fit_audit.get(name)
        for name in (
            "model_config_fingerprint",
            "X_train_fingerprint",
            "X_evaluation_fingerprint",
            "y_evaluation_fingerprint",
        )
    }
    exact_fields = (
        "sample_count",
        "correct_count",
        "high_confidence_error_count",
    )
    expected["sample_count"] = int(expected["validation_rows"])
    for name in exact_fields:
        if observed[name] != int(expected[name]):
            raise ValueError(
                f"M00 Official {name} mismatch: expected {expected[name]!r}, "
                f"observed {observed[name]!r}; "
                f"observed_provenance={observed_provenance!r}"
            )
    if not np.array_equal(
        np.asarray(observed["confusion_matrix"], dtype=np.int64),
        np.asarray(expected["confusion_matrix"], dtype=np.int64),
    ):
        raise ValueError(
            "M00 Official confusion_matrix mismatch: "
            f"expected {expected['confusion_matrix']!r}, "
            f"observed {observed['confusion_matrix']!r}; "
            f"observed_provenance={observed_provenance!r}"
        )
    reference_matrix = np.asarray(expected["confusion_matrix"], dtype=np.int64)
    rate_counts = {
        "accuracy": (int(expected["correct_count"]), int(expected["validation_rows"])),
        "recall_speaker_0": (int(reference_matrix[0, 0]), int(reference_matrix[0].sum())),
        "recall_speaker_1": (int(reference_matrix[1, 1]), int(reference_matrix[1].sum())),
    }
    for name, (numerator, denominator) in rate_counts.items():
        if not _rate_matches_count(expected[name], numerator, denominator):
            raise ValueError(
                f"M00 Official {name} reference mismatch: stored {expected[name]!r} "
                f"disagrees with exact count {numerator}/{denominator}; "
                f"observed {observed[name]!r}"
            )
    return observed


def require_existing_cache_files(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Fail on cache miss; never start extraction or create directories."""

    resolved = tuple(Path(path) for path in paths)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required cache file is missing and automatic extraction is disabled: "
            + ", ".join(missing)
        )
    return resolved


__all__ = [
    "CACHE_BACKEND",
    "CANDIDATE_CACHE_SCHEMA",
    "DECISION_TABLE_COLUMNS",
    "E05_EXTRA_FEATURE_NAMES",
    "E05_FEATURE_NAMES",
    "EXPECTED_INTERNAL_SPLIT",
    "EXPECTED_OFFICIAL_CALLS",
    "EXPECTED_OFFICIAL_ROWS",
    "FROZEN_E05_OFFICIAL_REFERENCE",
    "M00_LOGISTIC_REGRESSION_CONFIG",
    "M01_HGB_CONFIG",
    "M02_EXTRATREES_CONFIG",
    "M03_QDA_CONFIG",
    "M04_CONVERGENCE_CHECK_CONFIG",
    "M04_CONVERGENCE_CHECK_ID",
    "M04_MLP_CONFIG",
    "MFCC40_FEATURE_NAMES",
    "MFCC_CACHE_SCHEMA",
    "MFCC_CONFIG",
    "MANIFEST_CACHE_SCHEMA",
    "ModelExperimentResult",
    "PreparedComparison",
    "PromotionSelection",
    "SCREENING_MODEL_CONFIGS",
    "SCREENING_MODEL_IDS",
    "SCREENING_MODEL_NAMES",
    "StandaloneCachePaths",
    "StandaloneE05Cache",
    "ValidatedE05Data",
    "augment_model_report",
    "build_decision_table",
    "build_m00_model",
    "build_m01_model",
    "build_m04_convergence_check_model",
    "build_screening_comparisons",
    "build_screening_model",
    "build_standalone_diagnostic_report",
    "calculate_paired_prediction_changes",
    "calculate_prediction_metrics",
    "compare_model_results",
    "compare_official_model_results",
    "compare_model_candidate_to_reference",
    "compare_m04_convergence_check",
    "compare_to_frozen_e05_official",
    "config_fingerprint",
    "create_fixed_internal_split",
    "freeze_promotion_selection",
    "prepare_internal_comparison",
    "prepare_official_comparison",
    "record_key_signature",
    "require_existing_cache_files",
    "run_internal_m00",
    "run_internal_m01",
    "run_internal_m04_reference",
    "run_internal_m04_convergence_check",
    "run_internal_screening",
    "run_official_m00",
    "run_official_m01",
    "run_promoted_official_models",
    "standalone_group_split_indices",
    "standalone_training_cache_paths",
    "standalone_validation_cache_paths",
    "summarize_mlp_convergence",
    "load_standalone_e05_cache",
    "validate_e05_contract",
    "validate_frozen_e05_row_level_result",
    "validate_m00_official_reproducibility",
]
