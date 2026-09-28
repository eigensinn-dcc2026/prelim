"""One-utterance eGeMAPSv02 extraction and paired Stage 4-A preparation.

Importing this module does not extract audio, write a cache, or fit a model.
The caller explicitly invokes those operations after supplying aligned rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import opensmile
from sklearn.preprocessing import StandardScaler

from . import mission2_acoustic_feature_expansion as acoustic
from .mission2_model_experiments import (
    EXPECTED_INTERNAL_SPLIT,
    M04_MLP_CONFIG,
    calculate_paired_prediction_changes,
    calculate_prediction_metrics,
    config_fingerprint,
    create_fixed_internal_split,
    record_key_signature,
    standalone_group_split_indices,
)


REPRESENTATION_ID = "EGEMAPSV02_FUNCTIONALS_88D"
CACHE_SCHEMA = "mission2_egemaps_v1"
EXTRACTION_CONFIG = {
    "feature_set": "eGeMAPSv02",
    "feature_level": "Functionals",
    "sample_rate": 8_000,
    "channels": 0,
    "resample": False,
}


@dataclass(frozen=True)
class EgemapsData:
    X: np.ndarray
    record_keys: np.ndarray
    call_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]


@lru_cache(maxsize=1)
def _smile() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=EXTRACTION_CONFIG["sample_rate"],
        channels=EXTRACTION_CONFIG["channels"],
        resample=EXTRACTION_CONFIG["resample"],
    )


def get_egemaps_feature_names() -> tuple[str, ...]:
    names = tuple(_smile().feature_names)
    if len(names) != 88 or len(set(names)) != 88 or not all(names):
        raise ValueError("eGeMAPSv02 feature names must be 88 unique non-empty columns")
    return names


def extract_egemaps(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract only the supplied mono utterance; reject unavailable functionals."""

    signal = np.asarray(waveform)
    if sample_rate != 8_000:
        raise ValueError("eGeMAPSv02 Stage 4-A requires native 8 kHz audio")
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional mono array")
    if not np.issubdtype(signal.dtype, np.number) or not np.isfinite(signal).all():
        raise ValueError("waveform must contain finite numeric samples")
    smile = _smile()
    names = get_egemaps_feature_names()
    frame = smile.process_signal(signal, sample_rate)
    if frame.shape != (1, 88) or tuple(frame.columns) != names:
        raise ValueError("eGeMAPSv02 must return one row in the exact 88-feature order")
    raw = frame.to_numpy(copy=True)
    if not np.isfinite(raw).all():
        raise ValueError("eGeMAPSv02 returned NaN or inf; audio may be too short")
    values = raw.astype(np.float32)
    if values.shape != (1, 88) or not np.isfinite(values).all():
        raise ValueError("eGeMAPSv02 float32 conversion lost finite 88D values")
    return values[0], names


def _metadata(names: tuple[str, ...], keys: np.ndarray, calls: np.ndarray,
              source_provenance: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA,
        "representation_id": REPRESENTATION_ID,
        "opensmile_version": opensmile.__version__,
        "feature_set": EXTRACTION_CONFIG["feature_set"],
        "feature_level": EXTRACTION_CONFIG["feature_level"],
        "feature_count": 88,
        "feature_name_fingerprint": config_fingerprint({"names": names}),
        "extraction_config": dict(EXTRACTION_CONFIG),
        "config_fingerprint": config_fingerprint(EXTRACTION_CONFIG),
        "sample_rate": 8_000,
        "source_row_count": len(keys),
        "record_key_signature": record_key_signature(keys),
        "call_id_signature": config_fingerprint({"calls": calls.tolist()}),
        "source_provenance": source_provenance,
    }


def make_egemaps_data(
    X: np.ndarray, record_keys: Sequence[str], call_ids: Sequence[str],
    feature_names: Sequence[str], *, source_provenance: str,
) -> EgemapsData:
    """Validate an aligned extracted matrix before it enters cache or modeling."""

    matrix = np.asarray(X)
    raw_keys = tuple(record_keys)
    raw_calls = tuple(call_ids)
    if any(not isinstance(key, str) or not key for key in raw_keys) or len(set(raw_keys)) != len(raw_keys):
        raise ValueError("record_key must be unique non-empty strings")
    if any(not isinstance(call, str) or not call for call in raw_calls):
        raise ValueError("call_id must contain non-empty strings")
    keys = np.asarray(raw_keys, dtype=str)
    calls = np.asarray(raw_calls, dtype=str)
    names = tuple(feature_names)
    if names != get_egemaps_feature_names():
        raise ValueError("eGeMAPSv02 feature names/order mismatch")
    if matrix.ndim != 2 or matrix.shape != (len(keys), 88) or matrix.dtype != np.float32:
        raise ValueError("eGeMAPSv02 cache requires an N x 88 float32 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("eGeMAPSv02 cache contains NaN or inf")
    if keys.ndim != 1 or calls.shape != keys.shape or not isinstance(source_provenance, str) or not source_provenance:
        raise ValueError("record_key/call_id shape or source provenance mismatch")
    metadata = _metadata(names, keys, calls, source_provenance)
    return EgemapsData(matrix, keys, calls, names, metadata)


def build_egemaps_cache(
    path: str | Path, X: np.ndarray, record_keys: Sequence[str],
    call_ids: Sequence[str], feature_names: Sequence[str], *, source_provenance: str,
) -> EgemapsData:
    """Persist one fully validated cache; never extract implicitly."""

    data = make_egemaps_data(
        X, record_keys, call_ids, feature_names, source_provenance=source_provenance
    )
    with Path(path).open("xb") as handle:
        np.savez(
            handle, X=data.X, record_keys=data.record_keys,
            call_ids=data.call_ids, feature_names=np.asarray(data.feature_names),
            metadata=np.asarray(json.dumps(data.metadata, sort_keys=True)),
        )
    return data


def load_egemaps_cache(
    path: str | Path, expected_record_keys: Sequence[str],
    expected_call_ids: Sequence[str], *, source_provenance: str,
) -> EgemapsData:
    """Fail closed on stale metadata, row order, feature order, or values."""

    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"X", "record_keys", "call_ids", "feature_names", "metadata"}:
            raise ValueError("eGeMAPSv02 cache schema mismatch")
        matrix = stored["X"]
        keys = stored["record_keys"]
        calls = stored["call_ids"]
        names = tuple(stored["feature_names"].tolist())
        metadata = json.loads(stored["metadata"].item())
    if not np.array_equal(keys, np.asarray(expected_record_keys)):
        raise ValueError("eGeMAPSv02 record_key order mismatch")
    if not np.array_equal(calls, np.asarray(expected_call_ids)):
        raise ValueError("eGeMAPSv02 call_id order mismatch")
    data = make_egemaps_data(
        matrix, keys, calls, names, source_provenance=source_provenance
    )
    if metadata != data.metadata:
        raise ValueError("eGeMAPSv02 cache config/provenance metadata mismatch")
    return data


def run_standard_representation_comparison(
    prepared_59d: acoustic.PreparedRepresentation,
    egemaps: EgemapsData,
    y: Sequence[int], record_keys: Sequence[str], call_ids: Sequence[str],
    *, expected_split_contract: Mapping[str, int] | None = EXPECTED_INTERNAL_SPLIT,
) -> dict[str, Any]:
    """Fit two same-run M04-100 models only after all alignment checks pass."""

    expected_names_59 = (
        *acoustic.CANONICAL_52D_FEATURE_NAMES,
        *(name for block in acoustic.RESIDUAL_COMBINATION_SPECS["AF10_AF11_COMBINATION"]
          for name in acoustic.BLOCK_BY_ID[block].feature_names),
    )
    if prepared_59d.representation_id != "AF10_AF11_COMBINATION" or prepared_59d.feature_names != expected_names_59:
        raise ValueError("59D AF10+AF11 reference contract mismatch")
    expected_config = {**dict(M04_MLP_CONFIG), "max_iter": 100}
    if dict(acoustic.M04_ITER100_FEATURE_SCREENING_CONFIG) != expected_config:
        raise ValueError("M04-100 model config mismatch")
    labels = np.asarray(y)
    keys = np.asarray(record_keys)
    calls = np.asarray(call_ids)
    if labels.ndim != 1 or labels.shape != keys.shape or calls.shape != keys.shape:
        raise ValueError("row/label/call_id shape mismatch")
    if not np.array_equal(keys, egemaps.record_keys) or not np.array_equal(calls, egemaps.call_ids):
        raise ValueError("eGeMAPSv02 row record_key/call_id alignment mismatch")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("labels must be binary 0/1")
    if egemaps.metadata != _metadata(egemaps.feature_names, egemaps.record_keys,
                                     egemaps.call_ids, egemaps.metadata["source_provenance"]):
        raise ValueError("eGeMAPSv02 extraction config/provenance mismatch")
    if egemaps.feature_names != get_egemaps_feature_names() or egemaps.X.shape != (len(keys), 88):
        raise ValueError("eGeMAPSv02 feature/row contract mismatch")
    if egemaps.X.dtype != np.float32 or not np.isfinite(egemaps.X).all():
        raise ValueError("eGeMAPSv02 dtype/finite contract mismatch")
    train_idx, val_idx, audit = create_fixed_internal_split(
        calls, split_function=standalone_group_split_indices,
        expected_contract=expected_split_contract,
    )
    if not np.array_equal(prepared_59d.train_idx, train_idx) or not np.array_equal(prepared_59d.validation_idx, val_idx) or dict(prepared_59d.split_audit) != audit:
        raise ValueError("59D fixed call_id split mismatch")
    if not np.array_equal(prepared_59d.y_train, labels[train_idx]) or not np.array_equal(prepared_59d.y_evaluation, labels[val_idx]):
        raise ValueError("59D label alignment mismatch")
    if not np.array_equal(prepared_59d.training_record_keys, keys[train_idx]) or not np.array_equal(prepared_59d.evaluation_record_keys, keys[val_idx]):
        raise ValueError("59D record_key alignment mismatch")
    if prepared_59d.X_train_scaled.shape != (len(train_idx), 59) or prepared_59d.X_evaluation_scaled.shape != (len(val_idx), 59) or int(prepared_59d.scaler.n_samples_seen_) != len(train_idx):
        raise ValueError("59D scaler fit scope or dimension mismatch")
    if not np.isfinite(prepared_59d.X_train_scaled).all() or not np.isfinite(prepared_59d.X_evaluation_scaled).all():
        raise ValueError("59D scaled matrix contains NaN or inf")

    # The existing preparation helper intentionally whitelists handcrafted
    # names. Preserve that guard and apply its Training-only scaler policy here.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(egemaps.X[train_idx]).astype(np.float32, copy=False)
    X_val = scaler.transform(egemaps.X[val_idx]).astype(np.float32, copy=False)
    if int(scaler.n_samples_seen_) != len(train_idx) or not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("eGeMAPSv02 Training-only scaler contract failed")
    prepared_88d = acoustic.PreparedRepresentation(
        representation_id=REPRESENTATION_ID,
        feature_names=egemaps.feature_names,
        X_train_scaled=X_train,
        X_evaluation_scaled=X_val,
        y_train=labels[train_idx],
        y_evaluation=labels[val_idx],
        training_record_keys=keys[train_idx],
        evaluation_record_keys=keys[val_idx],
        train_idx=train_idx.copy(),
        validation_idx=val_idx.copy(),
        scaler=scaler,
        split_audit=audit,
    )
    reference = acoustic._run_prepared_representation(
        prepared_59d, experiment_id="AF10_AF11_COMBINATION", allow_fit=True
    )
    candidate = acoustic._run_prepared_representation(
        prepared_88d, experiment_id=REPRESENTATION_ID, allow_fit=True
    )
    if dict(reference.model_config) != dict(candidate.model_config) or dict(candidate.model_config) != expected_config:
        raise ValueError("M04-100 model config changed during fit")
    paired = calculate_paired_prediction_changes(
        y_true=labels[val_idx], m00_pred=reference.y_pred, m01_pred=candidate.y_pred,
        m00_record_keys=reference.fit_audit["validation_record_keys"],
        m01_record_keys=candidate.fit_audit["validation_record_keys"],
    )
    reports = {}
    for name, result, prepared in (("59D", reference, prepared_59d),
                                   ("eGeMAPSv02", candidate, prepared_88d)):
        train_metrics = calculate_prediction_metrics(
            prepared.y_train, result.model.predict(prepared.X_train_scaled)
        )
        reports[name] = {
            "representation": prepared.representation_id,
            "dimension": len(prepared.feature_names),
            "train_accuracy": train_metrics["accuracy"],
            "internal_validation_accuracy": result.metrics["accuracy"],
            "generalization_gap": train_metrics["accuracy"] - result.metrics["accuracy"],
            "confusion_matrix": result.metrics["confusion_matrix"],
            "recall_speaker_0": result.metrics["recall_speaker_0"],
            "recall_speaker_1": result.metrics["recall_speaker_1"],
        }
    return {
        "reports": reports,
        "both_correct": paired["both_correct"],
        "both_wrong": paired["both_wrong"],
        "59d_wrong_egemaps_correct": paired["m00_wrong_to_m01_correct"],
        "59d_correct_egemaps_wrong": paired["m00_correct_to_m01_wrong"],
        "net_correct_gain": paired["m00_wrong_to_m01_correct"] - paired["m00_correct_to_m01_wrong"],
    }
