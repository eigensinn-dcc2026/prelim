"""Cache-only, user-run comparison of clean 59D and eGeMAPSv02 88D.

No extraction, fitting, or writes occur at import. Existing project helpers
own the split, classifier, scaler policy, probability checks, and diagnostics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from . import mission2_acoustic_feature_expansion as acoustic
from . import mission2_model_experiments as models
from .mission2_clean_stage4 import load_clean_stage4_59d
from .mission2_experiments import HIGH_CONFIDENCE_THRESHOLD
from .mission2_standard_representation import load_egemaps_cache


CLEAN_FINGERPRINT = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
CLEAN_ACCURACY = 155434 / 177891
CLEAN_CM = [[72265, 13086], [9371, 83169]]
DURATION_BINS = (0., .5, 1., 2., 5., np.inf)
DURATION_LABELS = ("<0.5s", "0.5-1s", "1-2s", "2-5s", ">=5s")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_inputs(clean: Any, data: Any, manifest: pd.DataFrame):
    """Join by unique key, then enforce labels and calls without dropping rows.

    The NPZ has no labels. Its transfer manifest hashes must be verified by
    load_inputs; here the manifest's labels are compared to the clean cache.
    """
    keys = np.asarray(clean.record_keys).astype(str)
    other_keys = np.asarray(data.record_keys).astype(str)
    for values in (keys, other_keys, manifest.record_key.to_numpy()):
        if pd.isna(values).any() or len(np.unique(values)) != len(values):
            raise ValueError("alignment requires unique non-null record keys")
    if set(keys) != set(other_keys) or set(keys) != set(manifest.record_key):
        raise ValueError("record_key missing/extra mismatch")
    positions = pd.Index(other_keys).get_indexer(keys)
    tracking = manifest.set_index("record_key").loc[keys].reset_index()
    if (not np.array_equal(tracking.call_id.astype(str), clean.call_ids)
            or not np.array_equal(np.asarray(data.call_ids)[positions], clean.call_ids)):
        raise ValueError("call_id alignment mismatch")
    if not np.array_equal(tracking.speaker.to_numpy(), clean.y):
        raise ValueError("label alignment mismatch")
    return data.X[positions], tracking, {
        "status": "PASS", "rows": len(keys), "missing": 0, "extra": 0,
        "duplicate_keys": 0, "label_mismatches": 0, "call_mismatches": 0,
        "label_source": "transfer-manifest SHA256 + clean cache labels",
        "reordered_88d": not np.array_equal(keys, other_keys),
    }


def load_inputs(repo_root: Path, egemaps_path: Path, chain_root: Path):
    """Strictly load existing Training caches; never rebuild missing inputs."""
    paths = models.standalone_training_cache_paths(repo_root)
    for path in (egemaps_path, chain_root / "chain.json", paths.manifest_parquet,
                 paths.manifest_metadata):
        if not path.is_file():
            raise FileNotFoundError(f"CACHE_STOP: missing {path}")
    # Validate the stored ordering before any deterministic alignment. This
    # preserves the strict loader's metadata/order fingerprints unchanged.
    with np.load(egemaps_path, allow_pickle=False) as stored:
        keys, calls = stored["record_keys"], stored["call_ids"]
    data = load_egemaps_cache(egemaps_path, keys, calls,
                             source_provenance=CLEAN_FINGERPRINT)
    if (data.X.shape != (889358, 88) or data.X.dtype != np.float32
            or len(data.feature_names) != 88 or len(set(data.feature_names)) != 88
            or data.metadata.get("padded_row_count") != 3):
        raise ValueError("INTEGRITY_STOP: final eGeMAPS shape/schema/padding differs")
    identity = data.metadata.get("colab_resume", {}).get("source_identity", {})
    for path, name in ((paths.manifest_parquet, "manifest_sha256"),
                       (paths.manifest_metadata, "manifest_metadata_sha256")):
        if file_sha256(path) != identity.get(name):
            raise ValueError(f"ALIGNMENT_STOP: transfer {name} differs")
    manifest = pd.read_parquet(paths.manifest_parquet)
    manifest = manifest.loc[manifest["split"].astype(str) == "Training"].reset_index(drop=True)
    clean = load_clean_stage4_59d(chain_root, paths.manifest_parquet,
                                 paths.manifest_metadata, repo_root / "data/raw_original")
    if clean.provenance_fingerprint != CLEAN_FINGERPRINT:
        raise ValueError("INTEGRITY_STOP: clean 59D fingerprint differs")
    X88, tracking, alignment = align_inputs(clean, data, manifest)
    tracking["duration_ms"] = tracking.endAt_ms - tracking.startAt_ms
    if not np.array_equal(tracking.duration_ms.to_numpy() / 1000,
                          tracking.duration_sec.to_numpy()):
        raise ValueError("ALIGNMENT_STOP: duration tracking differs")
    return clean, data, X88, tracking, alignment


def prepare_pair(clean: Any, data: Any, X88: np.ndarray):
    train, validation, audit = models.create_fixed_internal_split(
        clean.call_ids, split_function=models.standalone_group_split_indices,
        expected_contract=models.EXPECTED_INTERNAL_SPLIT)
    reference = acoustic._prepare_representation(
        representation_id="AF10_AF11_COMBINATION", X=clean.X, y=clean.y,
        record_keys=clean.record_keys, feature_names=clean.feature_names,
        train_idx=train, validation_idx=validation, split_audit=audit)
    # The handcrafted whitelist stays intact. Only this strictly validated
    # standard representation uses its own PreparedRepresentation adapter.
    scaler = StandardScaler()
    candidate = acoustic.PreparedRepresentation(
        representation_id="EGEMAPSV02_FUNCTIONALS_88D", feature_names=data.feature_names,
        X_train_scaled=scaler.fit_transform(X88[train]),
        X_evaluation_scaled=scaler.transform(X88[validation]),
        y_train=clean.y[train], y_evaluation=clean.y[validation],
        training_record_keys=clean.record_keys[train],
        evaluation_record_keys=clean.record_keys[validation],
        train_idx=train.copy(), validation_idx=validation.copy(), scaler=scaler,
        split_audit=audit)
    audit = {**audit, "train_keys_hash": models.record_key_signature(clean.record_keys[train]),
             "validation_keys_hash": models.record_key_signature(clean.record_keys[validation]),
             "train_idx_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
             "validation_idx_sha256": hashlib.sha256(validation.tobytes()).hexdigest()}
    return reference, candidate, audit


def fit_prepared(prepared: Any, *, allow_fit: bool = False):
    result = acoustic._run_prepared_representation(
        prepared, experiment_id=prepared.representation_id, allow_fit=allow_fit)
    if (dict(result.model_config) != dict(acoustic.M04_ITER100_FEATURE_SCREENING_CONFIG)
            or result.fit_audit["scaler_fit_scope"] != "internal_train_only"
            or not np.array_equal(result.fit_audit["train_idx"], prepared.train_idx)
            or not np.array_equal(result.fit_audit["validation_idx"], prepared.validation_idx)):
        raise RuntimeError("FIT_STOP: model/scaler/split contract changed")
    predicted = (result.probabilities[:, 1] >= .5).astype(np.int64)
    if not np.array_equal(predicted, result.y_pred):
        raise RuntimeError("FIT_STOP: helper prediction differs from fixed >=0.5 threshold")
    return result


def verify_reproduction(result: Any) -> dict[str, Any]:
    if (not np.array_equal(result.metrics["confusion_matrix"], CLEAN_CM)
            or abs(result.metrics["accuracy"] - CLEAN_ACCURACY) > 1e-12):
        raise RuntimeError(f"REPRODUCTION_STOP: clean 59D differs: {result.metrics}")
    return {"status": "REPRODUCED_METRICS", "expected_accuracy": CLEAN_ACCURACY,
            "expected_confusion_matrix": CLEAN_CM,
            "historical_row_prediction_equality": "UNVERIFIABLE: historical row predictions unavailable",
            "accuracy_float_tolerance": 1e-12}


def fit_candidate(prepared: Any, reference: Any, *, allow_fit: bool = False):
    """A failed reference is terminal even if a caller enables fitting."""
    verify_reproduction(reference)
    if (not np.array_equal(reference.fit_audit["validation_record_keys"],
                           prepared.evaluation_record_keys)
            or not np.array_equal(reference.fit_audit["training_record_keys"],
                                  prepared.training_record_keys)
            or not np.array_equal(reference.y_true, prepared.y_evaluation)):
        raise ValueError("candidate/reference row identity mismatch")
    return fit_prepared(prepared, allow_fit=allow_fit)


def slice_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    duration = frame.duration_sec.to_numpy(dtype=float)
    if not np.isfinite(duration).all() or np.any(duration < 0):
        raise ValueError("duration diagnostics require finite nonnegative values")
    speaker0 = frame.y_true.to_numpy() == 0
    overlap = frame.overlap.to_numpy(dtype=bool)
    masks = {"all": np.ones(len(frame), dtype=bool), "speaker0": speaker0,
             "speaker1": ~speaker0, "overlap": overlap, "no_overlap": ~overlap}
    for label, left, right in zip(DURATION_LABELS, DURATION_BINS[:-1], DURATION_BINS[1:]):
        mask = (duration >= left) & (duration < right)
        masks[f"duration_{label}"] = mask
        masks[f"speaker0_duration_{label}"] = speaker0 & mask
        masks[f"speaker1_duration_{label}"] = ~speaker0 & mask
    masks["speaker0_0.5-1s_overlap"] = speaker0 & overlap & (duration >= .5) & (duration < 1)
    masks["speaker0_1-2s_overlap"] = speaker0 & overlap & (duration >= 1) & (duration < 2)
    return masks


def paired_slice_table(frame: pd.DataFrame) -> pd.DataFrame:
    paired = models.calculate_paired_prediction_changes(
        y_true=frame.y_true, m00_pred=frame.pred_59d, m01_pred=frame.pred_88d,
        m00_record_keys=frame.record_key, m01_record_keys=frame.record_key)
    first = frame.pred_59d.to_numpy() == frame.y_true.to_numpy()
    second = frame.pred_88d.to_numpy() == frame.y_true.to_numpy()
    rows = []
    for label, mask in slice_masks(frame).items():
        n = int(mask.sum())
        corrected = int((mask & ~first & second).sum())
        regressed = int((mask & first & ~second).sum())
        rows.append({"slice": label, "n": n,
                     "both_correct": int((mask & first & second).sum()),
                     "both_wrong": int((mask & ~first & ~second).sum()),
                     "correction": corrected, "regression": regressed,
                     "net_correction": corrected - regressed,
                     "correct_59d": int((mask & first).sum()),
                     "errors_59d": int((mask & ~first).sum()),
                     "accuracy_59d": float(first[mask].mean()) if n else None,
                     "correct_88d": int((mask & second).sum()),
                     "errors_88d": int((mask & ~second).sum()),
                     "accuracy_88d": float(second[mask].mean()) if n else None})
    if (rows[0]["correction"] != paired["m00_wrong_to_m01_correct"]
            or rows[0]["regression"] != paired["m00_correct_to_m01_wrong"]):
        raise AssertionError("paired summary reconciliation failed")
    return pd.DataFrame(rows)


def evaluate_result(result: Any, prepared: Any, tracking: pd.DataFrame):
    """Use the project diagnostic definitions and add requested common metrics."""
    diagnostics = acoustic.build_result_diagnostics(result, full_training_tracking=tracking)
    started = perf_counter()
    train_probability = models.validate_binary_probabilities(
        result.model.predict_proba(prepared.X_train_scaled),
        expected_rows=len(prepared.y_train), classes=result.model.classes_)
    train_prediction = (train_probability[:, 1] >= .5).astype(np.int64)
    train_seconds = perf_counter() - started
    train_metrics = models.calculate_prediction_metrics(prepared.y_train, train_prediction)
    probability = models.validate_binary_probabilities(result.probabilities,
        expected_rows=len(result.y_true), classes=result.model.classes_)
    confidence = probability.max(axis=1)
    correct = result.y_pred == result.y_true
    high_error = ~correct & (confidence >= HIGH_CONFIDENCE_THRESHOLD)
    low_row = diagnostics["confidence_summary"].loc[
        diagnostics["confidence_summary"]["slice"].astype(str) == "<0.60"]
    cm = np.asarray(result.metrics["confusion_matrix"])
    def mean(mask):
        return float(confidence[mask].mean()) if np.any(mask) else None
    report = {**result.metrics, "representation": prepared.representation_id,
              "dimension": len(prepared.feature_names), "train_accuracy": train_metrics["accuracy"],
              "internal_accuracy": result.metrics["accuracy"],
              "gap": train_metrics["accuracy"] - result.metrics["accuracy"],
              "errors": int((~correct).sum()), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]),
              "probability_shape": list(probability.shape), "probability_finite": True,
              "max_probability_sum_error": float(np.abs(probability.sum(axis=1) - 1).max()),
              "mean_confidence": float(confidence.mean()),
              "mean_confidence_correct": mean(correct), "mean_confidence_error": mean(~correct),
              "low_confidence_definition": "project confidence_summary <0.60 bucket (pd.cut right=True)",
              "low_confidence_accuracy": float(low_row.iloc[0].accuracy) if len(low_row) else None,
              "HCE_threshold": HIGH_CONFIDENCE_THRESHOLD, "HCE": int(high_error.sum()),
              "HCE_FP": int((high_error & (result.y_true == 0)).sum()),
              "HCE_FN": int((high_error & (result.y_true == 1)).sum()),
              "n_iter": int(result.model.n_iter_), "final_loss": float(result.model.loss_),
              "convergence_warning": bool(result.fit_audit["convergence_warning_raised"]),
              "warnings": list(result.fit_audit["warning_messages"]),
              "fit_seconds": result.fit_audit["fit_seconds"],
              "validation_inference_seconds": result.fit_audit["prediction_seconds"],
              "train_inference_seconds": train_seconds,
              "scaled_matrix_bytes": int(prepared.X_train_scaled.nbytes + prepared.X_evaluation_scaled.nbytes),
              "scaler_fit_scope": result.fit_audit["scaler_fit_scope"],
              "feature_names_hash": result.fit_audit["feature_fingerprint"],
              "prediction_hash": result.fit_audit["prediction_fingerprint"],
              "fit_array_hashes": {key: result.fit_audit[key] for key in (
                  "split_fingerprint", "scaler_fingerprint", "row_fingerprint",
                  "model_config_fingerprint", "probability_fingerprint")}}
    return report, diagnostics


def prediction_frame(reference: Any, candidate: Any, tracking: pd.DataFrame) -> pd.DataFrame:
    first_keys = reference.fit_audit["validation_record_keys"]
    if (not np.array_equal(first_keys, candidate.fit_audit["validation_record_keys"])
            or not np.array_equal(reference.y_true, candidate.y_true)):
        raise ValueError("paired row/label identity mismatch")
    frame = tracking.set_index("record_key").loc[first_keys].reset_index()
    frame = frame.loc[:, ["record_key", "call_id", "duration_sec", "startAt_ms", "endAt_ms"]]
    frame["overlap"] = models._overlap_mask(frame)
    frame["y_true"] = reference.y_true
    for suffix, result in (("59d", reference), ("88d", candidate)):
        frame[f"pred_{suffix}"] = result.y_pred
        frame[f"prob0_{suffix}"] = result.probabilities[:, 0]
        frame[f"prob1_{suffix}"] = result.probabilities[:, 1]
        frame[f"confidence_{suffix}"] = result.probabilities.max(axis=1)
    # Absolute boundaries are used above only to derive overlap; exported
    # diagnostic columns remain distinct from the two model feature schemas.
    return frame.drop(columns=["startAt_ms", "endAt_ms"])


def environment_info(repo_root: Path) -> dict[str, Any]:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
    return {"timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "python_executable": sys.executable, "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "git_branch": git("branch", "--show-current"), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"), "source_module_sha256": file_sha256(Path(__file__)),
            "canonical_model_sha256": file_sha256(Path(acoustic.__file__)),
            "canonical_metrics_sha256": file_sha256(Path(models.__file__))}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_artifacts(output: Path, summary: dict, predictions: pd.DataFrame,
                   comparison: pd.DataFrame, slices: pd.DataFrame) -> None:
    """Write to a fresh run directory; existing artifacts are never replaced."""
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    slices.to_csv(output / "paired_slices.csv", index=False)
    payload = _json_ready(summary)
    payload["artifact_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
