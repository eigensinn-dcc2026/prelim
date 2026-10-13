"""Fixed full-Training refit and Official Validation for the frozen Stage 14 model.

The module reuses Stage 14's 149D feature order, constructors, metric bundle and
equal-weight ensemble. Official Validation is used only after fitting, never for
scaler fitting, gradients, early stopping or checkpoint selection.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import uuid

import numpy as np
import pandas as pd


MODEL_IDS = ("tabm", "tabular_resnet", "gandalf")
MEMBER_IDS = tuple(f"{name}_F01_F02" for name in MODEL_IDS)
EXPECTED_TRAIN_ROWS = 889_358
EXPECTED_TRAIN_CALLS = 29_200
EXPECTED_VALIDATION_ROWS = 111_947
EXPECTED_VALIDATION_CALLS = 3_640
INPUT_DIMENSION = 149
THRESHOLD = 0.5
FINAL_RUN_PREFIX = "mission2_final_fulltrain_official_validation_"
VALIDATION_CACHE_SCHEMA = "mission2_final_official_validation_149d_v1"
FINAL_RUN_SCHEMA = "mission2_final_fulltrain_refit_v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_new(path: str | Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)


def _write_json_replace(path: str | Path, value: dict) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _array_fingerprint(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays are not supported in final refit fingerprints")
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _unicode_array(values) -> np.ndarray:
    """Normalize identifier columns for deterministic hashes and pickle-free NPZ."""
    array = np.asarray(values, dtype=str)
    if array.ndim != 1:
        raise ValueError("identifier arrays must be one-dimensional")
    return array


def _record_key_signature(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_manifest_rows(
    manifest: pd.DataFrame,
    split: str,
    *,
    expected_rows: int,
    expected_calls: int,
    raw_root: str | Path | None = None,
) -> pd.DataFrame:
    """Select a split and validate key, label, timestamp and crop alignment.

    Annotation text and every annotation field except speaker/startAt/endAt are
    discarded before feature extraction. Paths are tracked only to locate audio.
    """
    required = (
        "split", "record_key", "call_id", "annotation_row", "speaker",
        "audio_path", "json_path", "crop_start_frame", "crop_end_frame",
        "crop_ok", "startAt_ms", "endAt_ms",
    )
    missing = sorted(set(required) - set(manifest.columns))
    if missing:
        raise ValueError(f"source manifest is missing required columns: {missing}")
    source = manifest.loc[manifest["split"].astype(str) == split, list(required)].copy()
    source.reset_index(drop=True, inplace=True)
    if len(source) != expected_rows or source["call_id"].nunique() != expected_calls:
        raise ValueError(f"{split} row/call counts differ from the frozen contract")
    if source["record_key"].isna().any() or not source["record_key"].astype(str).is_unique:
        raise ValueError(f"{split} record_key is null or duplicated")
    identity = ("call_id", "annotation_row", "speaker", "audio_path", "json_path",
                "crop_start_frame", "crop_end_frame", "crop_ok", "startAt_ms", "endAt_ms")
    if source[list(identity)].isna().any().any():
        raise ValueError(f"{split} identity, labels or tracking contain null values")
    for column in ("call_id", "audio_path", "json_path"):
        if source[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{split} {column} contains an empty value")
    try:
        annotation_rows_raw = pd.to_numeric(source["annotation_row"], errors="raise").to_numpy(dtype=np.float64)
        labels_raw = pd.to_numeric(source["speaker"], errors="raise").to_numpy(dtype=np.float64)
        starts = pd.to_numeric(source["startAt_ms"], errors="raise").to_numpy(dtype=np.float64)
        ends = pd.to_numeric(source["endAt_ms"], errors="raise").to_numpy(dtype=np.float64)
        crop_starts = pd.to_numeric(source["crop_start_frame"], errors="raise").to_numpy(dtype=np.int64)
        crop_ends = pd.to_numeric(source["crop_end_frame"], errors="raise").to_numpy(dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{split} labels, annotation rows or crop metadata are invalid") from error
    if (not np.isfinite(annotation_rows_raw).all() or not np.isfinite(labels_raw).all()
            or np.any(annotation_rows_raw != np.floor(annotation_rows_raw))
            or np.any(labels_raw != np.floor(labels_raw))):
        raise ValueError(f"{split} annotation row or binary speaker label is not an integer")
    annotation_rows = annotation_rows_raw.astype(np.int64)
    labels = labels_raw.astype(np.int64)
    if np.any(annotation_rows < 0) or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{split} annotation row or binary speaker label contract differs")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends <= starts):
        raise ValueError(f"{split} startAt/endAt tracking is invalid")
    expected_keys = (split + "|" + source["call_id"].astype(str) + "|"
                     + pd.Series(annotation_rows, index=source.index).astype(str))
    if not source["record_key"].astype(str).reset_index(drop=True).equals(
            expected_keys.reset_index(drop=True)):
        raise ValueError(f"{split} record_key does not match call_id/annotation_row")
    if not source["crop_ok"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all():
        raise ValueError(f"{split} contains invalid audio crops")
    from .mission2 import milliseconds_to_frame
    from .mission2_model_experiments import MFCC_CONFIG

    rate = int(MFCC_CONFIG["sample_rate"])
    for index, (start, end) in enumerate(zip(starts, ends)):
        if (milliseconds_to_frame(float(start), rate) != int(crop_starts[index])
                or milliseconds_to_frame(float(end), rate) != int(crop_ends[index])
                or crop_starts[index] < 0 or crop_ends[index] <= crop_starts[index]):
            raise ValueError(f"{split} crop boundary differs from millisecond metadata at row {index}")
    source["annotation_row"] = annotation_rows
    source["speaker"] = labels
    source["startAt_ms"] = starts
    source["endAt_ms"] = ends
    source["crop_start_frame"] = crop_starts
    source["crop_end_frame"] = crop_ends
    source["record_key"] = source["record_key"].astype(str)
    source["call_id"] = source["call_id"].astype(str)
    if raw_root is not None:
        root = Path(raw_root).resolve(strict=True)
        for relative in set(source["audio_path"].astype(str)) | set(source["json_path"].astype(str)):
            path = (root / relative).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"source path escapes the raw-data root: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
    return source


def build_joint_matrix(clean_X, clean_names, y, f01, f02):
    """Use Stage 14's frozen Clean59→F01→F02 order for any one split."""
    from types import SimpleNamespace
    from . import mission2_stage14_ensemble as stage14

    rows = len(np.asarray(y))
    context = SimpleNamespace(
        clean_X=np.asarray(clean_X), feature_names_59d=tuple(clean_names),
        feature_matrices={"F01": np.asarray(f01), "F02": np.asarray(f02)},
        y=np.asarray(y),
    )
    matrix, names = stage14.build_joint_input(context)
    if matrix.shape != (rows, INPUT_DIMENSION):
        raise ValueError("final feature matrix must have 149 columns")
    return matrix, names


def derive_fixed_refit_epochs(evidence: dict[str, dict]) -> dict[str, int]:
    """Freeze each refit length to the saved Stage 14 best Internal epoch.

    GANDALF's saved Stage 14 evidence is a terminal snapshot, not epoch history;
    its best epoch is accepted only from the checkpoint callback metric.
    """
    if set(evidence) != set(MODEL_IDS):
        raise ValueError("saved Stage 14 epoch evidence must cover all three models")
    epochs = {}
    for model_id in MODEL_IDS:
        item = evidence[model_id]
        metrics = item.get("metrics", {})
        history = item.get("history")
        best_epoch = metrics.get("best_epoch")
        epochs_run = metrics.get("epochs_run")
        history_kind = metrics.get("history_kind")
        if (not isinstance(best_epoch, int) or best_epoch < 1
                or not isinstance(epochs_run, int) or epochs_run < best_epoch):
            raise ValueError(f"saved Stage 14 best epoch/epochs_run is invalid: {model_id}")
        if model_id in ("tabm", "tabular_resnet"):
            if (history_kind != "epoch_history" or not isinstance(history, pd.DataFrame)
                    or "epoch" not in history or "best_epoch" not in history
                    or len(history) != epochs_run
                    or int(history.epoch.iloc[-1]) != epochs_run
                    or int(history.best_epoch.iloc[-1]) != best_epoch):
                raise ValueError(f"saved Stage 14 epoch history does not support refit length: {model_id}")
        elif history_kind != "terminal_metrics_snapshot":
            raise ValueError("GANDALF Stage 14 history-kind evidence changed")
        epochs[model_id] = best_epoch
    return epochs


def attach_tracking_metadata(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach filename/time only after exact row, call and label alignment."""
    required_predictions = ("record_key", "call_id", "y_true", "probability_speaker1")
    required_manifest = ("record_key", "call_id", "speaker", "audio_path", "startAt_ms", "endAt_ms")
    if any(name not in predictions for name in required_predictions):
        raise ValueError("prediction frame lacks key/call/label/probability identity")
    if any(name not in manifest for name in required_manifest):
        raise ValueError("Validation manifest lacks key/call/label/time tracking")
    keys = manifest.record_key.astype(str).to_numpy()
    calls = manifest.call_id.astype(str).to_numpy()
    labels = manifest.speaker.to_numpy(dtype=np.int64)
    if (len(set(keys)) != len(keys)
            or not np.array_equal(predictions.record_key.astype(str).to_numpy(), keys)
            or not np.array_equal(predictions.call_id.astype(str).to_numpy(), calls)
            or not np.array_equal(predictions.y_true.to_numpy(dtype=np.int64), labels)):
        raise ValueError("prediction/Official Validation identity differs")
    result = predictions.copy()
    result["filename"] = manifest.audio_path.astype(str).map(lambda value: Path(value).name).to_numpy()
    result["startAt_ms"] = manifest.startAt_ms.to_numpy(dtype=np.float64)
    result["endAt_ms"] = manifest.endAt_ms.to_numpy(dtype=np.float64)
    return result


def build_final_ensemble(member_predictions: dict[str, pd.DataFrame],
                         validation_manifest: pd.DataFrame):
    """Reuse the existing Stage 14 E_TRG implementation (one-third each)."""
    from . import mission2_stage14_ensemble as stage14

    metrics, predictions = stage14.build_ensemble_predictions(member_predictions, "E_TRG")
    predictions = attach_tracking_metadata(predictions, validation_manifest)
    metrics["model_id"] = "ensemble_equal_probability_average"
    return metrics, predictions


def validate_official_prediction_alignment(predictions: dict[str, pd.DataFrame],
                                           manifest: pd.DataFrame) -> None:
    """Fail if any saved model or ensemble loses/reorders Official Validation rows."""
    if not predictions:
        raise ValueError("no Official Validation predictions were supplied")
    expected_keys = manifest.record_key.astype(str).to_numpy()
    expected_calls = manifest.call_id.astype(str).to_numpy()
    expected_labels = manifest.speaker.to_numpy(dtype=np.int64)
    expected_files = manifest.audio_path.astype(str).map(lambda value: Path(value).name).to_numpy()
    expected_starts = manifest.startAt_ms.to_numpy(dtype=np.float64)
    expected_ends = manifest.endAt_ms.to_numpy(dtype=np.float64)
    for name, frame in predictions.items():
        for column, expected in (("record_key", expected_keys), ("call_id", expected_calls),
                                 ("y_true", expected_labels), ("filename", expected_files),
                                 ("startAt_ms", expected_starts), ("endAt_ms", expected_ends)):
            if column not in frame or not np.array_equal(frame[column].to_numpy(), expected):
                raise ValueError(f"Official Validation identity differs for {name}/{column}")
        probability = frame.probability_speaker1.to_numpy(dtype=np.float64)
        if (len(frame) != len(manifest) or frame.record_key.astype(str).duplicated().any()
                or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise ValueError(f"Official Validation prediction rows/probabilities invalid: {name}")
        if not np.array_equal(frame.prediction.to_numpy(dtype=np.int64),
                              (probability >= THRESHOLD).astype(np.int64)):
            raise ValueError(f"Official Validation fixed 0.5 threshold differs: {name}")


def recompute_saved_prediction_metrics(predictions: pd.DataFrame) -> dict:
    """Recalculate scores from persisted probabilities and labels for final audit."""
    from . import mission2_acoustic_feature_grid as grid

    required = ("record_key", "call_id", "y_true", "probability_speaker1")
    if any(name not in predictions for name in required):
        raise ValueError("saved prediction frame lacks metric inputs")
    metrics, reconstructed = grid._metric_bundle(
        predictions.y_true.to_numpy(dtype=np.int64),
        predictions.probability_speaker1.to_numpy(dtype=np.float64),
        predictions.record_key.astype(str).to_numpy(),
        predictions.call_id.astype(str).to_numpy(),
    )
    if ("prediction" in predictions
            and not np.array_equal(predictions.prediction.to_numpy(dtype=np.int64),
                                   reconstructed.prediction.to_numpy(dtype=np.int64))):
        raise ValueError("saved prediction threshold labels do not reconstruct")
    if ("correct" in predictions
            and not np.array_equal(predictions.correct.to_numpy(dtype=bool),
                                   reconstructed.correct.to_numpy(dtype=bool))):
        raise ValueError("saved prediction correctness flags do not reconstruct")
    return metrics


def _raw_inventory_signature(source: pd.DataFrame, raw_root: str | Path) -> str:
    root = Path(raw_root).resolve(strict=True)
    digest = hashlib.sha256()
    paths = set(source.audio_path.astype(str)) | set(source.json_path.astype(str))
    for relative in sorted(paths):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("raw Validation path escapes its root") from error
        stat = path.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def build_official_validation_manifest(raw_root: str | Path, run_root: str | Path,
                                       *, expected_rows: int = EXPECTED_VALIDATION_ROWS,
                                       expected_calls: int = EXPECTED_VALIDATION_CALLS) -> dict:
    """Build a text-free Validation row manifest from the organizer WAV/JSON files.

    The JSON array position is retained as identity, while only startAt, endAt,
    and speaker are read from each utterance. No transcript or other annotation
    field enters the returned frame or any model feature.
    """
    from .mission2 import milliseconds_to_frame
    from .mission2_model_experiments import MFCC_CONFIG

    root = Path(raw_root).resolve(strict=True)
    validation_root = root / "Validation"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Official Validation folder is missing: {validation_root}")
    output = Path(run_root).resolve(strict=True) / "features"
    output.mkdir(exist_ok=True)
    manifest_path = output / "official_validation_manifest.parquet"
    metadata_path = output / "official_validation_manifest.json"
    if manifest_path.exists() or metadata_path.exists():
        if not manifest_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("partial Official Validation source manifest exists; inspect it before retry")
        manifest = pd.read_parquet(manifest_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = validate_manifest_rows(
            manifest, "Validation", expected_rows=expected_rows,
            expected_calls=expected_calls, raw_root=root,
        )
        if (metadata.get("schema_version") != "mission2_final_validation_source_manifest_v1"
                or metadata.get("source_inventory_signature") != _raw_inventory_signature(source, root)
                or metadata.get("artifact_sha256") != _sha256_file(manifest_path)):
            raise ValueError("saved Official Validation source manifest provenance differs")
        return {"manifest": source, "path": manifest_path,
                "metadata_path": metadata_path, "metadata": metadata}

    json_paths = sorted(validation_root.rglob("*.json"), key=lambda path: path.as_posix().casefold())
    wav_paths = sorted(validation_root.rglob("*.wav"), key=lambda path: path.as_posix().casefold())
    audio_by_stem = {}
    for path in wav_paths:
        key = path.stem.casefold()
        if key in audio_by_stem:
            raise ValueError(f"duplicate Official Validation WAV stem: {path.stem}")
        audio_by_stem[key] = path
    json_by_stem = {}
    for path in json_paths:
        key = path.stem.casefold()
        if key in json_by_stem:
            raise ValueError(f"duplicate Official Validation JSON stem: {path.stem}")
        json_by_stem[key] = path
    if set(audio_by_stem) != set(json_by_stem):
        raise ValueError("Official Validation WAV/JSON stem inventory differs")

    rate = int(MFCC_CONFIG["sample_rate"])
    rows = []
    for key in sorted(json_by_stem):
        json_path = json_by_stem[key]
        audio_path = audio_by_stem[key]
        call_id = json_path.stem
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        utterances = payload.get("utterances") if isinstance(payload, dict) else None
        if not isinstance(utterances, list):
            raise ValueError(f"Official Validation JSON lacks an utterances list: {json_path}")
        for annotation_row, utterance in enumerate(utterances):
            if not isinstance(utterance, dict):
                raise ValueError(f"invalid Official Validation utterance at {json_path}/{annotation_row}")
            try:
                start_ms = float(utterance["startAt"])
                end_ms = float(utterance["endAt"])
                speaker_raw = float(utterance["speaker"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"invalid startAt/endAt/speaker at {json_path}/{annotation_row}") from error
            if not np.isfinite(speaker_raw) or speaker_raw != np.floor(speaker_raw):
                raise ValueError(f"non-integer speaker label at {json_path}/{annotation_row}")
            speaker = int(speaker_raw)
            crop_start = milliseconds_to_frame(start_ms, rate)
            crop_end = milliseconds_to_frame(end_ms, rate)
            rows.append({
                "split": "Validation",
                "record_key": f"Validation|{call_id}|{annotation_row}",
                "call_id": call_id, "annotation_row": annotation_row,
                "speaker": speaker,
                "audio_path": audio_path.relative_to(root).as_posix(),
                "json_path": json_path.relative_to(root).as_posix(),
                "crop_start_frame": crop_start, "crop_end_frame": crop_end,
                "crop_ok": bool(np.isfinite(start_ms) and np.isfinite(end_ms)
                                 and start_ms >= 0 and end_ms > start_ms
                                 and crop_start >= 0 and crop_end > crop_start),
                "startAt_ms": start_ms, "endAt_ms": end_ms,
            })
    manifest = pd.DataFrame(rows)
    source = validate_manifest_rows(
        manifest, "Validation", expected_rows=expected_rows,
        expected_calls=expected_calls, raw_root=root,
    )
    inventory_signature = _raw_inventory_signature(source, root)
    with manifest_path.open("xb") as stream:
        source.to_parquet(stream, index=False)
    metadata = {
        "schema_version": "mission2_final_validation_source_manifest_v1",
        "source_split": "Validation", "source_inventory_signature": inventory_signature,
        "source_signature": inventory_signature,
        "row_count": len(source), "call_count": int(source.call_id.nunique()),
        "columns": list(source.columns), "text_used_for_model_input": False,
        "text_persisted": False,
        "annotation_fields_used": ["startAt", "endAt", "speaker"],
        "artifact_sha256": _sha256_file(manifest_path),
    }
    _write_json_new(metadata_path, metadata)
    return {"manifest": source, "path": manifest_path,
            "metadata_path": metadata_path, "metadata": metadata}


def _load_aligned_mfcc40(cache_path: Path, metadata_path: Path,
                         source: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Reuse the existing 40D Validation MFCC cache only after key/time/label checks."""
    from .mission2_model_experiments import MFCC_CONFIG

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_config = {key: int(value) for key, value in MFCC_CONFIG.items()}
    if (metadata.get("cache_schema_version") != "mission2_mfcc_mean_std_v1"
            or metadata.get("source_split") != "Validation"
            or metadata.get("utterance_count") != len(source)
            or metadata.get("source_call_count") != source.call_id.nunique()
            or metadata.get("feature_dimension") != 40
            or metadata.get("mfcc_config") != expected_config):
        raise ValueError("existing Validation MFCC cache metadata differs")
    with np.load(cache_path, allow_pickle=False) as saved:
        required = {"X", "y", "stem", "start_at", "end_at", "duration"}
        if not required.issubset(saved.files):
            raise ValueError("existing Validation MFCC cache members differ")
        values = saved["X"]
        labels = saved["y"].astype(np.int64, copy=False)
        calls = saved["stem"].astype(str)
        starts = saved["start_at"].astype(np.float64, copy=False)
        ends = saved["end_at"].astype(np.float64, copy=False)
        duration = saved["duration"].astype(np.float64, copy=False)
    rows = len(source)
    if (values.dtype != np.float32 or values.shape != (rows, 40)
            or any(array.shape != (rows,) for array in (labels, calls, starts, ends, duration))
            or not np.isfinite(values).all()):
        raise ValueError("existing Validation MFCC cache shape/dtype/finite contract differs")
    cache_identity = pd.DataFrame({"call_id": calls})
    cache_identity["annotation_row"] = cache_identity.groupby("call_id", sort=False).cumcount()
    cache_identity["record_key"] = (
        "Validation|" + cache_identity.call_id.astype(str) + "|"
        + cache_identity.annotation_row.astype(str)
    )
    if cache_identity.record_key.duplicated().any():
        raise ValueError("existing Validation MFCC cache has duplicate row keys")
    index_by_key = {key: index for index, key in enumerate(cache_identity.record_key.astype(str))}
    source_keys = source.record_key.astype(str).tolist()
    if set(source_keys) != set(index_by_key):
        raise ValueError("existing Validation MFCC key set differs from source manifest")
    order = np.fromiter((index_by_key[key] for key in source_keys), dtype=np.int64)
    if (not np.array_equal(calls[order], source.call_id.astype(str).to_numpy())
            or not np.array_equal(labels[order], source.speaker.to_numpy(dtype=np.int64))
            or not np.array_equal(starts[order], source.startAt_ms.to_numpy(dtype=np.float64))
            or not np.array_equal(ends[order], source.endAt_ms.to_numpy(dtype=np.float64))
            or not np.array_equal(duration[order], ends[order] - starts[order])):
        raise ValueError("existing Validation MFCC cache row/call/label/time identity differs")
    return np.asarray(values[order], dtype=np.float32), {
        "reuse_status": "REUSED_AFTER_EXACT_KEY_CALL_LABEL_TIME_CONFIG_CHECK",
        "cache_path": str(cache_path), "cache_sha256": _sha256_file(cache_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "raw_audio_content_hash_in_cache_metadata": False,
    }


def _load_official_feature_cache(cache_path: Path, metadata_path: Path,
                                 source: pd.DataFrame, source_manifest_sha256: str,
                                 inventory_signature: str):
    if cache_path.exists() != metadata_path.exists():
        raise RuntimeError("partial Official Validation feature cache exists; inspect it before retry")
    if not cache_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != VALIDATION_CACHE_SCHEMA
            or metadata.get("source_manifest_sha256") != source_manifest_sha256
            or metadata.get("source_inventory_signature") != inventory_signature
            or metadata.get("record_key_signature") != _record_key_signature(source.record_key)
            or metadata.get("artifact_sha256") != _sha256_file(cache_path)):
        raise ValueError("existing 149D Official Validation feature cache provenance differs")
    with np.load(cache_path, allow_pickle=False) as saved:
        required = {"X", "record_keys", "call_ids", "labels", "feature_names",
                    "filename", "start_at", "end_at"}
        if set(saved.files) != required:
            raise ValueError("existing 149D Official Validation cache members differ")
        matrix = saved["X"]
        keys = saved["record_keys"].astype(str)
        calls = saved["call_ids"].astype(str)
        labels = saved["labels"].astype(np.int64, copy=False)
        names = tuple(saved["feature_names"].astype(str).tolist())
        filenames = saved["filename"].astype(str)
        starts = saved["start_at"].astype(np.float64, copy=False)
        ends = saved["end_at"].astype(np.float64, copy=False)
    expected_keys = source.record_key.astype(str).to_numpy()
    expected_calls = source.call_id.astype(str).to_numpy()
    if (matrix.dtype != np.float32 or matrix.shape != (len(source), INPUT_DIMENSION)
            or not np.isfinite(matrix).all()
            or not np.array_equal(keys, expected_keys)
            or not np.array_equal(calls, expected_calls)
            or not np.array_equal(labels, source.speaker.to_numpy(dtype=np.int64))
            or not np.array_equal(filenames, source.audio_path.astype(str).map(lambda v: Path(v).name).to_numpy())
            or not np.array_equal(starts, source.startAt_ms.to_numpy(dtype=np.float64))
            or not np.array_equal(ends, source.endAt_ms.to_numpy(dtype=np.float64))):
        raise ValueError("existing 149D Official Validation cache row identity differs")
    return {"X": matrix, "record_keys": keys, "call_ids": calls, "y": labels,
            "feature_names": names, "manifest": source,
            "feature_cache_audit": metadata}


def load_or_extract_official_validation_features(
    manifest: pd.DataFrame,
    raw_root: str | Path,
    *,
    source_manifest_path: str | Path,
    source_manifest_metadata_path: str | Path,
    mfcc_cache_path: str | Path,
    mfcc_metadata_path: str | Path,
    run_root: str | Path,
    worker_count: int = 4,
    on_call_done=None,
) -> dict:
    """Reuse MFCC40 and create only missing frozen Validation feature blocks."""
    source = validate_manifest_rows(
        manifest, "Validation", expected_rows=EXPECTED_VALIDATION_ROWS,
        expected_calls=EXPECTED_VALIDATION_CALLS, raw_root=raw_root,
    )
    root = Path(raw_root).resolve(strict=True)
    output = Path(run_root).resolve(strict=True) / "features"
    output.mkdir(exist_ok=True)
    cache_path = output / "official_validation_149d.npz"
    metadata_path = output / "official_validation_149d.json"
    manifest_path = Path(source_manifest_path).resolve(strict=True)
    manifest_meta_path = Path(source_manifest_metadata_path).resolve(strict=True)
    manifest_sha = _sha256_file(manifest_path)
    metadata_sha = _sha256_file(manifest_meta_path)
    inventory_before = _raw_inventory_signature(source, root)
    cached = _load_official_feature_cache(
        cache_path, metadata_path, source, manifest_sha, inventory_before,
    )
    if cached is not None:
        return cached

    from . import mission2_acoustic_feature_expansion as acoustic
    from . import mission2_acoustic_feature_grid as grid
    from . import mission2_candidate_features as candidate
    from . import mission2_clean_stage4 as clean_stage4
    from .audio import read_wav_frames

    try:
        mfcc_cache = Path(mfcc_cache_path).resolve(strict=True)
        mfcc_meta = Path(mfcc_metadata_path).resolve(strict=True)
        mfcc40, mfcc_audit = _load_aligned_mfcc40(mfcc_cache, mfcc_meta, source)
    except (OSError, ValueError, KeyError) as error:
        # A stale feature cache is never partially aligned. Recompute its full
        # block from the validated Official Validation WAVs instead.
        data = clean_stage4._extract_mfcc40(source, root)
        mfcc40 = data["X"]
        mfcc_audit = {
            "reuse_status": "REEXTRACTED_CACHE_REJECTED",
            "rejected_cache_reason": f"{type(error).__name__}: {error}",
            "cache_path": str(mfcc_cache_path), "metadata_path": str(mfcc_metadata_path),
        }
    feature_names_e05 = clean_stage4.E05_CURRENT_NAMES
    e05_frame = candidate.extract_candidate_feature_frame(
        source, data_root=root, split="Validation", feature_names=feature_names_e05,
        on_call_done=(lambda done, total: on_call_done("E05", done, total)
                      if on_call_done is not None else None),
    )
    af_ids = tuple(clean_stage4.AF_BLOCKS.values())
    flat_af_ids = tuple(block_id for group in af_ids for block_id in group)
    af_frame = acoustic.extract_candidate_feature_frame(
        source, data_root=root, split="Validation", block_ids=flat_af_ids,
        worker_count=worker_count,
        on_call_done=(lambda done, total, call: on_call_done("AF", done, total)
                      if on_call_done is not None else None),
    )
    keys = _unicode_array(source.record_key)
    calls = _unicode_array(source.call_id)
    labels = source.speaker.to_numpy(dtype=np.int64)
    if (e05_frame.record_key.astype(str).tolist() != keys.tolist()
            or e05_frame.call_id.astype(str).tolist() != calls.tolist()
            or af_frame.record_key.astype(str).tolist() != keys.tolist()
            or af_frame.call_id.astype(str).tolist() != calls.tolist()):
        raise ValueError("Official Validation feature extraction changed canonical row order")
    clean_arrays = [mfcc40, e05_frame.loc[:, list(feature_names_e05)].to_numpy(dtype=np.float32)]
    for block_id in flat_af_ids:
        block_names = acoustic.BLOCK_BY_ID[block_id].feature_names
        clean_arrays.append(af_frame.loc[:, list(block_names)].to_numpy(dtype=np.float32))
    clean_names = clean_stage4.clean_59d_feature_names()
    clean_matrix = np.concatenate(clean_arrays, axis=1).astype(np.float32, copy=False)
    if (clean_matrix.shape != (len(source), 59) or not np.isfinite(clean_matrix).all()
            or len(clean_names) != 59):
        raise ValueError("Official Validation Clean59 matrix differs from frozen Stage 4 schema")

    f01 = np.empty((len(source), grid.BLOCK_BY_ID["F01"].dimension), dtype=np.float32)
    f02 = np.empty((len(source), grid.BLOCK_BY_ID["F02"].dimension), dtype=np.float32)
    for call_number, ((_call_id, audio_path), group) in enumerate(
            source.groupby(["call_id", "audio_path"], sort=False), start=1):
        audio = read_wav_frames(root / str(audio_path))
        if audio.sample_rate != 8_000:
            raise ValueError(f"unexpected Validation sample rate: {audio_path}")
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        for row in group.itertuples():
            start, end = int(row.crop_start_frame), int(row.crop_end_frame)
            if start < 0 or end <= start or end > mono.shape[0]:
                raise ValueError(f"invalid Official Validation crop: {row.record_key}")
            values, _ = grid._waveform_vectors(grid._mono_crop(mono[start:end], audio.sample_rate))
            f01[row.Index] = values["F01"]
            f02[row.Index] = values["F02"]
        if on_call_done is not None and (call_number % 250 == 0 or call_number == EXPECTED_VALIDATION_CALLS):
            on_call_done("F01_F02", call_number, EXPECTED_VALIDATION_CALLS)
    grid.validate_feature_block("F01", f01, keys, expected_keys=keys)
    grid.validate_feature_block("F02", f02, keys, expected_keys=keys)
    matrix, all_names = build_joint_matrix(clean_matrix, clean_names, labels, f01, f02)
    schema = {
        "clean_59d": list(clean_names),
        "F01": list(grid.BLOCK_BY_ID["F01"].feature_names),
        "F02": list(grid.BLOCK_BY_ID["F02"].feature_names),
    }
    feature_schema_fingerprint = grid._json_fingerprint(schema)
    if (feature_schema_fingerprint != "0e9834296216cf39b5715e9e80df20b56fba2ee70d2ffa84dfed90179ace1626"
            or len(all_names) != INPUT_DIMENSION):
        raise ValueError("Official Validation feature schema differs from accepted Stage 14 149D schema")
    inventory_after = _raw_inventory_signature(source, root)
    if inventory_after != inventory_before or _sha256_file(manifest_path) != manifest_sha:
        raise RuntimeError("Validation raw files or source manifest changed during feature extraction")
    feature_names_f01 = grid.BLOCK_BY_ID["F01"].feature_names
    feature_names_f02 = grid.BLOCK_BY_ID["F02"].feature_names
    with cache_path.open("xb") as stream:
        np.savez_compressed(
            stream, X=matrix, record_keys=keys, call_ids=calls, labels=labels,
            feature_names=np.asarray(all_names, dtype=str),
            filename=source.audio_path.astype(str).map(lambda value: Path(value).name).to_numpy(dtype=str),
            start_at=source.startAt_ms.to_numpy(dtype=np.float64),
            end_at=source.endAt_ms.to_numpy(dtype=np.float64),
        )
    cache_metadata = {
        "schema_version": VALIDATION_CACHE_SCHEMA,
        "source_split": "Validation", "source_manifest_sha256": manifest_sha,
        "source_manifest_metadata_sha256": metadata_sha,
        "manifest_recorded_source_signature": json.loads(manifest_meta_path.read_text(encoding="utf-8")).get("source_signature"),
        "source_inventory_signature": inventory_before,
        "record_key_signature": _record_key_signature(keys),
        "label_hash": _array_fingerprint(labels), "call_id_hash": _array_fingerprint(calls),
        "row_count": len(source), "call_count": int(source.call_id.nunique()),
        "feature_schema": schema, "feature_schema_fingerprint": feature_schema_fingerprint,
        "feature_names": list(all_names), "feature_dimension": len(all_names),
        "mfcc40_cache_audit": mfcc_audit,
        "extracted_blocks": ["E05", *flat_af_ids, "F01", "F02"],
        "official_validation_used_for_model_selection_or_fit": False,
        "artifact_sha256": _sha256_file(cache_path),
    }
    _write_json_new(metadata_path, cache_metadata)
    return {"X": matrix, "record_keys": keys, "call_ids": calls, "y": labels,
            "feature_names": tuple(all_names), "manifest": source,
            "feature_cache_audit": cache_metadata}


def validate_full_training_context(context) -> dict:
    """Require the exact complete organizer Training rows before refitting."""
    keys = np.asarray(context.record_keys).astype(str)
    calls = np.asarray(context.call_ids).astype(str)
    labels = np.asarray(context.y, dtype=np.int64)
    if (len(keys) != EXPECTED_TRAIN_ROWS or len(np.unique(calls)) != EXPECTED_TRAIN_CALLS
            or len(calls) != len(keys) or labels.shape != keys.shape
            or len(np.unique(keys)) != len(keys)
            or not np.isin(labels, (0, 1)).all()
            or not np.isfinite(labels).all()
            or not np.char.startswith(keys, "Training|").all()):
        raise ValueError("full organizer Training row/call/key/label contract differs")
    return {
        "rows": len(keys), "calls": int(len(np.unique(calls))),
        "record_key_signature": _record_key_signature(keys),
        "call_id_hash": _array_fingerprint(calls), "label_hash": _array_fingerprint(labels),
        "official_validation_used": False,
    }


def load_full_training_matrix(context, source_feature_run_root: str | Path):
    """Reuse the verified Stage 14 Clean59/F01/F02 Training matrices."""
    from . import mission2_stage14_ensemble as stage14

    context_audit = validate_full_training_context(context)
    source_root = Path(source_feature_run_root).resolve(strict=True)
    matrices, hashes, manifest_sha = stage14.load_joint_feature_blocks(source_root, context)
    matrix, names = stage14.build_joint_input(context)
    if (matrix.shape != (EXPECTED_TRAIN_ROWS, INPUT_DIMENSION)
            or matrix.dtype != np.float32 or not np.isfinite(matrix).all()
            or len(set(names)) != INPUT_DIMENSION):
        raise ValueError("full Training 149D feature matrix contract differs")
    schema = {
        "clean_59d": list(context.feature_names_59d),
        "F01": list(stage14.grid.BLOCK_BY_ID["F01"].feature_names),
        "F02": list(stage14.grid.BLOCK_BY_ID["F02"].feature_names),
    }
    fingerprint = stage14.grid._json_fingerprint(schema)
    expected = "0e9834296216cf39b5715e9e80df20b56fba2ee70d2ffa84dfed90179ace1626"
    if fingerprint != expected:
        raise ValueError("full Training feature schema differs from accepted Stage 14 schema")
    return matrix, tuple(names), {
        **context_audit, "feature_schema": schema,
        "feature_schema_fingerprint": fingerprint,
        "source_feature_run_root": str(source_root),
        "source_feature_cache_manifest_sha256": manifest_sha,
        "source_feature_block_sha256": hashes,
        "clean_fingerprint": context.clean_fingerprint,
        "feature_cache_blocks_loaded": sorted(matrices),
    }


def load_stage14_refit_evidence(context, stage14_run_root: str | Path,
                               source_feature_run_root: str | Path) -> dict:
    """Verify completed Stage 14 artifacts and derive the three fixed epochs."""
    from . import mission2_stage14_ensemble as stage14

    root = Path(stage14_run_root).resolve(strict=True)
    source = Path(source_feature_run_root).resolve(strict=True)
    audit = stage14.validate_stage14_run(context, root, source_run_root=source)
    contract = stage14.grid._read_json(root / "contract.json")
    if (contract.get("schema_version") != stage14.RUN_SCHEMA
            or contract.get("source_split") != "Training"
            or contract.get("official_validation_used") is not False
            or contract.get("feature_schema_fingerprint")
            != "0e9834296216cf39b5715e9e80df20b56fba2ee70d2ffa84dfed90179ace1626"):
        raise ValueError("saved Stage 14 run contract differs from the final refit contract")
    evidence = {}
    for member_id, model_id in zip(MEMBER_IDS, MODEL_IDS):
        metrics_path = root / "members" / member_id / "metrics.json"
        history_path = root / "members" / member_id / "history.csv"
        metrics = stage14.grid._read_json(metrics_path)
        config = stage14.grid._read_json(root / "members" / member_id / "config.json")
        expected_config = stage14.grid._model_config(model_id, INPUT_DIMENSION)
        if config != expected_config or config != contract["model_configs"].get(member_id):
            raise ValueError(f"Stage 14 saved model recipe/config differs: {member_id}")
        stage14.verify_completed_member(context, root, member_id)
        if not history_path.is_file():
            raise FileNotFoundError(f"Stage 14 training history is missing: {history_path}")
        history = pd.read_csv(history_path)
        evidence[model_id] = {"metrics": metrics, "history": history,
                              "metrics_sha256": _sha256_file(metrics_path),
                              "history_sha256": _sha256_file(history_path),
                              "checkpoint_sha256": _sha256_file(
                                  root / "members" / member_id / "checkpoint.pt"
                              ) if (root / "members" / member_id / "checkpoint.pt").is_file() else None}
    epochs = derive_fixed_refit_epochs(evidence)
    return {"run_root": str(root), "source_feature_run_root": str(source),
            "stage14_audit": audit, "contract_sha256": _sha256_file(root / "contract.json"),
            "fixed_refit_epochs": epochs, "members": evidence,
            "ensemble_id": "E_TRG", "ensemble_weights": [1 / 3, 1 / 3, 1 / 3]}


def create_run_root(output_parent: str | Path) -> Path:
    """Create a new, unique Drive output directory without touching prior runs."""
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_id = FINAL_RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    root = parent / run_id
    root.mkdir(exist_ok=False)
    (root / "members").mkdir()
    (root / "features").mkdir()
    return root


def initialize_final_run(
    run_root: str | Path,
    context,
    training_matrix: np.ndarray,
    training_feature_names: tuple[str, ...],
    training_audit: dict,
    validation_data: dict,
    stage14_evidence: dict,
    code_bundle_provenance: dict | None = None,
) -> dict:
    """Freeze source identities, feature schema, recipes and refit epochs."""
    from . import mission2_acoustic_feature_grid as grid

    root = Path(run_root).resolve(strict=True)
    train_audit = validate_full_training_context(context)
    val_manifest = validation_data["manifest"]
    val_keys = np.asarray(validation_data["record_keys"]).astype(str)
    val_calls = np.asarray(validation_data["call_ids"]).astype(str)
    val_labels = np.asarray(validation_data["y"], dtype=np.int64)
    train_keys = np.asarray(context.record_keys).astype(str)
    train_calls = np.asarray(context.call_ids).astype(str)
    if (np.asarray(training_matrix).shape != (EXPECTED_TRAIN_ROWS, INPUT_DIMENSION)
            or np.asarray(training_matrix).dtype != np.float32
            or tuple(training_feature_names) != tuple(validation_data["feature_names"])
            or not np.array_equal(val_keys, val_manifest.record_key.astype(str).to_numpy())
            or not np.array_equal(val_calls, val_manifest.call_id.astype(str).to_numpy())
            or not np.array_equal(val_labels, val_manifest.speaker.to_numpy(dtype=np.int64))):
        raise ValueError("final Training/Official Validation input rows or schema differ")
    if set(train_keys) & set(val_keys) or set(train_calls) & set(val_calls):
        raise ValueError("Training and Official Validation identity overlap")
    if (len(val_keys) != EXPECTED_VALIDATION_ROWS
            or len(np.unique(val_calls)) != EXPECTED_VALIDATION_CALLS):
        raise ValueError("Official Validation row/call counts differ from the frozen contract")
    epochs = stage14_evidence["fixed_refit_epochs"]
    stage14_contract = grid._read_json(Path(stage14_evidence["run_root"]) / "contract.json")
    model_configs = {member_id: deepcopy(stage14_contract["model_configs"][member_id])
                     for member_id in MEMBER_IDS}
    feature_schema = training_audit["feature_schema"]
    schema_fingerprint = grid._json_fingerprint(feature_schema)
    if schema_fingerprint != training_audit["feature_schema_fingerprint"]:
        raise ValueError("feature schema fingerprint changed before run initialization")
    validation_meta = validation_data["feature_cache_audit"]
    contract = {
        "schema_version": FINAL_RUN_SCHEMA, "run_id": root.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage14_run_root": stage14_evidence["run_root"],
        "source_stage14_contract_sha256": stage14_evidence["contract_sha256"],
        "source_feature_run_root": training_audit["source_feature_run_root"],
        "source_training_feature_cache_manifest_sha256": training_audit["source_feature_cache_manifest_sha256"],
        "source_training_feature_block_sha256": training_audit["source_feature_block_sha256"],
        "source_clean_fingerprint": training_audit["clean_fingerprint"],
        "source_code_provenance": code_bundle_provenance,
        "feature_dimension": INPUT_DIMENSION,
        "feature_schema_fingerprint": schema_fingerprint,
        "feature_schema": feature_schema,
        "training": {**train_audit,
                      "feature_matrix_sha256": _array_fingerprint(training_matrix),
                      "feature_name_order_sha256": hashlib.sha256(
                          "\n".join(training_feature_names).encode("utf-8")).hexdigest()},
        "cross_split_identity_overlap": {"record_key": 0, "call_id": 0},
        "official_validation": {
            "rows": len(val_keys), "calls": int(len(np.unique(val_calls))),
            "record_key_signature": _record_key_signature(val_keys),
            "label_hash": _array_fingerprint(val_labels),
            "feature_cache_sha256": validation_meta.get("artifact_sha256"),
            "source_manifest_sha256": validation_meta.get("source_manifest_sha256"),
            "feature_extraction_audit": validation_meta,
        },
        "models": list(MODEL_IDS), "member_ids": list(MEMBER_IDS),
        "model_configs": model_configs,
        "fixed_refit_epochs": epochs,
        "refit_epoch_policy": "saved_stage14_best_internal_validation_epoch_fixed_no_early_stopping",
        "scaler": {"class": "sklearn.preprocessing.StandardScaler",
                   "fit_scope": "all_organizer_training_only",
                   "fit_rows": EXPECTED_TRAIN_ROWS,
                   "official_validation_used_for_fit": False},
        "ensemble": {"id": "E_TRG", "rule": "probability_simple_average",
                     "weights": {member: 1 / 3 for member in MEMBER_IDS}},
        "threshold": THRESHOLD,
        "official_validation_used_for_training_or_selection": False,
    }
    _write_json_new(root / "contract.json", contract)
    schema_doc = {"feature_dimension": INPUT_DIMENSION,
                  "feature_schema_fingerprint": schema_fingerprint,
                  "feature_schema": feature_schema,
                  "feature_names": list(training_feature_names)}
    _write_json_new(root / "feature_schema.json", schema_doc)
    training_config = {
        "seed": 42, "threshold": THRESHOLD,
        "scaler_fit_scope": "all_organizer_training_only",
        "validation_data_used_for_fit": False,
        "ensemble": contract["ensemble"],
        "models": {
            model: {"stage14_recipe": model_configs[member],
                    "fixed_refit_epochs": epochs[model],
                    "epoch_source_best_internal_validation": stage14_evidence["members"][model]["metrics"]["best_epoch"],
                    "source_epochs_run": stage14_evidence["members"][model]["metrics"]["epochs_run"],
                    "source_history_kind": stage14_evidence["members"][model]["metrics"]["history_kind"],
                    "fit_mode": "fixed_epochs_no_official_validation_early_stopping"}
            for model, member in zip(MODEL_IDS, MEMBER_IDS)
        },
    }
    _write_json_new(root / "training_config.json", training_config)
    ledger = {"schema_version": "mission2_final_refit_ledger_v1",
              "run_id": root.name, "status": "READY",
              "members": {member: "PENDING" for member in MEMBER_IDS},
              "completed_member_count": 0, "next_member": MEMBER_IDS[0]}
    _write_json_new(root / "run_ledger.json", ledger)
    return contract


def fit_full_training_scaler(training_matrix: np.ndarray,
                             validation_matrix: np.ndarray):
    """Fit the shared scaler on all Training rows and transform both matrices."""
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(training_matrix)
    validation = np.asarray(validation_matrix)
    if (train.dtype != np.float32 or train.ndim != 2 or train.shape[1] != INPUT_DIMENSION
            or validation.dtype != np.float32 or validation.ndim != 2
            or validation.shape[1] != INPUT_DIMENSION
            or not np.isfinite(train).all() or not np.isfinite(validation).all()):
        raise ValueError("full-training scaler input shape/dtype/finite contract differs")
    scaler = StandardScaler().fit(train)
    if int(scaler.n_samples_seen_) != len(train):
        raise ValueError("StandardScaler did not fit all organizer Training rows")
    train_scaled = scaler.transform(train).astype(np.float32, copy=False)
    validation_scaled = scaler.transform(validation).astype(np.float32, copy=False)
    if not np.isfinite(train_scaled).all() or not np.isfinite(validation_scaled).all():
        raise ValueError("scaled full-training or Official Validation matrix is nonfinite")
    return scaler, train_scaled, validation_scaled


def _save_full_training_scaler(scaler, folder: Path, feature_names) -> None:
    _write_json_new(folder / "scaler.json", {
        "fit_scope": "all_organizer_training_only",
        "training_rows": int(scaler.n_samples_seen_),
        "feature_names": list(feature_names), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
    })


def record_environment(run_root: str | Path, *, device: str = "cuda") -> dict:
    """Record the explicit fit device; never switch to CPU implicitly."""
    import importlib.metadata
    import platform
    import sys
    import torch

    if device not in ("cuda", "cpu"):
        raise ValueError(f"unsupported explicit training device: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; no CPU fallback is permitted")
    packages = ("numpy", "pandas", "scikit-learn", "torch", "tabm",
                "rtdl-revisiting-models", "rtdl-num-embeddings", "pytorch-tabular")
    environment = {
        "status": "READY", "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cuda_build": torch.version.cuda if device == "cuda" else None,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "official_validation_used_for_fit": False,
    }
    path = Path(run_root).resolve(strict=True) / "environment.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != environment:
            raise RuntimeError("final refit environment differs from its saved environment")
    else:
        _write_json_new(path, environment)
    return environment


def _check_training_arrays(X_train, y_train, X_validation, feature_names):
    train = np.asarray(X_train)
    y = np.asarray(y_train, dtype=np.int64)
    validation = np.asarray(X_validation)
    if (train.dtype != np.float32 or train.shape != (len(y), INPUT_DIMENSION)
            or validation.dtype != np.float32 or validation.ndim != 2
            or validation.shape[1] != INPUT_DIMENSION
            or not np.isfinite(train).all() or not np.isfinite(validation).all()
            or not np.isin(y, (0, 1)).all() or len(set(np.unique(y))) != 2
            or len(feature_names) != INPUT_DIMENSION or len(set(feature_names)) != INPUT_DIMENSION):
        raise ValueError("fixed full-refit model input contract differs")


def _epoch_batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Keep every row while avoiding a one-row BatchNorm tail."""
    batches = [order[start:start + batch_size]
               for start in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        singleton = batches.pop()
        batches[-1] = np.concatenate((batches[-1], singleton))
    if not batches or sum(len(batch) for batch in batches) != len(order):
        raise RuntimeError("full-training batch construction lost or duplicated rows")
    return batches


def _fit_complex_fixed_epochs(model_id, config, X_train, y_train, *, device: str,
                              epochs: int, on_epoch_end=None):
    """Train TabM/ResNet for a fixed count without any validation input."""
    import torch
    from . import mission2_complex_tabular_screen as screen
    from . import mission2_acoustic_feature_grid as grid

    train = np.asarray(X_train)
    labels = np.asarray(y_train, dtype=np.int64)
    if (train.dtype != np.float32 or train.shape != (len(labels), INPUT_DIMENSION)
            or not np.isfinite(train).all() or not np.isin(labels, (0, 1)).all()
            or epochs < 1):
        raise ValueError("fixed-epoch complex-model training arrays differ")
    started = time.perf_counter()
    history = []
    with screen._deterministic_training(device, int(config["seed"])):
        model = grid._make_complex_model(model_id, config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"], betas=tuple(config["adam_betas"]),
            eps=config["adam_eps"],
        )
        rng = np.random.default_rng(int(config["seed"]))
        batch_size = int(config["batch_size"])
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(labels))
            loss_total = 0.0
            seen = 0
            for indices in _epoch_batches(order, batch_size):
                xb = torch.from_numpy(train[indices].copy()).to(device)
                yb = torch.from_numpy(labels[indices].astype(np.float32, copy=False)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = screen._logits(model_id, model, xb)
                if model_id == "tabm":
                    yb = yb[:, None].expand_as(logits)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite full-training loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise ValueError("non-finite full-training gradient")
                optimizer.step()
                loss_total += float(loss.detach().cpu()) * len(indices)
                seen += len(indices)
            if seen != len(labels):
                raise RuntimeError("fixed-epoch optimizer skipped Training rows")
            row = {"epoch": epoch, "train_loss": loss_total / seen,
                   "training_rows": seen, "validation_used": False}
            history.append(row)
            if on_epoch_end is not None:
                on_epoch_end(model_id, epoch, epochs, row["train_loss"])
    return model, pd.DataFrame(history), time.perf_counter() - started


def _gandalf_probability(model, X, feature_names) -> tuple[np.ndarray, str]:
    frame = pd.DataFrame(np.asarray(X), columns=list(feature_names))
    predicted = model.predict(frame)
    columns = [name for name in predicted.columns if str(name).endswith("_probability")]
    positive = [name for name in columns if re.search(r"(?:^|[_-])1(?:[_-]|$)", str(name))]
    if len(positive) == 1:
        column = positive[0]
    elif len(columns) == 1:
        # Stage 13 used the sole emitted probability as P(speaker=1); preserve
        # that class direction only when the library emits one probability.
        column = columns[0]
    else:
        raise ValueError(f"GANDALF speaker-1 probability column is ambiguous: {columns}")
    probability = predicted[column].to_numpy(dtype=np.float64)
    if (probability.shape != (len(X),) or not np.isfinite(probability).all()
            or np.any((probability < 0.0) | (probability > 1.0))):
        raise ValueError("GANDALF speaker-1 probabilities are invalid")
    return probability, str(column)


def _fit_gandalf_fixed_epochs(config, feature_names, X_train, y_train,
                              X_validation, *, device: str, epochs: int,
                              checkpoint_dir: Path):
    """Fit GANDALF on every Training row with stopping/checkpoints disabled."""
    import torch
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig

    train_values = np.asarray(X_train)
    labels = np.asarray(y_train, dtype=np.int64)
    validation_values = np.asarray(X_validation)
    if (train_values.dtype != np.float32 or train_values.shape != (len(labels), INPUT_DIMENSION)
            or validation_values.dtype != np.float32
            or validation_values.ndim != 2 or validation_values.shape[1] != INPUT_DIMENSION
            or not np.isfinite(train_values).all() or not np.isfinite(validation_values).all()
            or epochs < 1):
        raise ValueError("GANDALF fixed-epoch input contract differs")
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    train = pd.DataFrame(train_values, columns=list(feature_names))
    train["target"] = labels
    validation_stub = train.iloc[:1].copy()
    validation_features = pd.DataFrame(validation_values, columns=list(feature_names))
    model_config = GANDALFConfig(
        task="classification", gflu_stages=int(config["gflu_stages"]),
        gflu_dropout=float(config["gflu_dropout"]),
        gflu_feature_init_sparsity=float(config["gflu_feature_init_sparsity"]),
        learning_rate=float(config["learning_rate"]), metrics=["accuracy"],
    )
    trainer_config = TrainerConfig(
        max_epochs=int(epochs), batch_size=int(config["batch_size"]),
        accelerator="gpu" if device == "cuda" else "cpu", devices=1,
        early_stopping=None, checkpoints=None, load_best=False,
        deterministic=True, seed=int(config["seed"]), progress_bar="none",
        # A separate validation frame is required by this library's datamodule;
        # disable its loop so no holdout or Official Validation participates.
        trainer_kwargs={"limit_val_batches": 0},
    )
    data_config = DataConfig(
        target=["target"], continuous_cols=list(feature_names), categorical_cols=[],
        validation_split=0.0,
    )
    model = TabularModel(
        data_config=data_config, model_config=model_config,
        optimizer_config=OptimizerConfig(), trainer_config=trainer_config,
        verbose=False, suppress_lightning_logger=True,
    )
    started = time.perf_counter()
    trainer = model.fit(train=train, validation=validation_stub,
                        max_epochs=int(epochs), seed=int(config["seed"]))
    # Lightning reports the number of epochs run after fit returns; adding one
    # here incorrectly rejects a run that completed exactly the configured limit.
    completed_epochs = int(getattr(trainer, "current_epoch", -1))
    if completed_epochs != epochs:
        raise RuntimeError(f"GANDALF ran {completed_epochs} epochs, expected exactly {epochs}")
    fit_seconds = time.perf_counter() - started
    before, probability_column = _gandalf_probability(model, validation_features, feature_names)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite GANDALF checkpoint directory: {checkpoint_dir}")
    model.save_model(str(checkpoint_dir))
    loaded = TabularModel.load_model(str(checkpoint_dir))
    inference_started = time.perf_counter()
    after, reloaded_column = _gandalf_probability(loaded, validation_features, feature_names)
    inference_seconds = time.perf_counter() - inference_started
    if probability_column != reloaded_column or not np.allclose(before, after, rtol=0, atol=1e-7):
        raise ValueError("GANDALF checkpoint reload predictions differ")
    history = pd.DataFrame([{
        "history_kind": "terminal_metrics_snapshot", "completed_epochs": completed_epochs,
        "source_best_internal_epoch": int(epochs), "validation_used": False,
    }])
    return loaded, before, after, history, {
        "fit_seconds": fit_seconds, "epochs_run": completed_epochs,
        "official_validation_inference_seconds": inference_seconds,
        "checkpoint_reload": "EXACT_PASS" if np.array_equal(before, after) else "ATOL_1E-7_PASS",
        "probability_column": probability_column,
        "history_kind": "terminal_metrics_snapshot",
        "validation_loop_disabled": True,
    }


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): _sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in {"status.json", "artifact_manifest.json"}}


def _update_final_ledger(root: Path, member_id: str, status: str) -> None:
    path = root / "run_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if member_id not in ledger["members"]:
        raise KeyError(f"unknown final refit member: {member_id}")
    ledger["members"][member_id] = status
    complete = [name for name, state in ledger["members"].items() if state == "COMPLETE"]
    pending = [name for name, state in ledger["members"].items() if state == "PENDING"]
    ledger["completed_member_count"] = len(complete)
    ledger["next_member"] = pending[0] if pending else None
    ledger["status"] = (
        "FAILED_STOP" if status == "FAILED" else
        "ALL_MEMBERS_COMPLETE" if len(complete) == len(MEMBER_IDS) else
        "RUNNING" if status == "RUNNING" else "READY"
    )
    _write_json_replace(path, ledger)


def _load_final_member_predictions(root: Path, member_id: str,
                                  validation_data: dict) -> tuple[pd.DataFrame, dict]:
    """Verify a completed member's saved prediction and artifact hashes."""
    from . import mission2_acoustic_feature_grid as grid

    folder = root / "members" / member_id
    status_path = folder / "status.json"
    if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("status") != "COMPLETE":
        raise RuntimeError(f"Final refit member is not marked COMPLETE: {member_id}")
    manifest_path = folder / "artifact_manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("member_id") != member_id:
        raise ValueError(f"Final refit artifact manifest member id differs: {member_id}")
    expected_hashes = artifact_manifest.get("files", {})
    observed_hashes = _artifact_hashes(folder)
    if expected_hashes != observed_hashes:
        raise ValueError(f"Final refit member artifact hashes differ: {member_id}")
    predictions = pd.read_parquet(folder / "official_validation_predictions.parquet")
    predictions = attach_tracking_metadata(predictions, validation_data["manifest"])
    validate_official_prediction_alignment({member_id: predictions}, validation_data["manifest"])
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    recomputed = recompute_saved_prediction_metrics(predictions)
    for name in ("validation_accuracy", "recall_0", "recall_1", "confusion_matrix",
                 "tn", "fp", "fn", "tp", "hce", "threshold"):
        if metrics.get(name) != recomputed.get(name):
            raise ValueError(f"Saved member metric does not reconstruct: {member_id}/{name}")
    if (metrics.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS")
            or metrics.get("official_validation_used_for_fit") is not False):
        raise ValueError(f"Final refit checkpoint or split audit is incomplete: {member_id}")
    return predictions, metrics


def fit_final_member(
    run_root: str | Path,
    model_id: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_validation_scaled: np.ndarray,
    validation_data: dict,
    feature_names: tuple[str, ...],
    *,
    device: str = "cuda",
    on_epoch_end=None,
) -> dict:
    """Fit one frozen member on all Training rows and persist a reload-checked checkpoint."""
    from . import mission2_acoustic_feature_grid as grid
    from . import mission2_complex_tabular_screen as screen

    if model_id not in MODEL_IDS:
        raise KeyError(f"Unknown final refit model: {model_id}")
    root = Path(run_root).resolve(strict=True)
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    member_id = f"{model_id}_F01_F02"
    state = ledger["members"].get(member_id)
    if state == "COMPLETE":
        _load_final_member_predictions(root, member_id, validation_data)
        return {"status": "VERIFIED_EXISTING", "member_id": member_id}
    if state != "PENDING":
        raise RuntimeError(f"Final refit member is {state}; partial or failed fits are never overwritten")
    if contract.get("schema_version") != FINAL_RUN_SCHEMA:
        raise ValueError("Final refit contract schema differs")
    _check_training_arrays(X_train_scaled, y_train, X_validation_scaled, feature_names)
    if len(y_train) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Final refit must use every organizer Training row")
    if _array_fingerprint(np.asarray(y_train, dtype=np.int64)) != contract["training"]["label_hash"]:
        raise ValueError("Final refit labels differ from the verified complete Training context")
    schema_doc = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    if (tuple(schema_doc.get("feature_names", ())) != tuple(feature_names)
            or schema_doc.get("feature_schema_fingerprint") != contract["feature_schema_fingerprint"]):
        raise ValueError("Final refit feature names differ from the frozen 149D schema")
    validation_manifest = validation_data["manifest"]
    if (len(X_validation_scaled) != EXPECTED_VALIDATION_ROWS
            or not np.array_equal(validation_data["record_keys"],
                                  validation_manifest.record_key.astype(str).to_numpy())
            or not np.array_equal(validation_data["y"],
                                  validation_manifest.speaker.to_numpy(dtype=np.int64))):
        raise ValueError("Official Validation model input is not aligned to its manifest")
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if environment.get("status") != "READY" or environment.get("device") != device:
        raise RuntimeError("The explicitly selected device must be checked before fitting")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable; refusing an implicit CPU retry")
    config = deepcopy(contract["model_configs"][member_id])
    epochs = int(contract["fixed_refit_epochs"][model_id])
    folder = root / "members" / member_id
    folder.mkdir(exist_ok=False)
    _write_json_new(folder / "config.json", config)
    _write_json_new(folder / "environment.json", environment)
    scaler_path = root / "scaler.json"
    if not scaler_path.is_file():
        raise FileNotFoundError("Training-only scaler must be saved before any model fit")
    (folder / "scaler.json").write_bytes(scaler_path.read_bytes())
    _write_json_new(folder / "status.json", {
        "status": "RUNNING", "member_id": member_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "run_id": root.name,
        "training_rows": len(y_train), "fixed_epochs": epochs,
        "official_validation_used_for_fit": False,
    })
    _update_final_ledger(root, member_id, "RUNNING")
    try:
        if model_id in ("tabm", "tabular_resnet"):
            model, history, fit_seconds = _fit_complex_fixed_epochs(
                model_id, config, X_train_scaled, y_train, device=device,
                epochs=epochs, on_epoch_end=on_epoch_end,
            )
            checkpoint = folder / "checkpoint.pt"
            import torch
            with checkpoint.open("xb") as stream:
                torch.save({
                    "candidate": model_id, "config": config,
                    "feature_names": list(feature_names),
                    "state_dict": {name: value.detach().cpu()
                                   for name, value in model.state_dict().items()},
                }, stream)
            before = screen.predict_probability(
                model_id, model, X_validation_scaled, device=device,
                batch_size=int(config["batch_size"]),
            )
            model.cpu()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"] != config or saved["feature_names"] != list(feature_names):
                raise ValueError("Final checkpoint config or feature order differs")
            reloaded = grid._make_complex_model(model_id, config, device)
            reloaded.load_state_dict(saved["state_dict"], strict=True)
            prediction_started = time.perf_counter()
            probability = screen.predict_probability(
                model_id, reloaded, X_validation_scaled, device=device,
                batch_size=int(config["batch_size"]),
            )
            inference_seconds = time.perf_counter() - prediction_started
            if not np.array_equal(before, probability):
                raise ValueError("Final checkpoint reload probabilities differ exactly")
            fit = {"fit_seconds": fit_seconds, "epochs_run": len(history),
                   "checkpoint_reload": "EXACT_PASS",
                   "official_validation_inference_seconds": inference_seconds,
                   "history_kind": "epoch_history"}
        else:
            model, _before, probability, history, fit = _fit_gandalf_fixed_epochs(
                config, feature_names, X_train_scaled, y_train,
                X_validation_scaled, device=device, epochs=epochs,
                checkpoint_dir=folder / "checkpoint",
            )
        if int(fit["epochs_run"]) != epochs:
            raise ValueError("Final refit epoch count differs from its frozen Stage 14 decision")
        prediction_metrics, predictions = grid._metric_bundle(
            validation_data["y"], probability,
            validation_data["record_keys"], validation_data["call_ids"],
        )
        predictions = attach_tracking_metadata(predictions, validation_manifest)
        metrics = {
            **prediction_metrics,
            "member_id": member_id, "model_id": model_id,
            "input_dimension": INPUT_DIMENSION,
            "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
            "config_id": config.get("config_id"), "seed": int(config["seed"]),
            "training_rows": len(y_train), "official_validation_rows": len(probability),
            "fixed_epochs": epochs, "epochs_run": int(fit["epochs_run"]),
            "fit_seconds": float(fit["fit_seconds"]),
            "official_validation_inference_seconds": float(fit["official_validation_inference_seconds"]),
            "checkpoint_reload": fit["checkpoint_reload"],
            "history_kind": fit["history_kind"],
            "feature_names": list(feature_names),
            "scaler_fit_scope": "all_organizer_training_only",
            "official_validation_used_for_fit": False,
            "source_stage14_best_epoch": epochs,
        }
        history.to_csv(folder / "training_history.csv", index=False, mode="x")
        predictions.to_parquet(folder / "official_validation_predictions.parquet", index=False)
        _write_json_new(folder / "metrics.json", metrics)
        _write_json_new(folder / "artifact_manifest.json", {
            "schema_version": "mission2_final_member_artifacts_v1",
            "member_id": member_id, "files": _artifact_hashes(folder),
        })
        _write_json_replace(folder / "status.json", {
            "status": "COMPLETE", "member_id": member_id,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_reload": fit["checkpoint_reload"],
            "epochs_run": int(fit["epochs_run"]),
        })
        _update_final_ledger(root, member_id, "COMPLETE")
        _load_final_member_predictions(root, member_id, validation_data)
        return {"status": "COMPLETE", "member_id": member_id,
                "validation_accuracy": metrics["validation_accuracy"],
                "fit_seconds": metrics["fit_seconds"],
                "inference_seconds": metrics["official_validation_inference_seconds"]}
    except Exception as error:
        _write_json_replace(folder / "status.json", {
            "status": "FAILED", "member_id": member_id,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error),
        })
        _update_final_ledger(root, member_id, "FAILED")
        raise


def run_remaining_final_members(
    run_root: str | Path,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_validation_scaled: np.ndarray,
    validation_data: dict,
    feature_names: tuple[str, ...],
    *,
    device: str = "cuda",
    on_epoch_end=None,
) -> dict:
    """Fit the three saved members in frozen order; stop at the first failure."""
    results = []
    for model_id in MODEL_IDS:
        result = fit_final_member(
            run_root, model_id, X_train_scaled, y_train, X_validation_scaled,
            validation_data, feature_names, device=device, on_epoch_end=on_epoch_end,
        )
        results.append(result)
        print(f"{result['status']} {result['member_id']} "
              f"accuracy={result.get('validation_accuracy', float('nan'))}", flush=True)
    return {"status": "ALL_MEMBERS_COMPLETE", "results": results}


def _official_slice_table(predictions: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> pd.DataFrame:
    """Analyze duration/overlap only from identity-validated Official tracking metadata."""
    from . import mission2_model_experiments as models
    from . import mission2_standard_representation_comparison as diagnostics

    duration = ((manifest.endAt_ms.to_numpy(dtype=np.float64)
                 - manifest.startAt_ms.to_numpy(dtype=np.float64)) / 1000.0)
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("Official Validation duration metadata is invalid")
    overlap = models._overlap_mask(manifest)
    masks = diagnostics.slice_masks(pd.DataFrame({
        "duration_sec": duration, "overlap": overlap,
        "y_true": manifest.speaker.to_numpy(dtype=np.int64),
    }))
    rows = []
    for model_id, frame in predictions.items():
        correct = frame.correct.to_numpy(dtype=bool)
        for slice_name, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            rows.append({"model_id": model_id, "slice": slice_name,
                         "rows": int(mask.sum()),
                         "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return pd.DataFrame(rows)


def evaluate_final_run(run_root: str | Path, validation_data: dict) -> dict:
    """Create the fixed E_TRG Official report and independently audit saved predictions."""
    from . import mission2_acoustic_feature_grid as grid

    root = Path(run_root).resolve(strict=True)
    output_names = (
        "ensemble_predictions.parquet", "official_validation_comparison.csv",
        "official_validation_paired_errors.csv",
        "official_validation_duration_overlap_slices.csv",
        "official_validation_metrics.json", "final_audit.json",
        "artifact_manifest.json",
    )
    existing_outputs = [name for name in output_names if (root / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            "Final evaluation artifacts already exist; no overwrite/re-evaluation in this run: "
            + ", ".join(existing_outputs)
        )
    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    if any(ledger["members"].get(member_id) != "COMPLETE" for member_id in MEMBER_IDS):
        raise RuntimeError("Official Validation evaluation requires all three verified refits")
    manifest = validation_data["manifest"]
    member_predictions, member_metrics = {}, {}
    for member_id in MEMBER_IDS:
        frame, metrics = _load_final_member_predictions(root, member_id, validation_data)
        member_predictions[member_id] = frame
        member_metrics[member_id] = metrics
    ensemble_started = time.perf_counter()
    ensemble_metrics, ensemble = build_final_ensemble(member_predictions, manifest)
    ensemble_runtime = time.perf_counter() - ensemble_started
    ensemble_metrics["official_validation_inference_seconds"] = ensemble_runtime
    all_predictions = {**member_predictions, "E_TRG": ensemble}
    validate_official_prediction_alignment(all_predictions, manifest)
    ensemble_metrics["official_validation_used_for_training_or_selection"] = False
    ensemble.to_parquet(root / "ensemble_predictions.parquet", index=False)

    comparison_rows = []
    for member_id in MEMBER_IDS:
        comparison_rows.append({"model_id": member_id, **member_metrics[member_id]})
    comparison_rows.append({"model_id": "E_TRG", **ensemble_metrics})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root / "official_validation_comparison.csv", index=False)
    paired_rows = []
    for member_id in MEMBER_IDS:
        counts, regressions, corrections = grid._paired_counts(member_predictions[member_id], ensemble)
        paired_rows.append({
            "model_id": member_id, "ensemble_only_correct": counts["corrections"],
            "member_only_correct": counts["regressions"],
            "common_correct": counts["both_correct"], "common_error": counts["both_wrong"],
            "net_correction": counts["net_corrections"],
            "changed_rows": counts["changed_rows"],
            "correction_record_keys": ensemble.loc[corrections, "record_key"].astype(str).tolist(),
            "regression_record_keys": ensemble.loc[regressions, "record_key"].astype(str).tolist(),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "official_validation_paired_errors.csv", index=False)
    slices = _official_slice_table(all_predictions, manifest)
    slices.to_csv(root / "official_validation_duration_overlap_slices.csv", index=False)
    metrics_payload = {
        "run_id": root.name, "split": "Official Validation",
        "threshold": THRESHOLD,
        "models": {**member_metrics, "E_TRG": ensemble_metrics},
        "paired_errors": paired_rows,
        "duration_overlap_slices": slices.to_dict(orient="records"),
        "metric_source": "saved official_validation_predictions.parquet and ensemble_predictions.parquet",
    }
    _write_json_new(root / "official_validation_metrics.json", metrics_payload)

    recomputed = {}
    persisted_predictions = {}
    for member_id in MEMBER_IDS:
        frame = pd.read_parquet(root / "members" / member_id / "official_validation_predictions.parquet")
        frame = attach_tracking_metadata(frame, manifest)
        persisted_predictions[member_id] = frame
        recomputed[member_id] = recompute_saved_prediction_metrics(frame)
    persisted_ensemble = pd.read_parquet(root / "ensemble_predictions.parquet")
    persisted_ensemble = attach_tracking_metadata(persisted_ensemble, manifest)
    persisted_predictions["E_TRG"] = persisted_ensemble
    recomputed["E_TRG"] = recompute_saved_prediction_metrics(persisted_ensemble)
    validate_official_prediction_alignment(persisted_predictions, manifest)
    for name, observed in recomputed.items():
        original = member_metrics[name] if name in member_metrics else ensemble_metrics
        if any(observed[key] != original[key] for key in (
            "validation_accuracy", "confusion_matrix", "recall_0", "recall_1",
            "fp", "fn", "hce", "threshold")):
            raise ValueError(f"Saved Official Validation metrics do not independently reconstruct: {name}")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    audit = {
        "status": "FINAL_VERIFIED",
        "run_id": root.name,
        "feature_dimension": INPUT_DIMENSION,
        "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
        "training_rows": EXPECTED_TRAIN_ROWS, "training_calls": EXPECTED_TRAIN_CALLS,
        "official_validation_rows": EXPECTED_VALIDATION_ROWS,
        "official_validation_calls": EXPECTED_VALIDATION_CALLS,
        "train_validation_record_key_overlap": contract["cross_split_identity_overlap"]["record_key"],
        "train_validation_call_id_overlap": contract["cross_split_identity_overlap"]["call_id"],
        "official_validation_used_for_scaler_gradient_early_stopping_or_selection": False,
        "fixed_threshold": THRESHOLD,
        "ensemble": contract["ensemble"],
        "member_checkpoint_reload": {
            member_id: member_metrics[member_id]["checkpoint_reload"] for member_id in MEMBER_IDS
        },
        "saved_prediction_alignment": "PASS",
        "saved_prediction_metrics_recomputed": "PASS",
        "confusion_matrix_counts_sum_to_validation_rows": all(
            sum(sum(row) for row in result["confusion_matrix"]) == EXPECTED_VALIDATION_ROWS
            for result in recomputed.values()),
        "submission_inference_readiness": "NOT READY: shared 42D inference path is not integrated with this 149D run",
        "unverified_or_pending": [
            "shared submission inference integration and official sample-output parity",
            "artifact re-download verification after Drive sync completes",
        ],
    }
    if not audit["confusion_matrix_counts_sum_to_validation_rows"]:
        raise ValueError("Confusion matrix totals differ from Official Validation row count")
    _write_json_new(root / "final_audit.json", audit)
    _write_json_replace(root / "run_ledger.json", {
        **ledger, "status": "FINAL_VERIFIED", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_member_count": len(MEMBER_IDS), "next_member": None,
    })
    _write_json_replace(root / "status.json", {
        "status": "FINAL_VERIFIED", "run_id": root.name,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    root_hashes = _artifact_hashes(root)
    _write_json_new(root / "artifact_manifest.json", {
        "schema_version": "mission2_final_run_artifacts_v1",
        "run_id": root.name, "files": root_hashes,
    })
    return {"status": audit["status"], "run_id": root.name,
            "comparison": comparison, "paired_errors": paired, "slices": slices,
            "audit": audit}
