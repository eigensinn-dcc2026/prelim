"""Explicit, resumable eGeMAPSv02 extraction for the Stage 4 Notebook.

Importing this module does not read audio, create a cache, or run extraction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from .audio import read_wav_frames
from .mission2_standard_representation import (
    EgemapsData, _padded_rows_from_metadata, build_egemaps_cache,
    extract_egemaps, get_egemaps_feature_names, load_egemaps_cache,
)


PADDING_TARGET_SAMPLES = 800
SHORT_AUDIO_FALLBACK = "zero_pad_to_800_on_nonfinite_short_segment"
RECOVERY_FILES = (
    "matrix_88_partial.npy", "completed_mask.npy", "recovery_metadata.json",
    "record_keys.npy", "call_ids.npy",
)
CHECKPOINT_CALL_INTERVAL = 250
CRASH_JOURNAL = "committed_checkpoint.npz"


def extract_with_short_audio_fallback(
    waveform: np.ndarray, sample_rate: int,
) -> tuple[np.ndarray, tuple[str, ...], bool]:
    """Use the original crop first; retry only a warned short/nonfinite result."""

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", UserWarning)
        try:
            vector, names = extract_egemaps(waveform, sample_rate)
        except ValueError as error:
            warned_short = any("Segment too short" in str(item.message) for item in observed)
            if (sample_rate != 8_000 or len(waveform) >= PADDING_TARGET_SAMPLES
                    or str(error) != "eGeMAPSv02 returned NaN or inf; audio may be too short"
                    or not warned_short):
                raise
        else:
            if vector.shape != (88,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError("eGeMAPSv02 original crop did not yield finite float32 88D")
            return vector, names, False

    # Only the current utterance is extended, at the end, to a fixed 800 samples.
    padded = np.pad(waveform, (0, PADDING_TARGET_SAMPLES - len(waveform)))
    vector, names = extract_egemaps(padded, sample_rate)
    if vector.shape != (88,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError("eGeMAPSv02 padded retry did not yield finite float32 88D")
    return vector, names, True


def _recovery_metadata(
    keys: np.ndarray, names: tuple[str, ...], fingerprint: str,
    completed: int, padded_rows: list[tuple[str, int]],
) -> dict:
    return {
        "rows": len(keys),
        "dim": 88,
        "completed_rows": completed,
        "remaining_rows": len(keys) - completed,
        "feature_names": list(names),
        "clean_reference_fingerprint": fingerprint,
        "source_provenance": fingerprint,
        "short_audio_fallback": SHORT_AUDIO_FALLBACK,
        "padding_target_samples": PADDING_TARGET_SAMPLES,
        "sample_rate": 8_000,
        "padded_row_count": len(padded_rows),
        "padded_record_keys": [key for key, _ in padded_rows],
        "padded_original_sample_lengths": [length for _, length in padded_rows],
    }


def _padded_rows(
    metadata: dict, keys: np.ndarray, mask: np.ndarray,
) -> list[tuple[str, int]]:
    padded = _padded_rows_from_metadata(metadata)
    if padded is None:
        # The RAM-recovered legacy artifact predates the fallback policy.
        if any(name in metadata for name in ("sample_rate", "padded_row_count")):
            raise ValueError("STOP: partial padded-row metadata is incomplete")
        return []
    if metadata.get("sample_rate") != 8_000:
        raise ValueError("STOP: partial padding sample rate differs")
    positions = {key: index for index, key in enumerate(keys)}
    if (len(positions) != len(keys) or len({key for key, _ in padded}) != len(padded)
            or any(key not in positions or type(length) is not int
                   or not 0 < length < PADDING_TARGET_SAMPLES
                   or not mask[positions[key]] for key, length in padded)
            or padded != sorted(padded, key=lambda item: positions[item[0]])):
        raise ValueError("STOP: partial padded-row provenance differs")
    return padded


def _validate_partial(
    root: Path, keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    fingerprint: str, *, read_only: bool = False,
) -> tuple[np.memmap, np.memmap, list[tuple[str, int]]]:
    missing = [name for name in RECOVERY_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"STOP: partial recovery is incomplete: {missing}")
    metadata = json.loads((root / "recovery_metadata.json").read_text(encoding="utf-8"))
    matrix = np.load(root / "matrix_88_partial.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(root / "completed_mask.npy", mmap_mode="r", allow_pickle=False)
    # The user-saved RAM recovery has object-dtype string arrays. Only these
    # two local identity files need pickle loading; new checkpoints write Unicode.
    stored_keys = np.load(root / "record_keys.npy", allow_pickle=True)
    stored_calls = np.load(root / "call_ids.npy", allow_pickle=True)
    if matrix.shape != (len(keys), 88) or matrix.dtype != np.float32:
        raise ValueError("STOP: partial matrix shape/dtype differs")
    if mask.shape != (len(keys),) or mask.dtype != np.bool_:
        raise ValueError("STOP: partial completed_mask shape/dtype differs")
    if (stored_keys.shape != keys.shape
            or any(not isinstance(value, (str, np.str_)) for value in stored_keys)
            or not np.array_equal(stored_keys, keys)):
        raise ValueError("STOP: partial record_key order differs")
    if (stored_calls.shape != calls.shape
            or any(not isinstance(value, (str, np.str_)) for value in stored_calls)
            or not np.array_equal(stored_calls, calls)):
        raise ValueError("STOP: partial call_id order differs")
    completed = int(np.count_nonzero(mask))
    if (metadata.get("rows") != len(keys) or metadata.get("dim") != 88
            or metadata.get("completed_rows") != completed
            or metadata.get("remaining_rows") != len(keys) - completed
            or metadata.get("feature_names") != list(names)
            or metadata.get("clean_reference_fingerprint") != fingerprint
            or metadata.get("source_provenance", fingerprint) != fingerprint):
        raise ValueError("STOP: partial row/feature/provenance metadata differs")
    # Check in bounded chunks so the 889,358 x 88 matrix is not copied into RAM.
    for start in range(0, len(keys), 8192):
        stop = min(start + 8192, len(keys))
        finite_rows = np.isfinite(matrix[start:stop]).all(axis=1)
        if not np.array_equal(finite_rows, mask[start:stop]):
            raise ValueError(f"STOP: partial completed_mask/finite mismatch at row {start}")
    padded = _padded_rows(metadata, keys, mask)
    del matrix, mask
    return (
        np.load(root / "matrix_88_partial.npy", mmap_mode="r" if read_only else "r+", allow_pickle=False),
        np.load(root / "completed_mask.npy", mmap_mode="r" if read_only else "r+", allow_pickle=False),
        padded,
    )


def _create_partial(
    root: Path, keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    fingerprint: str,
) -> tuple[np.memmap, np.memmap, list[tuple[str, int]]]:
    root.mkdir(parents=True, exist_ok=False)
    matrix = np.lib.format.open_memmap(
        root / "matrix_88_partial.npy", mode="w+", dtype=np.float32,
        shape=(len(keys), 88),
    )
    matrix[:] = np.nan
    mask = np.lib.format.open_memmap(
        root / "completed_mask.npy", mode="w+", dtype=np.bool_, shape=(len(keys),),
    )
    mask[:] = False
    matrix.flush()
    mask.flush()
    for name, values in (("record_keys.npy", keys), ("call_ids.npy", calls)):
        with (root / name).open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
    (root / "recovery_metadata.json").write_text(
        json.dumps(_recovery_metadata(keys, names, fingerprint, 0, []), indent=2),
        encoding="utf-8",
    )
    return matrix, mask, []


def _checkpoint(
    root: Path, matrix: np.memmap, mask: np.memmap, keys: np.ndarray,
    names: tuple[str, ...], fingerprint: str, padded_by_key: dict[str, int],
) -> None:
    # A controlled extraction error reaches this checkpoint before propagating.
    matrix.flush()
    mask.flush()
    padded = [(key, padded_by_key[key]) for key in keys if key in padded_by_key]
    metadata = _recovery_metadata(
        keys, names, fingerprint, int(np.count_nonzero(mask)), padded,
    )
    previous = json.loads((root / "recovery_metadata.json").read_text(encoding="utf-8"))
    for field in ("resume_history", "windows_overlap_record_keys"):
        if field in previous:
            metadata[field] = previous[field]
    if (root / CRASH_JOURNAL).exists():
        # The atomic journal is the last durable commit boundary for a Colab
        # runtime kill. Its mask and metadata must describe the same rows.
        journal_tmp = root / f"{CRASH_JOURNAL}.tmp"
        with journal_tmp.open("wb") as stream:
            np.savez(stream, mask=np.asarray(mask),
                     metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(journal_tmp, root / CRASH_JOURNAL)
    temporary = root / "recovery_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, root / "recovery_metadata.json")


def validate_existing_partial(
    manifest: pd.DataFrame, recovery_root: str | Path,
    record_keys: np.ndarray, call_ids: np.ndarray, feature_names: tuple[str, ...],
    *, fingerprint: str, expected_rows: int = 889_358,
) -> dict:
    """Read-only strict check; a missing or stale partial is never created here."""
    keys = np.asarray(record_keys, dtype=str)
    calls = np.asarray(call_ids, dtype=str)
    if (len(manifest) != expected_rows or len(keys) != expected_rows
            or calls.shape != keys.shape
            or not np.array_equal(manifest.index.to_numpy(), np.arange(expected_rows))
            or not manifest["split"].astype(str).eq("Training").all()
            or not np.array_equal(manifest["record_key"].to_numpy(), keys)
            or not np.array_equal(manifest["call_id"].to_numpy(), calls)
            or tuple(feature_names) != get_egemaps_feature_names()):
        raise ValueError("STOP: Training manifest row/order/feature contract differs")
    matrix, mask, padded = _validate_partial(
        Path(recovery_root), keys, calls, tuple(feature_names), fingerprint, read_only=True,
    )
    metadata = json.loads((Path(recovery_root) / "recovery_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("source_provenance") != fingerprint
            or metadata.get("short_audio_fallback") != SHORT_AUDIO_FALLBACK
            or metadata.get("padding_target_samples") != PADDING_TARGET_SAMPLES
            or metadata.get("sample_rate") != 8_000):
        raise ValueError("STOP: partial clean provenance or short-audio policy differs")
    result = {"rows": expected_rows, "completed": int(np.count_nonzero(mask)),
              "remaining": int(np.count_nonzero(~mask)), "padded": len(padded),
              "fingerprint": fingerprint, "status": "PASS"}
    del matrix, mask
    return result


def fill_existing_partial(
    manifest: pd.DataFrame, raw_root: str | Path, recovery_root: str | Path,
    keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...], fingerprint: str,
) -> dict:
    """Resume only missing rows; leave final cache publication to a separate step."""
    root = Path(recovery_root)
    matrix, mask, padded = _validate_partial(root, keys, calls, names, fingerprint)
    padded_by_key = dict(padded)
    started = perf_counter()
    initial_completed = int(np.count_nonzero(mask))
    groups = manifest.groupby("audio_path", sort=False)
    try:
        for call_number, (audio_path, group) in enumerate(groups, start=1):
            if bool(np.all(mask[group.index.to_numpy()])):
                continue  # Fully completed calls require no WAV read.
            audio = read_wav_frames(Path(raw_root) / str(audio_path))
            if audio.sample_rate != 8_000 or audio.waveform.shape[0] != 1:
                raise ValueError(f"STOP: unexpected WAV contract: {audio_path}")
            for row in group.itertuples():
                if mask[row.Index]:
                    continue
                start, end = int(row.crop_start_frame), int(row.crop_end_frame)
                if start < 0 or end <= start or end > audio.waveform.shape[1]:
                    raise ValueError(f"STOP: invalid crop for {row.record_key}")
                crop = audio.waveform[0, start:end]
                try:
                    vector, returned_names, used_padding = extract_with_short_audio_fallback(
                        crop, audio.sample_rate,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"STOP: eGeMAPS extraction failed for {row.record_key} "
                        f"(samples={len(crop)})"
                    ) from error
                if returned_names != names:
                    raise ValueError(f"STOP: eGeMAPS feature order changed for {row.record_key}")
                matrix[row.Index] = vector
                if used_padding:
                    padded_by_key[str(row.record_key)] = len(crop)
                mask[row.Index] = True
            if call_number % CHECKPOINT_CALL_INTERVAL == 0:
                _checkpoint(root, matrix, mask, keys, names, fingerprint, padded_by_key)
                completed = int(np.count_nonzero(mask))
                elapsed = perf_counter() - started
                print("Calls visited:", call_number, "completed:", completed,
                      "remaining:", len(keys) - completed, "padded:", len(padded_by_key),
                      "elapsed seconds:", round(elapsed, 1),
                      "rows/sec:", round((completed - initial_completed) / max(elapsed, 1e-9), 2))
    finally:
        _checkpoint(root, matrix, mask, keys, names, fingerprint, padded_by_key)
    completed = int(np.count_nonzero(mask))
    return {"completed": completed, "remaining": len(keys) - completed,
            "padded": len(padded_by_key), "elapsed_seconds": perf_counter() - started}


def resume_egemaps_extraction(
    manifest: pd.DataFrame, raw_root: str | Path, recovery_root: str | Path,
    final_cache_path: str | Path, record_keys: np.ndarray, call_ids: np.ndarray,
    feature_names: tuple[str, ...], *, source_provenance: str,
    clean_reference_fingerprint: str, expected_rows: int = 889_358,
) -> EgemapsData:
    """Validate a recovery artifact, skip completed rows, then publish a full cache."""

    keys = np.asarray(record_keys, dtype=str)
    calls = np.asarray(call_ids, dtype=str)
    names = tuple(feature_names)
    if (keys.ndim != 1 or calls.ndim != 1 or len(keys) != expected_rows
            or calls.shape != keys.shape
            or len(manifest) != len(keys)
            or not np.array_equal(manifest.index.to_numpy(), np.arange(len(keys)))
            or not np.array_equal(manifest["record_key"].to_numpy(), keys)
            or not np.array_equal(manifest["call_id"].to_numpy(), calls)
            or names != get_egemaps_feature_names()
            or not isinstance(source_provenance, str) or not source_provenance
            or source_provenance != clean_reference_fingerprint):
        raise ValueError("STOP: current rows, feature order, or clean source contract differs")
    root, final_path = Path(recovery_root), Path(final_cache_path)
    if final_path.exists():
        raise FileExistsError(f"STOP: final cache already exists: {final_path}")
    if root.exists():
        matrix, mask, padded = _validate_partial(root, keys, calls, names, source_provenance)
        print("REUSING_PARTIAL_EGEMAPS_CACHE")
        print("Completed / remaining:", int(np.count_nonzero(mask)),
              len(keys) - int(np.count_nonzero(mask)))
    else:
        matrix, mask, padded = _create_partial(root, keys, calls, names, source_provenance)
    del matrix, mask, padded
    fill_existing_partial(manifest, raw_root, root, keys, calls, names, source_provenance)
    matrix, mask, padded = _validate_partial(root, keys, calls, names, source_provenance)

    if not bool(np.all(mask)):
        raise ValueError("STOP: final cache cannot publish before all rows complete")
    if not np.isfinite(matrix).all():
        raise ValueError("STOP: final matrix contains NaN or inf")
    if names != get_egemaps_feature_names():
        raise ValueError("STOP: final feature order differs")
    if not np.array_equal(manifest["record_key"].to_numpy(), keys):
        raise ValueError("STOP: final record_key order differs")
    if not np.array_equal(manifest["call_id"].to_numpy(), calls):
        raise ValueError("STOP: final call_id order differs")
    padded_rows = padded
    final_path.parent.mkdir(parents=True, exist_ok=True)
    build_egemaps_cache(
        final_path, matrix, keys, calls, names, source_provenance=source_provenance,
        padded_rows=padded_rows,
    )
    result = load_egemaps_cache(final_path, keys, calls, source_provenance=source_provenance)
    if (result.metadata["short_audio_fallback"] != SHORT_AUDIO_FALLBACK
            or result.metadata["padded_row_count"] != len(padded_rows)):
        raise ValueError("STOP: published cache lacks fallback provenance")
    return result
