"""One-utterance eGeMAPSv02 extraction and paired Stage 4-A preparation.

Importing this module does not extract audio, write a cache, or fit a model.
The caller explicitly invokes those operations after supplying aligned rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import opensmile
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import mission2_acoustic_feature_expansion as acoustic
from .mission2_model_experiments import (
    E05_EXTRA_FEATURE_NAMES,
    EXPECTED_INTERNAL_SPLIT,
    EXPECTED_OFFICIAL_CALLS,
    EXPECTED_OFFICIAL_ROWS,
    MFCC40_FEATURE_NAMES,
    MFCC_CONFIG,
    M04_MLP_CONFIG,
    calculate_paired_prediction_changes,
    calculate_prediction_metrics,
    config_fingerprint,
    create_fixed_internal_split,
    record_key_signature,
    standalone_group_split_indices,
    standalone_training_cache_paths,
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

PROVENANCE_EXACT = "EXACT_PROVENANCE_MATCH"
PROVENANCE_FILESYSTEM_ONLY = "FILESYSTEM_SIGNATURE_ONLY_MISMATCH"
PROVENANCE_STALE = "STALE_OR_DIFFERENT_CACHE"
PROVENANCE_UNVERIFIABLE = "UNVERIFIABLE"
RAW_CONTENT_SIGNATURE_KIND = "sha256_WAV_JSON_bytes_v1"


def audit_cache_provenance(
    *, cache_name: str, manifest_rows: pd.DataFrame, cache_rows: pd.DataFrame,
    metadata: Mapping[str, Any], manifest_metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, Any], feature_names: Sequence[str],
    label_column: str | None = None,
    verified_current_content_sha256: str | None = None,
    artifact_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit alignment separately from raw-content identity; never edit metadata.

    A matching key only identifies an annotation row. It says nothing about the
    waveform bytes used when the feature was extracted. A content digest must
    therefore be present on both sides before an old cache may be cleared.
    """
    result: dict[str, Any] = {
        "cache": cache_name, "rows": len(cache_rows), "keys": "unchecked",
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "call_id": "unchecked", "label": "unavailable", "schema": "unchecked",
        "schema_version": metadata.get("cache_schema_version"),
        "source_split": metadata.get("source_split"),
        "feature_names": metadata.get("feature_names", list(feature_names)),
        "source_signature": metadata.get("source_signature"),
        "manifest_source_signature": manifest_metadata.get("source_signature"),
        "source_inventory_signature": metadata.get("source_inventory_signature", metadata.get("source_signature")),
        "source_content_sha256": metadata.get("source_content_sha256"),
        "record_key_signature": metadata.get("record_key_signature"),
        "config_fingerprint": metadata.get(
            "feature_manifest_fingerprint",
            metadata.get("extraction_config", metadata.get("mfcc_config")),
        ),
        "parent_provenance": metadata.get("parent_provenance", "not recorded"),
        "status": PROVENANCE_UNVERIFIABLE, "recommended_action": "regenerate cache",
        "reason": "RAW_SOURCE_IDENTITY_NOT_RECORDED",
    }
    required = {"record_key", "call_id", *feature_names}
    if not required.issubset(cache_rows.columns) or not {"record_key", "call_id"}.issubset(manifest_rows.columns):
        result.update(status=PROVENANCE_STALE, reason="required columns differ")
        return result
    if len(cache_rows) != len(manifest_rows) or len(cache_rows) != expected_metadata.get("row_count", len(manifest_rows)):
        result.update(status=PROVENANCE_STALE, reason="row count differs")
        return result
    keys = cache_rows["record_key"].astype(str)
    manifest_keys = manifest_rows["record_key"].astype(str)
    if (cache_rows["record_key"].isna().any() or manifest_rows["record_key"].isna().any()
            or not keys.is_unique or not manifest_keys.is_unique):
        result.update(status=PROVENANCE_STALE, keys="duplicate/null", reason="record_key is not unique")
        return result
    if set(keys) != set(manifest_keys):
        result.update(status=PROVENANCE_STALE, keys="set mismatch", reason="record_key set differs")
        return result
    if not keys.reset_index(drop=True).equals(manifest_keys.reset_index(drop=True)):
        result.update(status=PROVENANCE_STALE, keys="order mismatch", reason="record_key order differs")
        return result
    result["keys"] = "exact"
    if not cache_rows["call_id"].astype(str).reset_index(drop=True).equals(
        manifest_rows["call_id"].astype(str).reset_index(drop=True)
    ):
        result.update(status=PROVENANCE_STALE, call_id="mismatch", reason="call_id differs")
        return result
    result["call_id"] = "exact"
    if label_column is not None:
        if label_column not in cache_rows or "speaker" not in manifest_rows or not np.array_equal(
            cache_rows[label_column].to_numpy(), manifest_rows["speaker"].to_numpy()
        ):
            result.update(status=PROVENANCE_STALE, label="mismatch", reason="labels differ")
            return result
        result["label"] = "exact"
    metadata_contract = {key: value for key, value in expected_metadata.items()
                         if key not in {"source_signature", "source_content_sha256"}}
    if any(metadata.get(key) != value for key, value in metadata_contract.items()):
        result.update(status=PROVENANCE_STALE, schema="mismatch", reason="schema/config/feature metadata differs")
        return result
    if list(cache_rows.columns) != ["record_key", "call_id", *feature_names] and cache_name != "MFCC40":
        result.update(status=PROVENANCE_STALE, schema="mismatch", reason="feature column order differs")
        return result
    result["schema"] = "exact"
    if "record_key_signature" in metadata and metadata["record_key_signature"] != record_key_signature(keys):
        result.update(status=PROVENANCE_STALE, reason="record_key signature differs")
        return result
    cache_content = metadata.get("source_content_sha256")
    manifest_content = manifest_metadata.get("source_content_sha256")
    digest_kind_matches = (
        metadata.get("source_content_signature_kind") == RAW_CONTENT_SIGNATURE_KIND
        and manifest_metadata.get("source_content_signature_kind") == RAW_CONTENT_SIGNATURE_KIND
    )
    digests_well_formed = all(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in (cache_content, manifest_content, verified_current_content_sha256)
    )
    if not digest_kind_matches or not digests_well_formed:
        cache_signature = metadata.get("source_signature")
        manifest_signature = manifest_metadata.get("source_signature")
        if cache_signature and manifest_signature and cache_signature != manifest_signature:
            result["reason"] = "SOURCE_SIGNATURE_MISMATCH"
        return result
    if cache_content != manifest_content or cache_content != verified_current_content_sha256:
        result.update(status=PROVENANCE_STALE, reason="raw content digest differs")
        return result
    if metadata.get("source_signature") != manifest_metadata.get("source_signature"):
        result.update(status=PROVENANCE_FILESYSTEM_ONLY,
                      reason="raw content digest matches; filesystem signature differs",
                      recommended_action="retain immutable cache; strict loader still requires a matching source signature")
    else:
        result.update(status=PROVENANCE_EXACT, reason="raw content digest and cache contract match",
                      recommended_action="strict load")
    return result


def audit_stage4_59d_dependencies(repo_root: str | Path) -> pd.DataFrame:
    """Read the six immutable Training artifacts and report every dependency.

    This is diagnostic only. It never calls a strict loader, extractor, or fit.
    Old caches without a build-time raw-content digest remain unverifiable.
    """
    root = Path(repo_root)
    paths = standalone_training_cache_paths(root)
    manifest_meta = json.loads(paths.manifest_metadata.read_text(encoding="utf-8"))
    manifest = pd.read_parquet(paths.manifest_parquet)
    source = manifest.loc[manifest["split"].astype(str) == "Training"].reset_index(drop=True)
    if len(source) != 889_358 or not source["record_key"].is_unique:
        raise ValueError("STOP: Training manifest row/key contract differs")
    # Historical metadata lacks this digest, so the expensive live hash is skipped.
    current_content_sha256 = None
    if manifest_meta.get("source_content_signature_kind") == RAW_CONTENT_SIGNATURE_KIND:
        current_content_sha256 = raw_content_signature(
            manifest, root / "data" / "raw_original"
        )
    reports = []
    with np.load(paths.mfcc_npz, allow_pickle=False) as bundle:
        calls = np.asarray(bundle["stem"]).astype(str)
        positions = pd.Series(calls).groupby(calls, sort=False).cumcount().astype(str)
        keys = "Training|" + pd.Series(calls) + "|" + positions
        mfcc = pd.DataFrame({"record_key": keys, "call_id": calls,
                             "y": np.asarray(bundle["y"])})
        if bundle["X"].shape != (889_358, 40) or bundle["X"].dtype != np.float32:
            raise ValueError("STOP: MFCC40 shape/dtype differs")
    mfcc_meta = json.loads(paths.mfcc_metadata.read_text(encoding="utf-8"))
    reports.append(audit_cache_provenance(
        cache_name="MFCC40", manifest_rows=source, cache_rows=mfcc,
        metadata=mfcc_meta, manifest_metadata=manifest_meta,
        expected_metadata={"cache_schema_version": "mission2_mfcc_mean_std_v1",
                           "source_split": "Training", "source_call_count": 29_200,
                           "utterance_count": 889_358, "feature_dimension": 40,
                           "mfcc_config": dict(MFCC_CONFIG)},
        feature_names=(), label_column="y",
        verified_current_content_sha256=current_content_sha256,
        artifact_path=paths.mfcc_npz, metadata_path=paths.mfcc_metadata,
    ))
    candidate_meta = json.loads(Path(paths.candidate_metadata).read_text(encoding="utf-8"))
    candidate = pd.read_parquet(Path(paths.candidate_parquet))
    candidate_names = ["zcr", "spectral_centroid", "spectral_rolloff",
                       "spectral_flatness", "silence_ratio"]
    reports.append(audit_cache_provenance(
        cache_name="E05 candidate", manifest_rows=source, cache_rows=candidate,
        metadata=candidate_meta, manifest_metadata=manifest_meta,
        expected_metadata={
            "cache_schema_version": "mission2_candidate_audio_features_v1",
            "source_split": "Training", "row_count": 889_358,
            "record_key_signature": record_key_signature(source["record_key"]),
            "feature_names": candidate_names, "feature_count": len(candidate_names),
            "waveform_only": True,
            "extraction_config": {"sample_rate": 8_000, "n_fft": 512,
                                  "hop_length": 160, "silence_threshold_dbfs": -40.0,
                                  "clipping_amplitude": 0.999},
        }, feature_names=candidate_names,
        verified_current_content_sha256=current_content_sha256,
        artifact_path=paths.candidate_parquet, metadata_path=paths.candidate_metadata,
    ))
    specs = (
        ("AF02", acoustic.ROUND1_BLOCK_IDS),
        ("AF04", acoustic.ROUND2_BLOCK_IDS),
        ("AF09", acoustic.NEXT_CANDIDATE_BLOCK_IDS),
        ("AF10/AF11", acoustic.RESIDUAL_TARGET_BLOCK_IDS),
    )
    for name, blocks in specs:
        data_path, metadata_path = acoustic.candidate_cache_paths(
            root / "data/processed/mission2_acoustic_feature_expansion",
            split="Training", block_ids=blocks,
        )
        cache_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(data_path)
        expected = acoustic._expected_cache_metadata(source, split="Training", block_ids=blocks)
        reports.append(audit_cache_provenance(
            cache_name=name, manifest_rows=source, cache_rows=frame,
            metadata=cache_meta, manifest_metadata=manifest_meta,
            expected_metadata=expected, feature_names=expected["feature_names"],
            verified_current_content_sha256=current_content_sha256,
            artifact_path=data_path, metadata_path=metadata_path,
        ))
    return pd.DataFrame(reports)


def _feature_value_differences(
    historical: np.ndarray, clean: np.ndarray, feature_names: Sequence[str],
    *, rtol: float = 1e-7, atol: float = 1e-8, chunk_rows: int = 32_768,
) -> dict[str, Any]:
    """Compare aligned model-input values in bounded row chunks.

    Exact differences drive the counts. A separate float tolerance is only a
    diagnostic; it never changes the exact-equality result.
    """
    old = np.asarray(historical)
    new = np.asarray(clean)
    names = tuple(feature_names)
    if (old.ndim != 2 or old.shape != new.shape or old.shape[0] == 0
            or old.shape[1] != len(names) or not names):
        raise ValueError("feature matrix shape/order differs")
    if chunk_rows < 1 or rtol < 0 or atol < 0:
        raise ValueError("comparison chunk/tolerance must be nonnegative")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("feature matrix contains NaN or inf")
    rows, columns = old.shape
    changed_rows = np.zeros(rows, dtype=bool)
    changed_per_column = np.zeros(columns, dtype=np.int64)
    absolute_sum = np.zeros(columns, dtype=np.float64)
    absolute_max = np.zeros(columns, dtype=np.float64)
    numerical_close = True
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        left = old[start:stop].astype(np.float64, copy=False)
        right = new[start:stop].astype(np.float64, copy=False)
        changed = left != right
        absolute = np.abs(left - right)
        changed_rows[start:stop] = changed.any(axis=1)
        changed_per_column += changed.sum(axis=0)
        absolute_sum += absolute.sum(axis=0)
        absolute_max = np.maximum(absolute_max, absolute.max(axis=0))
        numerical_close &= bool(np.isclose(left, right, rtol=rtol, atol=atol).all())
    changed_columns = sorted(
        (index for index, count in enumerate(changed_per_column) if count),
        key=lambda index: (-absolute_max[index], -changed_per_column[index], names[index]),
    )[:5]
    return {
        "exact_feature_matrix_equality": not bool(changed_rows.any()),
        "numerical_close": numerical_close,
        "differing_row_count": int(changed_rows.sum()),
        "differing_value_count": int(changed_per_column.sum()),
        "mean_abs_diff": float(absolute_sum.sum() / (rows * columns)),
        "max_abs_diff": float(absolute_max.max()),
        "top_changed_features": [
            {"feature": names[index],
             "differing_row_count": int(changed_per_column[index]),
             "mean_abs_diff": float(absolute_sum[index] / rows),
             "max_abs_diff": float(absolute_max[index])}
            for index in changed_columns
        ],
        "_changed_rows": changed_rows,
        "_absolute_sum": float(absolute_sum.sum()),
    }


def _assembled_feature_differences(
    blocks: Sequence[tuple[str, Mapping[str, Any] | None]],
    *, rows: int, feature_names: Sequence[str], clean_feature_names: Sequence[str],
) -> dict[str, Any]:
    """Compute concatenation metrics from ordered blocks without a second 59D matrix."""
    names = tuple(feature_names)
    result: dict[str, Any] = {
        "historical_shape": (rows, len(names)),
        "clean_shape": (rows, len(clean_feature_names)),
        "feature_order_equal": names == tuple(clean_feature_names),
        "historical_dtype": "float32", "clean_dtype": "float32",
        "comparison_dtype": "float32",
    }
    if not result["feature_order_equal"] or any(stats is None for _, stats in blocks):
        result.update(status="INCOMPLETE_HISTORICAL_EVIDENCE",
                      incomplete_blocks=[name for name, stats in blocks if stats is None])
        return result
    changed_rows = np.zeros(rows, dtype=bool)
    differing_values = 0
    absolute_sum = 0.0
    maximum = 0.0
    numerical_close = True
    changed_blocks = []
    for name, stats in blocks:
        assert stats is not None
        changed_rows |= stats["_changed_rows"]
        differing_values += stats["differing_value_count"]
        absolute_sum += stats["_absolute_sum"]
        maximum = max(maximum, stats["max_abs_diff"])
        numerical_close &= stats["numerical_close"]
        if not stats["exact_feature_matrix_equality"]:
            changed_blocks.append(name)
    result.update(
        status="COMPARABLE", exact_feature_matrix_equality=not bool(changed_rows.any()),
        numerical_close=bool(numerical_close),
        differing_row_count=int(changed_rows.sum()),
        differing_value_count=int(differing_values),
        mean_abs_diff=float(absolute_sum / (rows * len(names))),
        max_abs_diff=float(maximum), changed_blocks=changed_blocks,
    )
    return result


def _historical_row_alignment(
    old_keys: Sequence[str], clean_keys: Sequence[str],
    old_calls: Sequence[str], clean_calls: Sequence[str],
    old_labels: np.ndarray | None, clean_labels: np.ndarray,
) -> dict[str, bool | None]:
    historical_keys = np.asarray(old_keys).astype(str)
    current_keys = np.asarray(clean_keys).astype(str)
    return {
        "record_key_exact": (
            len(set(historical_keys)) == len(historical_keys)
            and len(historical_keys) == len(current_keys)
            and set(historical_keys) == set(current_keys)
        ),
        "record_key_order_equal": bool(np.array_equal(historical_keys, current_keys)),
        "call_id_exact": bool(np.array_equal(
            np.asarray(old_calls).astype(str), np.asarray(clean_calls).astype(str)
        )),
        "label_exact": None if old_labels is None else bool(np.array_equal(
            old_labels, clean_labels
        )),
    }


def audit_historical_vs_clean_feature_caches(
    repo_root: str | Path, clean_chain_root: str | Path, *,
    expected_fingerprint: str,
    expected_rows: int = EXPECTED_OFFICIAL_ROWS["Training"],
    expected_calls: int = EXPECTED_OFFICIAL_CALLS["Training"],
    include_reproduction_inputs: bool = False,
) -> dict[str, Any]:
    """Read six frozen historical sources and one strict clean chain exactly once.

    Historical parquet storage order is reported. Values are selected by the
    frozen model feature names and cast to float32 for the 52D/59D comparison,
    matching the original model-input contract. No extractor or fit is called.
    """
    from . import mission2_clean_stage4 as clean_module

    root = Path(repo_root)
    chain_root = Path(clean_chain_root)
    paths = standalone_training_cache_paths(root)
    clean = clean_module.load_clean_stage4_59d(
        chain_root, paths.manifest_parquet, paths.manifest_metadata,
        root / "data/raw_original",
        expected_rows=expected_rows, expected_calls=expected_calls,
    )
    if clean.provenance_fingerprint != expected_fingerprint:
        raise ValueError("clean chain provenance fingerprint differs")
    acoustic_root = root / "data/processed/mission2_acoustic_feature_expansion"
    old_af_paths = {
        "AF02": acoustic.candidate_cache_paths(
            acoustic_root, split="Training", block_ids=acoustic.ROUND1_BLOCK_IDS),
        "AF04": acoustic.candidate_cache_paths(
            acoustic_root, split="Training", block_ids=acoustic.ROUND2_BLOCK_IDS),
        "AF09": acoustic.candidate_cache_paths(
            acoustic_root, split="Training", block_ids=acoustic.NEXT_CANDIDATE_BLOCK_IDS),
        "AF10/AF11": acoustic.candidate_cache_paths(
            acoustic_root, split="Training", block_ids=acoustic.RESIDUAL_TARGET_BLOCK_IDS),
    }
    def af_names(block: str) -> tuple[str, ...]:
        return tuple(acoustic.BLOCK_BY_ID[block].feature_names)
    specs = (
        ("MFCC40", MFCC40_FEATURE_NAMES, paths.mfcc_npz, paths.mfcc_metadata,
         (chain_root / "mfcc40.npz",), (MFCC40_FEATURE_NAMES,)),
        ("E05 candidate", E05_EXTRA_FEATURE_NAMES, Path(paths.candidate_parquet),
         Path(paths.candidate_metadata), (chain_root / "e05_candidate.parquet",),
         (E05_EXTRA_FEATURE_NAMES,)),
        ("AF02", af_names("AF02_PITCH_VOICING"), *old_af_paths["AF02"],
         (chain_root / "af02.parquet",), (af_names("AF02_PITCH_VOICING"),)),
        ("AF04", af_names("AF04_CPP"), *old_af_paths["AF04"],
         (chain_root / "af04.parquet",), (af_names("AF04_CPP"),)),
        ("AF09", af_names("AF09_MULTIBAND_ENVELOPE_COHERENCE"), *old_af_paths["AF09"],
         (chain_root / "af09.parquet",),
         (af_names("AF09_MULTIBAND_ENVELOPE_COHERENCE"),)),
        ("AF10/AF11", (*af_names("AF10_SPEECH_ACTIVITY_BOUNDARY"),
                         *af_names("AF11_SPECTRAL_ENTROPY_DYNAMICS")),
         *old_af_paths["AF10/AF11"],
         (chain_root / "af10.parquet", chain_root / "af11.parquet"),
         (af_names("AF10_SPEECH_ACTIVITY_BOUNDARY"),
          af_names("AF11_SPECTRAL_ENTROPY_DYNAMICS"))),
    )
    expected_names = tuple(name for spec in specs for name in spec[1])
    if expected_names != clean.feature_names or len(expected_names) != 59:
        raise ValueError("clean 59D model feature order differs")

    table_rows: list[dict[str, Any]] = []
    comparable: dict[str, Mapping[str, Any] | None] = {}
    old_model_blocks: dict[str, np.ndarray] = {}
    e05_differences: list[dict[str, Any]] = []
    offset = 0
    for block, names, old_path, old_meta_path, new_paths, new_parts in specs:
        end = offset + len(names)
        row: dict[str, Any] = {
            "block": block, "historical_artifact_path": str(old_path),
            "clean_artifact_path": " | ".join(map(str, new_paths)),
            "historical_row_count": None, "clean_row_count": len(clean.y),
            "historical_feature_dimension": None,
            "clean_feature_dimension": len(names),
            "historical_stored_feature_dimension": None,
            "clean_stored_feature_dimension": None,
            "record_key_exact": None, "record_key_order_equal": None,
            "call_id_exact": None, "label_exact": None,
            "feature_names_order_equal": None,
            "historical_storage_order_equal": None,
            "historical_metadata_order_equal": None,
            "model_feature_order_equal": None,
            "historical_dtype": None, "clean_dtype": None,
            "comparison_dtype": "float32",
            "historical_finite": None, "clean_finite": None,
            "raw_cache_exact_equality": None,
            "exact_feature_matrix_equality": None, "numerical_close": None,
            "differing_row_count": None, "differing_value_count": None,
            "mean_abs_diff": None, "max_abs_diff": None,
            "top_changed_features": [], "status": "INCOMPLETE_HISTORICAL_EVIDENCE",
            "reason": None,
        }
        clean_parts = []
        clean_stored_dimension = 0
        for new_path, part_names in zip(new_paths, new_parts):
            if new_path.suffix == ".npz":
                with np.load(new_path, allow_pickle=False) as bundle:
                    values = bundle["X"]
                clean_stored_dimension += values.shape[1]
            else:
                frame = pd.read_parquet(new_path)
                if (not np.array_equal(frame["record_key"].astype(str), clean.record_keys)
                        or not np.array_equal(frame["call_id"].astype(str), clean.call_ids)):
                    raise ValueError(f"clean {block} key/call contract changed after strict load")
                values = frame.loc[:, list(part_names)].to_numpy()
                clean_stored_dimension += len(set(frame.columns) - {"record_key", "call_id"})
            clean_parts.append(values)
        clean_raw = np.concatenate(clean_parts, axis=1) if len(clean_parts) > 1 else clean_parts[0]
        row["clean_stored_feature_dimension"] = clean_stored_dimension
        row["clean_dtype"] = str(clean_raw.dtype)
        row["clean_finite"] = bool(np.isfinite(clean_raw).all())
        clean_input = np.asarray(clean_raw, dtype=np.float32)
        if not np.array_equal(clean_input, clean.X[:, offset:end]):
            raise ValueError(f"clean {block} values changed after strict load")
        try:
            metadata = json.loads(old_meta_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("historical metadata is not a JSON object")
            if old_path.suffix == ".npz":
                with np.load(old_path, allow_pickle=False) as bundle:
                    old_raw = bundle["X"]
                    old_calls = np.asarray(bundle["stem"]).astype(str)
                    old_labels = np.asarray(bundle["y"])
                old_keys = ("Training|" + pd.Series(old_calls) + "|"
                            + pd.Series(old_calls).groupby(old_calls, sort=False)
                            .cumcount().astype(str)).to_numpy()
                old_stored_dimension = old_raw.shape[1]
                storage_names = tuple(names)
                metadata_names = tuple(names)
            else:
                frame = pd.read_parquet(old_path)
                if not {"record_key", "call_id", *names} <= set(frame.columns):
                    raise ValueError("historical feature/key columns are missing")
                old_raw = frame.loc[:, list(names)].to_numpy()
                old_stored_dimension = len(set(frame.columns) - {
                    "record_key", "call_id", "y", "speaker"
                })
                old_keys = frame["record_key"].astype(str).to_numpy()
                old_calls = frame["call_id"].astype(str).to_numpy()
                if "y" in frame.columns:
                    old_labels = frame["y"].to_numpy()
                elif "speaker" in frame.columns:
                    old_labels = frame["speaker"].to_numpy()
                else:
                    old_labels = None
                storage_names = tuple(name for name in frame.columns if name in names)
                metadata_names = tuple(name for name in metadata.get("feature_names", ())
                                       if name in names)
                if len(metadata_names) != len(names) or set(metadata_names) != set(names):
                    raise ValueError("historical feature metadata lacks selected names")
        except Exception as exc:
            row["reason"] = f"historical artifact unreadable: {type(exc).__name__}: {exc}"
            table_rows.append(row)
            comparable[block] = None
            offset = end
            continue
        try:
            old_finite = bool(np.isfinite(old_raw).all())
            old_input = np.asarray(old_raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            row["reason"] = f"historical feature values are unreadable: {exc}"
            table_rows.append(row)
            comparable[block] = None
            offset = end
            continue
        row.update(
            historical_row_count=len(old_raw),
            historical_feature_dimension=old_raw.shape[1] if old_raw.ndim == 2 else None,
            historical_stored_feature_dimension=old_stored_dimension,
            historical_dtype=str(old_raw.dtype),
            historical_storage_order_equal=storage_names == tuple(names),
            historical_metadata_order_equal=metadata_names == tuple(names),
            model_feature_order_equal=tuple(names) == tuple(clean.feature_names[offset:end]),
            feature_names_order_equal=(storage_names == tuple(names)
                                       and metadata_names == tuple(names)
                                       and tuple(names) == tuple(clean.feature_names[offset:end])),
            historical_finite=old_finite,
            raw_cache_exact_equality=bool(np.array_equal(old_raw, clean_raw)),
        )
        row.update(_historical_row_alignment(
            old_keys, clean.record_keys, old_calls, clean.call_ids, old_labels, clean.y
        ))
        aligned = (old_raw.shape == clean_raw.shape and row["record_key_exact"]
                   and row["record_key_order_equal"] and row["call_id_exact"]
                   and row["label_exact"] is not False
                   and row["model_feature_order_equal"]
                   and row["historical_finite"] and row["clean_finite"])
        if not aligned:
            row["reason"] = "row/key/order/call/label/feature/finite contract differs"
            comparable[block] = None
        else:
            if not np.isfinite(old_input).all():
                row["reason"] = "historical float32 model input is nonfinite"
                comparable[block] = None
            else:
                stats = _feature_value_differences(old_input, clean_input, names)
                row.update({key: value for key, value in stats.items()
                            if not key.startswith("_")})
                row["status"] = "COMPARABLE"
                comparable[block] = stats
                if include_reproduction_inputs:
                    old_model_blocks[block] = old_input
                    if block == "E05 candidate":
                        for row_index, column_index in np.argwhere(old_input != clean_input):
                            e05_differences.append({
                                "row_index": int(row_index),
                                "record_key": str(clean.record_keys[row_index]),
                                "call_id": str(clean.call_ids[row_index]),
                                "label": int(clean.y[row_index]),
                                "feature_name": names[column_index],
                                "historical_value": float(old_input[row_index, column_index]),
                                "clean_value": float(clean_input[row_index, column_index]),
                                "absolute_difference": float(abs(
                                    float(old_input[row_index, column_index])
                                    - float(clean_input[row_index, column_index])
                                )),
                            })
        table_rows.append(row)
        offset = end

    names_52 = tuple(name for spec in specs[:5] for name in spec[1])
    summary_52 = _assembled_feature_differences(
        [(spec[0], comparable[spec[0]]) for spec in specs[:5]],
        rows=len(clean.y), feature_names=names_52,
        clean_feature_names=clean.feature_names[:52],
    )
    summary_59 = _assembled_feature_differences(
        [(spec[0], comparable[spec[0]]) for spec in specs],
        rows=len(clean.y), feature_names=expected_names,
        clean_feature_names=clean.feature_names,
    )
    if any(value is None for value in comparable.values()):
        conclusion = "INCOMPLETE_HISTORICAL_EVIDENCE"
    elif any(row["raw_cache_exact_equality"] is False for row in table_rows) or any(
        not stats["exact_feature_matrix_equality"] for stats in comparable.values()
    ):
        conclusion = "FEATURE_CACHE_DIFFERENCE_FOUND"
    else:
        conclusion = "FEATURE_CACHE_DIFFERENCE_EXCLUDED"
    result = {
        "block_audit": pd.DataFrame(table_rows),
        "assembled_52d": summary_52,
        "assembled_59d": summary_59,
        "conclusion": {
            "status": conclusion,
            "changed_model_input_blocks": [
                name for name, stats in comparable.items()
                if stats is not None and not stats["exact_feature_matrix_equality"]
            ],
            "raw_only_changed_blocks": [
                row["block"] for row in table_rows
                if row["raw_cache_exact_equality"] is False
                and comparable[row["block"]] is not None
                and comparable[row["block"]]["exact_feature_matrix_equality"]
            ],
            "interpretation": (
                "Historical artifact or alignment evidence is incomplete."
                if conclusion == "INCOMPLETE_HISTORICAL_EVIDENCE" else
                "Selected float32 model inputs differ; feature caches are a cause candidate, "
                "but historical prediction/model artifacts are needed for a causal conclusion."
                if any(stats is not None and not stats["exact_feature_matrix_equality"]
                       for stats in comparable.values()) else
                "Raw cache values differ, but selected float32 model inputs are exact; "
                "these value differences do not explain the Accuracy gap."
                if conclusion == "FEATURE_CACHE_DIFFERENCE_FOUND" else
                "Selected feature cache values are exact; they do not explain the Accuracy gap."
            ),
            "clean_provenance_fingerprint": clean.provenance_fingerprint,
            "historical_raw_content_identity": "UNVERIFIABLE",
            "rtol": 1e-7, "atol": 1e-8,
        },
    }
    if include_reproduction_inputs:
        if conclusion != "FEATURE_CACHE_DIFFERENCE_FOUND" and conclusion != "FEATURE_CACHE_DIFFERENCE_EXCLUDED":
            result["reproduction_inputs"] = None
        else:
            order = [spec[0] for spec in specs]
            historical_59d = np.concatenate(
                [old_model_blocks[name] for name in order], axis=1
            )
            if (historical_59d.shape != clean.X.shape
                    or not np.isfinite(historical_59d).all()
                    or tuple(expected_names) != clean.feature_names
                    or not np.array_equal(historical_59d[:, :40], old_model_blocks["MFCC40"])):
                raise ValueError("historical 59D assembly contract differs")
            result["reproduction_inputs"] = {
                "X_52d": historical_59d[:, :52], "X_59d": historical_59d,
                "y": clean.y, "record_keys": clean.record_keys,
                "call_ids": clean.call_ids, "feature_names": clean.feature_names,
                "clean_provenance_fingerprint": clean.provenance_fingerprint,
                "e05_differences": pd.DataFrame(e05_differences),
            }
    return result


def annotate_e05_fixed_split(
    differences: pd.DataFrame, train_idx: np.ndarray, validation_idx: np.ndarray,
    *, total_rows: int,
) -> pd.DataFrame:
    """Label only changed E05 rows using the same complete fixed split as fitting."""
    train = np.asarray(train_idx, dtype=np.int64)
    validation = np.asarray(validation_idx, dtype=np.int64)
    if (len(np.unique(np.concatenate((train, validation)))) != total_rows
            or set(np.concatenate((train, validation)).tolist()) != set(range(total_rows))):
        raise ValueError("fixed split does not partition historical rows")
    if "row_index" not in differences.columns:
        raise ValueError("E05 differences lack row_index")
    output = differences.copy()
    membership = np.full(total_rows, "internal-validation", dtype=object)
    membership[train] = "internal-train"
    indices = output["row_index"].to_numpy(dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= total_rows):
        raise ValueError("E05 difference row_index is out of bounds")
    output["fixed_split_membership"] = membership[indices]
    return output.drop(columns="row_index")


def reproduce_historical_stage4_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Fit exactly old-cache 52D and 59D with the frozen Stage 4 split/config."""
    from .mission2_clean_stage4 import clean_59d_feature_names
    from .mission2_acoustic_feature_expansion import (
        CANONICAL_52D_FEATURE_NAMES, CANONICAL_52D_PREPARED_ID,
        M04_ITER100_FEATURE_SCREENING_CONFIG,
    )

    X52 = np.asarray(inputs["X_52d"])
    X59 = np.asarray(inputs["X_59d"])
    y = np.asarray(inputs["y"], dtype=np.int64)
    keys = np.asarray(inputs["record_keys"]).astype(str)
    calls = np.asarray(inputs["call_ids"]).astype(str)
    names = tuple(inputs["feature_names"])
    n = len(y)
    if (X52.shape != (n, 52) or X59.shape != (n, 59)
            or X52.dtype != np.float32 or X59.dtype != np.float32
            or not np.array_equal(X52, X59[:, :52])
            or not np.isfinite(X59).all()
            or names[:52] != CANONICAL_52D_FEATURE_NAMES
            or names != clean_59d_feature_names()
            or len(keys) != n or len(set(keys)) != n
            or len(calls) != n or not set(np.unique(y)).issubset({0, 1})):
        raise ValueError("historical 52D/59D row, feature, label, or finite contract differs")
    expected_m04 = {**dict(M04_MLP_CONFIG), "max_iter": 100}
    if dict(M04_ITER100_FEATURE_SCREENING_CONFIG) != expected_m04:
        raise ValueError("historical M04-100 config differs")
    train_idx, validation_idx, split_audit = create_fixed_internal_split(
        calls, split_function=standalone_group_split_indices,
        expected_contract=(EXPECTED_INTERNAL_SPLIT
                           if n == EXPECTED_OFFICIAL_ROWS["Training"] else None),
    )
    if split_audit["seed"] != 42 or split_audit["call_overlap"] != 0:
        raise ValueError("historical fixed call_id split differs")
    prepared = [
        acoustic._prepare_representation(
            representation_id=representation_id, X=matrix, y=y,
            record_keys=keys, feature_names=feature_names,
            train_idx=train_idx, validation_idx=validation_idx,
            split_audit=split_audit,
        )
        for representation_id, matrix, feature_names in (
            (CANONICAL_52D_PREPARED_ID, X52, names[:52]),
            ("AF10_AF11_COMBINATION", X59, names),
        )
    ]
    # Both scalers and split contracts are checked before either model fit.
    for item in prepared:
        if (not np.array_equal(item.train_idx, train_idx)
                or not np.array_equal(item.validation_idx, validation_idx)
                or int(item.scaler.n_samples_seen_) != len(train_idx)
                or not np.array_equal(item.y_evaluation, y[validation_idx])
                or not np.array_equal(item.evaluation_record_keys, keys[validation_idx])):
            raise ValueError("historical train-only scaler or evaluation contract differs")
    differences = annotate_e05_fixed_split(
        inputs["e05_differences"], train_idx, validation_idx, total_rows=n,
    )
    results = [
        acoustic._run_prepared_representation(item, experiment_id=experiment_id,
                                              allow_fit=True)
        for item, experiment_id in zip(
            prepared, ("AF09_52D_HISTORICAL_CACHE", "AF10_AF11_59D_HISTORICAL_CACHE")
        )
    ]
    for result, item in zip(results, prepared):
        if (dict(result.model_config) != expected_m04
                or result.fit_audit["scaler_fit_scope"] != "internal_train_only"
                or result.fit_audit["seed"] != 42
                or not np.array_equal(result.fit_audit["train_idx"], train_idx)
                or not np.array_equal(result.fit_audit["validation_idx"], validation_idx)
                or not np.array_equal(result.y_true, y[validation_idx])
                or tuple(result.feature_names) != tuple(item.feature_names)):
            raise ValueError("historical M04-100 fit contract differs")
    return {"results": results, "e05_differences": differences,
            "split_audit": split_audit,
            "clean_provenance_fingerprint": inputs["clean_provenance_fingerprint"]}


def raw_content_signature(manifest: pd.DataFrame, raw_root: str | Path) -> str:
    """Hash current WAV and JSON bytes in path order for a future build-time proof."""
    required = {"audio_path", "json_path"}
    if not required.issubset(manifest.columns):
        raise ValueError("raw content proof requires WAV and JSON paths")
    root = Path(raw_root).resolve()
    relative_paths = set()
    for column in ("audio_path", "json_path"):
        if manifest[column].isna().any():
            raise ValueError(f"raw content proof has null {column}")
        relative_paths.update(manifest[column].astype(str))
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("raw content path escapes source root") from exc
        content = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.digest())
    return digest.hexdigest()


def find_historical_manifest_metadata(metadata_roots: Sequence[str | Path],
                                      old_source_signature: str) -> list[Path]:
    """Search only JSON metadata filenames; leave all artifacts untouched."""
    matches = []
    for root in metadata_roots:
        for path in Path(root).rglob("*manifest*.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("source_signature") == old_source_signature:
                matches.append(path)
    return matches


def compare_manifest_inventories(old: Mapping[str, Any],
                                 current: Mapping[str, Any]) -> dict[str, Any]:
    """Compare saved path/size/mtime, retaining an explicit proof boundary."""
    old_files = old.get("source_inventory")
    current_files = current.get("source_inventory")
    if not isinstance(old_files, list) or not isinstance(current_files, list):
        return {"status": PROVENANCE_UNVERIFIABLE,
                "reason": "one or both raw file inventories are unavailable"}
    def by_path(files: list[Any]) -> dict[str, Any]:
        indexed = {str(item["path"]): item for item in files}
        if len(indexed) != len(files):
            raise ValueError("raw file inventory contains duplicate paths")
        return indexed
    before, after = by_path(old_files), by_path(current_files)
    paths = sorted(set(before) | set(after))
    differences = []
    for path in paths:
        a, b = before.get(path), after.get(path)
        differences.append({
            "path": path,
            "size_before": None if a is None else a.get("size_bytes"),
            "size_after": None if b is None else b.get("size_bytes"),
            "mtime_before": None if a is None else a.get("mtime_ns"),
            "mtime_after": None if b is None else b.get("mtime_ns"),
            "sha256_before": None if a is None else a.get("sha256"),
            "sha256_after": None if b is None else b.get("sha256"),
        })
    changed = [row for row in differences if (
        row["size_before"] != row["size_after"] or
        row["mtime_before"] != row["mtime_after"] or
        row["sha256_before"] != row["sha256_after"]
    )]
    hashes_complete = all(before.get(path, {}).get("sha256") and after.get(path, {}).get("sha256")
                          for path in paths)
    content_equal = bool(paths) and hashes_complete and all(
        before[path]["sha256"] == after[path]["sha256"] for path in paths
    )
    return {"status": "INVENTORY_CONTENT_MATCH" if content_equal else PROVENANCE_UNVERIFIABLE,
            "differences": changed, "content_hashes_complete": hashes_complete}


def minimum_59d_rebuild_plan(audit: pd.DataFrame) -> pd.DataFrame:
    """Plan only the affected independent cache and its in-memory descendants."""
    descendants = {
        "MFCC40": "MFCC40 → E05 42D → canonical 52D → 59D",
        "E05 candidate": "E05 candidate → E05 42D → canonical 52D → 59D",
        "AF02": "AF02 → canonical 52D → 59D",
        "AF04": "AF04 → canonical 52D → 59D",
        "AF09": "AF09 → canonical 52D → 59D",
        "AF10/AF11": "AF10/AF11 → 59D",
    }
    rebuild_notes = {
        "MFCC40": "run Stage 4 B2 clean versioned chain builder",
        "E05 candidate": "run Stage 4 B2 with current v2 candidate semantics",
        "AF02": "run Stage 4 B2 with the unchanged AF02 block",
        "AF04": "run Stage 4 B2 with the unchanged AF04 block",
        "AF09": "run Stage 4 B2 with the unchanged AF09 block",
        "AF10/AF11": "run Stage 4 B2 with the unchanged AF10/AF11 blocks",
    }
    rows = []
    for row in audit.itertuples(index=False):
        if row.status != PROVENANCE_EXACT:
            action = (
                "raw identity proved; retain cache, but current strict loader still blocks this signature"
                if row.status == PROVENANCE_FILESYSTEM_ONLY else rebuild_notes[row.cache]
            )
            rows.append({"cache": row.cache, "status": row.status,
                         "dependent_rebuild": descendants[row.cache],
                         "action": action,
                         "preserve_existing_artifact": True})
    return pd.DataFrame(rows)


def audit_historical_af09_divergence(historical: Mapping[str, Any],
                                     current: Mapping[str, Any]) -> dict[str, Any]:
    """Compare supplied artifacts only; never reconstruct historical scores."""
    fields = ("source_provenance", "record_keys", "labels", "X_52d",
              "scaler_input", "train_idx", "validation_idx", "m04_config",
              "predictions", "confusion_matrix", "accuracy")
    missing = [name for name in fields if name not in historical or name not in current]
    if missing:
        return {"status": "UNRESOLVED_DUE_TO_MISSING_ARTIFACT", "missing": missing}
    equal = {}
    for name in fields:
        old, new = historical[name], current[name]
        equal[name] = bool(np.array_equal(np.asarray(old), np.asarray(new))) if name != "m04_config" else old == new
    if equal["X_52d"]:
        status = "SOURCE_SIGNATURE_EXCLUDED_FROM_AF09_DIVERGENCE"
    else:
        status = "FEATURE_DIFFERENCE_OBSERVED_CAUSE_UNRESOLVED"
    return {"status": status, "equal": equal}


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
              source_provenance: str,
              padded_rows: Sequence[tuple[str, int]] | None = None) -> dict[str, Any]:
    metadata = {
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
    if padded_rows is not None:
        positions = {key: index for index, key in enumerate(keys)}
        padded = list(padded_rows)
        if (len({key for key, _ in padded}) != len(padded)
                or any(key not in positions or type(length) is not int
                       or not 0 < length < 800 for key, length in padded)
                or padded != sorted(padded, key=lambda item: positions[item[0]])):
            raise ValueError("eGeMAPSv02 padded-row provenance is invalid or unordered")
        metadata.update({
            "short_audio_fallback": "zero_pad_to_800_on_nonfinite_short_segment",
            "padding_target_samples": 800,
            "sample_rate": 8_000,
            "padded_row_count": len(padded),
            "padded_record_keys": [key for key, _ in padded],
            "padded_original_sample_lengths": [length for _, length in padded],
        })
    return metadata


def _padded_rows_from_metadata(metadata: Mapping[str, Any]
                               ) -> list[tuple[str, int]] | None:
    fields = {
        "short_audio_fallback", "padding_target_samples", "padded_row_count",
        "padded_record_keys", "padded_original_sample_lengths",
    }
    if not fields.intersection(metadata):
        return None
    if not fields.issubset(metadata):
        raise ValueError("eGeMAPSv02 padded-row provenance is incomplete")
    keys = metadata["padded_record_keys"]
    lengths = metadata["padded_original_sample_lengths"]
    if (metadata["short_audio_fallback"] != "zero_pad_to_800_on_nonfinite_short_segment"
            or metadata["padding_target_samples"] != 800
            or not isinstance(keys, list) or not isinstance(lengths, list)
            or len(keys) != len(lengths)
            or type(metadata["padded_row_count"]) is not int
            or metadata["padded_row_count"] != len(keys)):
        raise ValueError("eGeMAPSv02 padded-row provenance contract mismatch")
    return list(zip(keys, lengths))


def _validate_colab_mismatch_approval(
    colab_resume: Mapping[str, Any], feature_names: tuple[str, ...],
    source_provenance: str,
) -> None:
    identity = colab_resume.get("source_identity", {})
    overlap = colab_resume.get("overlap_audit", {})
    approval = colab_resume.get("resume_approval", {})
    environment = colab_resume.get("environment", {})
    history = colab_resume.get("resume_history", [])
    if not all(isinstance(value, Mapping)
               for value in (identity, overlap, approval, environment)):
        raise ValueError("Colab resume mismatch approval provenance is invalid")
    fingerprint = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
    overlap_fields = (
        "sampled_rows", "sampled_calls", "padded_sampled", "exact_row_count",
        "numerically_close_row_count", "differing_row_count", "differing_value_count",
        "mean_abs_diff", "max_abs_diff",
    )
    valid_overlap = (
        overlap.get("status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
        and overlap.get("sampled_rows") == 100
        and overlap.get("sampled_calls") == 100
        and overlap.get("padded_sampled") is True
        and all(approval.get(key) == overlap.get(key) for key in overlap_fields)
        and isinstance(overlap.get("differing_value_count"), (int, np.integer))
        and not isinstance(overlap.get("differing_value_count"), bool)
        and overlap.get("differing_value_count", 121) <= 120
        and all(isinstance(overlap.get(key), (int, float, np.integer, np.floating))
                and np.isfinite(overlap.get(key)) and overlap.get(key) >= 0
                for key in ("mean_abs_diff", "max_abs_diff"))
        and overlap.get("mean_abs_diff", float("inf")) <= 2e-8
        and overlap.get("max_abs_diff", float("inf")) <= 3e-5
    )
    valid_identity = (
        identity.get("status") == "PASS"
        and identity.get("source_split") == "Training"
        and identity.get("raw_file_count") == 58_400
        and identity.get("rows") == 889_358
        and identity.get("clean_reference_fingerprint") == fingerprint
        and bool(identity.get("raw_content_sha256"))
        and source_provenance == fingerprint
    )
    runtime = environment.get("packages", {})
    valid_runtime = (
        runtime.get("opensmile") == "2.6.0"
        and approval.get("opensmile_version") == "2.6.0"
        and approval.get("runtime_provenance") == environment
        and approval.get("feature_dim") == 88
        and approval.get("feature_names") == list(feature_names)
        and len(feature_names) == 88
        and feature_names == get_egemaps_feature_names()
    )
    valid_approval = (
        approval.get("status") == "APPROVED"
        and approval.get("approval_type") == "APPROVE_CROSS_PLATFORM_MISMATCH"
        and approval.get("overlap_status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
        and approval.get("raw_file_count") == identity.get("raw_file_count")
        and approval.get("source_split") == identity.get("source_split")
        and approval.get("source_rows") == identity.get("rows")
        and approval.get("clean_reference_fingerprint") == fingerprint
        and approval.get("raw_content_sha256") == identity.get("raw_content_sha256")
        and approval.get("starting_completed_rows") == 470_255
        and approval.get("starting_remaining_rows") == 419_103
        and approval.get("fingerprint") == fingerprint
    )
    recorded = isinstance(history, list) and any(
        isinstance(event, Mapping)
        and event.get("resume_approval") == approval
        and event.get("overlap_status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
        and event.get("starting_completed_rows") == 470_255
        and event.get("starting_remaining_rows") == 419_103
        and event.get("raw_content_sha256") == identity.get("raw_content_sha256")
        for event in history
    )
    if not (valid_overlap and valid_identity and valid_runtime and valid_approval and recorded):
        raise ValueError("Colab resume mismatch approval provenance is invalid")


def make_egemaps_data(
    X: np.ndarray, record_keys: Sequence[str], call_ids: Sequence[str],
    feature_names: Sequence[str], *, source_provenance: str,
    padded_rows: Sequence[tuple[str, int]] | None = None,
    colab_resume: Mapping[str, Any] | None = None,
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
    metadata = _metadata(names, keys, calls, source_provenance, padded_rows)
    if colab_resume is not None:
        required = {"environment", "source_identity", "overlap_audit", "resume_history"}
        if not isinstance(colab_resume, Mapping) or not required.issubset(colab_resume):
            raise ValueError("Colab resume provenance is incomplete")
        overlap_status = colab_resume["overlap_audit"].get("status")
        if overlap_status == "CROSS_PLATFORM_FEATURE_MISMATCH":
            _validate_colab_mismatch_approval(colab_resume, names, source_provenance)
        elif overlap_status not in (
            "CROSS_PLATFORM_EXACT_MATCH", "CROSS_PLATFORM_NUMERICALLY_CLOSE",
        ):
            raise ValueError("Colab resume overlap audit is not approved")
        metadata["colab_resume"] = dict(colab_resume)
    return EgemapsData(matrix, keys, calls, names, metadata)


def build_egemaps_cache(
    path: str | Path, X: np.ndarray, record_keys: Sequence[str],
    call_ids: Sequence[str], feature_names: Sequence[str], *, source_provenance: str,
    padded_rows: Sequence[tuple[str, int]] | None = None,
    colab_resume: Mapping[str, Any] | None = None,
) -> EgemapsData:
    """Persist one fully validated cache; never extract implicitly."""

    data = make_egemaps_data(
        X, record_keys, call_ids, feature_names, source_provenance=source_provenance,
        padded_rows=padded_rows, colab_resume=colab_resume,
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
    padded_rows = _padded_rows_from_metadata(metadata)
    if not np.array_equal(keys, np.asarray(expected_record_keys)):
        raise ValueError("eGeMAPSv02 record_key order mismatch")
    if not np.array_equal(calls, np.asarray(expected_call_ids)):
        raise ValueError("eGeMAPSv02 call_id order mismatch")
    data = make_egemaps_data(
        matrix, keys, calls, names, source_provenance=source_provenance,
        padded_rows=padded_rows, colab_resume=metadata.get("colab_resume"),
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
    if egemaps.metadata != _metadata(
        egemaps.feature_names, egemaps.record_keys, egemaps.call_ids,
        egemaps.metadata["source_provenance"],
        _padded_rows_from_metadata(egemaps.metadata),
    ):
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
