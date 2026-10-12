"""Explicit, versioned Training cache chain for the Mission 2 Stage 4 reference.

Building reads raw WAVs and writes only a new directory. Importing this module
does neither. Historical v1/v2 loaders and artifacts remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import librosa
import numpy as np
import pandas as pd

from .audio import read_wav_frames
from .mission2 import milliseconds_to_frame
from . import mission2_acoustic_feature_expansion as acoustic
from . import mission2_candidate_features as candidate
from .mission2_model_experiments import (
    E05_EXTRA_FEATURE_NAMES, E05_FEATURE_NAMES,
    EXPECTED_INTERNAL_SPLIT, EXPECTED_OFFICIAL_CALLS,
    EXPECTED_OFFICIAL_ROWS, MFCC40_FEATURE_NAMES, MFCC_CONFIG,
    create_fixed_internal_split, record_key_signature,
    standalone_group_split_indices,
)
from .mission2_standard_representation import raw_content_signature


CLEAN_CHAIN_SCHEMA = "mission2_stage4_clean_chain_v1"
CLEAN_CACHE_SCHEMA = "mission2_stage4_clean_artifact_v1"
BUILDER_CONTRACT_VERSION = "stage4-clean-build-1"
E05_CURRENT_NAMES = E05_EXTRA_FEATURE_NAMES
AF_BLOCKS: dict[str, tuple[str, ...]] = {
    "af02": ("AF02_PITCH_VOICING",),
    "af04": ("AF04_CPP",),
    "af09": ("AF09_MULTIBAND_ENVELOPE_COHERENCE",),
    "af10": ("AF10_SPEECH_ACTIVITY_BOUNDARY",),
    "af11": ("AF11_SPECTRAL_ENTROPY_DYNAMICS",),
}
ARTIFACT_SUFFIX = {"mfcc40": ".npz", "e05_candidate": ".parquet",
                   **{name: ".parquet" for name in AF_BLOCKS}}


@dataclass(frozen=True)
class Clean59D:
    X: np.ndarray
    y: np.ndarray
    record_keys: np.ndarray
    call_ids: np.ndarray
    feature_names: tuple[str, ...]
    provenance_fingerprint: str
    chain_root: Path
    raw_source_verification: Mapping[str, Any] | None = None


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)


def _source_rows(manifest: pd.DataFrame, *, expected_rows: int,
                 expected_calls: int) -> pd.DataFrame:
    fields = ("split", "record_key", "call_id", "annotation_row", "speaker",
              "audio_path", "json_path", "crop_start_frame", "crop_end_frame",
              "crop_ok", "startAt_ms", "endAt_ms")
    missing = sorted(set(fields) - set(manifest.columns))
    if missing:
        raise ValueError(f"clean source manifest lacks columns: {missing}")
    source = manifest.loc[manifest["split"].astype(str) == "Training", list(fields)].reset_index(drop=True)
    if len(source) != expected_rows or source["call_id"].nunique() != expected_calls:
        raise ValueError("clean Training row/call count differs")
    if source["record_key"].isna().any() or not source["record_key"].is_unique:
        raise ValueError("clean Training record_key is null or duplicated")
    if source[["call_id", "audio_path", "json_path", "speaker", "startAt_ms", "endAt_ms",
               "crop_start_frame", "crop_end_frame", "crop_ok"]].isna().any().any():
        raise ValueError("clean Training source identity/tracking contains null")
    if source[["call_id", "audio_path", "json_path"]].astype(str).eq("").any().any():
        raise ValueError("clean Training source identity contains an empty value")
    if not source["crop_ok"].eq(True).all():
        raise ValueError("clean Training contains an invalid crop")
    if not set(source["speaker"].unique()).issubset({0, 1}):
        raise ValueError("clean Training labels are not binary")
    expected_keys = (
        "Training|" + source["call_id"].astype(str) + "|"
        + source["annotation_row"].astype(int).astype(str)
    )
    if not source["record_key"].astype(str).reset_index(drop=True).equals(expected_keys):
        raise ValueError("clean Training record_key contract differs")
    for row in source.itertuples(index=False):
        start = milliseconds_to_frame(float(row.startAt_ms), MFCC_CONFIG["sample_rate"])
        end = milliseconds_to_frame(float(row.endAt_ms), MFCC_CONFIG["sample_rate"])
        if start != int(row.crop_start_frame) or end != int(row.crop_end_frame):
            raise ValueError(f"baseline/manifest crop boundary differs: {row.record_key}")
    return source


def _row_signature(source: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in source.itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _order_signature(keys: pd.Series) -> str:
    digest = hashlib.sha256()
    for key in keys.astype(str):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _raw_inventory_signature(source: pd.DataFrame, raw_root: Path) -> str:
    """Record path/size/mtime separately from stable row identity."""
    root = raw_root.resolve()
    paths = set(source["audio_path"].astype(str)) | set(source["json_path"].astype(str))
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("raw source path escapes the Training root") from exc
        stat = path.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _extract_mfcc40(source: pd.DataFrame, raw_root: Path) -> dict[str, np.ndarray]:
    """Match the frozen baseline's crop, librosa call, mean/std and float32 order."""
    X = np.empty((len(source), 40), dtype=np.float32)
    grouped = source.groupby(["call_id", "audio_path"], sort=False)
    for (_call_id, audio_path), group in grouped:
        audio = read_wav_frames(raw_root / str(audio_path))
        if audio.sample_rate != MFCC_CONFIG["sample_rate"]:
            raise ValueError(f"unexpected sample rate: {audio_path}")
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        for row in group.itertuples():
            start, end = int(row.crop_start_frame), int(row.crop_end_frame)
            if start < 0 or end <= start or end > mono.shape[0]:
                raise ValueError(f"invalid MFCC crop: {row.record_key}")
            mfcc = librosa.feature.mfcc(
                y=mono[start:end], sr=audio.sample_rate,
                n_mfcc=MFCC_CONFIG["n_mfcc"], n_fft=MFCC_CONFIG["n_fft"],
                hop_length=MFCC_CONFIG["hop_length"],
            )
            vector = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float32)
            if vector.shape != (40,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid MFCC40 vector: {row.record_key}")
            X[row.Index] = vector
    y = source["speaker"].to_numpy(dtype=np.int64)
    start_at = source["startAt_ms"].to_numpy(dtype=np.float64)
    end_at = source["endAt_ms"].to_numpy(dtype=np.float64)
    return {"X": X, "y": y,
            "stem": np.asarray(source["call_id"].astype(str).tolist(), dtype=str),
            "start_at": start_at, "end_at": end_at, "duration": end_at - start_at}


def _artifact_contract(name: str, source: pd.DataFrame,
                       inventory_signature: str) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    if name == "mfcc40":
        return MFCC40_FEATURE_NAMES, {"mfcc_config": dict(MFCC_CONFIG),
                                      "aggregation": "time_axis_mean_then_std",
                                      "dtype": "float32"}
    if name == "e05_candidate":
        return E05_CURRENT_NAMES, candidate.expected_candidate_cache_metadata(
            source, split="Training", feature_names=E05_CURRENT_NAMES,
            source_signature=inventory_signature,
        )
    blocks = AF_BLOCKS[name]
    expected = acoustic._expected_cache_metadata(source, split="Training", block_ids=blocks)
    return tuple(expected["feature_names"]), expected


def _artifact_metadata(name: str, source: pd.DataFrame, base: Mapping[str, Any],
                       inventory_signature: str, artifact_hash: str,
                       parent: Mapping[str, Any], *,
                       dependency_versions: Mapping[str, str] | None = None) -> dict[str, Any]:
    names, extractor = _artifact_contract(name, source, inventory_signature)
    return {
        "cache_schema_version": CLEAN_CACHE_SCHEMA, "artifact_id": name,
        "builder_contract_version": BUILDER_CONTRACT_VERSION,
        "dependency_versions": dict(dependency_versions) if dependency_versions is not None else {
            "librosa": librosa.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "source_manifest_schema_version": base["source_manifest_schema_version"],
        "manifest_recorded_source_signature": base["manifest_recorded_source_signature"],
        "source_manifest_sha256": base["source_manifest_sha256"],
        "source_manifest_metadata_sha256": base["source_manifest_metadata_sha256"],
        "source_inventory_signature": inventory_signature,
        "source_content_sha256": base["source_content_sha256"],
        "source_content_proof": base["source_content_proof"],
        "source_row_signature": base["source_row_signature"],
        "record_key_signature": base["record_key_signature"],
        "record_key_order_signature": base["record_key_order_signature"],
        "source_split": "Training", "row_count": len(source),
        "feature_names": list(names), "feature_dimension": len(names),
        "config_fingerprint": _digest_json(extractor),
        "extractor_contract": extractor, "parent_provenance": dict(parent),
        "artifact_sha256": artifact_hash,
    }


def _derived_nodes(hashes: Mapping[str, str]) -> dict[str, Any]:
    """Keep extraction parents distinct from in-memory assembly parents."""
    e05 = {"parent_hashes": {"mfcc40": hashes["mfcc40"],
                             "e05_candidate": hashes["e05_candidate"]},
           "feature_names": list(E05_FEATURE_NAMES)}
    e05["fingerprint"] = _digest_json(e05)
    names_52 = clean_59d_feature_names()[:52]
    base_52 = {"parent_hashes": {"e05_42d": e05["fingerprint"],
                                 "af02": hashes["af02"], "af04": hashes["af04"],
                                 "af09": hashes["af09"]},
               "feature_names": list(names_52)}
    base_52["fingerprint"] = _digest_json(base_52)
    combined = {"parent_hashes": {"af09_52d": base_52["fingerprint"],
                                  "af10": hashes["af10"], "af11": hashes["af11"]},
                "feature_names": list(clean_59d_feature_names())}
    combined["fingerprint"] = _digest_json(combined)
    return {"e05_42d": e05, "af09_52d": base_52, "af10_af11_59d": combined}


def build_clean_stage4_chain(
    manifest_path: str | Path, manifest_metadata_path: str | Path,
    raw_root: str | Path, output_root: str | Path, *,
    expected_rows: int = EXPECTED_OFFICIAL_ROWS["Training"],
    expected_calls: int = EXPECTED_OFFICIAL_CALLS["Training"],
    source_content_sha256: str | None = None,
) -> Path:
    """Deliberate full extraction; publish only a complete new versioned chain."""
    manifest_path = Path(manifest_path)
    manifest_metadata_path = Path(manifest_metadata_path)
    raw_root = Path(raw_root)
    metadata = json.loads(manifest_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "mission2_raw_eda_v1":
        raise ValueError("clean source manifest schema differs")
    source = _source_rows(pd.read_parquet(manifest_path),
                          expected_rows=expected_rows, expected_calls=expected_calls)
    inventory = _raw_inventory_signature(source, raw_root)
    if source_content_sha256 is not None:
        if raw_content_signature(source, raw_root) != source_content_sha256:
            raise ValueError("optional raw content digest differs before extraction")
    base = {
        "source_manifest_schema_version": metadata["schema_version"],
        "manifest_recorded_source_signature": metadata.get("source_signature"),
        "source_manifest_sha256": _file_sha256(manifest_path),
        "source_manifest_metadata_sha256": _file_sha256(manifest_metadata_path),
        "source_row_signature": _row_signature(source),
        "source_content_sha256": source_content_sha256,
        "source_content_proof": (
            "verified_before_and_after_build" if source_content_sha256 is not None
            else "not_collected"
        ),
        "record_key_signature": record_key_signature(source["record_key"]),
        "record_key_order_signature": _order_signature(source["record_key"]),
    }
    version = _digest_json({"builder": BUILDER_CONTRACT_VERSION, "inventory": inventory,
                            "source": base})[:16]
    output_root = Path(output_root)
    final = output_root / f"stage4_clean_v1_{version}"
    if final.exists():
        raise FileExistsError(f"clean chain already exists; no overwrite: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{final.name}.building-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    parent = {**base, "source_inventory_signature": inventory}
    hashes: dict[str, str] = {}
    for name in ARTIFACT_SUFFIX:
        data_path = staging / f"{name}{ARTIFACT_SUFFIX[name]}"
        if name == "mfcc40":
            with data_path.open("xb") as stream:
                np.savez_compressed(stream, **_extract_mfcc40(source, raw_root))
        elif name == "e05_candidate":
            frame = candidate.extract_candidate_feature_frame(
                source, data_root=raw_root, split="Training", feature_names=E05_CURRENT_NAMES,
            )
            frame.to_parquet(data_path, index=False)
        else:
            frame = acoustic.extract_candidate_feature_frame(
                source, data_root=raw_root, split="Training", block_ids=AF_BLOCKS[name],
            )
            frame.to_parquet(data_path, index=False)
        hashes[name] = _file_sha256(data_path)
        artifact_meta = _artifact_metadata(
            name, source, base, inventory, hashes[name], parent
        )
        _write_json_new(staging / f"{name}.json", artifact_meta)
    if _raw_inventory_signature(source, raw_root) != inventory:
        raise RuntimeError(f"raw source inventory changed during extraction; inspect {staging}")
    if source_content_sha256 is not None and raw_content_signature(source, raw_root) != source_content_sha256:
        raise RuntimeError(f"raw content changed during extraction; inspect {staging}")
    if _file_sha256(manifest_path) != base["source_manifest_sha256"]:
        raise RuntimeError(f"source manifest changed during extraction; inspect {staging}")
    if _file_sha256(manifest_metadata_path) != base["source_manifest_metadata_sha256"]:
        raise RuntimeError(f"source manifest metadata changed during extraction; inspect {staging}")
    chain = {"schema_version": CLEAN_CHAIN_SCHEMA,
             "builder_contract_version": BUILDER_CONTRACT_VERSION,
             **base, "source_inventory_signature": inventory,
             "source_split": "Training", "row_count": len(source),
             "call_count": expected_calls, "artifact_hashes": hashes,
             "derived_nodes": _derived_nodes(hashes),
             "assembly_order": list(clean_59d_feature_names())}
    chain["provenance_fingerprint"] = _digest_json(chain)
    _write_json_new(staging / "chain.json", chain)
    # Publish only after the same strict contracts used by B3 accept every parent.
    load_clean_stage4_59d(
        staging, manifest_path, manifest_metadata_path, raw_root,
        expected_rows=expected_rows, expected_calls=expected_calls,
    )
    if final.exists():
        raise FileExistsError(f"clean chain appeared during build; inspect {staging}")
    staging.rename(final)
    return final


def load_or_build_clean_stage4_chain(
    manifest_path: str | Path, manifest_metadata_path: str | Path,
    raw_root: str | Path, output_root: str | Path, *,
    expected_rows: int = EXPECTED_OFFICIAL_ROWS["Training"],
    expected_calls: int = EXPECTED_OFFICIAL_CALLS["Training"],
) -> tuple[Clean59D, str]:
    """Reuse one completed chain after strict validation; build only if none exists.

    An ambiguous, incomplete, or stale chain requires inspection. Never replace it.
    """
    output_root = Path(output_root)
    existing = sorted(output_root.glob("stage4_clean_v1_*"))
    if len(existing) > 1:
        raise RuntimeError(f"STOP: multiple clean chains need explicit selection: {existing}")
    if existing:
        loaded = load_clean_stage4_59d(
            existing[0], manifest_path, manifest_metadata_path, raw_root,
            expected_rows=expected_rows, expected_calls=expected_calls,
        )
        return loaded, "REUSING_EXISTING_CLEAN_CHAIN"
    # An interrupted extraction is evidence to inspect, not permission to retry.
    if any(output_root.glob(".stage4_clean_v1_*.building-*")):
        raise RuntimeError("STOP: incomplete clean chain staging directory exists")
    built = build_clean_stage4_chain(
        manifest_path, manifest_metadata_path, raw_root, output_root,
        expected_rows=expected_rows, expected_calls=expected_calls,
    )
    loaded = load_clean_stage4_59d(
        built, manifest_path, manifest_metadata_path, raw_root,
        expected_rows=expected_rows, expected_calls=expected_calls,
    )
    return loaded, "BUILT_NEW_CLEAN_CHAIN"


def clean_59d_feature_names() -> tuple[str, ...]:
    return (
        *E05_FEATURE_NAMES,
        *acoustic.BLOCK_BY_ID["AF02_PITCH_VOICING"].feature_names,
        *acoustic.BLOCK_BY_ID["AF04_CPP"].feature_names,
        *acoustic.BLOCK_BY_ID["AF09_MULTIBAND_ENVELOPE_COHERENCE"].feature_names,
        *acoustic.BLOCK_BY_ID["AF10_SPEECH_ACTIVITY_BOUNDARY"].feature_names,
        *acoustic.BLOCK_BY_ID["AF11_SPECTRAL_ENTROPY_DYNAMICS"].feature_names,
    )


def load_clean_stage4_59d(
    chain_root: str | Path, manifest_path: str | Path,
    manifest_metadata_path: str | Path, raw_root: str | Path, *,
    expected_rows: int = EXPECTED_OFFICIAL_ROWS["Training"],
    expected_calls: int = EXPECTED_OFFICIAL_CALLS["Training"],
    transfer_identity_path: str | Path | None = None,
    fixed_validation_record_keys: Sequence[str] | None = None,
) -> Clean59D:
    """Load the frozen chain using filesystem identity or explicit transfer proof.

    The default retains the original filesystem/version checks. Supplying a
    transfer identity requires all raw WAV/JSON bytes to match that prior proof;
    recorded builder mtimes/versions remain historical metadata. This does not
    retroactively prove the bytes used at build time when that proof is absent.
    """
    chain_root = Path(chain_root)
    chain = json.loads((chain_root / "chain.json").read_text(encoding="utf-8"))
    fingerprint = chain.pop("provenance_fingerprint", None)
    if chain.get("schema_version") != CLEAN_CHAIN_SCHEMA or fingerprint != _digest_json(chain):
        raise ValueError("clean chain schema/fingerprint differs")
    if chain.get("builder_contract_version") != BUILDER_CONTRACT_VERSION:
        raise ValueError("clean builder contract differs")
    manifest_path = Path(manifest_path)
    manifest_metadata_path = Path(manifest_metadata_path)
    manifest_meta = json.loads(manifest_metadata_path.read_text(encoding="utf-8"))
    # The feature-grid caller supplies the already persisted Stage 6 keys. The
    # loader then validates against those keys without creating/re-splitting
    # the rows; its default behavior remains unchanged for existing callers.
    if fixed_validation_record_keys is None:
        source_manifest = pd.read_parquet(manifest_path)
    else:
        source_manifest = pd.read_parquet(
            manifest_path,
            filters=[("split", "==", "Training")],
        )
    source = _source_rows(source_manifest,
                          expected_rows=expected_rows, expected_calls=expected_calls)
    raw_verification = None
    cache_versions = None
    if transfer_identity_path is None:
        inventory = _raw_inventory_signature(source, Path(raw_root))
    else:
        from .mission2_egemaps_colab import verify_transfer_identity

        verified = verify_transfer_identity(source, manifest_path, manifest_metadata_path,
                                            raw_root, transfer_identity_path)
        if verified["clean_reference_fingerprint"] != fingerprint:
            raise ValueError("clean chain and transfer identity fingerprints differ")
        # Staging creates new mtimes. Preserve the builder's inventory in cache
        # metadata, and require the independent byte proof for the current raw.
        inventory = chain["source_inventory_signature"]
        cache_versions = json.loads((chain_root / "mfcc40.json").read_text(encoding="utf-8")).get("dependency_versions")
        if (not isinstance(cache_versions, dict) or set(cache_versions) != {"librosa", "numpy", "pandas"}
                or not all(isinstance(value, str) and value for value in cache_versions.values())):
            raise ValueError("clean cache builder dependency metadata differs")
        raw_verification = {
            "mode": "shared_training_transfer_sha256",
            "transfer_identity_sha256": _file_sha256(Path(transfer_identity_path)),
            "raw_content_sha256": verified["raw_content_sha256"],
            "raw_file_count": verified["raw_file_count"],
            "build_source_inventory_signature": inventory,
            "build_time_content_proof": chain.get("source_content_proof"),
            "cache_dependency_versions": cache_versions,
            "note": "Transfer-time raw bytes verified; historical build-time proof status is unchanged.",
        }
    base = {
        "source_manifest_schema_version": manifest_meta.get("schema_version"),
        "manifest_recorded_source_signature": manifest_meta.get("source_signature"),
        "source_manifest_sha256": _file_sha256(manifest_path),
        "source_manifest_metadata_sha256": _file_sha256(manifest_metadata_path),
        "source_row_signature": _row_signature(source),
        "source_content_sha256": chain.get("source_content_sha256"),
        "source_content_proof": chain.get("source_content_proof"),
        "record_key_signature": record_key_signature(source["record_key"]),
        "record_key_order_signature": _order_signature(source["record_key"]),
    }
    for key, value in {**base, "source_inventory_signature": inventory,
                       "source_split": "Training", "row_count": expected_rows,
                       "call_count": expected_calls,
                       "assembly_order": list(clean_59d_feature_names())}.items():
        if chain.get(key) != value:
            raise ValueError(f"clean chain source contract differs: {key}")
    if base["source_content_sha256"] is not None:
        if (base["source_content_proof"] != "verified_before_and_after_build"
                or raw_content_signature(source, Path(raw_root)) != base["source_content_sha256"]):
            raise ValueError("clean chain optional raw content proof differs")
    elif base["source_content_proof"] != "not_collected":
        raise ValueError("clean chain raw content proof status differs")
    if len(chain.get("artifact_hashes", {})) != len(ARTIFACT_SUFFIX):
        raise ValueError("clean chain artifact set differs")
    if chain.get("derived_nodes") != _derived_nodes(chain["artifact_hashes"]):
        raise ValueError("clean E05/52D/59D parent provenance differs")
    parent = {**base, "source_inventory_signature": inventory}
    frames = {}
    bundle = None
    for name, suffix in ARTIFACT_SUFFIX.items():
        data_path = chain_root / f"{name}{suffix}"
        hash_value = _file_sha256(data_path)
        if chain["artifact_hashes"].get(name) != hash_value:
            raise ValueError(f"clean artifact hash differs: {name}")
        observed = json.loads((chain_root / f"{name}.json").read_text(encoding="utf-8"))
        expected = _artifact_metadata(name, source, base, inventory, hash_value, parent,
                                      dependency_versions=cache_versions)
        if observed != expected:
            raise ValueError(f"clean artifact metadata differs: {name}")
        if name == "mfcc40":
            with np.load(data_path, allow_pickle=False) as stored:
                if set(stored.files) != {"X", "y", "stem", "start_at", "end_at", "duration"}:
                    raise ValueError("clean MFCC40 member contract differs")
                bundle = {member: stored[member] for member in stored.files}
        else:
            frames[name] = pd.read_parquet(data_path)
    if bundle is None:
        raise ValueError("clean MFCC40 artifact is missing")
    keys = source["record_key"].astype(str).to_numpy()
    calls = source["call_id"].astype(str).to_numpy()
    y = source["speaker"].to_numpy(dtype=np.int64)
    if (bundle["X"].shape != (expected_rows, 40) or bundle["X"].dtype != np.float32
            or not np.array_equal(bundle["y"], y)
            or not np.array_equal(bundle["stem"].astype(str), calls)
            or not np.array_equal(bundle["start_at"], source["startAt_ms"].to_numpy(dtype=np.float64))
            or not np.array_equal(bundle["end_at"], source["endAt_ms"].to_numpy(dtype=np.float64))
            or not np.array_equal(bundle["duration"], bundle["end_at"] - bundle["start_at"])):
        raise ValueError("clean MFCC40 row/label/tracking contract differs")
    arrays = [bundle["X"]]
    for name in ("e05_candidate", "af02", "af04", "af09", "af10", "af11"):
        frame = frames[name]
        names, _ = _artifact_contract(name, source, inventory)
        if (list(frame.columns) != ["record_key", "call_id", *names]
                or len(frame) != expected_rows
                or not frame["record_key"].astype(str).is_unique
                or not np.array_equal(frame["record_key"].astype(str).to_numpy(), keys)
                or not np.array_equal(frame["call_id"].astype(str).to_numpy(), calls)):
            raise ValueError(f"clean {name} key/call/feature order contract differs")
        values = frame.loc[:, list(names)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"clean {name} contains nonfinite values")
        arrays.append(values)
    X = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    if (X.shape != (expected_rows, 59) or not np.isfinite(X).all()
            or tuple(chain["assembly_order"]) != clean_59d_feature_names()):
        raise ValueError("clean 59D dimension/order/finite contract differs")
    if fixed_validation_record_keys is None:
        train_idx, validation_idx, split_audit = create_fixed_internal_split(
            calls, split_function=standalone_group_split_indices,
            expected_contract=EXPECTED_INTERNAL_SPLIT if expected_rows == EXPECTED_OFFICIAL_ROWS["Training"] else None,
        )
    else:
        persisted_keys = np.asarray(fixed_validation_record_keys).astype(str)
        if len(np.unique(persisted_keys)) != len(persisted_keys):
            raise ValueError("persisted Internal Validation record_key is duplicated")
        key_index = {key: index for index, key in enumerate(keys)}
        if any(key not in key_index for key in persisted_keys):
            raise ValueError("persisted Internal Validation key is absent from Training")
        validation_idx = np.sort(np.fromiter((key_index[key] for key in persisted_keys), dtype=np.int64))
        train_mask = np.ones(len(keys), dtype=bool)
        train_mask[validation_idx] = False
        train_idx = np.flatnonzero(train_mask)
        train_calls, validation_calls = set(calls[train_idx]), set(calls[validation_idx])
        split_audit = {
            "seed": 42, "test_size": 0.2, "total_rows": int(len(keys)),
            "train_rows": int(len(train_idx)), "validation_rows": int(len(validation_idx)),
            "train_calls": int(len(train_calls)), "validation_calls": int(len(validation_calls)),
            "call_overlap": int(len(train_calls & validation_calls)),
        }
        expected_split = EXPECTED_INTERNAL_SPLIT if expected_rows == EXPECTED_OFFICIAL_ROWS["Training"] else None
        if expected_split is not None and any(split_audit[name] != value for name, value in expected_split.items()):
            raise ValueError("persisted Internal Split row/call counts differ")
    if set(calls[train_idx]) & set(calls[validation_idx]) or split_audit["seed"] != 42:
        raise ValueError("clean 59D fixed call split contract differs")
    return Clean59D(X, y, keys, calls, clean_59d_feature_names(), fingerprint, chain_root,
                    raw_source_verification=raw_verification)
