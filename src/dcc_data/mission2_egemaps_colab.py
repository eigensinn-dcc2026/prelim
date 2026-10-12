"""Manual Colab audit and resume helpers; importing performs no extraction or I/O."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys
import tempfile
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd

from .audio import read_wav_frames
from .mission2_egemaps_resume import (
    CRASH_JOURNAL, PADDING_TARGET_SAMPLES, SHORT_AUDIO_FALLBACK, _validate_partial,
    extract_with_short_audio_fallback, fill_existing_partial,
    validate_existing_partial,
)
from .mission2_standard_representation import (
    build_egemaps_cache, get_egemaps_feature_names, load_egemaps_cache,
)

EXPECTED_FINGERPRINT = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
EXPECTED_ROWS = 889_358
APPROVAL_PHRASE = "APPROVE_NUMERICALLY_CLOSE"
MISMATCH_APPROVAL_PHRASE = "APPROVE_CROSS_PLATFORM_MISMATCH"


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute()
            or any(part in ("", ".", "..") or ":" in part for part in value.rstrip("/").split("/"))):
        raise ValueError(f"STOP: unsafe archive/manifest path: {value}")
    return path


def inspect_shared_training_archives(
    manifest: pd.DataFrame, archive_paths: list[str | Path],
) -> dict:
    """Read ZIP directories only; require full, unique Training WAV/JSON coverage.

    Google Drive downloads can prepend a folder name. Only that prefix is
    removed; filenames and the path starting at Training must match exactly.
    No basename guessing, encoding repair, or Validation fallback is allowed.
    """
    if manifest.empty or not manifest["split"].astype(str).eq("Training").all():
        raise ValueError("STOP: archive staging requires a Training-only manifest")
    expected = set()
    for column, suffix in (("audio_path", ".wav"), ("json_path", ".json")):
        for value, call_id in manifest[[column, "call_id"]].drop_duplicates().itertuples(index=False, name=None):
            path = _safe_relative_path(str(value))
            if path.parts[0] != "Training" or path.suffix != suffix or path.stem != str(call_id):
                raise ValueError(f"STOP: manifest path/call mismatch: {value}")
            expected.add(str(path))
        if (manifest.groupby("call_id")[column].nunique().ne(1).any()
                or manifest.groupby(column)["call_id"].nunique().ne(1).any()):
            raise ValueError(f"STOP: manifest call/path alignment differs: {column}")
    # Check the user-visible Drive path before resolving it. In Colab, a broken
    # Drive shortcut can resolve to a meaningless ``/content/drive/ `` path;
    # retaining the supplied path makes the access failure actionable.
    supplied_archives = [Path(path).expanduser() for path in archive_paths]
    if not supplied_archives or len(set(supplied_archives)) != len(supplied_archives):
        raise ValueError("STOP: missing or duplicate archive paths")
    for archive_path in supplied_archives:
        if not archive_path.is_file():
            raise ValueError(
                "STOP: shared Training archive is unavailable at the configured Drive path: "
                f"{archive_path}"
            )
    archives = [path.resolve() for path in supplied_archives]
    members = {}
    archive_summary = []
    for archive_path in archives:
        if not archive_path.is_file():
            raise ValueError(f"STOP: shared archive missing: {archive_path}")
        samples = []
        matched = 0
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    path = _safe_relative_path(info.filename)
                    if len(samples) < 3 and not info.is_dir():
                        samples.append(info.filename)
                    if "Training" not in path.parts or info.is_dir():
                        continue
                    if path.parts.count("Training") != 1:
                        raise ValueError(f"STOP: ambiguous Training path: {info.filename}")
                    relative = str(PurePosixPath(*path.parts[path.parts.index("Training"):]))
                    if path.suffix.lower() not in (".wav", ".json"):
                        continue
                    if relative not in expected:
                        raise ValueError(f"STOP: archive path mismatch: {info.filename}")
                    if relative in members:
                        raise ValueError(f"STOP: duplicate Training member: {relative}")
                    if (info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16)
                            or info.file_size <= 0):
                        raise ValueError(f"STOP: encrypted/symlink/empty member: {info.filename}")
                    members[relative] = (str(archive_path), info.filename, info.file_size, info.CRC)
                    matched += 1
        except zipfile.BadZipFile as error:
            raise ValueError(f"STOP: invalid or split ZIP; do not concatenate blindly: {archive_path}") from error
        archive_summary.append({"archive": str(archive_path), "matched_files": matched,
                                "example_members": samples})
    missing = sorted(expected - members.keys())
    if missing:
        raise ValueError(f"STOP: {len(missing)} manifest WAV/JSON files missing from archives: {missing[:5]}")
    return {"files": len(members), "wav_files": manifest["audio_path"].nunique(),
            "json_files": manifest["json_path"].nunique(),
            "uncompressed_bytes": sum(item[2] for item in members.values()),
            "archives": archive_summary, "members": members}


def prepare_shared_training_raw(
    manifest: pd.DataFrame, archive_paths: list[str | Path], raw_root: str | Path,
    *, reserve_bytes: int = 2 * 1024**3,
) -> dict:
    """Stage manifest files to temporary local storage without changing partials.

    Inspect all archives before writes. Existing files must be byte-identical;
    publish each staged file atomically without replacing an existing target.
    The original loader and transfer-time SHA-256 guard remain unchanged.
    """
    plan = inspect_shared_training_archives(manifest, archive_paths)
    root = Path(raw_root).resolve()
    targets = {}
    needed = 0
    for relative, (_, _, size, _) in plan["members"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"STOP: staging path escapes root: {relative}")
        if target.exists() and not target.is_file():
            raise ValueError(f"STOP: existing staging target is not a file: {target}")
        targets[relative] = target
        if not target.exists():
            needed += size
    disk_root = root
    while not disk_root.exists():
        disk_root = disk_root.parent
    free = shutil.disk_usage(disk_root).free
    if reserve_bytes < 0 or free < needed + reserve_bytes:
        raise ValueError(f"STOP: insufficient local disk space: need {needed + reserve_bytes}, free {free}")

    # Check *all* existing files before staging any new one; never repair by overwrite.
    for summary in plan["archives"]:
        with zipfile.ZipFile(summary["archive"]) as archive:
            for relative, (source, member, size, crc) in plan["members"].items():
                if source != summary["archive"]:
                    continue
                info = archive.getinfo(member)
                if (info.file_size, info.CRC) != (size, crc):
                    raise ValueError(f"STOP: archive changed since inspection: {member}")
                target = targets[relative]
                if target.exists():
                    digest = hashlib.sha256()
                    with archive.open(info) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if target.stat().st_size != size or _sha256(target) != digest.hexdigest():
                        raise ValueError(f"STOP: existing staged file content differs: {relative}")
    for summary in plan["archives"]:
        with zipfile.ZipFile(summary["archive"]) as archive:
            for relative, (source, member, size, crc) in plan["members"].items():
                if source != summary["archive"] or targets[relative].exists():
                    continue
                info = archive.getinfo(member)
                if (info.file_size, info.CRC) != (size, crc):
                    raise ValueError(f"STOP: archive changed since inspection: {member}")
                target = targets[relative]
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        with archive.open(info) as source_stream:
                            shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                    if temporary.stat().st_size != size:
                        raise ValueError(f"STOP: truncated staged member: {member}")
                    os.link(temporary, target)  # Atomic, and fails if target already exists.
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        print("TRAINING_STAGE_ARCHIVE_DONE", Path(summary["archive"]).name,
              "matched files:", summary["matched_files"], flush=True)
    return {"status": "PASS", "raw_root": str(root), "files": plan["files"],
            "wav_files": plan["wav_files"], "json_files": plan["json_files"],
            "uncompressed_bytes": plan["uncompressed_bytes"], "partial_written": False}


def environment_provenance() -> dict:
    """Record the actual runtime, including the CPU-oriented openSMILE version."""
    from importlib import metadata

    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "opensmile", "pyarrow"):
        packages[name] = metadata.version(name)
    return {"python": sys.version, "platform": platform.platform(),
            "os": os.name, "cpu_count": os.cpu_count(), "packages": packages}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_digest(manifest: pd.DataFrame, raw_root: Path) -> tuple[str, int]:
    """Hash each unique raw WAV/JSON by relative path, without trusting mtimes."""
    root = raw_root.resolve()
    paths = set(manifest["audio_path"].astype(str)) | set(manifest["json_path"].astype(str))
    digest = hashlib.sha256()
    for relative in sorted(paths):
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"STOP: raw file missing or escapes root: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(target).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def _key_order_digest(manifest: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for key in manifest["record_key"].astype(str):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_local_transfer_identity(
    manifest: pd.DataFrame, manifest_path: str | Path,
    manifest_metadata_path: str | Path, raw_root: str | Path,
    chain_path: str | Path, output_path: str | Path,
) -> dict:
    """Run locally after stopping Windows extraction, then upload this proof."""
    chain = json.loads(Path(chain_path).read_text(encoding="utf-8"))
    manifest_sha = _sha256(Path(manifest_path))
    manifest_meta_sha = _sha256(Path(manifest_metadata_path))
    if (chain.get("provenance_fingerprint") != EXPECTED_FINGERPRINT
            or chain.get("source_manifest_sha256") != manifest_sha
            or chain.get("source_manifest_metadata_sha256") != manifest_meta_sha
            or chain.get("record_key_order_signature") != _key_order_digest(manifest)
            or len(manifest) != EXPECTED_ROWS
            or not manifest["split"].astype(str).eq("Training").all()):
        raise ValueError("STOP: local clean chain or manifest identity differs")
    raw_sha, raw_files = _raw_digest(manifest, Path(raw_root))
    result = {"schema": "mission2_egemaps_transfer_identity_v1",
              "clean_reference_fingerprint": EXPECTED_FINGERPRINT,
              "manifest_sha256": manifest_sha,
              "manifest_metadata_sha256": manifest_meta_sha,
              "record_key_order_signature": _key_order_digest(manifest),
              "raw_content_sha256": raw_sha, "raw_file_count": raw_files,
              "source_split": "Training", "rows": EXPECTED_ROWS}
    with Path(output_path).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def verify_transfer_identity(
    manifest: pd.DataFrame, manifest_path: str | Path,
    manifest_metadata_path: str | Path, raw_root: str | Path,
    identity_path: str | Path,
) -> dict:
    """Full content check; it can be slow and is deliberately separate from benchmark."""
    expected = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    if (expected.get("schema") != "mission2_egemaps_transfer_identity_v1"
            or expected.get("clean_reference_fingerprint") != EXPECTED_FINGERPRINT
            or expected.get("rows") != EXPECTED_ROWS
            or expected.get("source_split") != "Training"
            or len(manifest) != EXPECTED_ROWS
            or not manifest["split"].astype(str).eq("Training").all()
            or expected.get("manifest_sha256") != _sha256(Path(manifest_path))
            or expected.get("manifest_metadata_sha256") != _sha256(Path(manifest_metadata_path))
            or expected.get("record_key_order_signature") != _key_order_digest(manifest)):
        raise ValueError("STOP: transferred manifest identity differs")
    raw_sha, raw_files = _raw_digest(manifest, Path(raw_root))
    if (raw_sha != expected.get("raw_content_sha256")
            or raw_files != expected.get("raw_file_count")):
        raise ValueError("STOP: transferred raw WAV/JSON content differs")
    return {**expected, "status": "PASS"}


def select_overlap_rows(
    manifest: pd.DataFrame, mask: np.ndarray, padded_keys: list[str],
    *, count: int = 100,
) -> np.ndarray:
    """Deterministically span completed crop lengths and include a padded row."""
    completed = np.flatnonzero(mask)
    if len(completed) < count:
        raise ValueError("STOP: not enough completed rows for overlap audit")
    lengths = (manifest["crop_end_frame"].to_numpy(dtype=np.int64)
               - manifest["crop_start_frame"].to_numpy(dtype=np.int64))
    ordered = completed[np.argsort(lengths[completed], kind="stable")]
    selected = ordered[np.linspace(0, len(ordered) - 1, count, dtype=np.int64)]
    if padded_keys:
        positions = pd.Index(manifest["record_key"]).get_indexer(padded_keys)
        padded = next((int(index) for index in positions if index >= 0 and mask[index]), None)
        if padded is not None and padded not in selected:
            selected[-1] = padded
    if len(np.unique(selected)) != count:
        raise ValueError("STOP: overlap selection is not unique")
    return np.sort(selected)


def _extract_rows(
    manifest: pd.DataFrame, raw_root: Path, positions: np.ndarray,
    names: tuple[str, ...],
) -> tuple[dict[int, np.ndarray], int]:
    selected = manifest.iloc[positions]
    vectors = {}
    for audio_path, group in selected.groupby("audio_path", sort=False):
        audio = read_wav_frames(raw_root / str(audio_path))
        if audio.sample_rate != 8_000 or audio.waveform.shape[0] != 1:
            raise ValueError(f"STOP: unexpected WAV contract: {audio_path}")
        for row in group.itertuples():
            start, end = int(row.crop_start_frame), int(row.crop_end_frame)
            if start < 0 or end <= start or end > audio.waveform.shape[1]:
                raise ValueError(f"STOP: invalid crop: {row.record_key}")
            vector, returned_names, _ = extract_with_short_audio_fallback(
                audio.waveform[0, start:end], audio.sample_rate,
            )
            if returned_names != names:
                raise ValueError("STOP: eGeMAPS feature order differs")
            vectors[int(row.Index)] = vector
    return vectors, selected["audio_path"].nunique()


def compare_overlap_values(local: np.ndarray, rerun: np.ndarray,
                           names: tuple[str, ...]) -> dict:
    """Keep exact equality separate from tolerance-based diagnostics."""
    if local.shape != rerun.shape or local.ndim != 2 or local.shape[1] != len(names):
        raise ValueError("STOP: overlap matrices/feature names differ")
    if not np.isfinite(local).all() or not np.isfinite(rerun).all():
        raise ValueError("STOP: overlap contains nonfinite values")
    exact = np.all(local == rerun, axis=1)
    close = np.isclose(local, rerun, rtol=1e-7, atol=1e-8)
    differences = np.abs(local.astype(np.float64) - rerun.astype(np.float64))
    changed = np.count_nonzero(local != rerun, axis=0)
    top = np.argsort(changed)[::-1][:10]
    status = ("CROSS_PLATFORM_EXACT_MATCH" if bool(np.all(exact)) else
              "CROSS_PLATFORM_NUMERICALLY_CLOSE" if bool(np.all(close)) else
              "CROSS_PLATFORM_FEATURE_MISMATCH")
    return {"status": status, "exact_row_count": int(np.count_nonzero(exact)),
            "numerically_close_row_count": int(np.count_nonzero(np.all(close, axis=1))),
            "differing_row_count": int(np.count_nonzero(~exact)),
            "differing_value_count": int(np.count_nonzero(local != rerun)),
            "mean_abs_diff": float(differences.mean()),
            "max_abs_diff": float(differences.max()),
            "top_changed_features": [
                {"name": names[index], "changed_values": int(changed[index]),
                 "max_abs_diff": float(differences[:, index].max())}
                for index in top if changed[index] > 0],
            "rtol": 1e-7, "atol": 1e-8}


def audit_overlap(
    manifest: pd.DataFrame, raw_root: str | Path, recovery_root: str | Path,
    keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    *, expected_rows: int = EXPECTED_ROWS,
) -> dict:
    validate_existing_partial(manifest, recovery_root, keys, calls, names,
                              fingerprint=EXPECTED_FINGERPRINT, expected_rows=expected_rows)
    matrix, mask, padded = _validate_partial(
        Path(recovery_root), keys, calls, names, EXPECTED_FINGERPRINT, read_only=True,
    )
    recovery_meta = json.loads((Path(recovery_root) / "recovery_metadata.json").read_text(encoding="utf-8"))
    fixed_keys = recovery_meta.get("windows_overlap_record_keys")
    if fixed_keys is None:
        positions = select_overlap_rows(manifest, mask, [key for key, _ in padded])
    else:
        if not isinstance(fixed_keys, list) or len(fixed_keys) != 100 or len(set(fixed_keys)) != 100:
            raise ValueError("STOP: frozen Windows overlap set differs")
        positions = pd.Index(keys).get_indexer(fixed_keys)
        if np.any(positions < 0) or not np.all(mask[positions]):
            raise ValueError("STOP: frozen Windows overlap rows are unavailable")
    observed, calls_used = _extract_rows(manifest, Path(raw_root), positions, names)
    local = np.asarray(matrix[positions])
    colab = np.stack([observed[int(index)] for index in positions])
    comparison = compare_overlap_values(local, colab, names)
    return {**comparison, "sampled_rows": len(positions), "sampled_calls": calls_used,
            "sampled_record_keys": [str(keys[index]) for index in positions],
            "padded_available": bool(padded),
            "padded_sampled": any(str(keys[index]) in {key for key, _ in padded}
                                  for index in positions)}


def benchmark_extraction(
    manifest: pd.DataFrame, raw_root: str | Path, recovery_root: str | Path,
    keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    *, count: int = 250, expected_rows: int = EXPECTED_ROWS,
) -> dict:
    summary = validate_existing_partial(
        manifest, recovery_root, keys, calls, names,
        fingerprint=EXPECTED_FINGERPRINT, expected_rows=expected_rows,
    )
    _, mask, _ = _validate_partial(
        Path(recovery_root), keys, calls, names, EXPECTED_FINGERPRINT, read_only=True,
    )
    missing = np.flatnonzero(~mask)
    if len(missing) < count:
        raise ValueError("STOP: not enough incomplete rows for benchmark")
    positions = missing[np.linspace(0, len(missing) - 1, count, dtype=np.int64)]
    started = perf_counter()
    _, call_count = _extract_rows(manifest, Path(raw_root), positions, names)
    elapsed = perf_counter() - started
    return {"rows": count, "calls": int(call_count), "elapsed_seconds": elapsed,
            "rows_per_second": count / elapsed, "calls_per_second": call_count / elapsed,
            "remaining_rows": summary["remaining"],
            "estimated_remaining_seconds": summary["remaining"] * elapsed / count,
            "cache_written": False}


def _mismatch_approval_matches(approval: dict, overlap: dict, identity: dict,
                               environment: dict, feature_names: tuple[str, ...]) -> bool:
    overlap_fields = (
        "sampled_rows", "sampled_calls", "padded_sampled", "exact_row_count",
        "numerically_close_row_count", "differing_row_count", "differing_value_count",
        "mean_abs_diff", "max_abs_diff",
    )
    return (
        approval.get("approval_type") == MISMATCH_APPROVAL_PHRASE
        and approval.get("overlap_status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
        and all(approval.get(field) == overlap.get(field) for field in overlap_fields)
        and identity.get("status") == "PASS"
        and approval.get("raw_file_count") == identity.get("raw_file_count") == 58_400
        and approval.get("source_split") == identity.get("source_split") == "Training"
        and approval.get("source_rows") == identity.get("rows") == EXPECTED_ROWS
        and approval.get("clean_reference_fingerprint") == identity.get("clean_reference_fingerprint") == EXPECTED_FINGERPRINT
        and approval.get("raw_content_sha256") == identity.get("raw_content_sha256")
        and approval.get("opensmile_version") == environment.get("packages", {}).get("opensmile") == "2.6.0"
        and approval.get("feature_names") == list(feature_names)
    )


def _validate_mismatch_override(partial: dict, overlap: dict, identity: dict,
                                environment: dict | None) -> None:
    names = tuple(partial.get("feature_names", ()))
    if (environment is None or environment.get("packages", {}).get("opensmile") != "2.6.0"
            or partial.get("feature_dim") != 88
            or len(names) != 88 or names != get_egemaps_feature_names()):
        raise ValueError("STOP: mismatch override runtime/feature contract differs")
    if (identity.get("source_split") != "Training"
            or identity.get("raw_file_count") != 58_400
            or identity.get("rows") != EXPECTED_ROWS
            or identity.get("clean_reference_fingerprint") != EXPECTED_FINGERPRINT
            or not identity.get("raw_content_sha256")):
        raise ValueError("STOP: mismatch override source identity differs")
    if (overlap.get("status") != "CROSS_PLATFORM_FEATURE_MISMATCH"
            or overlap.get("sampled_rows") != 100
            or overlap.get("sampled_calls") != 100
            or overlap.get("padded_sampled") is not True
            or any(not isinstance(overlap.get(key), (int, np.integer))
                   for key in ("exact_row_count", "numerically_close_row_count",
                               "differing_row_count", "differing_value_count"))
            or any(not np.isfinite(overlap.get(key, np.nan))
                   for key in ("mean_abs_diff", "max_abs_diff"))
            or overlap.get("mean_abs_diff", float("inf")) < 0
            or overlap.get("max_abs_diff", float("inf")) < 0
            or overlap.get("differing_value_count", EXPECTED_ROWS + 1) > 120
            or overlap.get("mean_abs_diff", float("inf")) > 2e-8
            or overlap.get("max_abs_diff", float("inf")) > 3e-5):
        raise ValueError("STOP: mismatch override overlap contract differs")
    if (partial.get("rows") != EXPECTED_ROWS
            or partial.get("fingerprint") != EXPECTED_FINGERPRINT):
        raise ValueError("STOP: mismatch override partial identity differs")

    # Re-approval after interruption is allowed only if a prior explicit
    # approval records the same source and overlap contract.
    initial_state = (partial.get("completed") == 470_255
                     and partial.get("remaining") == 419_103)
    continued_state = (
        partial.get("completed", 0) >= 470_255
        and partial.get("remaining", EXPECTED_ROWS + 1) <= 419_103
        and any(_mismatch_approval_matches(
            event.get("resume_approval", {}), overlap, identity, environment, names,
        ) and event.get("starting_completed_rows") == 470_255
          and event.get("starting_remaining_rows") == 419_103
          for event in partial.get("resume_history", []) if isinstance(event, dict))
    )
    if not initial_state and not continued_state:
        raise ValueError("STOP: mismatch override requires the verified original partial state")


def approve_resume(partial: dict, overlap: dict, identity: dict,
                   *, explicit_approval: str | None = None,
                   environment: dict | None = None) -> dict:
    if (partial.get("status") != "PASS" or identity.get("status") != "PASS"
            or partial.get("fingerprint") != EXPECTED_FINGERPRINT
            or (overlap.get("status") != "CROSS_PLATFORM_FEATURE_MISMATCH"
                and identity.get("clean_reference_fingerprint") != EXPECTED_FINGERPRINT)
            or overlap.get("sampled_rows") != 100
            or (overlap.get("status") != "CROSS_PLATFORM_FEATURE_MISMATCH"
                and overlap.get("padded_available") is True
                and overlap.get("padded_sampled") is not True)):
        raise ValueError("STOP: partial/source/overlap prerequisite is incomplete")
    status = overlap.get("status")
    if status == "CROSS_PLATFORM_NUMERICALLY_CLOSE" and explicit_approval != APPROVAL_PHRASE:
        raise ValueError("STOP: numerically-close overlap needs explicit user approval")
    if status == "CROSS_PLATFORM_FEATURE_MISMATCH":
        if explicit_approval != MISMATCH_APPROVAL_PHRASE:
            raise ValueError("STOP: cross-platform mismatch requires explicit user approval")
        _validate_mismatch_override(partial, overlap, identity, environment)
        names = list(partial["feature_names"])
        return {"status": "APPROVED", "approval_type": MISMATCH_APPROVAL_PHRASE,
                "overlap_status": status,
                **{key: overlap[key] for key in (
                    "sampled_rows", "sampled_calls", "padded_sampled", "exact_row_count",
                    "numerically_close_row_count", "differing_row_count", "differing_value_count",
                    "mean_abs_diff", "max_abs_diff")},
                "runtime_provenance": environment,
                "opensmile_version": environment["packages"]["opensmile"],
                "feature_dim": len(names), "feature_names": names,
                "raw_file_count": identity["raw_file_count"],
                "source_split": identity["source_split"], "source_rows": identity["rows"],
                "clean_reference_fingerprint": identity["clean_reference_fingerprint"],
                "raw_content_sha256": identity["raw_content_sha256"],
                "starting_completed_rows": 470_255,
                "starting_remaining_rows": 419_103,
                "fingerprint": EXPECTED_FINGERPRINT,
                "short_audio_fallback": SHORT_AUDIO_FALLBACK,
                "padding_target_samples": PADDING_TARGET_SAMPLES, "sample_rate": 8000}
    if status not in ("CROSS_PLATFORM_EXACT_MATCH", "CROSS_PLATFORM_NUMERICALLY_CLOSE"):
        raise ValueError("STOP: cross-platform feature mismatch; do not mix matrices")
    return {"status": "APPROVED", "overlap_status": status,
            "numerical_user_approval": status == "CROSS_PLATFORM_NUMERICALLY_CLOSE",
            "raw_content_sha256": identity["raw_content_sha256"],
            "fingerprint": EXPECTED_FINGERPRINT,
            "short_audio_fallback": SHORT_AUDIO_FALLBACK,
            "padding_target_samples": PADDING_TARGET_SAMPLES, "sample_rate": 8000}


def recover_interrupted_checkpoint(recovery_root: str | Path) -> str:
    """Explicitly roll back only rows after the last atomic Colab checkpoint."""
    root = Path(recovery_root)
    journal = root / CRASH_JOURNAL
    if not journal.is_file():
        return "NO_COLAB_JOURNAL; run strict validation"
    with np.load(journal, allow_pickle=False) as stored:
        if set(stored.files) != {"mask", "metadata"}:
            raise ValueError("STOP: crash journal schema differs")
        committed = stored["mask"]
        metadata = json.loads(stored["metadata"].item())
    matrix = np.load(root / "matrix_88_partial.npy", mmap_mode="r+", allow_pickle=False)
    mask = np.load(root / "completed_mask.npy", mmap_mode="r+", allow_pickle=False)
    if (committed.shape != mask.shape or committed.dtype != np.bool_
            or matrix.shape != (len(mask), 88) or matrix.dtype != np.float32
            or metadata.get("completed_rows") != int(np.count_nonzero(committed))
            or metadata.get("clean_reference_fingerprint") != EXPECTED_FINGERPRINT):
        raise ValueError("STOP: crash journal identity/shape differs")
    for start in range(0, len(mask), 8192):
        stop = min(start + 8192, len(mask))
        if not np.isfinite(matrix[start:stop][committed[start:stop]]).all():
            raise ValueError("STOP: committed checkpoint matrix is nonfinite")
        # Uncommitted rows may have been partly written when Colab disconnected.
        matrix_chunk = matrix[start:stop]
        matrix_chunk[~committed[start:stop]] = np.nan
        mask[start:stop] = committed[start:stop]
    matrix.flush()
    mask.flush()
    temporary = root / "recovery_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, root / "recovery_metadata.json")
    return f"ROLLED_BACK_TO_COMMITTED_CHECKPOINT: {metadata['completed_rows']} rows"


def _start_crash_journal(root: Path) -> None:
    matrix, mask, _ = _validate_partial(
        root, np.load(root / "record_keys.npy", allow_pickle=True).astype(str),
        np.load(root / "call_ids.npy", allow_pickle=True).astype(str),
        get_egemaps_feature_names(), EXPECTED_FINGERPRINT, read_only=True,
    )
    del matrix
    metadata = json.loads((root / "recovery_metadata.json").read_text(encoding="utf-8"))
    temporary = root / f"{CRASH_JOURNAL}.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, mask=np.asarray(mask),
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, root / CRASH_JOURNAL)


def _assert_snapshot_completed_rows(
    snapshot_matrix: np.ndarray, snapshot_mask: np.ndarray,
    working_matrix: np.ndarray, working_mask: np.ndarray,
) -> None:
    if (snapshot_matrix.ndim != 2 or working_matrix.shape != snapshot_matrix.shape
            or snapshot_mask.shape != (snapshot_matrix.shape[0],)
            or working_mask.shape != snapshot_mask.shape
            or snapshot_mask.dtype != np.bool_ or working_mask.dtype != np.bool_
            or not working_mask[snapshot_mask].all()):
        raise ValueError("STOP: snapshot completed rows were not preserved")
    for start in range(0, len(snapshot_mask), 8192):
        stop = min(start + 8192, len(snapshot_mask))
        completed = snapshot_mask[start:stop]
        if not np.array_equal(snapshot_matrix[start:stop][completed],
                              working_matrix[start:stop][completed]):
            raise ValueError("STOP: snapshot completed feature rows changed")


def _validate_mismatch_finalization_approval(
    partial: dict, approval: dict, overlap: dict, identity: dict,
    environment: dict, feature_names: tuple[str, ...], resume_history: list[dict],
) -> str:
    if (approval.get("status") != "APPROVED"
            or approval.get("approval_type") != MISMATCH_APPROVAL_PHRASE):
        raise ValueError("STOP: finalization lacks mismatch manual approval")
    history_recorded = any(
        isinstance(event, dict)
        and event.get("resume_approval") == approval
        and event.get("starting_completed_rows") == 470_255
        and event.get("starting_remaining_rows") == 419_103
        and event.get("overlap_status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
        and event.get("raw_content_sha256") == identity.get("raw_content_sha256")
        for event in resume_history
    )
    if not history_recorded:
        raise ValueError("STOP: mismatch approval/history is absent from working partial")
    if (identity.get("status") != "PASS"
            or approval.get("runtime_provenance", {}).get("packages", {}).get("opensmile") != "2.6.0"
            or environment.get("packages", {}).get("opensmile") != "2.6.0"
            or approval.get("opensmile_version") != "2.6.0"
            or approval.get("runtime_provenance") != environment):
        raise ValueError("STOP: finalization source/runtime identity differs")
    partial = {**partial, "resume_history": resume_history}
    _validate_mismatch_override(partial, overlap, identity, environment)
    if (partial.get("rows") != EXPECTED_ROWS
            or partial.get("completed") != EXPECTED_ROWS
            or partial.get("remaining") != 0
            or partial.get("feature_dim") != 88
            or tuple(partial.get("feature_names", ())) != tuple(feature_names)
            or len(feature_names) != 88
            or tuple(feature_names) != get_egemaps_feature_names()):
        raise ValueError("STOP: final mismatch partial row/feature contract differs")
    if (approval.get("starting_completed_rows") != 470_255
            or approval.get("starting_remaining_rows") != 419_103):
        raise ValueError("STOP: final mismatch resume starting state differs")
    return "PASS"


def resume_approved_partial(
    manifest: pd.DataFrame, raw_root: str | Path, recovery_root: str | Path,
    keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    approval: dict, overlap: dict, identity: dict, environment: dict,
) -> dict:
    if (approval.get("status") != "APPROVED"
            or approval.get("overlap_status") != overlap.get("status")
            or approval.get("raw_content_sha256") != identity.get("raw_content_sha256")
            or approval.get("fingerprint") != EXPECTED_FINGERPRINT
            or approval.get("short_audio_fallback") != SHORT_AUDIO_FALLBACK
            or approval.get("padding_target_samples") != PADDING_TARGET_SAMPLES
            or approval.get("sample_rate") != 8000
            or overlap.get("status") not in ("CROSS_PLATFORM_EXACT_MATCH",
                                               "CROSS_PLATFORM_NUMERICALLY_CLOSE",
                                               "CROSS_PLATFORM_FEATURE_MISMATCH")
            or (overlap.get("status") == "CROSS_PLATFORM_NUMERICALLY_CLOSE"
                and approval.get("numerical_user_approval") is not True)
            or (overlap.get("status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
                and approval.get("approval_type") != MISMATCH_APPROVAL_PHRASE)):
        raise ValueError("STOP: resume approval contract differs")
    current = validate_existing_partial(manifest, recovery_root, keys, calls, names,
                                         fingerprint=EXPECTED_FINGERPRINT)
    root = Path(recovery_root)
    metadata_path = root / "recovery_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if overlap.get("status") == "CROSS_PLATFORM_FEATURE_MISMATCH":
        current.update(feature_dim=len(names), feature_names=list(names),
                       resume_history=metadata.get("resume_history", []))
        if (not _mismatch_approval_matches(approval, overlap, identity, environment, names)
                or approval.get("runtime_provenance") != environment
                or approval.get("feature_dim") != 88):
            raise ValueError("STOP: mismatch resume approval/source/runtime differs")
        _validate_mismatch_override(current, overlap, identity, environment)
        if (approval.get("starting_completed_rows") != 470_255
                or approval.get("starting_remaining_rows") != 419_103):
            raise ValueError("STOP: mismatch resume starting state differs")
        print("START_COMPLETED =", approval["starting_completed_rows"], flush=True)
        print("START_REMAINING =", approval["starting_remaining_rows"], flush=True)
        print("CURRENT_COMPLETED =", current["completed"],
              "CURRENT_REMAINING =", current["remaining"], flush=True)
    history = metadata.get("resume_history", [])
    history.append({"platform": environment["platform"],
                    "opensmile_version": environment["packages"]["opensmile"],
                    "starting_completed": metadata["completed_rows"],
                    "overlap_status": overlap["status"],
                    "raw_content_sha256": identity["raw_content_sha256"],
                    **({"starting_completed_rows": approval["starting_completed_rows"],
                        "starting_remaining_rows": approval["starting_remaining_rows"],
                        "resume_approval": approval}
                       if overlap["status"] == "CROSS_PLATFORM_FEATURE_MISMATCH" else {})})
    metadata["resume_history"] = history
    frozen_keys = metadata.get("windows_overlap_record_keys")
    if frozen_keys is not None and frozen_keys != overlap["sampled_record_keys"]:
        raise ValueError("STOP: Windows overlap set changed between sessions")
    metadata["windows_overlap_record_keys"] = overlap["sampled_record_keys"]
    temporary = root / "recovery_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    _start_crash_journal(root)
    return fill_existing_partial(manifest, raw_root, root, keys, calls, names,
                                 EXPECTED_FINGERPRINT)


def publish_approved_cache(
    manifest: pd.DataFrame, recovery_root: str | Path, final_path: str | Path,
    keys: np.ndarray, calls: np.ndarray, names: tuple[str, ...],
    approval: dict, overlap: dict, identity: dict, environment: dict,
    *, snapshot_root: str | Path | None = None,
) -> dict:
    if Path(final_path).exists():
        raise FileExistsError(f"STOP: final cache already exists: {final_path}")
    if (approval.get("status") != "APPROVED"
            or approval.get("overlap_status") != overlap.get("status")
            or approval.get("fingerprint") != EXPECTED_FINGERPRINT
            or approval.get("short_audio_fallback") != SHORT_AUDIO_FALLBACK
            or approval.get("padding_target_samples") != PADDING_TARGET_SAMPLES
            or approval.get("sample_rate") != 8000
            or identity.get("status") != "PASS"
            or identity.get("clean_reference_fingerprint") != EXPECTED_FINGERPRINT
            or overlap.get("status") not in ("CROSS_PLATFORM_EXACT_MATCH",
                                               "CROSS_PLATFORM_NUMERICALLY_CLOSE",
                                               "CROSS_PLATFORM_FEATURE_MISMATCH")
            or (overlap.get("status") == "CROSS_PLATFORM_NUMERICALLY_CLOSE"
                and approval.get("numerical_user_approval") is not True)
            or (overlap.get("status") == "CROSS_PLATFORM_FEATURE_MISMATCH"
                and approval.get("approval_type") != MISMATCH_APPROVAL_PHRASE)):
        raise ValueError("STOP: final publish lacks cross-platform approval")
    summary = validate_existing_partial(manifest, recovery_root, keys, calls, names,
                                        fingerprint=EXPECTED_FINGERPRINT)
    if summary["remaining"] != 0:
        raise ValueError("STOP: final cache requires all 889358 finite rows")
    matrix, _, padded = _validate_partial(
        Path(recovery_root), keys, calls, names, EXPECTED_FINGERPRINT, read_only=True,
    )
    recovery_meta = json.loads((Path(recovery_root) / "recovery_metadata.json").read_text(encoding="utf-8"))
    if overlap.get("status") == "CROSS_PLATFORM_FEATURE_MISMATCH":
        final_summary = {**summary, "feature_dim": matrix.shape[1],
                         "feature_names": list(names)}
        _validate_mismatch_finalization_approval(
            final_summary, approval, overlap, identity, environment, names,
            recovery_meta.get("resume_history", []),
        )
        if snapshot_root is None:
            raise ValueError("STOP: mismatch finalization requires the read-only Windows snapshot")
        snapshot_matrix, snapshot_mask, _ = _validate_partial(
            Path(snapshot_root), keys, calls, names, EXPECTED_FINGERPRINT, read_only=True,
        )
        if int(np.count_nonzero(snapshot_mask)) != 470_255:
            raise ValueError("STOP: Windows snapshot completed-row count differs")
        _assert_snapshot_completed_rows(snapshot_matrix, snapshot_mask, matrix,
                                        np.load(Path(recovery_root) / "completed_mask.npy",
                                                mmap_mode="r", allow_pickle=False))
        del snapshot_matrix, snapshot_mask
    extra = {"environment": environment, "source_identity": identity,
             "overlap_audit": overlap, "resume_approval": approval,
             "resume_history": recovery_meta.get("resume_history", [])}
    if not extra["resume_history"] or identity.get("raw_content_sha256") != approval.get("raw_content_sha256"):
        raise ValueError("STOP: final cache lacks approved resume history/source identity")
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    build_egemaps_cache(final_path, matrix, keys, calls, names,
                        source_provenance=EXPECTED_FINGERPRINT,
                        padded_rows=padded, colab_resume=extra)
    result = load_egemaps_cache(final_path, keys, calls,
                                source_provenance=EXPECTED_FINGERPRINT)
    return {"path": str(final_path), "rows": len(result.record_keys),
            "padded": result.metadata["padded_row_count"],
            "provenance_fingerprint": EXPECTED_FINGERPRINT}
