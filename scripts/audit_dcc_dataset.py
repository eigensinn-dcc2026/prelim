#!/usr/bin/env python3
"""Read-only audit of the DCC Google Drive snapshot and local download shards.

The script never moves, renames, deletes, or rewrites source data.  Google Drive
is enumerated with gdown's ``skip_download=True`` mode, so file bodies are not
downloaded.  Audit artifacts are created only in a new output directory; an
existing output path is treated as an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


MANIFEST_VERSION = "1.0"
OUTPUT_FILES = (
    "local_manifest.csv",
    "remote_manifest.csv",
    "comparison.csv",
    "pair_audit.csv",
    "duplicate_groups.csv",
    "integrity_issues.csv",
    "audit_summary.json",
    "audit_report.md",
)
DOWNLOAD_CHUNK_RE = re.compile(r"^대학부 데이터-\d{8}T\d{6}Z-1-\d{3}$")
REGION_FOLDER_RE = re.compile(r"^(TS|TL|VS|VL)_([^_]+)_구급$")
DRIVE_FOLDER_ID_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
DATA_EXTENSIONS = {".wav", ".json"}

LOCAL_FIELDS = (
    "manifest_version",
    "snapshot_at",
    "download_chunk",
    "split",
    "data_role",
    "region",
    "region_status",
    "region_or_subfolder_raw",
    "relative_path",
    "relative_path_from_raw",
    "filename",
    "base_identifier",
    "extension",
    "size_bytes",
    "sha256",
    "zero_byte",
    "official_distribution_status",
    "training_eligibility_status",
    "training_eligibility_reason",
    "json_encoding",
    "json_parse_status",
    "json_error",
    "wav_open_success",
    "wav_channels",
    "wav_sample_rate",
    "wav_frame_count",
    "wav_error",
)

REMOTE_FIELDS = (
    "manifest_version",
    "snapshot_at",
    "drive_folder_id",
    "drive_file_id",
    "split",
    "data_role",
    "region",
    "region_status",
    "region_or_subfolder_raw",
    "relative_path",
    "filename",
    "base_identifier",
    "extension",
    "remote_size",
    "remote_size_status",
    "official_distribution_status",
    "training_eligibility_status",
)

COMPARISON_FIELDS = (
    "comparison_key",
    "split",
    "region",
    "relative_path",
    "filename",
    "remote_present",
    "local_present",
    "comparison_status",
    "download_chunks",
)

PAIR_FIELDS = (
    "split",
    "region",
    "base_identifier",
    "wav_count",
    "json_count",
    "wav_paths",
    "json_paths",
    "pair_status",
)

DUPLICATE_FIELDS = (
    "duplicate_kind",
    "group_id",
    "split",
    "filename",
    "relative_path",
    "sha256",
    "source_chunk_count",
    "source_chunks",
    "member_count",
    "member_paths",
)

INTEGRITY_FIELDS = (
    "issue_type",
    "severity",
    "split",
    "region",
    "relative_path",
    "download_chunk",
    "detail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-url", required=True)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(2, (os.cpu_count() or 4))),
        help="Local file inspection workers (default: %(default)s)",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def normalize_slashes(value: str) -> str:
    return value.replace("\\", "/")


def extract_drive_folder_id(url: str) -> str:
    match = DRIVE_FOLDER_ID_RE.search(urlparse(url).path)
    if not match:
        raise ValueError(f"Drive folder ID not found in URL: {url}")
    return match.group(1)


def canonical_parts(parts: Iterable[str]) -> tuple[str, list[str]]:
    cleaned = [part for part in parts if part not in ("", ".")]
    for index, part in enumerate(cleaned):
        if part in ("Training", "Validation"):
            return part, cleaned[index:]
    return "unknown", cleaned


def classify_path(parts: list[str], extension: str) -> dict[str, str]:
    split, canonical = canonical_parts(parts)
    region = ""
    region_status = "unmapped"
    region_raw = ""
    prefix = ""
    for part in canonical:
        match = REGION_FOLDER_RE.fullmatch(part)
        if match:
            prefix, region = match.groups()
            region_status = "mapped_from_path"
            region_raw = part
            break

    if prefix in ("TS", "VS"):
        data_role = "source_audio"
    elif prefix in ("TL", "VL"):
        data_role = "label_json"
    elif extension == ".wav":
        data_role = "source_audio_unmapped"
    elif extension == ".json":
        data_role = "label_json_unmapped"
    else:
        data_role = "other"

    relative_path = "/".join(canonical)
    return {
        "split": split,
        "relative_path": relative_path,
        "data_role": data_role,
        "region": region,
        "region_status": region_status,
        "region_or_subfolder_raw": region_raw,
    }


def training_status(
    split: str, region: str, region_status: str, official_status: str
) -> tuple[str, str]:
    if split == "Validation":
        return (
            "ineligible_validation_split",
            "Validation data is prohibited from model training.",
        )
    if region_status != "mapped_from_path":
        return (
            "pending_region_mapping",
            "Region could not be derived from the preserved source path.",
        )
    if region != "서울":
        return (
            "ineligible_non_seoul_region",
            "Published rule allows only Seoul data for Training.",
        )
    if official_status != "organizer_confirmed_official":
        return (
            "pending_official_distribution",
            "Seoul Training path detected, but official distribution scope is pending.",
        )
    return "eligible", "Organizer-confirmed official Seoul Training data."


def inspect_environment(args: argparse.Namespace) -> tuple[Path, Path]:
    local_root = args.local_root.resolve()
    output_dir = args.output_dir.resolve()
    if not local_root.is_dir():
        raise FileNotFoundError(f"Local root does not exist: {local_root}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    planned = [output_dir / name for name in OUTPUT_FILES]
    collisions = [path for path in planned if path.exists()]
    if collisions:
        raise FileExistsError(f"Output collision(s): {collisions}")
    print("Planned new audit files:")
    for path in planned:
        print(f"  {path}")
    print("Source data policy: read-only")
    print("Remote policy: metadata/path enumeration only (no file body download)")
    return local_root, output_dir


def enumerate_remote(
    drive_url: str, snapshot_at: str
) -> tuple[list[dict[str, Any]], str]:
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("gdown is required for remote metadata enumeration") from exc

    drive_folder_id = extract_drive_folder_id(drive_url)
    print("Enumerating current Google Drive snapshot without downloading file bodies...")
    entries = gdown.download_folder(
        url=drive_url,
        skip_download=True,
        quiet=True,
        use_cookies=False,
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        remote_path = PurePosixPath(normalize_slashes(str(entry.path)))
        extension = remote_path.suffix.lower()
        classified = classify_path(list(remote_path.parts), extension)
        official = "current_remote_candidate"
        eligibility, _ = training_status(
            classified["split"],
            classified["region"],
            classified["region_status"],
            official,
        )
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "snapshot_at": snapshot_at,
                "drive_folder_id": drive_folder_id,
                "drive_file_id": str(entry.id),
                "split": classified["split"],
                "data_role": classified["data_role"],
                "region": classified["region"],
                "region_status": classified["region_status"],
                "region_or_subfolder_raw": classified["region_or_subfolder_raw"],
                "relative_path": classified["relative_path"],
                "filename": remote_path.name,
                "base_identifier": remote_path.stem,
                "extension": extension,
                "remote_size": "",
                "remote_size_status": "unavailable",
                "official_distribution_status": official,
                "training_eligibility_status": eligibility,
            }
        )
    rows.sort(key=lambda row: (row["split"], row["relative_path"].casefold()))
    print(f"Remote entries enumerated: {len(rows):,}")
    return rows, drive_folder_id


def enumerate_local(local_root: Path, snapshot_at: str) -> list[dict[str, Any]]:
    print("Enumerating local download shards...")
    rows: list[dict[str, Any]] = []
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in DATA_EXTENSIONS:
            continue
        relative = path.relative_to(local_root)
        parts = list(relative.parts)
        download_chunk = parts[0] if parts and DOWNLOAD_CHUNK_RE.fullmatch(parts[0]) else ""
        classified = classify_path(parts, extension)
        rows.append(
            {
                "_path": path,
                "manifest_version": MANIFEST_VERSION,
                "snapshot_at": snapshot_at,
                "download_chunk": download_chunk,
                "split": classified["split"],
                "data_role": classified["data_role"],
                "region": classified["region"],
                "region_status": classified["region_status"],
                "region_or_subfolder_raw": classified["region_or_subfolder_raw"],
                "relative_path": classified["relative_path"],
                "relative_path_from_raw": relative.as_posix(),
                "filename": path.name,
                "base_identifier": path.stem,
                "extension": extension,
                "size_bytes": path.stat().st_size,
                "sha256": "",
                "zero_byte": bool_text(path.stat().st_size == 0),
                "official_distribution_status": "unknown",
                "training_eligibility_status": "unknown",
                "training_eligibility_reason": "",
                "json_encoding": "not_applicable",
                "json_parse_status": "not_applicable",
                "json_error": "",
                "wav_open_success": "not_applicable",
                "wav_channels": "",
                "wav_sample_rate": "",
                "wav_frame_count": "",
                "wav_error": "",
            }
        )
    rows.sort(key=lambda row: row["relative_path_from_raw"].casefold())
    print(f"Local data entries enumerated: {len(rows):,}")
    return rows


def decode_json_bytes(data: bytes) -> tuple[str | None, str, str]:
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig"), "utf-8-sig", ""
        except UnicodeDecodeError as exc:
            return None, "utf-8-sig", f"UnicodeDecodeError: {exc}"
    try:
        return data.decode("utf-8"), "utf-8", ""
    except UnicodeDecodeError as utf8_error:
        try:
            from charset_normalizer import from_bytes  # type: ignore

            best = from_bytes(data).best()
            if best is None or best.encoding is None:
                return None, "unknown", f"UnicodeDecodeError: {utf8_error}"
            return str(best), best.encoding, ""
        except Exception as detect_error:  # pragma: no cover - environment dependent
            return (
                None,
                "unknown",
                f"UnicodeDecodeError: {utf8_error}; detection_error: {detect_error}",
            )


def inspect_local_file(row: dict[str, Any]) -> dict[str, Any]:
    path: Path = row["_path"]
    extension = row["extension"]
    result: dict[str, Any] = {}

    if extension == ".json":
        try:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            text, encoding, decode_error = decode_json_bytes(data)
            result["sha256"] = digest
            result["json_encoding"] = encoding
            if text is None:
                result["json_parse_status"] = "decode_error"
                result["json_error"] = decode_error
            else:
                try:
                    json.loads(text)
                    result["json_parse_status"] = "success"
                    result["json_error"] = ""
                except json.JSONDecodeError as exc:
                    result["json_parse_status"] = "parse_error"
                    result["json_error"] = f"JSONDecodeError: {exc}"
        except Exception as exc:
            result["sha256"] = ""
            result["json_encoding"] = "unreadable"
            result["json_parse_status"] = "read_error"
            result["json_error"] = f"{type(exc).__name__}: {exc}"
        return result

    if extension == ".wav":
        try:
            with path.open("rb") as stream:
                header = stream.read(12)
                stream.seek(0)
                try:
                    with wave.open(stream, "rb") as wav_file:
                        channels = wav_file.getnchannels()
                        sample_rate = wav_file.getframerate()
                        frame_count = wav_file.getnframes()
                    valid = channels > 0 and sample_rate > 0 and frame_count > 0
                    result["wav_open_success"] = bool_text(valid)
                    result["wav_channels"] = channels
                    result["wav_sample_rate"] = sample_rate
                    result["wav_frame_count"] = frame_count
                    result["wav_error"] = "" if valid else "invalid_nonpositive_parameters"
                except wave.Error as exc:
                    riff_wave = header[:4] == b"RIFF" and header[8:12] == b"WAVE"
                    prefix = "unsupported_codec_or_wave_parser" if riff_wave else "invalid_wav_header"
                    result["wav_open_success"] = "false"
                    result["wav_error"] = f"{prefix}: {exc}"
        except Exception as exc:
            result["wav_open_success"] = "false"
            result["wav_error"] = f"{type(exc).__name__}: {exc}"

        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            result["sha256"] = digest.hexdigest()
        except Exception as exc:
            result["sha256"] = ""
            existing = result.get("wav_error", "")
            result["wav_error"] = "; ".join(
                item for item in (existing, f"hash_error: {type(exc).__name__}: {exc}") if item
            )
        return result

    return result


def inspect_local_files(rows: list[dict[str, Any]], workers: int) -> None:
    print(f"Hashing and validating local files with {workers} worker(s)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(inspect_local_file, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                row.update(future.result())
            except Exception as exc:  # defensive: retain the manifest row
                row["sha256"] = ""
                if row["extension"] == ".json":
                    row["json_parse_status"] = "worker_error"
                    row["json_error"] = f"{type(exc).__name__}: {exc}"
                elif row["extension"] == ".wav":
                    row["wav_open_success"] = "false"
                    row["wav_error"] = f"worker_error: {type(exc).__name__}: {exc}"
            completed += 1
            if completed % 2000 == 0 or completed == len(rows):
                print(f"  inspected {completed:,}/{len(rows):,}")


def comparison_key(row: dict[str, Any]) -> str:
    return str(row["relative_path"]).casefold()


def apply_statuses(
    local_rows: list[dict[str, Any]], remote_rows: list[dict[str, Any]]
) -> None:
    remote_keys = {comparison_key(row) for row in remote_rows}
    for row in local_rows:
        official = (
            "current_remote_candidate"
            if comparison_key(row) in remote_keys
            else "local_only_unverified"
        )
        row["official_distribution_status"] = official
        eligibility, reason = training_status(
            row["split"], row["region"], row["region_status"], official
        )
        row["training_eligibility_status"] = eligibility
        row["training_eligibility_reason"] = reason


def build_comparison(
    local_rows: list[dict[str, Any]], remote_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    local_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    remote_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local_rows:
        local_by_key[comparison_key(row)].append(row)
    for row in remote_rows:
        remote_by_key[comparison_key(row)].append(row)

    result: list[dict[str, Any]] = []
    for key in sorted(set(local_by_key) | set(remote_by_key)):
        local_members = local_by_key.get(key, [])
        remote_members = remote_by_key.get(key, [])
        representative = (remote_members or local_members)[0]
        if remote_members and local_members:
            status = "matched"
        elif remote_members:
            status = "remote_only"
        else:
            status = "local_only"
        result.append(
            {
                "comparison_key": key,
                "split": representative["split"],
                "region": representative["region"],
                "relative_path": representative["relative_path"],
                "filename": representative["filename"],
                "remote_present": bool_text(bool(remote_members)),
                "local_present": bool_text(bool(local_members)),
                "comparison_status": status,
                "download_chunks": "|".join(
                    sorted({row["download_chunk"] for row in local_members if row["download_chunk"]})
                ),
            }
        )
    return result


def build_pair_audit(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in local_rows:
        groups[(row["split"], row["base_identifier"].casefold())].append(row)

    result: list[dict[str, Any]] = []
    for (split, _), members in sorted(groups.items()):
        wavs = [row for row in members if row["extension"] == ".wav"]
        jsons = [row for row in members if row["extension"] == ".json"]
        statuses: list[str] = []
        if not wavs:
            statuses.append("json_without_wav")
        if not jsons:
            statuses.append("wav_without_json")
        if len(wavs) > 1:
            statuses.append("multiple_wav")
        if len(jsons) > 1:
            statuses.append("multiple_json")
        if len(wavs) == 1 and len(jsons) == 1:
            statuses.append("matched")
        result.append(
            {
                "split": split,
                "region": "|".join(sorted({row["region"] for row in members if row["region"]})),
                "base_identifier": members[0]["base_identifier"],
                "wav_count": len(wavs),
                "json_count": len(jsons),
                "wav_paths": "|".join(row["relative_path_from_raw"] for row in wavs),
                "json_paths": "|".join(row["relative_path_from_raw"] for row in jsons),
                "pair_status": "|".join(statuses),
            }
        )
    return result


def duplicate_row(
    kind: str, group_id: str, members: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "duplicate_kind": kind,
        "group_id": group_id,
        "split": "|".join(sorted({row["split"] for row in members})),
        "filename": "|".join(sorted({row["filename"] for row in members})),
        "relative_path": "|".join(sorted({row["relative_path"] for row in members})),
        "sha256": "|".join(sorted({row["sha256"] for row in members if row["sha256"]})),
        "source_chunk_count": len({row["download_chunk"] for row in members if row["download_chunk"]}),
        "source_chunks": "|".join(
            sorted({row["download_chunk"] for row in members if row["download_chunk"]})
        ),
        "member_count": len(members),
        "member_paths": "|".join(row["relative_path_from_raw"] for row in members),
    }


def build_duplicates(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sequence = 0

    filename_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    path_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local_rows:
        filename_groups[
            (row["split"], row["extension"], row["filename"].casefold())
        ].append(row)
        path_groups[row["relative_path"].casefold()].append(row)
        if row["sha256"]:
            hash_groups[row["sha256"]].append(row)

    for members in filename_groups.values():
        if len(members) > 1:
            sequence += 1
            output.append(duplicate_row("duplicate_filename", f"dup-{sequence:06d}", members))
    for members in path_groups.values():
        if len(members) > 1:
            sequence += 1
            output.append(duplicate_row("duplicate_relative_path", f"dup-{sequence:06d}", members))
    for members in hash_groups.values():
        if len(members) > 1 and len({row["filename"].casefold() for row in members}) > 1:
            sequence += 1
            output.append(
                duplicate_row(
                    "duplicate_content_different_filename", f"dup-{sequence:06d}", members
                )
            )
    output.sort(key=lambda row: (row["duplicate_kind"], row["group_id"]))
    return output


def build_integrity_issues(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in local_rows:
        candidates: list[tuple[str, str, str]] = []
        if row["zero_byte"] == "true":
            candidates.append(("zero_byte", "error", "File size is zero."))
        if row["extension"] == ".json" and row["json_parse_status"] != "success":
            candidates.append(("json_integrity", "error", row["json_error"]))
        if row["extension"] == ".wav" and row["wav_open_success"] != "true":
            severity = "warning" if str(row["wav_error"]).startswith("unsupported_codec") else "error"
            candidates.append(("wav_integrity", severity, row["wav_error"]))
        if not row["sha256"]:
            candidates.append(("sha256_missing", "error", "SHA-256 could not be calculated."))
        if row["region_status"] != "mapped_from_path":
            candidates.append(("unmapped_region", "warning", "Region folder pattern did not match."))
        for issue_type, severity, detail in candidates:
            issues.append(
                {
                    "issue_type": issue_type,
                    "severity": severity,
                    "split": row["split"],
                    "region": row["region"],
                    "relative_path": row["relative_path_from_raw"],
                    "download_chunk": row["download_chunk"],
                    "detail": detail,
                }
            )
    return issues


def nested_counts(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    counts = Counter(tuple(str(row.get(field, "")) for field in fields) for row in rows)
    output: list[dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        item = {field: value for field, value in zip(fields, key)}
        item["count"] = count
        output.append(item)
    return output


def build_summary(
    snapshot_at: str,
    drive_url: str,
    drive_folder_id: str,
    local_rows: list[dict[str, Any]],
    remote_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    local_only_keys = {
        row["comparison_key"]
        for row in comparison_rows
        if row["comparison_status"] == "local_only"
    }
    local_only_rows = [row for row in local_rows if comparison_key(row) in local_only_keys]
    comparison_counts = Counter(row["comparison_status"] for row in comparison_rows)
    pair_counts = Counter(row["pair_status"] for row in pair_rows)
    duplicate_counts = Counter(row["duplicate_kind"] for row in duplicate_rows)
    integrity_counts = Counter(row["issue_type"] for row in integrity_rows)

    return {
        "manifest_version": MANIFEST_VERSION,
        "snapshot_at": snapshot_at,
        "snapshot_note": (
            "Current Google Drive state only; not a guaranteed historical snapshot of the "
            "2026-08-23 download."
        ),
        "drive_url": drive_url,
        "drive_folder_id": drive_folder_id,
        "remote_file_count": len(remote_rows),
        "local_file_count": len(local_rows),
        "remote_counts": nested_counts(remote_rows, ("split", "region", "extension")),
        "local_counts": nested_counts(local_rows, ("split", "region", "extension")),
        "local_counts_by_chunk": nested_counts(
            local_rows, ("download_chunk", "split", "region", "extension")
        ),
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "local_only_counts": nested_counts(local_only_rows, ("split", "region", "extension")),
        "local_only_by_chunk": nested_counts(
            local_only_rows, ("download_chunk", "split", "region", "extension")
        ),
        "pair_status_counts": dict(sorted(pair_counts.items())),
        "duplicate_group_counts": dict(sorted(duplicate_counts.items())),
        "integrity_issue_counts": dict(sorted(integrity_counts.items())),
        "zero_byte_count": sum(row["zero_byte"] == "true" for row in local_rows),
        "json_parse_success_count": sum(
            row["extension"] == ".json" and row["json_parse_status"] == "success"
            for row in local_rows
        ),
        "json_parse_failure_count": sum(
            row["extension"] == ".json" and row["json_parse_status"] != "success"
            for row in local_rows
        ),
        "wav_header_success_count": sum(
            row["extension"] == ".wav" and row["wav_open_success"] == "true"
            for row in local_rows
        ),
        "wav_header_failure_count": sum(
            row["extension"] == ".wav" and row["wav_open_success"] != "true"
            for row in local_rows
        ),
        "sha256_missing_count": sum(not row["sha256"] for row in local_rows),
        "official_distribution_status_counts": dict(
            sorted(Counter(row["official_distribution_status"] for row in local_rows).items())
        ),
        "training_eligibility_status_counts": dict(
            sorted(Counter(row["training_eligibility_status"] for row in local_rows).items())
        ),
        "structure_change_gate": {
            "passed": False,
            "reason": "Organizer confirmation or an official manifest is still required.",
        },
    }


def markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "_없음_"
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def organizer_message() -> str:
    return """안녕하세요.

현재 제공된 Google Drive 공유 폴더를 확인하면 다음과 같습니다.

- Training: WAV 5,500개, JSON 5,500개
- Validation: WAV 3,640개, JSON 3,640개

그러나 2026-08-23에 동일 공유 Drive의 ‘대학부 데이터’ 전체 폴더를
다운로드한 로컬 사본에는 다음 파일이 있습니다.

- Training: WAV 29,200개, JSON 29,200개
- Validation: WAV 3,640개, JSON 3,640개

현재 원격 파일 중 로컬에 없는 파일은 확인되지 않았으며,
Training에서만 로컬에 WAV/JSON 23,700쌍이 추가로 존재합니다.
로컬 데이터는 Google Drive 전체 폴더 다운로드 과정에서
‘대학부 데이터-20260823T080055Z-1-001’부터 ‘-012’까지의
여러 다운로드 조각으로 나뉘어 저장됐습니다.

아래 내용을 확인 부탁드립니다.

1. 공식적으로 배포된 Training 데이터의 전체 수량은 몇 쌍인가요?
2. 2026-08-23 전후 공유 Drive의 Training 폴더 구성 변경이 있었나요?
3. 29,200쌍이 당시 공식 배포본이었다면 전체를 공식 배포 원본으로
   보존해야 하는지 확인 부탁드립니다.
4. 29,200쌍 안에 서울 외 지역 데이터가 포함되어 있다면 해당 데이터도
   공식 배포 원본에는 포함되지만 실제 모델 학습에는 사용할 수 없는
   데이터로 이해하면 될까요?
5. 실제 모델 Training에 사용할 수 있는 범위는 문제 안내대로
   서울 Training 데이터만인지 다시 확인 부탁드립니다.
6. 가능하다면 공식 파일 manifest 또는 파일명·크기·checksum 목록을
   제공해 주실 수 있을까요?

감사합니다."""


def build_report(summary: dict[str, Any]) -> str:
    comparison = summary["comparison_counts"]
    duplicates = summary["duplicate_group_counts"]
    pairs = summary["pair_status_counts"]
    integrity = summary["integrity_issue_counts"]
    return f"""# DCC 데이터 원격·로컬 무결성 감사 보고서

## 감사 범위

- Snapshot 시각(UTC): `{summary['snapshot_at']}`
- Google Drive: `{summary['drive_url']}`
- 원격 조회 방식: `gdown.download_folder(..., skip_download=True)`
- 원격 파일 본문 다운로드: 없음
- 로컬 원본 파일 변경: 없음
- 주의: 이 원격 manifest는 현재 Drive 상태이며 2026-08-23 당시의 역사적 snapshot을 보장하지 않는다.

## 핵심 결과

- 원격 파일: {summary['remote_file_count']:,}개
- 로컬 파일: {summary['local_file_count']:,}개
- matched: {comparison.get('matched', 0):,}개
- remote-only: {comparison.get('remote_only', 0):,}개
- local-only: {comparison.get('local_only', 0):,}개
- 0-byte: {summary['zero_byte_count']:,}개
- JSON parse 성공/실패: {summary['json_parse_success_count']:,} / {summary['json_parse_failure_count']:,}
- WAV header 성공/실패: {summary['wav_header_success_count']:,} / {summary['wav_header_failure_count']:,}
- SHA-256 미생성: {summary['sha256_missing_count']:,}개

### 원격 split·지역·형식별 수량

{markdown_table(summary['remote_counts'], ('split', 'region', 'extension', 'count'))}

### 로컬 split·지역·형식별 수량

{markdown_table(summary['local_counts'], ('split', 'region', 'extension', 'count'))}

### Local-only 분포

{markdown_table(summary['local_only_counts'], ('split', 'region', 'extension', 'count'))}

### Local-only 다운로드 조각별 분포

{markdown_table(summary['local_only_by_chunk'], ('download_chunk', 'split', 'region', 'extension', 'count'))}

> WAV와 대응 JSON이 서로 다른 다운로드 조각에 존재할 수 있으므로 위 조각별 수량은 파일 단위 집계다.

## Pairing

```json
{json.dumps(pairs, ensure_ascii=False, indent=2)}
```

## 중복 그룹

```json
{json.dumps(duplicates, ensure_ascii=False, indent=2)}
```

내용 중복은 삭제 근거로 사용하지 않는다.

## 무결성 이슈

```json
{json.dumps(integrity, ensure_ascii=False, indent=2)}
```

WAV 표준 라이브러리가 지원하지 않는 codec은 손상으로 자동 판정하지 않는다.

## 공식 배포와 학습 허용 상태

- `official_distribution_status`와 `training_eligibility_status`는 독립적으로 기록했다.
- 현재 원격 일치 파일은 `current_remote_candidate`이며 주최측 확인 전까지 공식 확정 상태가 아니다.
- 로컬 전용 파일은 `local_only_unverified`이며 삭제·격리·이동 대상이라고 판정하지 않는다.
- Validation은 학습 불가로 표시한다.
- 서울 Training도 공식 배포 범위가 확인되기 전까지 `pending_official_distribution`이다.

## 구조 변경 게이트

- [ ] 공식 Training 범위에 대한 주최측 확인 또는 공식 manifest 확보
- [x] 공식 배포 상태와 학습 허용 상태를 별도 컬럼으로 기록
- [x] split·지역·하위 경로·다운로드 조각별 집계
- [x] WAV-JSON exact stem pairing 감사
- [x] 파일명·상대 경로 중복 감사
- [x] SHA-256 내용 중복 감사
- [x] 0-byte 검사
- [x] JSON 인코딩·parse 검사
- [x] WAV header 검사
- [{'x' if comparison.get('remote_only', 0) == 0 else ' '}] 현재 원격 manifest의 모든 파일이 로컬에 존재
- [x] local-only의 경로·지역·조각별 분포 기록
- [x] 로컬 SHA-256 및 provenance manifest 저장

**현재 구조 변경 게이트: 통과하지 않음.** 주최측 확인 또는 공식 manifest가 필요하다.

## 주최측 문의 문안

```text
{organizer_message()}
```

## 답변별 후속 처리

```text
주최측 답변 또는 공식 manifest 확보
├─ Case A: 29,200쌍 전체가 공식 배포본
│  ├─ 전체 원본 보존
│  ├─ 서울 Training만 학습 적격 판정
│  └─ 서울 외 데이터는 보존하되 학습 경로 차단
├─ Case B: 공식 Training이 5,500쌍
│  ├─ 현재 원격 일치 항목을 official candidate로 유지
│  └─ local-only 23,700쌍은 이동·삭제 없이 unverified 유지
├─ Case C: 공식 manifest/checksum 제공
│  ├─ 공식 목록을 최우선 기준으로 재감사
│  └─ 누락·불일치 항목을 덮어쓰지 않고 별도 보고
└─ Case D: 수량 확답 없음
   ├─ 001~012 원본과 양쪽 snapshot manifest 유지
   └─ 불명확한 데이터는 학습 경로에서 차단
```
"""


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    local_rows: list[dict[str, Any]],
    remote_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.audit-tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        write_csv(temporary / "local_manifest.csv", local_rows, LOCAL_FIELDS)
        write_csv(temporary / "remote_manifest.csv", remote_rows, REMOTE_FIELDS)
        write_csv(temporary / "comparison.csv", comparison_rows, COMPARISON_FIELDS)
        write_csv(temporary / "pair_audit.csv", pair_rows, PAIR_FIELDS)
        write_csv(temporary / "duplicate_groups.csv", duplicate_rows, DUPLICATE_FIELDS)
        write_csv(temporary / "integrity_issues.csv", integrity_rows, INTEGRITY_FIELDS)
        (temporary / "audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "audit_report.md").write_text(build_report(summary), encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    local_root, output_dir = inspect_environment(args)
    snapshot_at = utc_timestamp()

    remote_rows, drive_folder_id = enumerate_remote(args.drive_url, snapshot_at)
    local_rows = enumerate_local(local_root, snapshot_at)
    inspect_local_files(local_rows, args.workers)
    apply_statuses(local_rows, remote_rows)

    comparison_rows = build_comparison(local_rows, remote_rows)
    pair_rows = build_pair_audit(local_rows)
    duplicate_rows = build_duplicates(local_rows)
    integrity_rows = build_integrity_issues(local_rows)
    summary = build_summary(
        snapshot_at,
        args.drive_url,
        drive_folder_id,
        local_rows,
        remote_rows,
        comparison_rows,
        pair_rows,
        duplicate_rows,
        integrity_rows,
    )

    for row in local_rows:
        row.pop("_path", None)
    write_outputs(
        output_dir,
        local_rows,
        remote_rows,
        comparison_rows,
        pair_rows,
        duplicate_rows,
        integrity_rows,
        summary,
    )
    print(f"Audit completed. Results: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Audit interrupted; source data was not modified.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
