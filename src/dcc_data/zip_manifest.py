"""Locate wav/json pairs inside zip-chunked AI-Hub downloads without
extracting everything.

The team's dataset was downloaded from Google Drive as several separate zip
chunks; a wav file and its paired json can land in *different* chunks --
this mirrors ``dcc_data.manifest.build_manifest``, which makes the same
assumption for an already-extracted directory tree ("Download shard
boundaries are deliberately ignored during pairing"). This module does the
equivalent pairing directly against zip *entry names* (reading only each
zip's central directory, never file content) so a teammate low on local
disk space can extract just the files one baseline run actually needs --
e.g. a 3,000-call Training sample -- instead of the entire distribution.

Region/split detection reuses the exact same rules as the rest of the
pipeline (``guards.split_from_path``, ``mission2_features.is_seoul_path``)
since both operate on path strings and need no real file on disk.
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .guards import UnknownSplitError, split_from_path
from .mission2_features import is_seoul_path


@dataclass(frozen=True)
class ZipEntryLocation:
    zip_path: Path
    entry_name: str


@dataclass(frozen=True)
class ZipCallEntry:
    split: str
    stem: str
    wav: ZipEntryLocation
    json: ZipEntryLocation


def index_zip_entries(zip_paths: list[Path]) -> list[ZipCallEntry]:
    """Scan every zip's entry list and pair wav<->json by filename stem.

    An entry whose path has no exactly-one Training/Validation component
    (directory entries, stray non-data files) is silently skipped rather
    than raising, since zips routinely contain such entries.
    """

    groups: dict[tuple[str, str], dict[str, ZipEntryLocation]] = {}
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for entry_name in archive.namelist():
                path = PurePosixPath(entry_name)
                suffix = path.suffix.lower()
                if suffix not in (".wav", ".json"):
                    continue
                try:
                    split = split_from_path(entry_name)
                except UnknownSplitError:
                    continue
                key = (split, path.stem)
                group = groups.setdefault(key, {})
                role = "wav" if suffix == ".wav" else "json"
                group[role] = ZipEntryLocation(zip_path, entry_name)

    entries = [
        ZipCallEntry(split=split, stem=stem, wav=group["wav"], json=group["json"])
        for (split, stem), group in groups.items()
        if "wav" in group and "json" in group
    ]
    return entries


def select_calls(
    entries: list[ZipCallEntry], *, split: str, seoul_only: bool, n_calls: int, seed: int
) -> list[ZipCallEntry]:
    """Sample up to ``n_calls`` calls from ``split`` (0/negative = take all)."""

    pool = [entry for entry in entries if entry.split == split]
    if seoul_only:
        pool = [
            entry
            for entry in pool
            if is_seoul_path(entry.wav.entry_name) and is_seoul_path(entry.json.entry_name)
        ]
    pool = sorted(pool, key=lambda entry: entry.stem)  # deterministic order before sampling
    if n_calls <= 0 or n_calls >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n_calls)


def extract_calls(entries: list[ZipCallEntry], output_dir: Path) -> None:
    """Extract exactly the wav+json entries for ``entries``.

    Each file lands at its own entry name's relative path under
    ``output_dir`` -- i.e. whatever Training/Validation/region structure
    the zip itself used -- so the result is a normal (just much smaller)
    data root that build_manifest.py can scan afterward.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    opened: dict[Path, zipfile.ZipFile] = {}
    try:
        for call in entries:
            for location in (call.wav, call.json):
                archive = opened.get(location.zip_path)
                if archive is None:
                    archive = zipfile.ZipFile(location.zip_path)
                    opened[location.zip_path] = archive
                archive.extract(location.entry_name, path=output_dir)
    finally:
        for archive in opened.values():
            archive.close()
