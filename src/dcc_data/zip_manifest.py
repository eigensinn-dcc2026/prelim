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
import shutil
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


def extract_all_entries(
    zip_paths: list[Path],
    output_dir: Path,
    *,
    split: str,
    seoul_only: bool,
    skip_existing: bool = True,
) -> int:
    """Extract every wav/json entry matching ``split``/``seoul_only`` from
    every zip, independent of whether its wav<->json partner also lives
    inside one of these same zips.

    ``select_calls`` + ``extract_calls`` require both halves of a call to be
    paired *within the scanned zip set itself* -- which silently drops a
    call whenever one half was already extracted separately outside any zip
    (e.g. a chunk a teammate unzipped by hand before this tool existed, or a
    Drive "download folder as zip" split that put a call's audio and label
    in different numbered chunks) while its other half still lives inside a
    zip. Real-world example that surfaced this: a call's wav sat inside zip
    chunk -012 while its json had already been extracted loose from a
    different chunk -- ``index_zip_entries`` never saw both halves at once,
    so the wav was silently never extracted.

    Manifest-time pairing across the *final* combined output_dir (whatever
    this extracts, plus anything already there from elsewhere) is what
    actually determines completeness, so this function does not try to pair
    anything itself -- it just extracts every entry whose path passes the
    split/region filter, whether or not a partner is visible from the zip
    set alone. Returns the number of files actually extracted (excludes
    already-present ones when ``skip_existing``).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for entry_name in archive.namelist():
                path = PurePosixPath(entry_name)
                if path.suffix.lower() not in (".wav", ".json"):
                    continue
                try:
                    entry_split = split_from_path(entry_name)
                except UnknownSplitError:
                    continue
                if entry_split != split:
                    continue
                if seoul_only and not is_seoul_path(entry_name):
                    continue
                # Drop everything before the Training/Validation component --
                # a Drive "download folder as zip" split wraps entries in the
                # source folder's own name (e.g. "대학부 데이터/Training/..."),
                # and different zips can use different wrapper names. Without
                # stripping it, the same call extracted via a different code
                # path (e.g. the already-merged loose chunk) lands at a
                # different depth than this one, so build_manifest's pairing
                # still finds both eventually, but a second extraction run can
                # end up writing the *same* entry a second time at the
                # wrapper-prefixed path even though it already exists at the
                # clean one -- silently doubling disk usage and risking a
                # DuplicateIdentifierError once both copies are visible.
                split_index = path.parts.index(entry_split)
                relative = PurePosixPath(*path.parts[split_index:])
                target = output_dir / relative
                if skip_existing and target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry_name) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                extracted_count += 1
    return extracted_count


def _call_already_extracted(call: ZipCallEntry, output_dir: Path) -> bool:
    return (output_dir / call.wav.entry_name).is_file() and (output_dir / call.json.entry_name).is_file()


def extract_calls(entries: list[ZipCallEntry], output_dir: Path, *, skip_existing: bool = True) -> int:
    """Extract exactly the wav+json entries for ``entries``.

    Each file lands at its own entry name's relative path under
    ``output_dir`` -- i.e. whatever Training/Validation/region structure
    the zip itself used -- so the result is a normal (just much smaller)
    data root that build_manifest.py can scan afterward.

    ``skip_existing`` (default True) makes this resumable: a call whose wav
    AND json are both already present at their target path is left alone
    instead of re-extracted. Re-running the exact same selection after an
    interrupted run (e.g. a Colab disconnect partway through a multi-GB
    extraction) then only fetches whatever is still missing, rather than
    starting over from zero. Returns the number of calls actually extracted
    (excludes skipped ones).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    opened: dict[Path, zipfile.ZipFile] = {}
    extracted_count = 0
    try:
        for call in entries:
            if skip_existing and _call_already_extracted(call, output_dir):
                continue
            for location in (call.wav, call.json):
                archive = opened.get(location.zip_path)
                if archive is None:
                    archive = zipfile.ZipFile(location.zip_path)
                    opened[location.zip_path] = archive
                archive.extract(location.entry_name, path=output_dir)
            extracted_count += 1
    finally:
        for archive in opened.values():
            archive.close()
    return extracted_count
