"""Fail-fast guards for manifest integrity and split isolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


TRAINING = "Training"
VALIDATION = "Validation"
VALID_SPLITS = frozenset({TRAINING, VALIDATION})


class DataGuardError(ValueError):
    """Base class for DCC data guard failures."""


class UnknownSplitError(DataGuardError):
    """Raised when a path does not contain exactly one recognized split."""


class ManifestPairingError(DataGuardError):
    """Raised when a stem does not have exactly one WAV and one JSON file."""


class DuplicateIdentifierError(DataGuardError):
    """Raised when a split contains a duplicate identifier."""


class CrossSplitIdentifierError(DataGuardError):
    """Raised when an identifier appears in both Training and Validation."""


class SplitContaminationError(DataGuardError):
    """Raised when data from an unexpected split reaches a guarded path."""


def split_from_path(path: str | Path) -> str:
    """Return the exact split component from a relative or absolute path."""

    candidates = [part for part in Path(path).parts if part in VALID_SPLITS]
    if len(candidates) != 1:
        raise UnknownSplitError(
            f"Expected exactly one Training/Validation path component, got "
            f"{candidates!r}: {path}"
        )
    return candidates[0]


def validate_manifest_rows(
    rows: Iterable[Any], data_root: str | Path | None = None
) -> None:
    """Validate uniqueness, path/split agreement, and optional file existence."""

    materialized = list(rows)
    root = Path(data_root).resolve() if data_root is not None else None
    seen: set[tuple[str, str]] = set()
    stems_by_split = {TRAINING: set(), VALIDATION: set()}

    for row in materialized:
        if row.split not in VALID_SPLITS:
            raise UnknownSplitError(f"Unknown manifest split: {row.split!r}")

        key = (row.split, row.stem.casefold())
        if key in seen:
            raise DuplicateIdentifierError(f"Duplicate manifest key: {key}")
        seen.add(key)
        stems_by_split[row.split].add(row.stem.casefold())

        for role, relative_path, suffix in (
            ("wav", row.wav_path, ".wav"),
            ("json", row.json_path, ".json"),
        ):
            path = Path(relative_path)
            if path.suffix.lower() != suffix:
                raise ManifestPairingError(
                    f"{role} path has the wrong extension for {key}: {relative_path}"
                )
            actual_split = split_from_path(path)
            if actual_split != row.split:
                raise SplitContaminationError(
                    f"Manifest split {row.split!r} disagrees with {role} path "
                    f"split {actual_split!r}: {relative_path}"
                )
            if root is not None and not (root / path).is_file():
                raise FileNotFoundError(root / path)

    overlap = stems_by_split[TRAINING] & stems_by_split[VALIDATION]
    if overlap:
        preview = sorted(overlap)[:10]
        raise CrossSplitIdentifierError(
            f"{len(overlap)} stem(s) occur in both splits; examples: {preview}"
        )


def ensure_split(rows: Iterable[Any], expected_split: str) -> list[Any]:
    """Materialize rows and fail if any row is from another split."""

    if expected_split not in VALID_SPLITS:
        raise UnknownSplitError(f"Unknown expected split: {expected_split!r}")
    materialized = list(rows)
    wrong = [row for row in materialized if row.split != expected_split]
    if wrong:
        examples = [(row.split, row.stem) for row in wrong[:10]]
        raise SplitContaminationError(
            f"Expected only {expected_split}, found {len(wrong)} foreign row(s): "
            f"{examples}"
        )
    return materialized


def ensure_training_rows(rows: Iterable[Any]) -> list[Any]:
    """Return rows only when every row belongs to Training."""

    return ensure_split(rows, TRAINING)


def ensure_training_batch(batch: Mapping[str, Any]) -> None:
    """Fail immediately if a batch contains a non-Training provenance split."""

    if "split" not in batch:
        raise SplitContaminationError("Training batch has no split provenance")
    value = batch["split"]
    splits = [value] if isinstance(value, str) else list(value)
    foreign = [split for split in splits if split != TRAINING]
    if foreign:
        raise SplitContaminationError(
            f"Training batch contains non-Training split(s): {foreign[:10]}"
        )
