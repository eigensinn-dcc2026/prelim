"""목록 무결성과 split 분리를 지키는, 즉시 실패(fail-fast) 검사."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


TRAINING = "Training"
VALIDATION = "Validation"
VALID_SPLITS = frozenset({TRAINING, VALIDATION})


class DataGuardError(ValueError):
    """DCC 데이터 검사 실패의 기본 클래스."""


class UnknownSplitError(DataGuardError):
    """경로에 인식되는 split이 정확히 하나 들어 있지 않을 때 발생한다."""


class ManifestPairingError(DataGuardError):
    """stem 하나에 WAV와 JSON이 정확히 하나씩 없을 때 발생한다."""


class DuplicateIdentifierError(DataGuardError):
    """한 split 안에 식별자가 중복될 때 발생한다."""


class CrossSplitIdentifierError(DataGuardError):
    """같은 식별자가 Training과 Validation 양쪽에 나타날 때 발생한다."""


class SplitContaminationError(DataGuardError):
    """예상하지 못한 split의 데이터가 검사 대상 경로에 들어왔을 때 발생한다."""


def split_from_path(path: str | Path) -> str:
    """상대 경로나 절대 경로에서 split 구성 요소를 정확히 돌려준다."""

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
    """유일성, 경로와 split의 일치, (선택) 파일 존재 여부를 검사한다."""

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
    """행을 모두 꺼내 보고, 다른 split의 행이 하나라도 있으면 실패한다."""

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
    """모든 행이 Training에 속할 때만 행을 돌려준다."""

    return ensure_split(rows, TRAINING)


def ensure_training_batch(batch: Mapping[str, Any]) -> None:
    """배치에 Training이 아닌 출처의 split가 섞여 있으면 즉시 실패한다."""

    if "split" not in batch:
        raise SplitContaminationError("Training batch has no split provenance")
    value = batch["split"]
    splits = [value] if isinstance(value, str) else list(value)
    foreign = [split for split in splits if split != TRAINING]
    if foreign:
        raise SplitContaminationError(
            f"Training batch contains non-Training split(s): {foreign[:10]}"
        )
