"""데이터 안전장치: manifest 가 올바른지, Training 과 Validation 이 섞이지 않았는지 바로 검사하고 문제가 있으면 멈춘다.

대회 규정상 Validation 은 학습(역전파)에 쓸 수 없다. 학습 경로로 들어가는 데이터는 이 검사들을 거친다.

(제출 폴더 정리 메모: 쓰이지 않는 배치 검사 함수 ``ensure_training_batch`` 는 뺐다.)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


TRAINING = "Training"
VALIDATION = "Validation"
VALID_SPLITS = frozenset({TRAINING, VALIDATION})


class DataGuardError(ValueError):
    """데이터 안전장치 오류의 기본 클래스."""


class UnknownSplitError(DataGuardError):
    """경로에 Training/Validation 이 정확히 하나 들어 있지 않을 때."""


class ManifestPairingError(DataGuardError):
    """같은 이름(stem)에 WAV 하나와 JSON 하나가 정확히 짝지어지지 않을 때."""


class DuplicateIdentifierError(DataGuardError):
    """한 분할 안에 같은 이름이 두 번 나올 때."""


class CrossSplitIdentifierError(DataGuardError):
    """같은 이름이 Training 과 Validation 양쪽에 나올 때."""


class SplitContaminationError(DataGuardError):
    """기대하지 않은 분할의 데이터가 들어왔을 때 (예: 학습 경로에 Validation)."""


def split_from_path(path: str | Path) -> str:
    """경로에서 Training/Validation 폴더 이름을 정확히 하나 찾아 돌려준다. 없거나 둘 이상이면 멈춘다."""

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
    """manifest 행들을 검사한다: 이름 중복, 경로와 분할의 일치, 확장자, (data_root 를 주면) 파일 존재, 분할 간 이름 겹침."""

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
    """행들을 목록으로 만들고, 하나라도 다른 분할이면 멈춘다."""

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
    """모든 행이 Training 일 때만 돌려준다 (학습 데이터 입구의 검사)."""

    return ensure_split(rows, TRAINING)
