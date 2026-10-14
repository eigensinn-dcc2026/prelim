"""manifest(데이터 위치 목록) 만들기·저장·불러오기. 라벨이나 대사 내용 없이 WAV/JSON 의 위치만 담는다.

같은 통화의 WAV 와 JSON 을 (분할, 파일 이름) 으로 짝짓는다. 내려받은 조각(zip)의 경계는 일부러 무시한다.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .guards import (
    DuplicateIdentifierError,
    ManifestPairingError,
    split_from_path,
    validate_manifest_rows,
)


# manifest CSV 형식 버전과 컬럼
MANIFEST_VERSION = "1"
MANIFEST_COLUMNS = (
    "manifest_version",
    "split",
    "stem",
    "wav_path",
    "json_path",
)


@dataclass(frozen=True)
class ManifestRow:
    """통화 하나: 분할(Training/Validation), 파일 이름(stem), WAV 와 JSON 의 상대 경로."""

    split: str
    stem: str
    wav_path: str
    json_path: str


def build_manifest(data_root: str | Path) -> list[ManifestRow]:
    """``data_root`` 아래의 모든 WAV/JSON 을 ``(분할, 파일 이름)`` 으로 짝짓는다.

    경로는 ``data_root`` 기준 상대 경로(POSIX 형식)다. 내려받은 조각의 경계는 짝짓기에서 일부러 무시한다.
    같은 이름이 두 번 나오거나 짝이 없으면 멈추고, 마지막에 guards.validate_manifest_rows 로 한 번 더 검사한다.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    groups: dict[tuple[str, str], dict[str, object]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".json"}:
            continue
        relative = path.relative_to(root)
        split = split_from_path(relative)
        key = (split, path.stem.casefold())
        group = groups.setdefault(
            key,
            {"stem": path.stem, "wav": [], "json": []},
        )
        role = path.suffix.lower().lstrip(".")
        role_paths = group[role]
        assert isinstance(role_paths, list)
        role_paths.append(relative.as_posix())

    rows: list[ManifestRow] = []
    for (split, _), group in sorted(groups.items()):
        wav_paths = group["wav"]
        json_paths = group["json"]
        assert isinstance(wav_paths, list) and isinstance(json_paths, list)
        if len(wav_paths) > 1 or len(json_paths) > 1:
            raise DuplicateIdentifierError(
                f"Duplicate files for {(split, group['stem'])}: "
                f"wav={wav_paths}, json={json_paths}"
            )
        if len(wav_paths) != 1 or len(json_paths) != 1:
            raise ManifestPairingError(
                f"Unpaired files for {(split, group['stem'])}: "
                f"wav={wav_paths}, json={json_paths}"
            )
        rows.append(
            ManifestRow(
                split=split,
                stem=str(group["stem"]),
                wav_path=wav_paths[0],
                json_path=json_paths[0],
            )
        )

    validate_manifest_rows(rows, root)
    return rows


def write_manifest(
    rows: list[ManifestRow], output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """라벨과 대사 없이 위치만 담은 CSV 를 쓴다(항상 같은 순서). 임시 파일에 다 쓴 뒤 이름을 바꾼다."""

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "manifest_version": MANIFEST_VERSION,
                        "split": row.split,
                        "stem": row.stem,
                        "wav_path": row.wav_path,
                        "json_path": row.json_path,
                    }
                )
        os.replace(temporary_name, output)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def load_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    validate_paths: bool = True,
) -> list[ManifestRow]:
    """manifest CSV 를 불러오고 안전장치 검사를 다시 한다(``data_root`` 를 주면 파일 존재까지 확인)."""

    path = Path(manifest_path)
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"Unexpected manifest columns: {reader.fieldnames}; "
                f"expected {MANIFEST_COLUMNS}"
            )
        for line_number, item in enumerate(reader, start=2):
            if item["manifest_version"] != MANIFEST_VERSION:
                raise ValueError(
                    f"Unsupported manifest version on line {line_number}: "
                    f"{item['manifest_version']!r}"
                )
            rows.append(
                ManifestRow(
                    split=item["split"],
                    stem=item["stem"],
                    wav_path=item["wav_path"],
                    json_path=item["json_path"],
                )
            )

    validation_root = data_root if validate_paths else None
    if validate_paths and validation_root is None:
        raise ValueError("data_root is required when validate_paths=True")
    validate_manifest_rows(rows, validation_root)
    return rows
