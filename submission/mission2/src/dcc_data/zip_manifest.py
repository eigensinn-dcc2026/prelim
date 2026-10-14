"""대회 원본 데이터(여러 zip 조각) 풀기.

팀 데이터는 Google Drive 에서 여러 zip 조각으로 받았고, 한 통화의 WAV 와 JSON 이 *서로 다른* 조각에 들어 있을 수 있다.

- ``extract_all_entries``("전부 풀기"): 분할·지역 조건에 맞는 WAV/JSON 을 짝과 상관없이 모두 푼다. 짝짓기는 나중에
  build_manifest 가 풀린 전체 폴더를 보고 한다. **최종 학습·검증 노트북은 모두 이 방식(--n-train-calls 0)을 썼다.**
- ``index_zip_entries`` + ``select_calls`` + ``extract_calls``("일부만 뽑기"): zip 안의 파일 목록만 읽어 짝을 찾고 일부 통화만 푼다.
  디스크가 부족할 때를 위한 기능인데, 짝의 한쪽이 zip 밖(이미 풀린 조각)에 있거나 다른 조각에서 짝을 못 찾으면 그 통화를 빠뜨린다
  (실제로 공식 Validation 3,640통화 중 1,541통화만 풀린 적이 있다). 전체 데이터가 필요할 때는 쓰지 않는다.

분할·지역 판단은 다른 코드와 똑같은 규칙(``guards.split_from_path``, ``mission2_features.is_seoul_path``)을 쓴다.
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
    """zip 파일 하나 안의 항목 하나의 위치."""

    zip_path: Path
    entry_name: str


@dataclass(frozen=True)
class ZipCallEntry:
    """같은 zip 묶음 안에서 짝을 찾은 통화 하나 (WAV 위치 + JSON 위치)."""

    split: str
    stem: str
    wav: ZipEntryLocation
    json: ZipEntryLocation


def index_zip_entries(zip_paths: list[Path]) -> list[ZipCallEntry]:
    """("일부만 뽑기"용) 모든 zip 의 항목 목록을 읽어 WAV 와 JSON 을 파일 이름으로 짝짓는다.

    경로에 Training/Validation 이 정확히 하나 없는 항목(폴더 항목, 데이터가 아닌 파일)은 zip 에 흔하므로 오류 없이 건너뛴다.
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
    """("일부만 뽑기"용) ``split`` 에서 최대 ``n_calls`` 통화를 뽑는다(0 이하면 전부). 뽑기 전 이름순 정렬로 재현 가능하게 한다."""

    pool = [entry for entry in entries if entry.split == split]
    if seoul_only:
        pool = [
            entry
            for entry in pool
            if is_seoul_path(entry.wav.entry_name) and is_seoul_path(entry.json.entry_name)
        ]
    pool = sorted(pool, key=lambda entry: entry.stem)  # 뽑기 전에 순서를 고정한다(재현 가능)
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
    """("전부 풀기", 최종 과정이 쓴 방식) 모든 zip 에서 ``split``/``seoul_only`` 조건에 맞는 WAV/JSON 항목을 짝과 상관없이 모두 푼다.

    "일부만 뽑기"는 한 통화의 두 파일이 *읽은 zip 묶음 안에서* 짝지어져야 해서, 한쪽이 이미 따로 풀려 있거나 다른 조각에 있으면
    그 통화를 조용히 빠뜨린다(실제 사례: WAV 는 -012 조각 안에, JSON 은 다른 조각에서 이미 풀려 있어서 WAV 가 끝내 풀리지 않음).

    완전한지는 결국 풀린 전체 폴더로 manifest 를 만들 때 정해지므로, 이 함수는 짝짓기를 하지 않고 조건에 맞는 항목을 모두 푼다.
    실제로 푼 파일 수를 돌려준다(``skip_existing`` 이면 이미 있는 파일은 세지 않음).
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
                # Training/Validation 앞의 경로는 버린다. Drive 의 "폴더를 zip 으로 받기"는 항목 앞에 원래 폴더 이름
                # (예: "대학부 데이터/Training/...")을 붙이고, zip 마다 그 이름이 다를 수 있다. 버리지 않으면 같은 통화가
                # 다른 깊이에 두 번 풀려 디스크를 두 배로 쓰고, 두 사본이 함께 보일 때 DuplicateIdentifierError 가 날 수 있다.
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
    """통화의 WAV 와 JSON 이 둘 다 이미 풀려 있는지."""

    return (output_dir / call.wav.entry_name).is_file() and (output_dir / call.json.entry_name).is_file()


def extract_calls(entries: list[ZipCallEntry], output_dir: Path, *, skip_existing: bool = True) -> int:
    """("일부만 뽑기"용) ``entries`` 의 WAV+JSON 만 정확히 푼다.

    각 파일은 zip 항목 이름 그대로의 상대 경로에 풀리므로, 결과는 build_manifest.py 가 읽을 수 있는 (작은) 데이터 폴더가 된다.
    ``skip_existing``(기본 True)이면 WAV 와 JSON 이 모두 이미 있는 통화는 건너뛴다(Colab 이 끊긴 뒤 다시 실행해도 남은 것만 푼다).
    실제로 푼 통화 수를 돌려준다.
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
