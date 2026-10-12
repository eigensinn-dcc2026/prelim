"""전체를 풀지 않고, zip 조각으로 나뉜 AI-Hub 다운로드 안에서 wav/json 짝을 찾는다.

팀 데이터셋은 Google Drive에서 여러 개의 zip 조각으로 나뉘어 내려받았고,
wav 파일과 짝인 json이 *서로 다른* 조각에 들어갈 수 있다.
이미 풀어 둔 디렉터리 트리에 같은 가정을 쓰는 ``dcc_data.manifest.build_manifest``
("짝지을 때 다운로드 조각 경계는 일부러 무시한다")와 같은 방식이다.
이 모듈은 zip의 *엔트리 이름*만으로 같은 짝짓기를 한다(각 zip의 central directory만 읽고
파일 내용은 읽지 않는다). 그래서 로컬 디스크가 부족한 팀원도 전체 배포본 대신,
baseline 한 번에 실제로 필요한 파일(예: Training 3,000통화 샘플)만 풀 수 있다.

지역/split 판별은 나머지 파이프라인과 똑같은 규칙(``guards.split_from_path``,
``mission2_features.is_seoul_path``)을 그대로 쓴다. 둘 다 경로 문자열만 보므로
디스크에 실제 파일이 없어도 된다.
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
    """모든 zip의 엔트리 목록을 훑어 파일 이름 stem 기준으로 wav<->json을 짝짓는다.

    경로에 Training/Validation 구성요소가 정확히 하나 있지 않은 엔트리(디렉터리 엔트리,
    데이터가 아닌 파일)는 zip에 흔히 들어 있으므로, 예외를 내지 않고 조용히 건너뛴다.
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
    """``split``에서 최대 ``n_calls``개 통화를 뽑는다 (0 이하이면 전부)."""

    pool = [entry for entry in entries if entry.split == split]
    if seoul_only:
        pool = [
            entry
            for entry in pool
            if is_seoul_path(entry.wav.entry_name) and is_seoul_path(entry.json.entry_name)
        ]
    pool = sorted(pool, key=lambda entry: entry.stem)  # 샘플링 전에 순서를 고정해 항상 같은 결과가 나오게 한다
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
    """모든 zip에서 ``split``/``seoul_only`` 조건에 맞는 wav/json 엔트리를 전부 푼다.
    짝이 되는 wav<->json이 같은 zip들 안에 있는지는 따지지 않는다.

    ``select_calls`` + ``extract_calls``는 통화의 두 파일이 *스캔한 zip 집합 안에서* 모두
    짝지어져야 한다. 그래서 한쪽이 이미 zip 밖에 따로 풀려 있고(예: 이 도구가 생기기 전에
    팀원이 손으로 푼 조각, 또는 Drive "폴더를 zip으로 내려받기"가 한 통화의 오디오와 라벨을
    서로 다른 번호의 조각에 넣은 경우) 다른 한쪽만 zip 안에 있으면, 그 통화가 조용히 빠진다.
    실제로 이 문제가 드러난 사례: 한 통화의 wav는 zip 조각 -012 안에 있었는데 json은 이미
    다른 조각에서 따로 풀려 있었다. ``index_zip_entries``는 두 파일을 한 번에 보지 못했고,
    그래서 wav가 조용히 풀리지 않았다.

    통화가 완전한지는 *최종적으로 합쳐진* output_dir(여기서 푼 파일 + 다른 곳에서 이미
    있던 파일) 전체에 manifest를 만들 때의 짝짓기가 결정한다. 그래서 이 함수는 직접 짝을
    짓지 않고, split/지역 필터를 통과하는 엔트리를 짝이 zip 안에서 보이든 안 보이든 전부 푼다.
    실제로 푼 파일 수를 돌려준다 (``skip_existing``이면 이미 있던 파일은 제외).
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
                # Training/Validation 구성요소 앞의 경로는 모두 버린다.
                # Drive "폴더를 zip으로 내려받기"는 엔트리를 원본 폴더 이름으로 감싸고
                # (예: "대학부 데이터/Training/..."), zip마다 감싸는 이름이 다를 수 있다.
                # 이걸 떼지 않으면, 다른 경로로 풀린 같은 통화(예: 이미 합쳐 둔 조각)가
                # 이 파일과 다른 깊이에 놓인다. build_manifest의 짝짓기는 결국 둘 다
                # 찾지만, 두 번째로 푸는 실행이 이미 깨끗한 경로에 있는 *같은* 엔트리를
                # 감싼 이름이 붙은 경로에 한 번 더 써 버릴 수 있다. 그러면 디스크 사용량이
                # 조용히 두 배가 되고, 두 복사본이 모두 보이는 순간 DuplicateIdentifierError가
                # 날 수 있다.
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
    """``entries``에 해당하는 wav+json 엔트리만 정확히 푼다.

    각 파일은 ``output_dir`` 아래에서 자기 엔트리 이름의 상대 경로에 놓인다. 즉 zip이 쓰던
    Training/Validation/지역 구조가 그대로 유지되므로, 결과는 (훨씬 작을 뿐) 보통의 데이터
    루트이고 build_manifest.py가 나중에 그대로 스캔할 수 있다.

    ``skip_existing``(기본 True)이면 중간부터 이어서 할 수 있다. wav와 json이 둘 다 이미 목표
    경로에 있는 통화는 다시 풀지 않고 그냥 둔다. 그래서 실행이 끊긴 뒤(예: 수 GB를 푸는 도중
    Colab 연결이 끊긴 경우) 같은 선택으로 다시 실행하면, 처음부터 다시 하지 않고 아직 없는
    파일만 가져온다. 실제로 푼 통화 수를 돌려준다 (건너뛴 통화는 제외).
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
