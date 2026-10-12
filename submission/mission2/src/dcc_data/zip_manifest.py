"""zip 조각으로 나뉜 AI-Hub 내려받기 안에서, 전부 풀지 않고 wav/json 쌍을 찾는다.

팀의 데이터셋은 Google Drive에서 여러 개의 zip 조각으로 내려받았고, wav 파일과 짝이 되는 json이 *서로 다른*
조각에 들어갈 수 있다. 이는 이미 풀어 둔 디렉터리 트리에 같은 가정을 두는 ``dcc_data.manifest.build_manifest``
("내려받은 조각의 경계는 짝짓기에서 일부러 무시한다")와 같다. 이 모듈은 zip의 *항목 이름*만 가지고
(각 zip의 중앙 디렉터리만 읽고 파일 내용은 읽지 않는다) 같은 짝짓기를 직접 한다. 그래서 로컬 디스크가
부족한 팀원이 전체 배포본 대신, 기준선 실행 한 번에 실제로 필요한 파일(예: Training 3,000통화 표본)만
풀 수 있다.

지역/split 판별은 파이프라인의 나머지와 똑같은 규칙(``guards.split_from_path``,
``mission2_features.is_seoul_path``)을 다시 쓴다. 둘 다 경로 문자열만 다루므로 디스크에 실제 파일이
없어도 된다.
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
    """모든 zip의 항목 목록을 훑어 파일 이름(stem)으로 wav<->json을 짝짓는다.

        경로에 Training/Validation 구성 요소가 정확히 하나 없는 항목(디렉터리 항목, 데이터가 아닌 파일)은
        zip에 흔히 들어 있으므로 오류를 내지 않고 조용히 건너뛴다.
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
    """``split``에서 최대 ``n_calls``개 통화를 뽑는다(0 또는 음수는 전부)."""

    pool = [entry for entry in entries if entry.split == split]
    if seoul_only:
        pool = [
            entry
            for entry in pool
            if is_seoul_path(entry.wav.entry_name) and is_seoul_path(entry.json.entry_name)
        ]
    pool = sorted(pool, key=lambda entry: entry.stem)  # 뽑기 전에 순서를 항상 같게 정렬한다
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
    """``split``/``seoul_only``에 맞는 모든 wav/json 항목을 모든 zip에서 푼다. 그 wav<->json의 짝이 같은
        zip 안에 있는지와는 상관없다.

        ``select_calls``와 ``extract_calls``는 통화의 두 부분이 *훑은 zip 묶음 안에서* 짝지어져야 한다. 그래서
        통화의 한쪽이 이미 어떤 zip 바깥에 따로 풀려 있고(예: 이 도구가 생기기 전에 팀원이 손으로 푼 조각, 또는
        Drive의 "폴더를 zip으로 내려받기"가 한 통화의 오디오와 라벨을 번호가 다른 조각에 나눈 경우) 다른 한쪽은
        아직 zip 안에 있으면, 그 통화를 조용히 빠뜨린다. 이 문제가 드러난 실제 예: 통화의 wav는 zip 조각 -012 안에
        있었는데 json은 다른 조각에서 이미 따로 풀려 있었다. ``index_zip_entries``는 두 부분을 한꺼번에 본 적이
        없어서 wav는 조용히 풀리지 않았다.

        *최종적으로 합쳐진* output_dir(이 함수가 푸는 것에 다른 곳에서 이미 있던 것까지 더한 것) 전체에 대한
        목록 단계의 짝짓기가 실제로 완전성을 결정한다. 그래서 이 함수는 스스로 짝짓기를 하려 하지 않고, split/지역
        거르개를 통과한 모든 항목을 zip 묶음만 봐서 짝이 보이든 말든 그냥 푼다. 실제로 푼 파일 수를 돌려준다
        (``skip_existing``이면 이미 있던 것은 제외).
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
                # Training/Validation 구성 요소 앞의 모든 것을 버린다 --
                # Drive의 "폴더를 zip으로 내려받기"는 조각을 나눌 때 항목을
                # 원본 폴더 자신의 이름으로 감싼다(예: "대학부 데이터/Training/..."),
                # 그리고 zip마다 감싸는 이름이 다를 수 있다. 이것을
                # 떼어 내지 않으면, 같은 통화를 다른 코드 경로(예: 이미 합쳐진
                # 낱개 조각)로 푼 것이 이것과
                # 다른 깊이에 놓인다. 그래서 build_manifest의 짝짓기는
                # 결국 둘 다 찾지만, 두 번째로 푸는 실행은
                # 이미 깨끗한 경로에 있는데도 *같은* 항목을 감싼 이름이 붙은 경로에
                # 한 번 더 쓰게 될 수 있다.
                # 그러면 디스크 사용량이 조용히 두 배가 되고, 두 사본이 모두 보이게 되면
                # DuplicateIdentifierError가 날 위험이 있다.
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
    """``entries``에 해당하는 wav+json 항목을 정확히 그것만 푼다.

        각 파일은 ``output_dir`` 아래에서 자기 항목 이름의 상대 경로에 놓인다. 즉 zip 자체가 쓴
        Training/Validation/지역 구조를 그대로 따르므로, 결과는 (훨씬 작을 뿐) 보통의 데이터 루트가 되어
        build_manifest.py가 이후에 훑을 수 있다.

        ``skip_existing``(기본 True)은 이어서 실행할 수 있게 한다: wav와 json이 둘 다 이미 목표 경로에 있는
        통화는 다시 풀지 않고 둔다. 중단된 실행(예: 수 GB 추출 도중의 Colab 연결 끊김) 뒤에 똑같은 선택으로
        다시 실행하면, 처음부터가 아니라 아직 없는 것만 가져온다. 실제로 푼 통화 수를 돌려준다(건너뛴 것은 제외).
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
