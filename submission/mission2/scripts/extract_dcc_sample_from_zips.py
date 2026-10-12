#!/usr/bin/env python3
"""Mission 2 baseline에 실제로 필요한 wav/json 파일만, 팀이 받은 zip 조각 AI-Hub 다운로드에서
바로 꺼낸다. 전체를 푼 데이터셋(수십 GB)을 담을 로컬 디스크 공간이 없고, 예를 들어
Training 3,000통화 샘플과 Validation 몇 통화만 필요한 컴퓨터를 위한 스크립트다.

먼저 모든 zip의 파일 목록을 훑어(압축은 풀지 않고 각 아카이브의 central directory만 읽음)
조각을 가로질러 파일 이름 stem으로 wav<->json을 짝짓고, 선택된 통화만 푼다.

사용법:
    python scripts/extract_dcc_sample_from_zips.py \\
        --zip-dir "C:\\...\\대학부 데이터 모음" \\
        --output-dir "C:\\dcc_data_sample" \\
        --n-train-calls 3000 --n-val-calls 50 --seed 42

다 푼 뒤에는 평소처럼 scripts/build_manifest.py를 --data-root에 --output-dir을 주고 실행한다.
푼 트리는 나머지 파이프라인이 기대하는 Training/Validation 구조 그대로이고, 파일 수만 훨씬 적다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.guards import TRAINING, VALIDATION  # noqa: E402
from dcc_data.zip_manifest import (  # noqa: E402
    extract_all_entries,
    extract_calls,
    index_zip_entries,
    select_calls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", type=Path, required=True, help="directory containing the *.zip download chunks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train-calls", type=int, default=3000)
    parser.add_argument(
        "--n-val-calls",
        type=int,
        default=50,
        help="ignored when --n-train-calls <= 0 (Validation is then extracted in full too)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # glob이 아니라 rglob을 쓴다. Drive "폴더를 zip으로 내려받기"는 실제 조각을 한 단계
    # 더 깊이 넣는 경우가 많다 (예: "<part>/대학부 데이터/<chunk>.zip").
    # 12개 zip이 --zip-dir 바로 아래에 평평하게 놓이는 게 아니다.
    zip_paths = sorted(args.zip_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"no .zip files found under {args.zip_dir}")

    if args.n_train_calls <= 0:
        # "전부 가져오기": wav<->json 짝이 zip 집합만으로 보이는지와 상관없이 조건에 맞는
        # 엔트리를 모두 푼다. index_zip_entries()+select_calls()는 스캔한 zip 안에 통화의
        # 두 파일이 *모두* 있어야 그 통화를 인식한다. 그래서 다른 한쪽이 이미 zip 밖에
        # 따로 풀려 있으면 그 통화가 조용히 빠진다 (실제 사례: Drive "zip으로 내려받기"가
        # 한 통화의 wav는 한 번호의 조각에, json은 팀원이 이미 손으로 푼 다른 조각에 넣었다.
        # index_zip_entries가 두 파일을 함께 보지 못해 wav가 풀리지 않았다). 짝짓기는
        # 전부 build_manifest.py에 맡긴다. 거기서는 *합쳐진* output_dir(이 zip들 + 이미 있던
        # 파일)을 보므로, 통화가 완전한지 실제로 판단할 수 있는 유일한 곳이다.
        print(f"{len(zip_paths)}개 zip에서 Seoul Training 전체 추출 중 (페어링 여부 무관, 이미 있는 파일은 건너뜀)...")
        newly_extracted = extract_all_entries(zip_paths, args.output_dir, split=TRAINING, seoul_only=True)
        print(f"Training 신규 추출 파일: {newly_extracted}개")
        # 조각을 가로지르는 같은 짝짓기 문제가 Validation에도 있다 (통화의 라벨은 이미 풀어 둔
        # 조각에 있고 오디오는 이 zip들 중 하나에 있을 수 있고, 그 반대도 마찬가지다). 게다가
        # Validation은 전체 양이 적다 (팀 전체 약 2,120통화). 그래서 zip만으로 짝지은 불완전할 수
        # 있는 결과에서 --n-val-calls개를 뽑지 않고, 이것도 전부 가져온다.
        newly_extracted_val = extract_all_entries(zip_paths, args.output_dir, split=VALIDATION, seoul_only=False)
        print(f"Validation 신규 추출 파일: {newly_extracted_val}개 (전체, --n-val-calls 무시됨)")
    else:
        print(f"{len(zip_paths)}개 zip에서 항목 목록 스캔 중 (압축 해제 없음)...")
        entries = index_zip_entries(zip_paths)
        print(f"wav/json 쌍을 찾은 통화: {len(entries)}개")
        train_calls = select_calls(
            entries, split=TRAINING, seoul_only=True, n_calls=args.n_train_calls, seed=args.seed
        )
        print(f"Seoul Training 선택: {len(train_calls)}개 통화 (신규 추출 {extract_calls(train_calls, args.output_dir)}개)")
        val_calls = select_calls(
            entries, split=VALIDATION, seoul_only=False, n_calls=args.n_val_calls, seed=args.seed
        )
        print(f"Validation 선택: {len(val_calls)}개 통화 (신규 추출 {extract_calls(val_calls, args.output_dir)}개)")

    print(
        "-> 실제로 몇 개 통화가 완전한 쌍으로 잡혔는지는 다음 단계(build_manifest.py)의 "
        "Training/Validation 행 수가 진짜 기준입니다 (여기 숫자는 파일 단위라 다를 수 있음)."
    )
    print(f"이제 이 경로로 build_manifest.py를 실행하세요: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
