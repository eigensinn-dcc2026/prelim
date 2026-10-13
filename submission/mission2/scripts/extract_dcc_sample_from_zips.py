#!/usr/bin/env python3
"""대회 원본 zip 조각들에서 미션 2 에 필요한 wav/json 을 바로 풀어낸다.

최종 학습·검증 노트북은 모두 "전부 풀기"(``--n-train-calls 0``)를 썼다: 서울 Training 전체와 Validation 전체를 짝과 상관없이 모두 풀고,
짝짓기는 다음 단계 scripts/build_manifest.py 가 풀린 폴더 전체를 보고 한다.

    python scripts/extract_dcc_sample_from_zips.py --zip-dir "<Drive>/데이터 크리에이터 캠프/대학부 데이터" \
        --output-dir /content/dcc_data_full --n-train-calls 0

``--n-train-calls`` 가 1 이상이면 "일부만 뽑기"가 된다(디스크가 부족한 PC 에서 일부 통화만 시험할 때). 이 방식은 zip 안의 파일 목록만 읽어
WAV 와 JSON 을 짝짓기 때문에, 짝의 한쪽이 다른 조각에 있거나 이미 풀려 있으면 그 통화를 빠뜨린다(공식 Validation 에서 3,640통화 중
1,541통화만 풀린 적이 있다). 전체 데이터가 필요할 때는 쓰지 않는다.

풀린 폴더는 원래와 같은 Training/Validation 구조라서, 이후 --data-root 를 이 --output-dir 로 주고 build_manifest.py 를 실행하면 된다.
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
    # glob 이 아니라 rglob: Drive 의 "폴더를 zip 으로 받기"는 실제 조각을 한 단계 더 깊이 넣는 경우가 많다
    # (예: "<part>/대학부 데이터/<chunk>.zip").
    zip_paths = sorted(args.zip_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"no .zip files found under {args.zip_dir}")

    if args.n_train_calls <= 0:
        # "전부 풀기": 짝이 zip 묶음 안에서 보이든 말든 조건에 맞는 항목을 모두 푼다.
        # index_zip_entries()+select_calls() 는 두 파일이 모두 읽은 zip 안에 있어야 통화를 알아보므로, 한쪽이 이미 따로
        # 풀려 있으면 그 통화를 조용히 빠뜨린다. 짝짓기는 풀린 폴더 전체를 보는 build_manifest.py 에 맡긴다.
        print(f"{len(zip_paths)}개 zip에서 Seoul Training 전체 추출 중 (페어링 여부 무관, 이미 있는 파일은 건너뜀)...")
        newly_extracted = extract_all_entries(zip_paths, args.output_dir, split=TRAINING, seoul_only=True)
        print(f"Training 신규 추출 파일: {newly_extracted}개")
        # Validation 에도 같은 문제가 있고(라벨은 풀린 조각에, 소리는 zip 안에, 또는 반대), Validation 은 양이 작으므로
        # --n-val-calls 로 뽑지 않고 전부 푼다.
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
