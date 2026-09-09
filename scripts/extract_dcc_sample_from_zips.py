#!/usr/bin/env python3
"""Extract only the wav/json files a Mission 2 baseline actually needs,
directly out of the team's zip-chunked AI-Hub download -- for a machine
whose local disk doesn't have room for the full extracted dataset (tens of
GB) but only needs, say, a 3,000-call Training sample plus a handful of
Validation calls.

Scans every zip's file list first (no extraction, just reads each archive's
central directory) to pair wav<->json by filename stem across chunks, then
extracts only the selected calls.

Usage:
    python scripts/extract_dcc_sample_from_zips.py \\
        --zip-dir "C:\\...\\대학부 데이터 모음" \\
        --output-dir "C:\\dcc_data_sample" \\
        --n-train-calls 3000 --n-val-calls 50 --seed 42

Afterward, run scripts/build_manifest.py with --data-root pointed at
--output-dir as usual -- the extracted tree has the same Training/Validation
structure the rest of the pipeline expects, just far fewer files in it.
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
from dcc_data.zip_manifest import extract_calls, index_zip_entries, select_calls  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", type=Path, required=True, help="directory containing the *.zip download chunks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train-calls", type=int, default=3000)
    parser.add_argument("--n-val-calls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # rglob (not glob): a Drive "download folder as zip" split often nests each
    # real chunk one level deeper (e.g. "<part>/대학부 데이터/<chunk>.zip") instead
    # of dropping all 12 zips flat under --zip-dir.
    zip_paths = sorted(args.zip_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"no .zip files found under {args.zip_dir}")

    print(f"{len(zip_paths)}개 zip에서 항목 목록 스캔 중 (압축 해제 없음)...")
    entries = index_zip_entries(zip_paths)
    print(f"wav/json 쌍을 찾은 통화: {len(entries)}개")

    train_calls = select_calls(entries, split=TRAINING, seoul_only=True, n_calls=args.n_train_calls, seed=args.seed)
    val_calls = select_calls(entries, split=VALIDATION, seoul_only=False, n_calls=args.n_val_calls, seed=args.seed)
    print(f"Seoul Training 선택: {len(train_calls)}개 통화")
    print(f"Validation 선택: {len(val_calls)}개 통화")

    print(f"선택된 파일을 {args.output_dir}에 추출 중 (이미 있는 파일은 건너뜀)...")
    selected = train_calls + val_calls
    newly_extracted = extract_calls(selected, args.output_dir)
    skipped = len(selected) - newly_extracted

    total_bytes = sum(
        (args.output_dir / location.entry_name).stat().st_size
        for call in selected
        for location in (call.wav, call.json)
    )
    print(
        f"추출 완료: 신규 {newly_extracted}개 / 이미 있어서 건너뜀 {skipped}개 "
        f"/ 전체 {len(selected)}개 통화, 총 {total_bytes / 1e9:.2f} GB"
    )
    if newly_extracted < len(selected):
        print("-> 도중에 끊겼다가 재실행한 경우여도 정상입니다 (같은 명령을 그대로 다시 실행하면 이어서 받습니다).")
    print(f"이제 이 경로로 build_manifest.py를 실행하세요: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
