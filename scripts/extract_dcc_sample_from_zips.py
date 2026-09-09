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
    # rglob (not glob): a Drive "download folder as zip" split often nests each
    # real chunk one level deeper (e.g. "<part>/대학부 데이터/<chunk>.zip") instead
    # of dropping all 12 zips flat under --zip-dir.
    zip_paths = sorted(args.zip_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"no .zip files found under {args.zip_dir}")

    if args.n_train_calls <= 0:
        # "Take everything" -- extract every matching entry independent of
        # whether its wav<->json partner is visible from the zip set alone.
        # index_zip_entries()+select_calls() require BOTH halves inside the
        # scanned zips to recognize a call at all; that silently drops any
        # call whose other half was already extracted separately outside a
        # zip (real case: a Drive "download as zip" split put a call's wav in
        # one numbered chunk and its json in a different one that a teammate
        # had already unzipped by hand -- the wav was never extracted because
        # index_zip_entries never saw both halves together). Pairing is left
        # entirely to build_manifest.py, which sees the *combined* output_dir
        # (these zips plus anything already there) and is the only place that
        # can actually tell a call is complete.
        print(f"{len(zip_paths)}개 zip에서 Seoul Training 전체 추출 중 (페어링 여부 무관, 이미 있는 파일은 건너뜀)...")
        newly_extracted = extract_all_entries(zip_paths, args.output_dir, split=TRAINING, seoul_only=True)
        print(f"Training 신규 추출 파일: {newly_extracted}개")
        # Same cross-chunk pairing bug applies to Validation (a call's label
        # can sit in an already-extracted loose chunk while its audio is in
        # one of these zips, or vice versa), and Validation total volume is
        # small (~2,120 calls team-wide) -- so take all of it too rather than
        # sampling --n-val-calls from a possibly-incomplete zip-only pairing.
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
