#!/usr/bin/env python3
"""Mission 2 L2 입력 캐시를 만든다: 발화마다 로그 멜 스펙트로그램 .npy 하나와, 인덱스 parquet
(call_id, utt_idx, speaker, npy_path, 그리고 만들 때의 split / n_mels / hop_length / n_fft / sample_rate.
이것은 train_mission2_cnn.py가 인덱스로 학습하기 전에 확인하는 출처 정보이다).

scripts/build_mission2_dataset.py의 뽑기/목록 처리와 같고, 출력 형식만 다르다(스펙트로그램은 평평한 표의
한 행에 들어가지 않는다). L1 특징 캐시에 더해 왜 이것이 필요한지는 dcc_data.mission2_spectrogram의 모듈
설명을 보라.

사용법:
    python scripts/build_mission2_spectrograms.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw \\
        --output-dir data/m2_spectrograms \\
        --n-calls 3000 --seed 42 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission2_features import HOP_LENGTH  # noqa: E402
from dcc_data.mission2_spectrogram import (  # noqa: E402
    N_MELS,
    build_seoul_training_spectrograms,
    build_seoul_validation_spectrograms,
)

BUILDERS = {"training": build_seoul_training_spectrograms, "validation": build_seoul_validation_spectrograms}

# --output-dir/--index-output/--failures-log를 주지 않았을 때 각 split의 캐시가 가는 곳. 두
# split이 기본값을 공유하면 절대 안 된다: Training 경로에 덮어쓴 Validation 캐시는
# 기본 경로를 읽는 모든 것에게 Training 인덱스처럼 보이게 된다(Training 도장이 찍히지 않은 인덱스를
# 거부하는 train_mission2_cnn.py도 참고).
DEFAULT_CACHE_PATHS = {
    "training": {
        "output_dir": Path("data/m2_spectrograms"),
        "index_output": Path("data/m2_spectrogram_index.parquet"),
        "failures_log": Path("data/m2_spectrogram_build_failures.jsonl"),
    },
    "validation": {
        "output_dir": Path("data/m2_spectrograms_validation"),
        "index_output": Path("data/m2_spectrogram_index_validation.parquet"),
        "failures_log": Path("data/m2_spectrogram_build_failures_validation.jsonl"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=list(BUILDERS), default="training",
        help="training (default, unchanged behavior) or validation -- the official Validation "
        "folder, eval-only, never for a training loop's .fit()/backprop.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['output_dir']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['output_dir']} (validation)",
    )
    parser.add_argument("--n-calls", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index-output", type=Path, default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['index_output']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['index_output']} (validation)",
    )
    parser.add_argument(
        "--failures-log", type=Path, default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['failures_log']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['failures_log']} (validation)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--hop-length", type=int, default=HOP_LENGTH,
        help=f"STFT hop length in samples (default {HOP_LENGTH}, same as L1's features). "
        "A smaller value gives more time frames for the same audio -- pass a distinct "
        "--output-dir/--index-output so this doesn't overwrite the default-resolution set.",
    )
    parser.add_argument(
        "--n-mels", type=int, default=N_MELS,
        help=f"number of mel frequency bands (default {N_MELS}). A larger value gives more "
        "frequency resolution for the same audio -- pass a distinct --output-dir/--index-output "
        "so this doesn't overwrite the default-resolution set.",
    )
    args = parser.parse_args()
    for name, default in DEFAULT_CACHE_PATHS[args.split].items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)
    build_spectrograms = BUILDERS[args.split]

    try:
        from tqdm import tqdm

        bar = tqdm(unit="call")
    except ImportError:
        bar = None

    def on_call_done(done: int, total: int) -> None:
        if bar is not None:
            if bar.total is None:
                bar.total = total
            bar.update(done - bar.n)

    start = time.monotonic()
    index_rows, failures = build_spectrograms(
        rows,
        args.data_root,
        args.output_dir,
        n_calls=args.n_calls,
        seed=args.seed,
        on_call_done=on_call_done,
        max_workers=args.workers,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    if bar is not None:
        bar.close()

    import pandas as pd

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_parquet(args.index_output, index=False)

    args.failures_log.parent.mkdir(parents=True, exist_ok=True)
    with args.failures_log.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - start
    print(f"Spectrograms saved (utterances): {len(index_rows)}")
    print(f"Failures: {len(failures)} (see {args.failures_log})")
    print(f"Index parquet: {args.index_output}")
    print(f"Spectrogram dir: {args.output_dir}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
