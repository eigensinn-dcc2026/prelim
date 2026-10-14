#!/usr/bin/env python3
"""DCC 2026 Mission 1 (팀 eigensinn): 신고자의 성별(F/M)을 예측한다.

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

라벨 JSON에서는 ``startAt`` / ``endAt`` / ``speaker``만 읽어 WAV에서 신고자의 음성을 잘라 낸다.
출력 열은 ``audio file name,gender``이다.
"""

from __future__ import annotations

# numpy/torch를 불러오기 전에 스레드 수를 먼저 정해야 한다.
import os

_N_THREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mission1_model as model_code  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./outputs/mission1.csv"))
    args = parser.parse_args(argv)

    frame = model_code.predict(args.audio_dir, args.label_dir, args.ckpt_path)
    if tuple(frame.columns) != model_code.COLUMNS or frame.empty or frame.isna().any().any():
        raise ValueError("Mission 1 output does not match the official CSV format")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Mission 1 CSV saved: {args.output} (rows={len(frame)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
