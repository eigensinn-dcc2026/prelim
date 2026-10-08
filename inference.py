#!/usr/bin/env python3
"""DCC 2026 prelim unified inference entry point (team eigensinn).

Official command (one run per mission, one checkpoint per mission):

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission{N}.csv

The mission is taken from ``--mission`` if given, otherwise from the
``mission{N}`` part of the ``--output`` file name. Each mission module exposes
``predict(audio_dir, label_dir, ckpt_path) -> pandas.DataFrame`` whose columns
are exactly the official CSV columns for that mission; this file only
dispatches, validates the columns and writes the CSV.
"""

from __future__ import annotations

# Thread settings must be set before numpy/torch are imported anywhere.
import os

_N_THREADS = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_inference.registry import (  # noqa: E402
    CSV_ENCODING,
    load_predict,
    resolve_mission,
    validate_frame,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mission",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="optional; defaults to the mission{N} part of --output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mission = resolve_mission(args.output, args.mission)
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")

    predict = load_predict(mission)
    frame = predict(args.audio_dir, args.label_dir, args.ckpt_path)
    validate_frame(mission, frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding=CSV_ENCODING)
    print(f"Mission {mission} CSV saved: {args.output} (rows={len(frame)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())