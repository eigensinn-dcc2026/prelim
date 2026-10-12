#!/usr/bin/env python3
"""Mission 1 standalone inference CLI (kept for smoke tests).

    python scripts/mission1_inference.py \
        --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

The logic lives in ``src/dcc_inference/mission1.py`` and is shared with the
root ``inference.py``; this wrapper keeps the old CLI and output encoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_inference.mission1 import predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = predict(args.audio_dir, args.label_dir, args.ckpt_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Mission 1 CSV saved: {args.output} (rows={len(frame)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())