"""Offline M1-XVEC-001 inference using only audio and caller intervals."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    from mission1_model import predict
    frame = predict(args.audio_dir, args.label_dir, args.ckpt_path,
                    device=args.device, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Mission 1: {len(frame)} calls -> {args.output}")


if __name__ == "__main__":
    main()
