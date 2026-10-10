"""Standalone official Mission 2 entry point (selected weights supplied separately)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.ckpt_path.is_file():
        parser.error("M2-RERUN-001 selected CNN ensemble checkpoint is missing: " + str(args.ckpt_path))
    from mission2_model import predict
    frame = predict(args.audio_dir, args.label_dir, args.ckpt_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Mission 2: {len(frame)} utterances -> {args.output}")


if __name__ == "__main__":
    main()
