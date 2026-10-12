#!/usr/bin/env python3
"""DCC 2026 Mission 3 (team eigensinn): predict patient symptoms from call transcripts.

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission3.csv

Only the label JSON transcripts are read (``--audio_dir`` is accepted for the
common command and ignored). Each symptom gets ``probability >= 0.5 -> 1``; no
other post-processing. Output columns: ``label file name,symptom``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import mission3_model as model_code  # noqa: E402

COLUMNS = ["label file name", "symptom"]
BATCH_SIZE = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./outputs/mission3.csv"))
    args = parser.parse_args(argv)

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")
    json_paths = sorted(args.label_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no JSON files under {args.label_dir}")

    started = time.time()
    texts = [model_code.read_text(path) for path in json_paths]
    probs = model_code.predict_probabilities(args.ckpt_path, texts, BATCH_SIZE)
    binary = (probs >= model_code.THRESHOLD).astype(int)

    frame = pd.DataFrame(
        {COLUMNS[0]: [path.name for path in json_paths],
         COLUMNS[1]: [model_code.symptom_cell(row) for row in binary]},
        dtype=object,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Mission 3 CSV saved: {args.output} (rows={len(frame)}, {time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
