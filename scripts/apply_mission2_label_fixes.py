#!/usr/bin/env python3
"""Patch an already-built index parquet's ``speaker`` column with the
confirmed label-swap fixes (dcc_data.mission2_features.corrected_speaker),
in place, without touching audio or the source JSON.

Exists so a fix to CONFIRMED_LABEL_SWAPPED_CALL_IDS/PARTIAL_LABEL_SWAP_
FLIP_FROM_UTT_IDX doesn't require re-running the ~80-minute full
spectrogram build (dcc_data.mission2_spectrogram) or L1 feature build
(dcc_data.mission2_features.build_seoul_training_dataset) -- both already
carry call_id/utt_idx/speaker as plain columns, and speaker is the only
thing that changes.

Works on either L1's or L2's index parquet (same call_id/utt_idx/speaker
columns either way).

Usage:
    python scripts/apply_mission2_label_fixes.py \\
        --index data/m2_spectrogram_index_full.parquet \\
        --output data/m2_spectrogram_index_full.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402

from dcc_data.mission2_features import corrected_speaker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="pass the same path as --index to overwrite in place")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.index)

    fixed = [
        corrected_speaker(call_id, int(utt_idx), int(speaker))
        for call_id, utt_idx, speaker in zip(df["call_id"], df["utt_idx"], df["speaker"])
    ]
    n_changed = int((pd.Series(fixed) != df["speaker"]).sum())
    df["speaker"] = fixed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"{len(df)} rows, {n_changed} speaker value(s) changed")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
