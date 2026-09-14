#!/usr/bin/env python3
"""Build just the Mission 2 utterance-level index (call_id, utt_idx,
startAt, endAt, speaker, duration_sec) -- no audio, no spectrograms, no
53/58/61-dim features. Fast (JSON parsing only).

Exists so error analysis on a specific data pull (e.g. the L2 spectrogram
build's zip-extracted sample) can compute overlap flags
(dcc_data.mission2_analysis.compute_overlap_flags, which needs startAt/
endAt) against an index guaranteed to cover the *same* (call_id, utt_idx)
pairs -- joining against an index built from a different data pull (e.g.
an older L1 run against a since-expired shared Drive) can silently miss
most rows if the two pulls sampled different calls.

Usage:
    python scripts/build_mission2_utterance_index.py \\
        --manifest data/manifests/dcc_manifest_full.csv \\
        --data-root /content/dcc_full \\
        --output data/m2_utterance_index_only.parquet
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

from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission2_features import build_utterance_records, sample_calls, select_seoul_training_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--n-calls", type=int, default=0, help="0 = all (matches build_mission2_spectrograms.py's default usage)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/m2_utterance_index_only.parquet"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)
    seoul_rows = select_seoul_training_rows(rows)
    sampled_rows = sample_calls(seoul_rows, args.n_calls, args.seed)
    print(f"{len(sampled_rows)} Seoul/Training calls")

    records = []
    failures = 0
    for row in sampled_rows:
        try:
            records.extend(build_utterance_records(row, args.data_root))
        except Exception as exc:  # noqa: BLE001 -- one bad call's JSON must not abort the whole build
            failures += 1
            print(f"[warn] skipping {row.stem}: {type(exc).__name__}: {exc}", file=sys.stderr)

    df = pd.DataFrame(
        [
            {
                "call_id": r.call_id,
                "utt_idx": r.utt_idx,
                "startAt": r.start_at,
                "endAt": r.end_at,
                "speaker": r.speaker,
                "duration_sec": r.duration_sec,
            }
            for r in records
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Utterances: {len(df)}  Failures: {failures}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
