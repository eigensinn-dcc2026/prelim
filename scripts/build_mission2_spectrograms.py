#!/usr/bin/env python3
"""Build the Mission 2 L2 input cache: one log-mel spectrogram .npy per
utterance, plus an index parquet (call_id, utt_idx, speaker, npy_path).

Mirrors scripts/build_mission2_dataset.py's sampling/manifest handling;
only the output format differs (a spectrogram doesn't fit a flat tabular
row). See dcc_data.mission2_spectrogram's module docstring for why this
exists on top of the L1 feature cache.

Usage:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=list(BUILDERS), default="training",
        help="training (default, unchanged behavior) or validation -- the official Validation "
        "folder, eval-only, never for a training loop's .fit()/backprop.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/m2_spectrograms"))
    parser.add_argument("--n-calls", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-output", type=Path, default=Path("data/m2_spectrogram_index.parquet"))
    parser.add_argument("--failures-log", type=Path, default=Path("data/m2_spectrogram_build_failures.jsonl"))
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
    return parser.parse_args()


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
