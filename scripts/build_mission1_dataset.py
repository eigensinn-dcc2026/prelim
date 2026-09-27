#!/usr/bin/env python3
"""Build the Mission 1 call-level 52-dim acoustic feature cache.

Only Seoul rows of the requested official split are used. Training and
Validation caches are built separately so Validation samples cannot enter
the Training parquet.

Usage (from the repository root):

    python scripts/build_mission1_dataset.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw \\
        --split Training

    python scripts/build_mission1_dataset.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw \\
        --split Validation \\
        --features-output data/m1_features_validation.parquet \\
        --failures-log data/m1_build_failures_validation.jsonl
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

from dcc_data.guards import TRAINING, VALIDATION  # noqa: E402
from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission1_features import ACOUSTIC_FEATURE_COLUMNS, build_mission1_feature_dataset  # noqa: E402

FEATURE_TABLE_COLUMNS = (
    "stem",
    "split",
    "gender",
    "n_caller_utterances",
    *ACOUSTIC_FEATURE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=(TRAINING, VALIDATION), default=TRAINING)
    parser.add_argument(
        "--n-calls",
        type=int,
        default=0,
        help="sample this many calls (0 = all Seoul rows of the split)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--features-output",
        type=Path,
        default=Path("data/m1_features_train.parquet"),
    )
    parser.add_argument(
        "--failures-log",
        type=Path,
        default=Path("data/m1_build_failures.jsonl"),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=200,
        help="write partial outputs every N calls (0 disables)",
    )
    return parser.parse_args()


def _write_features(feature_rows: list[dict], features_output: Path) -> None:
    import pandas as pd

    features_output.parent.mkdir(parents=True, exist_ok=True)
    features_df = pd.DataFrame(feature_rows, columns=list(FEATURE_TABLE_COLUMNS))
    features_df.to_parquet(features_output, index=False)


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)

    feature_rows: list[dict] = []
    failures: list[dict] = []

    try:
        from tqdm import tqdm

        bar = tqdm(unit="call")
    except ImportError:
        bar = None

    last_flush_call = 0

    def on_call_done(done: int, total: int) -> None:
        nonlocal last_flush_call
        if bar is not None:
            if bar.total is None:
                bar.total = total
            bar.update(done - bar.n)
        if args.flush_every > 0 and done - last_flush_call >= args.flush_every:
            _write_features(feature_rows, args.features_output)
            last_flush_call = done

    start = time.monotonic()
    build_mission1_feature_dataset(
        rows,
        args.data_root,
        args.split,
        n_calls=args.n_calls,
        seed=args.seed,
        feature_sink=feature_rows,
        failure_sink=failures,
        on_call_done=on_call_done,
    )
    if bar is not None:
        bar.close()

    _write_features(feature_rows, args.features_output)

    args.failures_log.parent.mkdir(parents=True, exist_ok=True)
    with args.failures_log.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - start
    print(f"Split: {args.split}")
    print(f"Feature rows: {len(feature_rows)}")
    print(f"Failures: {len(failures)} (see {args.failures_log})")
    print(f"Features parquet: {args.features_output}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
