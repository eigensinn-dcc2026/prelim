#!/usr/bin/env python3
"""Build the official Mission 2 Validation feature cache (53-dim, Seoul-only).

Unlike scripts/build_mission2_dataset.py (Training-only, gated by
dcc_data.guards.ensure_training_rows), this reads the untouched official
Validation/1.원천데이터/VS_서울_구급 + Validation/2.라벨링데이터/VL_서울_구급
folders directly and pairs wav/json by filename stem via pair_by_stem()
(lenient -- some stems are unpaired in the raw download; manifest.build_manifest's
strict all-or-nothing pairing would raise on that, so it's not used here).

For evaluation/early-stopping/checkpoint-selection only -- nothing downstream
of this cache may call .fit()/backprop on it (see the contest rules and
dcc_data.mission2_features.select_seoul_validation_rows).

Usage:
    python scripts/build_mission2_validation_features.py \\
        --validation-root "<path to>/Validation" \\
        --features-output data/local_cache/m2_features_validation_official.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402

from dcc_data.manifest import ManifestRow  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_seoul_validation_dataset,
    pair_by_stem,
)

AUDIO_SUBDIR = "1.원천데이터/VS_서울_구급"
LABEL_SUBDIR = "2.라벨링데이터/VL_서울_구급"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-root", type=Path, required=True, help="path to the .../Validation folder")
    parser.add_argument("--index-output", type=Path, default=Path("data/local_cache/m2_utterances_validation_official.parquet"))
    parser.add_argument("--features-output", type=Path, default=Path("data/local_cache/m2_features_validation_official.parquet"))
    parser.add_argument("--failures-log", type=Path, default=Path("reports/mission2/m2_build_failures_validation.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.validation_root.resolve()
    audio_dir = data_root / AUDIO_SUBDIR
    label_dir = data_root / LABEL_SUBDIR

    pairs = pair_by_stem(audio_dir, label_dir)
    print(f"Paired {len(pairs)} calls by stem (audio dir has more/fewer -- unmatched stems are dropped, logged below).")

    wav_only = {p.stem for p in audio_dir.glob("*.wav")} - {stem for stem, _, _ in pairs}
    json_only = {p.stem for p in label_dir.glob("*.json")} - {stem for stem, _, _ in pairs}
    if wav_only or json_only:
        print(f"Unpaired: {len(wav_only)} wav-only, {len(json_only)} json-only (skipped, not counted as failures).")

    rows = [
        ManifestRow(
            split="Validation",
            stem=stem,
            wav_path=f"{AUDIO_SUBDIR}/{stem}.wav",
            json_path=f"{LABEL_SUBDIR}/{stem}.json",
        )
        for stem, _, _ in pairs
    ]

    def on_call_done(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total} calls processed")

    index_records, feature_rows, failures = build_seoul_validation_dataset(
        rows, data_root, max_workers=args.workers, on_call_done=on_call_done
    )

    print(f"Utterance index rows: {len(index_records)}")
    print(f"Feature rows (extraction succeeded): {len(feature_rows)}")
    print(f"Failures: {len(failures)}")

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(r) for r in index_records]).to_parquet(args.index_output, index=False)

    features_df = pd.DataFrame(feature_rows)
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"missing feature columns in output: {missing}")
    features_df.to_parquet(args.features_output, index=False)

    if failures:
        with args.failures_log.open("w", encoding="utf-8") as stream:
            for failure in failures:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    print(f"Index saved: {args.index_output}")
    print(f"Features saved: {args.features_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
