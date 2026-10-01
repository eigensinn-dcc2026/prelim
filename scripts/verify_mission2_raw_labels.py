#!/usr/bin/env python3
"""Check a Mission 2 cache's speaker labels against the provided label files; optionally restore them.

The organizers ruled (2026-10-01) that Mission 2 training must use the provided ``speaker`` labels as they
are -- no correction, and no data selection, on the basis of ``text``. A cache built before that ruling may
carry corrected labels (six Training calls, 109 utterances, were corrected in the earlier L1/L2 runs), so
run this on any such cache before training on it. It works on an L2 spectrogram index and on an L1
utterance/feature table alike: anything with ``call_id``, ``utt_idx`` and ``speaker`` columns.

    # just check (exit 0 = every label equals the provided one, 1 = some differ, 2 = could not be checked)
    python scripts/verify_mission2_raw_labels.py \\
        --index data/m2_spectrogram_index.parquet --label-root data/raw/Training/2.라벨링데이터

    # write a copy with the provided labels restored (an L2 spectrogram index is also stamped, so that
    # train_mission2_cnn.py will accept it); --stamp-training-defaults is only for a cache built before
    # provenance existed that you know is a Training-only, default-settings cache
    python scripts/verify_mission2_raw_labels.py \\
        --index m2_spectrogram_index_full.parquet --label-root .../Training/2.라벨링데이터 \\
        --output m2_spectrogram_index_full_provided_labels.parquet --stamp-training-defaults

Only ``_id`` and ``speaker`` are read from the label files, never ``text``; the report names call ids and
counts only. Reading the full Training folder (29,200 label files) takes several minutes.
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

from dcc_data.guards import DataGuardError  # noqa: E402
from dcc_data.mission2_raw_labels import (  # noqa: E402
    LabelComparison,
    LabelVerificationError,
    compare_index_labels,
    load_provided_speakers,
    restore_provided_labels,
)

EXIT_MATCHES, EXIT_DIFFERS, EXIT_CANNOT_CHECK = 0, 1, 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="the cache's parquet (call_id, utt_idx, speaker, ...)")
    parser.add_argument("--label-root", type=Path, required=True, help="the provided Training label folder (the .json files)")
    parser.add_argument("--output", type=Path, default=None, help="write a copy with the provided labels restored here")
    parser.add_argument(
        "--stamp-training-defaults", action="store_true",
        help="for an L2 spectrogram index built before provenance existed: add the missing split/n_mels/hop_length/"
        "n_fft/sample_rate columns as Training with the default settings (40 mels, hop 160, FFT 512, 8 kHz). "
        "Only right if the cache really is a Training-only, default-settings one.",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() == args.index.resolve():
        parser.error("--output must be a new file, not --index itself")
    if args.stamp_training_defaults and args.output is None:
        parser.error("--stamp-training-defaults only makes sense together with --output")
    return args


def _print_comparison(comparison: LabelComparison) -> None:
    print(f"rows checked: {comparison.n_rows} ({comparison.n_calls} calls)")
    print(f"rows whose speaker differs from the provided label: {comparison.mismatched_rows}")
    if comparison.mismatched_by_call:
        for call_id, count in sorted(comparison.mismatched_by_call.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"  {call_id}: {count} row(s)")
        if len(comparison.mismatched_by_call) > 20:
            print(f"  ... and {len(comparison.mismatched_by_call) - 20} more call(s)")
    if comparison.unverifiable_rows:
        print(f"rows that could not be found in the label files: {comparison.unverifiable_rows}")
        for call_id in sorted(comparison.unverifiable_by_call)[:10]:
            print(f"  {call_id}: {comparison.unverifiable_by_call[call_id]} row(s)")


def main() -> int:
    args = parse_args()
    try:
        index_df = pd.read_parquet(args.index)
        provided = load_provided_speakers(
            args.label_root, on_progress=lambda done, total: print(f"  read {done}/{total} label files", flush=True)
        )
        print(f"provided labels: {len(provided)} calls under {args.label_root}")
        comparison = compare_index_labels(index_df, provided)
        _print_comparison(comparison)

        if comparison.unverifiable_rows:
            print("CANNOT CHECK: some rows are not in these label files (wrong --label-root, or an index from another split).")
            return EXIT_CANNOT_CHECK

        if args.output is None:
            if comparison.matches:
                print("OK: every speaker label equals the provided one.")
                return EXIT_MATCHES
            print("DIFFERS: re-run with --output to write a copy with the provided labels restored.")
            return EXIT_DIFFERS

        restored = restore_provided_labels(index_df, provided, stamp_training_defaults=args.stamp_training_defaults)
    except LabelVerificationError as error:
        print(f"CANNOT CHECK: {error}")
        return EXIT_CANNOT_CHECK
    except (ValueError, DataGuardError) as error:  # e.g. no provenance and --stamp-training-defaults not given
        print(f"CANNOT WRITE: {error}")
        return EXIT_CANNOT_CHECK

    args.output.parent.mkdir(parents=True, exist_ok=True)
    restored.to_parquet(args.output, index=False)
    print(f"restored {comparison.mismatched_rows} label(s); wrote {args.output}")
    return EXIT_MATCHES


if __name__ == "__main__":
    raise SystemExit(main())
