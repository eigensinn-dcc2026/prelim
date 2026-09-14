#!/usr/bin/env python3
"""Scan an error-analysis CSV (L1's or L2's) for calls whose low accuracy
is best explained by a label error (whole-call swap or mid-call
discontinuity), not by the call being genuinely hard for the model.

See dcc_data.mission2_analysis.find_label_discontinuity_candidates's
docstring for the algorithm. Found by hand first: the CNN error analysis's
worst-10-calls list included 651e4c754428de718c773b56 (one of the 5
calls already "fixed" as a whole-call swap) still at 28.57% accuracy --
turned out only utt_idx 0-5 were mislabeled in the direction we fixed;
6+ needed the *opposite* flip. This scans every call (not just a
hand-picked worst-10) for that same signature.

Usage:
    python scripts/find_mission2_label_discontinuities.py \\
        --error-csv reports/mission2/m2_cnn_error_analysis.csv \\
        --output reports/mission2/m2_label_discontinuities.json
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

from dcc_data.mission2_analysis import find_label_discontinuity_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--error-csv", type=Path, required=True, help="output of analyze_mission2_errors.py or analyze_mission2_cnn_errors.py")
    parser.add_argument("--min-utterances", type=int, default=5)
    parser.add_argument("--min-corrected-accuracy", type=float, default=0.9)
    parser.add_argument("--min-improvement", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("reports/mission2/m2_label_discontinuities.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.error_csv)

    candidates = find_label_discontinuity_candidates(
        df,
        min_utterances=args.min_utterances,
        min_corrected_accuracy=args.min_corrected_accuracy,
        min_improvement=args.min_improvement,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{df['call_id'].nunique()} calls scanned, {len(candidates)} label-discontinuity candidate(s) found")
    for c in candidates:
        whole_call = c["flip_from_utt_idx"] == 0
        kind = "전체 뒤집기" if whole_call else f"utt_idx {c['flip_from_utt_idx']}부터 뒤집기"
        print(
            f"  {c['call_id']}: raw={c['raw_accuracy']:.4f} -> corrected={c['corrected_accuracy']:.4f} "
            f"(n={c['n_utterances']}, {kind})"
        )
    print(f"\nOutput: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
