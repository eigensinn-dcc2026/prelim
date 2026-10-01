#!/usr/bin/env python3
"""Scan Seoul/Training calls for suspected speaker-label swaps.

A handful of calls have their 대원(speaker 0)/신고자(speaker 1) labels
reversed for the *entire* call at the source -- see
reports/mission2/m2_label_swap_audit.md for how the first ones were found
and confirmed by reading transcript text (never fed to the model; see
dcc_data.mission2_features's module docstring on why "text" is off-limits as
a feature -- this is a one-time data-quality audit, not modeling).

This is a cheap, no-audio first pass (JSON only) meant to narrow a
dataset-wide search down to a short list a human can then read/listen to --
see dcc_data.mission2_analysis.find_label_swap_candidates's docstring for
its known blind spots (misses greetings that don't match the fixed phrase;
occasionally flags a legitimate 119-to-119 inter-region transfer call).

Usage (from the repository root; run scripts/build_manifest.py first if
--manifest does not exist yet):

    python scripts/scan_mission2_label_swaps.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw \\
        --output reports/mission2/m2_label_swap_candidates.json
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

from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission2_analysis import find_label_swap_candidates  # noqa: E402
from dcc_data.mission2_features import select_seoul_training_rows  # noqa: E402

# Calls already known to look swapped (listed in reports/mission2/m2_label_noise_suspected_calls.md). They are only
# left out of this scan's candidate list so they are not reported twice; nothing here changes a label -- the
# organizers ruled (2026-10-01) that labels must be used as provided.
KNOWN_SUSPECTED_CALL_IDS = frozenset(
    {
        "651e5550bfd63782eb3866a5",
        "651e4b1829d82455557a23b2",
        "651e50c72f06ed4a6e31e3d4",
        "651e4c754428de718c773dab",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mission2/m2_label_swap_candidates.json")
    )
    return parser.parse_args()


def _load_calls(rows, data_root: Path) -> list[tuple[str, list[dict], str, str]]:
    calls = []
    for row in rows:
        json_path = data_root / row.json_path
        try:
            with json_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the whole scan
            print(f"skip (unreadable): {json_path}: {type(exc).__name__}: {exc}")
            continue
        utterances = document.get("utterances")
        if not isinstance(utterances, list):
            continue
        call_id = str(document.get("_id") or json_path.stem)
        calls.append((call_id, utterances, row.wav_path, row.json_path))
    return calls


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)
    seoul_training_rows = select_seoul_training_rows(rows)
    print(f"{len(seoul_training_rows)} Seoul/Training calls in manifest")

    loaded = _load_calls(seoul_training_rows, args.data_root)
    path_by_call_id = {call_id: (wav_path, json_path) for call_id, _, wav_path, json_path in loaded}
    calls_for_scan = [(call_id, utterances) for call_id, utterances, _, _ in loaded]

    candidates = find_label_swap_candidates(calls_for_scan, exclude_call_ids=KNOWN_SUSPECTED_CALL_IDS)
    for candidate in candidates:
        wav_path, json_path = path_by_call_id[candidate["call_id"]]
        candidate["wav_path"] = wav_path
        candidate["json_path"] = json_path

    print(f"checked={len(loaded)} candidates={len(candidates)} (already-known suspected calls excluded)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
