#!/usr/bin/env python3
"""Build the location-only DCC manifest without modifying raw data."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.manifest import build_manifest, load_manifest, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest after the caller has inspected it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_manifest(args.data_root)
    output = write_manifest(rows, args.output, overwrite=args.overwrite)
    loaded = load_manifest(output, data_root=args.data_root)
    counts = Counter(row.split for row in loaded)
    print(f"Manifest: {output.resolve()}")
    print(f"Rows: {len(loaded)}")
    for split in sorted(counts):
        print(f"{split}: {counts[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
