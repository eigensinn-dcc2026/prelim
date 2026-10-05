"""Run the deep-dive EDA that quantifies what the three missions depend on.

Reads every call JSON once (Training 29,200 / Validation 3,640), reuses the
waveform sample cache and ``eda_full_summary.json`` produced by
``scripts/eda_dcc_dataset.py``, and writes:

* ``reports/eda/eda_deep_summary.json`` -- aggregate-only statistics
* ``reports/eda/figures/deep_*.png``    -- evidence figures
* ``reports/eda/eda_report.md``         -- regenerated with the deep-dive section

Examples::

    python scripts/eda_dcc_dataset.py      # prerequisite (full summary + cache)
    python scripts/eda_deep_dive.py
    python scripts/eda_deep_dive.py --json-max-train 50 --json-max-val 20 --no-report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dcc_data.eda_deep import build_deep_summary, load_deep_rows  # noqa: E402
from dcc_data.eda_deep_report import render_deep_figures  # noqa: E402
from dcc_data.eda_report import build_markdown_report  # noqa: E402
from dcc_data.manifest import ManifestRow, load_manifest  # noqa: E402


EDA_DIR = REPO_ROOT / "reports" / "eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "manifests" / "dcc_manifest_v1.csv"),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DCC_DATA_ROOT", str(REPO_ROOT / "data" / "raw_original")),
    )
    parser.add_argument("--full-summary", default=str(EDA_DIR / "eda_full_summary.json"))
    parser.add_argument(
        "--waveform-cache", default=str(EDA_DIR / "waveform_sample_features.parquet")
    )
    parser.add_argument("--output", default=str(EDA_DIR / "eda_deep_summary.json"))
    parser.add_argument("--report-output", default=str(EDA_DIR / "eda_report.md"))
    parser.add_argument("--figures-dir", default=str(EDA_DIR / "figures"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-max-train", type=int, default=0, help="Development limit; 0 = all.")
    parser.add_argument("--json-max-val", type=int, default=0, help="Development limit; 0 = all.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--no-report", action="store_true", help="Do not regenerate eda_report.md."
    )
    return parser.parse_args()


def _limit(rows: list[ManifestRow], limit: int) -> list[ManifestRow]:
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    full_summary_path = Path(args.full_summary)
    waveform_cache = Path(args.waveform_cache)
    for label, path, hint in (
        ("Data directory", data_root, ""),
        ("Manifest", Path(args.manifest), "; prepare it with scripts/build_manifest.py"),
        ("Full EDA summary", full_summary_path, "; run scripts/eda_dcc_dataset.py first"),
        ("Waveform cache", waveform_cache, "; run scripts/eda_dcc_dataset.py first"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}{hint}")

    manifest = load_manifest(args.manifest, validate_paths=False)
    rows = [
        *_limit([row for row in manifest if row.split == "Training"], args.json_max_train),
        *_limit([row for row in manifest if row.split == "Validation"], args.json_max_val),
    ]

    print("[1/4] Reading call JSON")
    records, detail = load_deep_rows(rows, data_root)
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    waveform_frame = pd.read_parquet(waveform_cache)

    print("[2/4] Deep-dive statistics")
    summary = build_deep_summary(
        records, detail, waveform_frame, full_summary, seed=args.seed
    )
    summary["meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("[3/4] Figures")
    if not args.skip_plots:
        render_deep_figures(summary, records, Path(args.figures_dir))

    print("[4/4] Markdown report")
    if not args.no_report:
        Path(args.report_output).write_text(
            build_markdown_report(full_summary, deep=summary), encoding="utf-8"
        )

    print(f"[summary] {output_path}")
    if not args.no_report:
        print(f"[report]  {args.report_output}")


if __name__ == "__main__":
    main()
