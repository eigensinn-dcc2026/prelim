"""Run exhaustive DCC label/WAV-header EDA plus sampled waveform analysis.

Default execution profiles all 29,200 Training and 3,640 Validation call pairs.
WAV samples are decoded only for a deterministic gender-stratified sample.
The preserved legacy file ``reports/eda/eda_summary.json`` is never overwritten.

Examples::

    python scripts/eda_dcc_dataset.py
    python scripts/eda_dcc_dataset.py --json-max-train 10 --json-max-val 10 \
        --waveform-sample-train 4 --waveform-sample-val 2
    python scripts/eda_dcc_dataset.py --skip-waveform
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dcc_data.audio import read_wav_info  # noqa: E402
from dcc_data.eda import (  # noqa: E402
    analyze_split,
    compare_splits,
    run_waveform_sample,
)
from dcc_data.eda_report import build_markdown_report, render_figures  # noqa: E402
from dcc_data.manifest import ManifestRow, load_manifest  # noqa: E402


LEGACY_EDA_PATH = REPO_ROOT / "reports" / "eda" / "eda_summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "eda" / "eda_full_summary.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "eda" / "eda_report.md"
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "eda" / "figures"
DEFAULT_WAV_HEADER_CACHE = REPO_ROOT / "reports" / "eda" / "wav_header_cache.csv"
DEFAULT_WAVEFORM_CACHE = (
    REPO_ROOT / "reports" / "eda" / "waveform_sample_features.parquet"
)
WAV_HEADER_FIELDS = (
    "split",
    "stem",
    "wav_path",
    "file_size",
    "mtime_ns",
    "channels",
    "sample_rate",
    "sample_width",
    "frame_count",
    "duration_s",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "manifests" / "dcc_manifest_v1.csv"),
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--wav-header-cache", default=str(DEFAULT_WAV_HEADER_CACHE))
    parser.add_argument("--waveform-cache", default=str(DEFAULT_WAVEFORM_CACHE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-max-train",
        type=int,
        default=0,
        help="Development limit; 0 profiles all Training rows.",
    )
    parser.add_argument(
        "--json-max-val",
        type=int,
        default=0,
        help="Development limit; 0 profiles all Validation rows.",
    )
    parser.add_argument("--waveform-sample-train", type=int, default=1000)
    parser.add_argument("--waveform-sample-val", type=int, default=400)
    parser.add_argument("--skip-waveform", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--rebuild-wav-cache", action="store_true")
    return parser.parse_args()


# Kept as public helpers because the existing unit tests and timestamp checker
# use these deterministic sampling/profile primitives.
def sample_rows(rows: list[ManifestRow], n: int, rng: random.Random) -> list[ManifestRow]:
    if n <= 0 or n >= len(rows):
        return rows
    return rng.sample(rows, n)


def summarize_durations(values_ms: list[float]) -> dict:
    if not values_ms:
        return {}
    seconds = sorted(value / 1000 for value in values_ms)
    n = len(seconds)

    def percentile(p: float) -> float:
        index = min(n - 1, max(0, int(round(p * (n - 1)))))
        return seconds[index]

    return {
        "n": n,
        "min_s": round(seconds[0], 2),
        "median_s": round(statistics.median(seconds), 2),
        "mean_s": round(statistics.fmean(seconds), 2),
        "p90_s": round(percentile(0.90), 2),
        "max_s": round(seconds[-1], 2),
    }


def summarize_values(values: list[float], unit: str) -> dict:
    if not values:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "unit": unit,
        }
    ordered = sorted(float(value) for value in values)
    n = len(ordered)

    def percentile(p: float) -> float:
        index = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ordered[index]

    return {
        "n": n,
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "unit": unit,
    }


def empty_split_block() -> dict:
    return {"common": {}, "mission1": {}, "mission2": {}, "mission3": {}}


def limit_rows(rows: list[ManifestRow], n: int) -> list[ManifestRow]:
    if n <= 0 or n >= len(rows):
        return rows
    return rows[:n]


def scope_meta(total: int, used: int, limit: int) -> dict:
    limited = 0 < limit < total
    return {
        "mode": "limited" if limited else "full",
        "n_manifest": total,
        "n_used": used,
        "json_max": None if limit <= 0 else limit,
    }


def assert_not_legacy_output(path: Path) -> None:
    if path.resolve() == LEGACY_EDA_PATH.resolve():
        raise ValueError(
            "Refusing to overwrite the preserved legacy EDA file: "
            f"{LEGACY_EDA_PATH}. Pass a different --output path."
        )


def cache_row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["split"], row["stem"])


def load_wav_header_cache(
    cache_path: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    if not cache_path.is_file():
        return {}
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != WAV_HEADER_FIELDS:
            return {}
        return {
            cache_row_key(item): {
                field: item.get(field, "") for field in WAV_HEADER_FIELDS
            }
            for item in reader
        }


def write_wav_header_cache(
    cache_path: Path, rows: dict[tuple[str, str], dict[str, str]]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=WAV_HEADER_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(
                {field: rows[key].get(field, "") for field in WAV_HEADER_FIELDS}
            )


def _blank_wav_audio_fields() -> dict[str, str]:
    return {
        "channels": "",
        "sample_rate": "",
        "sample_width": "",
        "frame_count": "",
        "duration_s": "",
    }


def stat_wav_file(wav_path: Path) -> tuple[str, str, str]:
    try:
        stat = wav_path.stat()
    except OSError as exc:
        return "", "", f"{type(exc).__name__}: {exc}"
    return str(stat.st_size), str(stat.st_mtime_ns), ""


def inspect_wav_header(wav_path: Path) -> dict[str, str]:
    file_size, mtime_ns, stat_error = stat_wav_file(wav_path)
    if stat_error:
        return {
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            **_blank_wav_audio_fields(),
            "error": stat_error,
        }
    result = {
        "file_size": file_size,
        "mtime_ns": mtime_ns,
        **_blank_wav_audio_fields(),
        "error": "",
    }
    try:
        info = read_wav_info(wav_path)
    except Exception as exc:  # noqa: BLE001 - cached as data-quality result
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        {
            "channels": str(info.channels),
            "sample_rate": str(info.sample_rate),
            "sample_width": str(info.sample_width),
            "frame_count": str(info.frame_count),
            "duration_s": f"{info.frame_count / info.sample_rate:.10g}",
        }
    )
    return result


def wav_cache_is_fresh(
    cached: dict[str, str], file_size: str, mtime_ns: str
) -> bool:
    return (
        bool(cached)
        and bool(file_size)
        and bool(mtime_ns)
        and cached.get("file_size") == file_size
        and cached.get("mtime_ns") == mtime_ns
    )


def read_or_build_wav_header_cache(
    rows: list[ManifestRow],
    data_root: Path,
    cache_path: Path,
    *,
    rebuild: bool = False,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, int]]:
    """Read headers with size/mtime invalidation; waveform samples stay untouched."""

    merged = load_wav_header_cache(cache_path)
    stats = {"n_requested": len(rows), "n_reused": 0, "n_computed": 0, "n_error": 0}
    for index, row in enumerate(rows, start=1):
        key = (row.split, row.stem)
        wav_path = data_root / row.wav_path
        file_size, mtime_ns, stat_error = stat_wav_file(wav_path)
        cached = merged.get(key, {})
        if not rebuild and not stat_error and wav_cache_is_fresh(
            cached, file_size, mtime_ns
        ):
            record = dict(cached)
            record["wav_path"] = row.wav_path
            merged[key] = record
            stats["n_reused"] += 1
        else:
            record = {
                "split": row.split,
                "stem": row.stem,
                "wav_path": row.wav_path,
                **inspect_wav_header(wav_path),
            }
            merged[key] = record
            stats["n_computed"] += 1
        if record.get("error"):
            stats["n_error"] += 1
        if index % 2000 == 0 or index == len(rows):
            print(f"  wav header {index:,}/{len(rows):,}")
    write_wav_header_cache(cache_path, merged)
    return merged, stats


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_waveform_cache(rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    report_path = Path(args.report_output)
    figures_dir = Path(args.figures_dir)
    cache_path = Path(args.wav_header_cache)
    waveform_cache_path = Path(args.waveform_cache)
    assert_not_legacy_output(output_path)

    all_rows = load_manifest(args.manifest, validate_paths=False)
    train_rows = [row for row in all_rows if row.split == "Training"]
    val_rows = [row for row in all_rows if row.split == "Validation"]
    train_used = limit_rows(train_rows, args.json_max_train)
    val_used = limit_rows(val_rows, args.json_max_val)
    used_rows = [*train_used, *val_used]

    print("[1/5] WAV header profile")
    headers, wav_stats = read_or_build_wav_header_cache(
        used_rows, data_root, cache_path, rebuild=args.rebuild_wav_cache
    )

    print("[2/5] Exhaustive JSON profile")
    training = analyze_split(train_used, data_root, headers)
    validation = analyze_split(val_used, data_root, headers)

    waveform_rows: list[dict] = []
    if args.skip_waveform:
        waveform_summary: dict = {"skipped": True}
    else:
        print("[3/5] Stratified waveform sample")
        train_features, _train_failures, train_wave_summary = run_waveform_sample(
            training.profiles,
            data_root,
            args.waveform_sample_train,
            args.seed,
        )
        val_features, _val_failures, val_wave_summary = run_waveform_sample(
            validation.profiles,
            data_root,
            args.waveform_sample_val,
            args.seed + 1,
        )
        waveform_rows = [*train_features, *val_features]
        _write_waveform_cache(waveform_rows, waveform_cache_path)
        waveform_summary = {
            "skipped": False,
            "cache_path": _portable_path(waveform_cache_path),
            "training": train_wave_summary,
            "validation": val_wave_summary,
        }

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "manifest_path": _portable_path(Path(args.manifest)),
            "data_root_redacted": True,
            "legacy_eda_path": _portable_path(LEGACY_EDA_PATH),
            "legacy_eda_recomputed": False,
            "training_scope": scope_meta(
                len(train_rows), len(train_used), args.json_max_train
            ),
            "validation_scope": scope_meta(
                len(val_rows), len(val_used), args.json_max_val
            ),
            "wav_header_cache_path": _portable_path(cache_path),
            "wav_header_cache_run": {
                **wav_stats,
                "rebuild": bool(args.rebuild_wav_cache),
            },
            "waveform_sampling": {
                "strategy": "proportional gender stratification within split",
                "training_requested": args.waveform_sample_train,
                "validation_requested": args.waveform_sample_val,
            },
        },
        "training": training.summary,
        "validation": validation.summary,
        "cross_split": compare_splits(training, validation),
        "waveform_sample": waveform_summary,
    }

    print("[4/5] JSON summary and figures")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not args.skip_plots:
        render_figures(result, figures_dir)

    print("[5/5] Markdown report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(result), encoding="utf-8")

    print(f"[summary] {output_path}")
    print(f"[report]  {report_path}")
    if not args.skip_plots:
        print(f"[figures] {figures_dir}")
    if waveform_rows:
        print(f"[waveform cache] {waveform_cache_path}")


if __name__ == "__main__":
    main()
