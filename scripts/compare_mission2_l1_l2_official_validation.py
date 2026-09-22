#!/usr/bin/env python3
"""Compare L1 (LightGBM) and L2 (CNN) per-utterance on official Validation.

Both models were evaluated separately (scripts/evaluate_mission2_l1_on_
official_validation.py, scripts/evaluate_mission2_l2_on_official_validation.py)
but neither saved per-utterance predictions, so there was no way to ask
"which utterances did CNN fix that GBM missed" -- only compare aggregate
accuracy. This re-runs both models' inference (read-only, no training/
tuning) and joins the results by (call_id, utt_idx) against the same
overlap/duration breakdown scripts/analyze_mission2_errors.py uses for L1,
so the two models' error patterns are directly comparable.

Usage:
    python scripts/compare_mission2_l1_l2_official_validation.py \\
        --l1-ckpt checkpoints/mission2/m2_baseline_lgbm_tuned_postfix2.pkl \\
        --l1-features data/local_cache/m2_features_validation_official.parquet \\
        --l2-ckpt checkpoints/mission2/m2_cnn_large.pt \\
        --l2-index data/local_cache/m2_spectrogram_index_validation.parquet \\
        --l2-spectrogram-dir data/m2_spectrograms_validation \\
        --utterance-index data/local_cache/m2_utterances_validation_official.parquet
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dcc_data.mission2_analysis import bucket_duration, compute_overlap_flags  # noqa: E402
from dcc_data.mission2_cnn import SpeakerCNNLarge, SpectrogramDataset, collate_batch, evaluate  # noqa: E402
from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l1-ckpt", type=Path, required=True)
    parser.add_argument("--l1-features", type=Path, required=True)
    parser.add_argument("--l2-ckpt", type=Path, required=True)
    parser.add_argument("--l2-index", type=Path, required=True)
    parser.add_argument("--l2-spectrogram-dir", type=Path, required=True)
    parser.add_argument("--utterance-index", type=Path, required=True, help="has startAt/endAt for overlap flags")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mission2/m2_l1_l2_official_validation_comparison.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("reports/mission2/m2_l1_l2_official_validation_comparison.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- L1 predictions -----------------------------------------------
    l1_model, l1_meta = load_model(args.l1_ckpt)
    l1_feature_columns = tuple(l1_meta["feature_columns"])
    if l1_feature_columns != FEATURE_COLUMNS:
        raise ValueError("L1 checkpoint feature_columns do not match the current feature extractor")
    l1_df = pd.read_parquet(args.l1_features)
    X_l1 = l1_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    l1_pred = l1_model.predict(X_l1)
    l1_result = l1_df[["call_id", "utt_idx", "speaker"]].copy()
    l1_result["l1_pred"] = l1_pred
    print(f"L1: {len(l1_result)} utterances, accuracy={float((l1_result['speaker'] == l1_result['l1_pred']).mean()):.4f}")

    # --- L2 predictions -------------------------------------------------
    l2_index_df = pd.read_parquet(args.l2_index)
    loader = torch.utils.data.DataLoader(
        SpectrogramDataset(l2_index_df, args.l2_spectrogram_dir),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0,
    )
    l2_model = SpeakerCNNLarge()
    checkpoint = torch.load(args.l2_ckpt, map_location="cpu")
    if "best_model_state_dict" in checkpoint:
        l2_model.load_state_dict(checkpoint["best_model_state_dict"])
    else:
        l2_model.load_state_dict(checkpoint["model_state_dict"])
    l2_accuracy, l2_preds, l2_labels = evaluate(l2_model, loader, "cpu")
    print(f"L2: {len(l2_index_df)} utterances, accuracy={l2_accuracy:.4f}")
    if not np.array_equal(l2_labels, l2_index_df["speaker"].to_numpy()):
        raise ValueError("L2 dataset iteration order does not match l2_index_df row order")
    l2_result = l2_index_df[["call_id", "utt_idx", "speaker", "duration_sec"]].copy()
    l2_result["l2_pred"] = l2_preds

    # --- overlap flags ----------------------------------------------------
    utt_index = pd.read_parquet(args.utterance_index).rename(columns={"start_at": "startAt", "end_at": "endAt"})
    overlap_df = compute_overlap_flags(utt_index)

    # --- join everything ------------------------------------------------
    merged = l1_result.merge(l2_result[["call_id", "utt_idx", "l2_pred", "duration_sec"]], on=["call_id", "utt_idx"], how="inner")
    merged = merged.merge(overlap_df, on=["call_id", "utt_idx"], how="left")
    if merged["overlaps_other"].isna().any():
        print(f"[warn] {int(merged['overlaps_other'].isna().sum())} utterances had no overlap-flag match", file=sys.stderr)
    merged["duration_bucket"] = bucket_duration(merged["duration_sec"])
    merged["l1_correct"] = merged["speaker"] == merged["l1_pred"]
    merged["l2_correct"] = merged["speaker"] == merged["l2_pred"]

    if len(merged) != len(l1_result) or len(merged) != len(l2_result):
        print(
            f"[warn] row counts differ after join: l1={len(l1_result)} l2={len(l2_result)} merged={len(merged)}",
            file=sys.stderr,
        )

    def group_stats(by) -> dict:
        table = merged.groupby(by, observed=True)[["l1_correct", "l2_correct"]].agg(["mean", "count"])
        return {
            str(key): {
                "l1_accuracy": float(row[("l1_correct", "mean")]),
                "l2_accuracy": float(row[("l2_correct", "mean")]),
                "n": int(row[("l1_correct", "count")]),
            }
            for key, row in table.iterrows()
        }

    # 2x2: did L1/L2 agree with the truth, independently
    quadrant_counts = (
        merged.groupby(["l1_correct", "l2_correct"], observed=True).size().rename("n").reset_index()
    )
    quadrants = {
        f"l1_{'correct' if row.l1_correct else 'wrong'}_l2_{'correct' if row.l2_correct else 'wrong'}": int(row.n)
        for row in quadrant_counts.itertuples()
    }

    # Same 2x2, but only within the overlap==True subset -- the whole point
    # of L2 existing is to test whether raw time-frequency structure handles
    # overlapping voices better than L1's summary statistics.
    overlap_subset = merged[merged["overlaps_other"] == True]  # noqa: E712
    overlap_quadrant_counts = (
        overlap_subset.groupby(["l1_correct", "l2_correct"], observed=True).size().rename("n").reset_index()
    )
    overlap_quadrants = {
        f"l1_{'correct' if row.l1_correct else 'wrong'}_l2_{'correct' if row.l2_correct else 'wrong'}": int(row.n)
        for row in overlap_quadrant_counts.itertuples()
    }

    summary = {
        "n_utterances": len(merged),
        "l1_accuracy": float(merged["l1_correct"].mean()),
        "l2_accuracy": float(merged["l2_correct"].mean()),
        "by_overlap": group_stats("overlaps_other"),
        "by_duration_bucket": group_stats("duration_bucket"),
        "by_true_class": group_stats("speaker"),
        "agreement_quadrants": quadrants,
        "agreement_quadrants_overlap_only": overlap_quadrants,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"L1 accuracy: {summary['l1_accuracy']:.4f} / L2 accuracy: {summary['l2_accuracy']:.4f}")
    print()
    print("겹침 여부:")
    for key, stats in summary["by_overlap"].items():
        print(f"  overlaps_other={key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("발화 길이 구간:")
    for key, stats in summary["by_duration_bucket"].items():
        print(f"  {key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("실제 speaker:")
    for key, stats in summary["by_true_class"].items():
        print(f"  speaker={key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("전체 동의/불일치 구조:", quadrants)
    print("겹침 구간만:", overlap_quadrants)
    print()
    print(f"per-utterance 상세: {args.output}")
    print(f"요약 JSON: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
