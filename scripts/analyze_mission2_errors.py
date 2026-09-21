#!/usr/bin/env python3
"""Mission 2 baseline error analysis: where is the 16.22% coming from?

Re-derives the exact same call_id-grouped validation split used during
training (same seed/test_size -> dcc_data.mission2_train.make_group_split),
re-predicts on those held-out utterances with the saved checkpoint, and
breaks the errors down by:

  - overlap: does this utterance's [startAt, endAt) overlap another
    utterance in the same call? (the 37.1% overlap rate the team EDA found,
    ignored during baseline feature extraction -- this checks whether it's
    actually hurting accuracy)
  - duration bucket: are short utterances disproportionately wrong?
  - true class: reproduces the confusion-matrix asymmetry (대원 vs 신고자)
  - prediction confidence: are wrong predictions confidently wrong, or
    borderline/uncertain?
  - per-call error concentration: are errors spread evenly across calls,
    or clustered in a handful of "hard" calls?

Usage:
    python scripts/analyze_mission2_errors.py \\
        --features data/m2_features_train.parquet \\
        --index data/m2_utterances_train.parquet \\
        --ckpt-path checkpoints/mission2/m2_baseline_lgbm.pkl
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

from dcc_data.mission2_analysis import DURATION_LABELS, bucket_duration, compute_overlap_flags  # noqa: E402
from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import load_model, make_group_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("data/m2_features_train.parquet"))
    parser.add_argument("--index", type=Path, default=Path("data/m2_utterances_train.parquet"))
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/mission2/m2_baseline_lgbm.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("reports/mission2/m2_error_analysis.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/mission2/m2_error_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_df = pd.read_parquet(args.features)
    index_df = pd.read_parquet(args.index)
    model, meta = load_model(args.ckpt_path)

    feature_columns = tuple(meta["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("checkpoint feature_columns do not match the current feature extractor")

    train_idx, val_idx = make_group_split(features_df, seed=args.seed, test_size=args.test_size)
    val_df = features_df.iloc[val_idx][["call_id", "utt_idx", "speaker", "duration_sec"]].copy()

    X_val = features_df.iloc[val_idx][list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    predictions = model.predict(X_val)
    val_df["predicted"] = predictions
    val_df["correct"] = val_df["speaker"] == val_df["predicted"]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)
        # probability the model assigned to the class it actually predicted --
        # low values mean the model was guessing near 50/50, not confidently wrong.
        val_df["confidence"] = proba[np.arange(len(predictions)), predictions]
    else:
        val_df["confidence"] = np.nan

    overlap_df = compute_overlap_flags(index_df)
    val_df = val_df.merge(overlap_df, on=["call_id", "utt_idx"], how="left")
    if val_df["overlaps_other"].isna().any():
        missing = int(val_df["overlaps_other"].isna().sum())
        print(f"[warn] {missing} validation utterance(s) had no match in --index; overlap flag left as NaN", file=sys.stderr)

    val_df["duration_bucket"] = bucket_duration(val_df["duration_sec"])

    overall_accuracy = float(val_df["correct"].mean())

    def group_error_rate(by: str) -> dict:
        # dropna=False: groupby's default (dropna=True) silently excludes
        # NaN-group rows from the result entirely -- for "overlaps_other"
        # specifically, that's the utterances that had no match in --index
        # (already warned about above), so the per-bucket counts here used
        # to not sum to n_validation_utterances with no indication why.
        # Surfaced as an explicit "nan" bucket instead of dropped.
        table = val_df.groupby(by, observed=True, dropna=False)["correct"].agg(["mean", "count"])
        return {
            str(key): {"accuracy": float(row["mean"]), "n": int(row["count"])}
            for key, row in table.iterrows()
        }

    by_overlap = group_error_rate("overlaps_other")
    by_duration = group_error_rate("duration_bucket")
    by_true_class = group_error_rate("speaker")

    confidence_correct = float(val_df.loc[val_df["correct"], "confidence"].mean())
    confidence_wrong = float(val_df.loc[~val_df["correct"], "confidence"].mean())

    per_call = val_df.groupby("call_id")["correct"].agg(["mean", "count"])
    per_call = per_call[per_call["count"] >= 3]  # ignore calls with too few val utterances to be meaningful
    worst_calls = per_call.sort_values("mean").head(10)

    summary = {
        "n_validation_utterances": int(len(val_df)),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_overlap": by_overlap,
        "accuracy_by_duration_bucket": by_duration,
        "accuracy_by_true_class": by_true_class,
        "mean_confidence_when_correct": confidence_correct,
        "mean_confidence_when_wrong": confidence_wrong,
        "worst_calls_top10": [
            {"call_id": str(call_id), "accuracy": float(row["mean"]), "n_utterances": int(row["count"])}
            for call_id, row in worst_calls.iterrows()
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validation utterances: {summary['n_validation_utterances']}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print()
    print("정확도 -- 겹침 여부:")
    for key, stats in by_overlap.items():
        print(f"  overlaps_other={key}: accuracy={stats['accuracy']:.4f} (n={stats['n']})")
    print()
    print("정확도 -- 발화 길이 구간:")
    for label in DURATION_LABELS:
        if label in by_duration:
            stats = by_duration[label]
            print(f"  {label}: accuracy={stats['accuracy']:.4f} (n={stats['n']})")
    print()
    print("정확도 -- 실제 speaker:")
    for key, stats in by_true_class.items():
        print(f"  speaker={key}: accuracy={stats['accuracy']:.4f} (n={stats['n']})")
    print()
    print(f"평균 확신도(confidence) -- 정답일 때: {confidence_correct:.4f}, 오답일 때: {confidence_wrong:.4f}")
    print()
    print("오류가 가장 집중된 통화 top 10 (val 발화 3개 이상):")
    for row in summary["worst_calls_top10"]:
        print(f"  call_id={row['call_id']}: accuracy={row['accuracy']:.4f} (n={row['n_utterances']})")
    print()
    print(f"per-utterance 상세: {args.output}")
    print(f"요약 JSON: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
