#!/usr/bin/env python3
"""Mission 2 L2 (CNN) error analysis -- same breakdown as L1's
scripts/analyze_mission2_errors.py, so the two are directly comparable.

Re-derives the exact same call_id-grouped validation split used during
training (dcc_data.mission2_train.make_group_split -- the CNN's own
spectrogram index carries the same call_id/utt_idx/speaker/duration_sec
columns as L1's feature parquet, so the split reproduces identically given
the same seed/test_size), re-predicts on those held-out utterances with
the saved checkpoint's *best* epoch weights, and breaks errors down by:

  - overlap: does this utterance's [startAt, endAt) overlap another
    utterance in the same call?
  - duration bucket: are short utterances disproportionately wrong?
  - true class: reproduces the confusion-matrix asymmetry (대원 vs 신고자)
  - prediction confidence: are wrong predictions confidently wrong, or
    borderline/uncertain?
  - per-call error concentration

The CNN's own spectrogram index (see dcc_data.mission2_spectrogram) does
not carry startAt/endAt (only duration_sec), so overlap flags come from
--utterance-index -- the L1 utterance-index parquet built by the same
select_seoul_training_rows()+sample_calls(seed, n_calls=0) sampling, which
covers the same (call_id, utt_idx) pairs.

Usage:
    python scripts/analyze_mission2_cnn_errors.py \\
        --index data/m2_spectrogram_index_full.parquet \\
        --spectrogram-dir data/m2_spectrograms_full \\
        --utterance-index data/m2_utterances_train_full.parquet \\
        --ckpt-path checkpoints/mission2/m2_cnn_full.pt
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
import torch  # noqa: E402

from dcc_data.mission2_analysis import DURATION_LABELS, bucket_duration, compute_overlap_flags  # noqa: E402
from dcc_data.mission2_cnn import SpeakerCNN, SpeakerCNNLarge, SpectrogramDataset, collate_batch, predict_with_confidence  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402

MODEL_CLASSES = {"small": SpeakerCNN, "large": SpeakerCNNLarge}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path("data/m2_spectrogram_index_full.parquet"))
    parser.add_argument("--spectrogram-dir", type=Path, default=Path("data/m2_spectrograms_full"))
    parser.add_argument("--utterance-index", type=Path, default=Path("data/m2_utterances_train_full.parquet"))
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/mission2/m2_cnn_full.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="small", choices=list(MODEL_CLASSES), help="must match the --model the checkpoint was trained with")
    parser.add_argument("--output", type=Path, default=Path("reports/mission2/m2_cnn_error_analysis.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/mission2/m2_cnn_error_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    df = pd.read_parquet(args.index)
    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    val_df = df.iloc[val_idx][["call_id", "utt_idx", "speaker", "duration_sec"]].reset_index(drop=True).copy()

    val_loader = torch.utils.data.DataLoader(
        SpectrogramDataset(df.iloc[val_idx], args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model = MODEL_CLASSES[args.model]()
    model.load_state_dict(checkpoint["best_model_state_dict"])
    print(f"Loaded checkpoint: best_epoch={checkpoint.get('best_epoch')} best_val_acc={checkpoint.get('best_val_acc'):.4f}")

    predicted, true_labels, confidence = predict_with_confidence(model, val_loader, device=args.device)
    if not (true_labels == val_df["speaker"].to_numpy()).all():
        raise AssertionError("prediction order does not match val_df row order -- DataLoader must not shuffle")

    val_df["predicted"] = predicted
    val_df["confidence"] = confidence
    val_df["correct"] = val_df["speaker"] == val_df["predicted"]

    utterance_index_df = pd.read_parquet(args.utterance_index)
    overlap_df = compute_overlap_flags(utterance_index_df)
    val_df = val_df.merge(overlap_df, on=["call_id", "utt_idx"], how="left")
    if val_df["overlaps_other"].isna().any():
        missing = int(val_df["overlaps_other"].isna().sum())
        print(f"[warn] {missing} validation utterance(s) had no match in --utterance-index; overlap flag left as NaN", file=sys.stderr)

    val_df["duration_bucket"] = bucket_duration(val_df["duration_sec"])

    overall_accuracy = float(val_df["correct"].mean())

    def group_error_rate(by: str) -> dict:
        table = val_df.groupby(by, observed=True)["correct"].agg(["mean", "count"])
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
    per_call = per_call[per_call["count"] >= 3]
    worst_calls = per_call.sort_values("mean").head(10)

    summary = {
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_val_acc": checkpoint.get("best_val_acc"),
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
