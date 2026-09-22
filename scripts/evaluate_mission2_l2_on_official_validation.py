#!/usr/bin/env python3
"""Evaluate a saved L2 (CNN) checkpoint against the official Validation set.

Mirrors scripts/evaluate_mission2_l1_on_official_validation.py exactly in
spirit: the model was trained/checkpoint-selected entirely against Training
data (plus its own internal GroupShuffleSplit holdout), and this is the
first time it sees official Validation -- a one-shot check, not a tuning
signal (see DECISIONS [007] in docs/decisions.md). Nothing here re-trains
or re-tunes the model.

Usage:
    python scripts/evaluate_mission2_l2_on_official_validation.py \\
        --ckpt-path checkpoints/mission2/m2_cnn_large.pt \\
        --index data/m2_spectrogram_index_validation.parquet \\
        --spectrogram-dir data/m2_spectrograms_validation
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
from sklearn.metrics import confusion_matrix  # noqa: E402

from dcc_data.mission2_cnn import (  # noqa: E402
    SpeakerCNN,
    SpeakerCNNLarge,
    SpeakerCNNLargeWithDuration,
    SpectrogramDataset,
    SpectrogramWithDurationDataset,
    collate_batch,
    collate_batch_with_duration,
    evaluate,
)

MODEL_CLASSES = {"small": SpeakerCNN, "large": SpeakerCNNLarge, "large-duration": SpeakerCNNLargeWithDuration}
DATASET_CLASSES = {"small": SpectrogramDataset, "large": SpectrogramDataset, "large-duration": SpectrogramWithDurationDataset}
COLLATE_FNS = {"small": collate_batch, "large": collate_batch, "large-duration": collate_batch_with_duration}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="validation spectrogram index parquet")
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=str, default="large", choices=list(MODEL_CLASSES),
        help="must match the architecture the checkpoint was trained with -- the shipped "
        "91.22%% internal-split result used 'large' (SpeakerCNNLarge).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-output", type=Path, default=Path("reports/mission2/m2_l2_official_validation_result.json")
    )
    parser.add_argument(
        "--predictions-output", type=Path, default=None,
        help="optional: also save a per-utterance (call_id, utt_idx, speaker, predicted) CSV -- "
        "e.g. for joining against another model's predictions on the same split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    df = pd.read_parquet(args.index)

    dataset_cls = DATASET_CLASSES[args.model]
    collate_fn = COLLATE_FNS[args.model]
    loader = torch.utils.data.DataLoader(
        dataset_cls(df, args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model_cls = MODEL_CLASSES[args.model]
    model = model_cls()
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")  # matches train_mission2_cnn.py's own loading
    # Same "best" vs. pre-resume-support checkpoint distinction
    # train_mission2_cnn.py's --resume path handles -- see its docstring.
    if "best_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["best_model_state_dict"])
        source_epoch = checkpoint.get("best_epoch")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        source_epoch = checkpoint.get("epoch")

    accuracy, preds, labels = evaluate(model, loader, args.device)
    if not np.array_equal(labels, df["speaker"].to_numpy()):
        raise ValueError("dataset iteration order does not match --index row order")

    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        df[["call_id", "utt_idx", "speaker"]].assign(predicted=preds).to_csv(
            args.predictions_output, index=False
        )
        print(f"Per-utterance predictions: {args.predictions_output}")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    recall_daewon = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else float("nan")
    recall_singoja = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else float("nan")

    result = {
        "n_utterances": len(df),
        "n_calls": int(df["call_id"].nunique()),
        "model": args.model,
        "checkpoint": str(args.ckpt_path),
        "checkpoint_source_epoch": source_epoch,
        "accuracy": accuracy,
        "recall_daewon_0": recall_daewon,
        "recall_singoja_1": recall_singoja,
        "recall_gap_singoja_minus_daewon": recall_singoja - recall_daewon,
        "confusion_matrix": cm.tolist(),
    }

    print(f"Official Validation: {result['n_calls']} calls / {result['n_utterances']} utterances")
    print(f"Model: {args.model} (checkpoint epoch: {source_epoch})")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"대원(0) recall:  {recall_daewon:.4f}")
    print(f"신고자(1) recall: {recall_singoja:.4f}")
    print(f"격차:            {recall_singoja - recall_daewon:+.4f}")
    print(f"Confusion matrix: {cm.tolist()}")

    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.log_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
