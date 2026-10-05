#!/usr/bin/env python3
"""Train the Mission 2 L2 model: a small CNN over log-mel spectrograms.

Reads the spectrogram index produced by scripts/build_mission2_spectrograms.py
(and refuses any index that isn't stamped as built from Training -- see
dcc_data.mission2_spectrogram.assert_training_spectrogram_index),
reuses the exact same call_id-grouped 80/20 split as the L1 baseline
(dcc_data.mission2_train.make_group_split) so accuracy is comparable, and
trains dcc_data.mission2_cnn.SpeakerCNN.

Uses ReduceLROnPlateau (halves the LR once val accuracy stops improving)
and always reports/saves the *best* epoch's checkpoint and metrics, not
whatever the last epoch happened to be -- a first Colab run showed val_acc
swinging 20+ points between adjacent epochs late in training (train_loss
still falling smoothly the whole time, the classic "LR too high" symptom),
and the final epoch can land mid-oscillation, far worse than the peak
already reached.

--resume continues from --model-output instead of starting over: a second
Colab run (same 3,000-call sample) hit its best (86.11%) on the very last
of 20 epochs, meaning it likely hadn't converged -- re-running 20 fresh
epochs to get back to where training already was would waste ~20 minutes
on GPU. The checkpoint carries the *last* epoch's model/optimizer/scheduler
state (to resume the LR schedule and Adam's momentum exactly where they
were) separately from the *best* epoch's model state (for inference), so a
resumed run that doesn't beat the standing best still doesn't overwrite it.
A resumed run saves a checkpoint stamped with *its* index provenance, so --resume first checks that the
file it continues from records exactly the same config (model, mel settings, label source, split, seeds --
dcc_data.mission2_checkpoint.require_resumable_checkpoint) and refuses one that records nothing, e.g. any
file written before the config was stored.

CPU works for a small smoke run; for anything close to the full dataset,
run this on Colab GPU (pass --device cuda) the same way the team already
runs scripts/build_mission2_dataset.py there.

Usage:
    python scripts/train_mission2_cnn.py \\
        --index data/m2_spectrogram_index.parquet \\
        --spectrogram-dir data/m2_spectrograms \\
        --epochs 20 --device cuda \\
        --model-output checkpoints/mission2/m2_cnn.pt

    # continue training the same checkpoint for 20 more epochs:
    python scripts/train_mission2_cnn.py ... --epochs 20 --resume
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, recall_score  # noqa: E402

from dcc_data.mission2_cnn import (  # noqa: E402
    SpeakerCNN,
    SpeakerCNNLarge,
    SpeakerCNNLargeWithDuration,
    SpectrogramDataset,
    SpectrogramWithDurationDataset,
    collate_batch,
    collate_batch_with_duration,
    evaluate,
    move_optimizer_state_to_device,
    train_with_best_checkpoint,
)
from dcc_data.mission2_checkpoint import build_checkpoint_metadata, require_resumable_checkpoint  # noqa: E402
from dcc_data.mission2_spectrogram import assert_training_spectrogram_index  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402

MODEL_CLASSES = {"small": SpeakerCNN, "large": SpeakerCNNLarge, "large-duration": SpeakerCNNLargeWithDuration}
# large-duration's loader yields (specs, duration, label) instead of (specs,
# label) -- see SpeakerCNNLargeWithDuration's docstring -- so its Dataset/
# collate_fn differ too; every other model shares the plain pair.
DATASET_CLASSES = {"small": SpectrogramDataset, "large": SpectrogramDataset, "large-duration": SpectrogramWithDurationDataset}
COLLATE_FNS = {"small": collate_batch, "large": collate_batch, "large-duration": collate_batch_with_duration}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path("data/m2_spectrogram_index.parquet"))
    parser.add_argument("--spectrogram-dir", type=Path, default=Path("data/m2_spectrograms"))
    parser.add_argument("--seed", type=int, default=42, help="seed for the call_id-grouped train/val split only (make_group_split)")
    parser.add_argument(
        "--torch-seed", type=int, default=None,
        help="seed for model init + DataLoader shuffle order (defaults to --seed). Kept separate from --seed "
        "so repeated runs can keep the same held-out val split while varying only these two sources of "
        "run-to-run noise -- collate_batch pads each batch to its own longest utterance, so an uncontrolled "
        "shuffle order changes what a short utterance is batched with, not just its weight-init luck.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20, help="how many MORE epochs to run this invocation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=2, help="epochs with no val_acc improvement before halving lr")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model", type=str, default="small", choices=list(MODEL_CLASSES),
        help="small=SpeakerCNN (25.7k params), large=SpeakerCNNLarge (422k params), "
        "large-duration=SpeakerCNNLarge + this utterance's own duration_sec as an explicit input",
    )
    parser.add_argument("--model-output", type=Path, default=Path("checkpoints/mission2/m2_cnn.pt"))
    parser.add_argument("--resume", action="store_true", help="continue from --model-output instead of a fresh model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    torch_seed = args.torch_seed if args.torch_seed is not None else args.seed
    random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    train_shuffle_generator = torch.Generator()
    train_shuffle_generator.manual_seed(torch_seed)
    print(f"torch_seed={torch_seed} (model init + train shuffle order); split seed={args.seed}")

    df = pd.read_parquet(args.index)
    # Refuse anything that isn't stamped as built from Training: the official Validation folder must
    # never reach the loss/backprop, even by pointing --index at the wrong cache.
    index_provenance = assert_training_spectrogram_index(df)
    print(f"index provenance: {index_provenance}")
    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"train={len(train_df)} val={len(val_df)} train_calls={train_df['call_id'].nunique()} val_calls={val_df['call_id'].nunique()}")

    dataset_cls = DATASET_CLASSES[args.model]
    collate_fn = COLLATE_FNS[args.model]
    train_loader = torch.utils.data.DataLoader(
        dataset_cls(train_df, args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers,
        generator=train_shuffle_generator,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_cls(val_df, args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model_cls = MODEL_CLASSES[args.model]
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    # What this run will stamp on the checkpoint it saves. Built up front because --resume must first prove that
    # the file it continues from was trained with exactly this (see require_resumable_checkpoint): otherwise
    # old weights trained on other labels/settings would leave this run stamped with today's provenance.
    checkpoint_metadata = build_checkpoint_metadata(
        model_name=args.model,
        model_class=model_cls.__name__,
        index_provenance=index_provenance,
        split_seed=args.seed,
        test_size=args.test_size,
        torch_seed=torch_seed,
    )

    prev_best_val_acc = -1.0
    prev_best_model_state = None
    prev_best_epoch = 0
    epochs_already_trained = 0

    if args.resume:
        if not args.model_output.exists():
            raise SystemExit(f"--resume given but {args.model_output} does not exist")
        checkpoint = torch.load(args.model_output, map_location="cpu")
        require_resumable_checkpoint(checkpoint, checkpoint_metadata, args.model_output)
        model.load_state_dict(checkpoint["last_model_state_dict"])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, args.device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        prev_best_val_acc = checkpoint["best_val_acc"]
        prev_best_model_state = checkpoint["best_model_state_dict"]
        prev_best_epoch = checkpoint["best_epoch"]
        epochs_already_trained = checkpoint["total_epochs_trained"]
        print(
            f"Resumed from {args.model_output}: {epochs_already_trained} epochs already trained, "
            f"standing best val_acc={prev_best_val_acc:.4f} (epoch {prev_best_epoch})"
        )

    def on_epoch_end(record: dict) -> None:
        lr = optimizer.param_groups[0]["lr"]
        absolute_epoch = epochs_already_trained + record["epoch"]
        print(f"epoch {absolute_epoch:3d}  train_loss={record['train_loss']:.4f}  val_acc={record['val_acc']:.4f}  lr={lr:.2e}")

    start = time.time()
    result = train_with_best_checkpoint(
        model, train_loader, val_loader, optimizer,
        epochs=args.epochs, device=args.device, scheduler=scheduler, on_epoch_end=on_epoch_end,
    )
    elapsed = time.time() - start
    total_epochs_trained = epochs_already_trained + args.epochs

    improved = result["best_val_acc"] > prev_best_val_acc
    if improved:
        best_val_acc = result["best_val_acc"]
        best_model_state = result["best_model_state"]
        best_epoch = epochs_already_trained + result["best_epoch"]
        best_preds, best_labels = result["best_val_preds"], result["best_val_labels"]
    else:
        best_val_acc = prev_best_val_acc
        best_model_state = prev_best_model_state
        best_epoch = prev_best_epoch
        best_preds = best_labels = None

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_model_state_dict": best_model_state,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "last_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "total_epochs_trained": total_epochs_trained,
            # What this checkpoint was trained with -- scripts/mission2_inference.py refuses to run it if
            # that differs from how it would run it (see dcc_data.mission2_checkpoint).
            "metadata": checkpoint_metadata,
        },
        args.model_output,
    )

    print(f"\nBest epoch so far: {best_epoch}/{total_epochs_trained}  val_acc={best_val_acc:.4f}  (this run: {elapsed:.1f}s)")
    if not improved:
        print("(this run did not beat the standing best -- checkpoint's best/resume state both updated, recall/confusion matrix below are the standing best's)")
        eval_model = model_cls()
        eval_model.load_state_dict(best_model_state)
        _, best_preds, best_labels = evaluate(eval_model, val_loader, device=args.device)
    recall_0 = recall_score(best_labels, best_preds, pos_label=0)
    recall_1 = recall_score(best_labels, best_preds, pos_label=1)
    print(f"Recall 대원(0)={recall_0:.4f} 신고자(1)={recall_1:.4f}")
    print(f"Confusion matrix: {confusion_matrix(best_labels, best_preds, labels=[0, 1]).tolist()}")
    print(f"Model saved: {args.model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
