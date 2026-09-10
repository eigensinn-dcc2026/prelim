#!/usr/bin/env python3
"""Train the Mission 2 L2 model: a small CNN over log-mel spectrograms.

Reads the spectrogram index produced by scripts/build_mission2_spectrograms.py,
reuses the exact same call_id-grouped 80/20 split as the L1 baseline
(dcc_data.mission2_train.make_group_split) so accuracy is comparable, and
trains dcc_data.mission2_cnn.SpeakerCNN.

CPU works for a small smoke run; for anything close to the full dataset,
run this on Colab GPU (pass --device cuda) the same way the team already
runs scripts/build_mission2_dataset.py there.

Usage:
    python scripts/train_mission2_cnn.py \\
        --index data/m2_spectrogram_index.parquet \\
        --spectrogram-dir data/m2_spectrograms \\
        --epochs 10 --device cuda \\
        --model-output checkpoints/mission2/m2_cnn.pt
"""

from __future__ import annotations

import argparse
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

from dcc_data.mission2_cnn import SpeakerCNN, SpectrogramDataset, collate_batch, evaluate, train_one_epoch  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path("data/m2_spectrogram_index.parquet"))
    parser.add_argument("--spectrogram-dir", type=Path, default=Path("data/m2_spectrograms"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-output", type=Path, default=Path("checkpoints/mission2/m2_cnn.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    df = pd.read_parquet(args.index)
    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"train={len(train_df)} val={len(val_df)} train_calls={train_df['call_id'].nunique()} val_calls={val_df['call_id'].nunique()}")

    train_loader = torch.utils.data.DataLoader(
        SpectrogramDataset(train_df, args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        SpectrogramDataset(val_df, args.spectrogram_dir),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers,
    )

    model = SpeakerCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device=args.device)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device=args.device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, args.model_output)

    recall_0 = recall_score(val_labels, val_preds, pos_label=0)
    recall_1 = recall_score(val_labels, val_preds, pos_label=1)
    print(f"Final val accuracy: {val_acc:.4f} (best: {best_acc:.4f})")
    print(f"Recall 대원(0)={recall_0:.4f} 신고자(1)={recall_1:.4f}")
    print(f"Confusion matrix: {confusion_matrix(val_labels, val_preds, labels=[0, 1]).tolist()}")
    print(f"Model saved: {args.model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
