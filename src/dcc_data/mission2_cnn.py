"""Mission 2 L2 model: a small CNN over log-mel spectrograms.

See dcc_data.mission2_spectrogram's module docstring for why (L1's
per-utterance summary statistics plateaued without closing the two error
modes that matter -- duration-bucket bias and the overlap gap).

Kept intentionally small (3 conv blocks) -- this is a first correctness
pass, not a tuned architecture; the team can widen/deepen it once the
pipeline is confirmed to work on Colab GPU (this module is exercised here
only with tiny CPU-only smoke tests, see tests/test_mission2_cnn.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


class SpectrogramDataset(torch.utils.data.Dataset):
    """Reads precomputed .npy spectrograms named by an index DataFrame's
    ``npy_path`` column (relative to ``spectrogram_dir``) -- see
    dcc_data.mission2_spectrogram.build_seoul_training_spectrograms, which
    produces exactly this index. Loading a saved array is far cheaper than
    re-decoding audio + recomputing the spectrogram every epoch.
    """

    def __init__(self, index_df: pd.DataFrame, spectrogram_dir: str | Path) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.spectrogram_dir = Path(spectrogram_dir)

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.index_df.iloc[idx]
        spec = np.load(self.spectrogram_dir / row["npy_path"])
        return torch.from_numpy(spec), int(row["speaker"])


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad every spectrogram in the batch to the batch's own max frame count.

    Padding uses SPEC_FLOOR_DB (silence under mel_spectrogram's own
    normalization), never 0.0 -- 0 dB would read as this utterance's
    *loudest* frame instead of padding.
    """

    specs, labels = zip(*batch)
    n_mels = specs[0].shape[0]
    max_frames = max(spec.shape[1] for spec in specs)

    padded = torch.full((len(specs), 1, n_mels, max_frames), SPEC_FLOOR_DB, dtype=torch.float32)
    for i, spec in enumerate(specs):
        padded[i, 0, :, : spec.shape[1]] = spec
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, labels_tensor


class SpeakerCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def train_one_epoch(model: nn.Module, loader, optimizer, device: str) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for specs, labels in loader:
        specs = specs.to(device)
        logits = model(specs)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    accuracy = float((preds == labels_arr).mean())
    return accuracy, preds, labels_arr
