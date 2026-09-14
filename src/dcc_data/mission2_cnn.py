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


class SpeakerCNNLarge(nn.Module):
    """4 conv blocks, 32->64->128->256 channels (SpeakerCNN's 3 blocks,
    16->32->64, plateaued at 89.6-89.7% val accuracy on the full 29,200-
    call dataset -- LR decayed to 6.25e-05, val_acc pinned in an ~0.3pp
    band for 10 straight epochs, i.e. converged for that capacity, not for
    lack of training). A separate class, not a resize of SpeakerCNN, so an
    existing SpeakerCNN checkpoint still loads/resumes untouched; pass
    --model large to scripts/train_mission2_cnn.py to use this one (with a
    fresh --model-output, since its state_dict shape doesn't match
    SpeakerCNN's).
    """

    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Move every tensor in ``optimizer.state`` (e.g. Adam's exp_avg/
    exp_avg_sq momentum buffers) to ``device``, in place.

    Needed after ``optimizer.load_state_dict()`` from a checkpoint loaded
    with ``map_location="cpu"``: the optimizer's internal state tensors are
    separate from the model's parameters, so moving the model with
    ``model.to(device)`` does *not* touch them -- the first ``step()`` after
    a resume then fails with "Expected all tensors to be on the same
    device" once the model's params are on cuda but the loaded momentum
    buffers are still on cpu.
    """

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


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


@torch.no_grad()
def predict_with_confidence(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like evaluate(), plus a confidence score per prediction: the softmax
    probability the model assigned to the class it actually predicted --
    mirrors L1's predict_proba-based confidence column in
    scripts/analyze_mission2_errors.py, for the same error-analysis use
    (are wrong predictions confidently wrong, or borderline?). Always in
    [0.5, 1.0] for a 2-class argmax.
    """

    model.to(device)
    model.eval()
    all_preds, all_labels, all_confidence = [], [], []
    for specs, labels in loader:
        specs = specs.to(device)
        logits = model(specs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confidence = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_confidence.append(confidence.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_confidence)


def train_with_best_checkpoint(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    *,
    epochs: int,
    device: str,
    scheduler=None,
    on_epoch_end=None,
) -> dict:
    """Train for ``epochs`` epochs, tracking whichever epoch had the best
    val accuracy -- not just the last one.

    A real run (see reports/mission2/, the Colab smoke test that motivated
    this) can be unstable late in training (val_acc swinging 20+ points
    between adjacent epochs while train_loss keeps falling smoothly -- a
    classic "learning rate too high" symptom) without ``epochs`` being
    tuned to stop exactly at the peak. Reporting metrics from whatever
    epoch happened to run last is actively misleading when that epoch is
    mid-oscillation and much worse than the best one already reached and
    saved.

    ``scheduler`` (e.g. ReduceLROnPlateau), if given, is stepped once per
    epoch with that epoch's val accuracy. ``on_epoch_end``, if given, is
    called once per epoch with that epoch's history dict (for a caller that
    wants to print live progress -- this function stays pure/side-effect
    free otherwise, for testability).

    Returns a dict: best_val_acc, best_val_preds, best_val_labels,
    best_epoch, best_model_state (state_dict, CPU tensors), history (list
    of {epoch, train_loss, val_acc}).
    """

    best_val_acc = -1.0
    best_val_preds: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None
    best_epoch = 0
    best_model_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device=device)
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
        history.append(epoch_record)

        if scheduler is not None:
            scheduler.step(val_acc)
        if on_epoch_end is not None:
            on_epoch_end(epoch_record)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "best_val_acc": best_val_acc,
        "best_val_preds": best_val_preds,
        "best_val_labels": best_val_labels,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "history": history,
    }
