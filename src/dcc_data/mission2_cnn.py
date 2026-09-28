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


class SpectrogramWithDurationDataset(SpectrogramDataset):
    """Like SpectrogramDataset, but also yields each utterance's own
    ``duration_sec`` (its own startAt/endAt span -- allowed, see
    SpeakerCNNLargeWithDuration) alongside its spectrogram and label.

    A separate class (rather than adding duration to SpectrogramDataset
    itself) so every existing SpectrogramDataset/collate_batch caller --
    the shipped 91.22% pipeline in particular -- is untouched.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, int]:
        row = self.index_df.iloc[idx]
        spec = np.load(self.spectrogram_dir / row["npy_path"])
        return torch.from_numpy(spec), float(row["duration_sec"]), int(row["speaker"])


def collate_batch_with_duration(
    batch: list[tuple[torch.Tensor, float, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like collate_batch, plus stacking each item's own duration_sec into a
    (batch,) float tensor.
    """

    specs, durations, labels = zip(*batch)
    n_mels = specs[0].shape[0]
    max_frames = max(spec.shape[1] for spec in specs)

    padded = torch.full((len(specs), 1, n_mels, max_frames), SPEC_FLOOR_DB, dtype=torch.float32)
    for i, spec in enumerate(specs):
        padded[i, 0, :, : spec.shape[1]] = spec
    durations_tensor = torch.tensor(durations, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, durations_tensor, labels_tensor


def collate_batch_with_lengths(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like collate_batch, plus each spectrogram's own frame count (before
    padding) as a (batch,) long tensor -- what SpeakerCNNLargeMasked needs to
    tell real frames from padding.
    """

    padded, labels = collate_batch(batch)
    lengths = torch.tensor([spec.shape[1] for spec, _ in batch], dtype=torch.long)
    return padded, lengths, labels


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


class SpeakerCNNLargeWithDuration(nn.Module):
    """SpeakerCNNLarge's conv stack, plus the utterance's own duration_sec
    concatenated into the classifier head as an explicit scalar input.

    Why: the L2 error analysis found the remaining 대원/신고자 gap
    concentrated in short, content-free backchannel utterances ("네"/"예").
    SpeakerCNNLarge already sees a short utterance's true frame count, but
    only implicitly, diluted by the batch's own zero-padding -- for a very
    short utterance, 3 rounds of MaxPool(stride 2) can leave only 1-2
    surviving timesteps before AdaptiveAvgPool2d. Giving the model duration
    directly (log1p-compressed, since raw seconds are strongly right-skewed)
    lets it condition on "this is a very short utterance" without relying on
    that weak, indirect signal. duration_sec is the current utterance's own
    endAt-startAt span -- not derived from any other utterance in the call,
    so this stays within the isolated-fragment rule (see
    [[project-dcc2026-contest-rules]]).

    A separate class from SpeakerCNNLarge (not a modification) so the
    already-trained 91.22% checkpoint and scripts/train_mission2_cnn.py's
    default path are untouched -- its classifier's first Linear has a
    different input width (257, not 256), so an old checkpoint's state_dict
    wouldn't load into it anyway. Pass --model large-duration to
    scripts/train_mission2_cnn.py to train this one.
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
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        duration_feature = torch.log1p(duration.to(x.dtype)).unsqueeze(1)
        x = torch.cat([x, duration_feature], dim=1)
        return self.classifier(x)


class SpeakerCNNLargeMasked(SpeakerCNNLarge):
    """SpeakerCNNLarge, but the final average over time skips padded frames.

    SpeakerCNNLarge's AdaptiveAvgPool2d averages over every time step of the
    batch tensor, so a short utterance's pooled feature is diluted by however
    much padding its batch neighbours forced on it -- and training (padded to
    the batch max) sees different padding than one-utterance-at-a-time
    inference does. That mismatch is what cost 89.73% vs 90.89% in
    M2-VAL-004/005 until inference padded to a fixed 300 frames. Here the
    pool only averages the ``lengths``-derived valid time steps, so the
    result no longer depends on how much padding follows the utterance
    (convolutions near the boundary still see a few padded frames; BatchNorm
    running stats still include them).

    Same modules as SpeakerCNNLarge (only forward differs), so a
    SpeakerCNNLarge checkpoint loads into it directly. ``lengths`` are each
    spectrogram's own frame counts -- see collate_batch_with_lengths; it is
    the current utterance's own frame count, so this stays within the
    isolated-fragment rule (see [[project-dcc2026-contest-rules]]).
    """

    # Each of features' 3 MaxPool2d(2, ceil_mode=True) maps a valid length L to
    # ceil(L/2), and ceil(ceil(ceil(L/2)/2)/2) == ceil(L/8).
    TIME_DOWNSAMPLE = 8

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        valid_steps = ((lengths + self.TIME_DOWNSAMPLE - 1) // self.TIME_DOWNSAMPLE).clamp(min=1)
        positions = torch.arange(x.shape[3], device=x.device)
        mask = (positions[None, :] < valid_steps[:, None].to(x.device)).to(x.dtype)[:, None, None, :]
        pooled = (x * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) * x.shape[2])
        return self.classifier(pooled)


class _ResidualBlock(nn.Module):
    """Two 3x3 convs plus a shortcut. ``stride`` 2 halves both axes with
    ceil semantics (out = ceil(in/2)), matching MaxPool2d(2, ceil_mode=True),
    so a stack of these downsamples by the same factor as SpeakerCNNLarge.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class SpeakerResNet(nn.Module):
    """A small ResNet over the same log-mel input as SpeakerCNNLarge.

    Same widths (32->64->128->256) and same 8x downsampling as
    SpeakerCNNLarge, but each stage is a residual block (two convs + a
    shortcut) instead of one conv, and downsampling is a stride-2 conv
    instead of MaxPool. The stem is a plain stride-1 3x3 conv with no pooling
    ("small stem"): eunkyo00's team reported that swapping ResNet18's 7x7
    stride-2 stem + maxpool for this took an ImageNet-init ResNet18 from
    88.60% to 90.25% on official Validation (their own numbers; the 90.25%
    epoch and threshold were picked on Validation) -- a spectrogram this
    small (40 mels, tens of frames) can't afford the early 4x reduction.
    Trained from scratch here (no ImageNet weights). About 3x
    SpeakerCNNLarge's parameters (1.26M vs 0.42M), so a win here is depth +
    shortcuts + capacity together, not shortcuts alone.
    Same head as SpeakerCNNLarge. A separate class so no existing checkpoint
    is affected; --model resnet in scripts/train_mission2_cnn.py.
    """

    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            _ResidualBlock(32, 32, stride=1),
            _ResidualBlock(32, 64, stride=2),
            _ResidualBlock(64, 128, stride=2),
            _ResidualBlock(128, 256, stride=2),
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
    """``loader`` yields ``(*model_inputs, labels)`` tuples -- ``(specs,
    labels)`` for SpeakerCNN/SpeakerCNNLarge, or ``(specs, duration,
    labels)`` for SpeakerCNNLargeWithDuration -- and ``model(*model_inputs)``
    is called accordingly, so this one loop drives either.
    """

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for *inputs, labels in loader:
        inputs = [t.to(device) for t in inputs]
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(*inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """See train_one_epoch()'s docstring for ``loader``'s tuple shape."""

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for *inputs, labels in loader:
        inputs = [t.to(device) for t in inputs]
        logits = model(*inputs)
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

    See train_one_epoch()'s docstring for ``loader``'s tuple shape.
    """

    model.to(device)
    model.eval()
    all_preds, all_labels, all_confidence = [], [], []
    for *inputs, labels in loader:
        inputs = [t.to(device) for t in inputs]
        logits = model(*inputs)
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
