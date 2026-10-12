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


class PackedSpectrogramDataset(torch.utils.data.Dataset):
    """SpectrogramDataset's items, read from one packed array instead of one .npy file per utterance.

    ``packed_path`` is the .npy written by scripts/pack_mission2_spectrograms.py: every utterance's
    spectrogram transposed to (n_frames, n_mels) and stacked along the frame axis, stored as float16.
    ``index_df`` row i's spectrogram is ``packed[frame_offset : frame_offset + n_frames].T``. One memory-mapped
    file avoids opening ~700k small files every epoch, and float16 halves it (~7 GB for the full Training set)
    so it fits in Colab's page cache. float16 costs nothing under --amp, whose first conv rounds its input to
    float16 anyway.

    Items are returned as float32, exactly the (n_mels, n_frames) tensors SpectrogramDataset returns, so
    collate_batch and the models are unchanged. The file is opened lazily, in whichever (worker) process
    first reads from it.
    """

    def __init__(self, index_df: pd.DataFrame, packed_path: str | Path) -> None:
        index_df = index_df.reset_index(drop=True)
        self.frame_offsets = index_df["frame_offset"].to_numpy(dtype=np.int64)
        self.n_frames = index_df["n_frames"].to_numpy(dtype=np.int64)
        self.speakers = index_df["speaker"].to_numpy(dtype=np.int64)
        self.packed_path = Path(packed_path)
        self._packed: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.speakers)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._packed is None:
            self._packed = np.load(self.packed_path, mmap_mode="r")
        start = self.frame_offsets[idx]
        frames = self._packed[start : start + self.n_frames[idx]]
        return torch.from_numpy(np.ascontiguousarray(frames.T, dtype=np.float32)), int(self.speakers[idx])


def collate_batch(
    batch: list[tuple[torch.Tensor, int]], pad_to_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad every spectrogram in the batch to the batch's own max frame count.

    Padding uses SPEC_FLOOR_DB (silence under mel_spectrogram's own
    normalization), never 0.0 -- 0 dB would read as this utterance's
    *loudest* frame instead of padding.

    ``pad_to_multiple`` > 1 rounds that length up to a multiple of it (at most ``pad_to_multiple - 1`` more
    silent frames, on batches already padded to ~400 frames on average). Far fewer distinct input shapes is what
    lets cudnn.benchmark pick and cache a fast kernel per shape; 1 keeps the exact batch max.
    """

    specs, labels = zip(*batch)
    n_mels = specs[0].shape[0]
    max_frames = max(spec.shape[1] for spec in specs)
    max_frames = -(-max_frames // pad_to_multiple) * pad_to_multiple

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


def _autocast(device: str, amp: bool):
    """float16 autocast on cuda when ``amp``; a no-op context otherwise (also on cpu, where it is never used)."""

    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda")


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, scaler=None) -> float:
    """``loader`` yields ``(*model_inputs, labels)`` tuples -- ``(specs,
    labels)`` for SpeakerCNN/SpeakerCNNLarge, or ``(specs, duration,
    labels)`` for SpeakerCNNLargeWithDuration -- and ``model(*model_inputs)``
    is called accordingly, so this one loop drives either.

    ``scaler`` (a torch.amp.GradScaler) turns on mixed precision: the forward pass runs under float16 autocast
    and the loss is scaled so small float16 gradients don't round to zero. None keeps plain float32.
    """

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp = scaler is not None
    # Summed on the device: a .item() per batch would make the CPU wait for the GPU every step.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    n = 0
    for *inputs, labels in loader:
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with _autocast(device, amp):
            logits = model(*inputs)
            loss = criterion(logits, labels)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().double() * len(labels)
        n += len(labels)
    return total_loss.item() / n


def predicted_classes(logits: torch.Tensor) -> torch.Tensor:
    """The contest's decision rule for a (batch, 2) logit tensor: class 1 iff P(speaker=1) >= 0.5.

    That is ``logit1 >= logit0``. ``argmax`` differs only at an exact tie, where it returns class 0 -- the
    submission script (scripts/mission2_inference.py) already decides ties as class 1, and the accuracy that
    picks the best epoch must be computed the same way.
    """

    return (logits[:, 1] >= logits[:, 0]).long()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, amp: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """See train_one_epoch()'s docstring for ``loader``'s tuple shape. ``amp`` runs the forward pass under
    float16 autocast (cuda only), the same as training with a GradScaler does."""

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for *inputs, labels in loader:
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        with _autocast(device, amp):
            logits = model(*inputs).float()
        all_preds.append(predicted_classes(logits).cpu().numpy())
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
    [0.5, 1.0] for a 2-class decision (an exact tie is class 1, at 0.5).

    See train_one_epoch()'s docstring for ``loader``'s tuple shape.
    """

    model.to(device)
    model.eval()
    all_preds, all_labels, all_confidence = [], [], []
    for *inputs, labels in loader:
        inputs = [t.to(device) for t in inputs]
        logits = model(*inputs)
        probs = torch.softmax(logits, dim=1)
        preds = predicted_classes(logits)
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
    scaler=None,
    should_stop=None,
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

    ``scaler`` (a torch.amp.GradScaler) trains and evaluates in mixed precision (see train_one_epoch).
    ``should_stop``, if given, is called after each epoch (after on_epoch_end) with that epoch's history dict;
    returning True ends training there, before ``epochs`` -- for early stopping, which needs to know about
    epochs from earlier resumed runs that this function never sees.

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
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device, scaler=scaler)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device=device, amp=scaler is not None)
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

        if should_stop is not None and should_stop(epoch_record):
            break

    return {
        "best_val_acc": best_val_acc,
        "best_val_preds": best_val_preds,
        "best_val_labels": best_val_labels,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "history": history,
    }
