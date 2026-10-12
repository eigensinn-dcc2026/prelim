"""WavLM utterance probability for Mission 2, matching the SSL training forward pass.

The training scripts (M2-SSL-003 base, M2-SSL-004 large) feed the model only the
target utterance: read that ``startAt``/``endAt`` span, average channels, resample
to 16 kHz, then take up to eight 3-second windows (1.5 s stride, last window
flush with the end). Each window is repeat-padded to 3 seconds and normalized
on its own. The utterance logit is the mean of the window logits; the reporter
probability is a sigmoid. Nothing from another utterance in the same call is used.

``torch.set_num_threads`` is left at the process default.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000
WIN = 48_000  # 3 seconds at 16 kHz
STRIDE = SR * 3 // 2  # 1.5 seconds
MAX_WINDOWS = 8

PROB_ENSEMBLE_FORMAT = "mission2_prob_ensemble_v1"
COMBINE_PROB_MEAN = "prob_mean"
COMBINE_WEIGHTED = "weighted_prob_mean"
ALLOWED_COMBINES = (COMBINE_PROB_MEAN, COMBINE_WEIGHTED)
FIXED_THRESHOLD = 0.5


def window_starts(
    length: int,
    win: int = WIN,
    stride: int = STRIDE,
    max_windows: int = MAX_WINDOWS,
) -> list[int]:
    """Start indices of the training/eval sliding windows. ``length`` is in samples at ``SR``."""

    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, stride))
    if starts[-1] != length - win:
        starts.append(length - win)
    if len(starts) > max_windows:
        last = max_windows - 1
        starts = [starts[round(i * (len(starts) - 1) / last)] for i in range(max_windows)]
    return starts


def normalize(x: torch.Tensor) -> torch.Tensor:
    std = x.std(unbiased=False)
    return (x - x.mean()) / std if float(std) >= 1e-5 else x * 0


def fit_length(x: torch.Tensor, rng=None, start=None, win: int = WIN) -> torch.Tensor:
    """Repeat a short window up to ``win`` samples, or crop a long one, then normalize.

    ``start`` selects the crop (eval passes each sliding-window start). With neither
    ``start`` nor ``rng``, a long signal is center-cropped, matching the training helper.
    """

    if len(x) < win:
        x = x.repeat(math.ceil(win / max(len(x), 1)))[:win]
    elif len(x) > win:
        if start is not None:
            offset = int(start)
        elif rng is not None:
            offset = rng.randrange(len(x) - win + 1)
        else:
            offset = (len(x) - win) // 2
        x = x[offset : offset + win]
    return normalize(x)


def frames_for_interval(start_at_ms, end_at_ms, sample_rate: int, frame_count: int) -> tuple[int, int]:
    """Half-open frame crop used by the SSL training ``load_segment`` (not the CNN slicer)."""

    start = round(float(start_at_ms) * sample_rate / 1000)
    end = round(float(end_at_ms) * sample_rate / 1000)
    start = min(max(int(start), 0), frame_count)
    end = min(max(int(end), 0), frame_count)
    return start, end


def read_utterance_signal(wav_path, start_at_ms, end_at_ms, target_sr: int = SR) -> torch.Tensor:
    """Target-utterance waveform at ``target_sr``. Empty spans become a length-1 zero, as in training."""

    from torchaudio.functional import resample

    from dcc_data.audio import read_wav_frames, read_wav_info

    info = read_wav_info(wav_path)
    start, end = frames_for_interval(start_at_ms, end_at_ms, info.sample_rate, info.frame_count)
    if end <= start:
        return torch.zeros(1)
    segment = read_wav_frames(wav_path, start, end)
    signal = torch.from_numpy(np.ascontiguousarray(segment.waveform)).mean(0)
    if segment.sample_rate != target_sr:
        signal = resample(signal, segment.sample_rate, target_sr)
    return signal


class WeightedMeanStdClassifier(nn.Module):
    """Layer-weighted hidden states, mean‖std, L2-normalized embedding, one logit.

    This is the training ``Classifier.forward``: softmax over ``layer_weights``,
    time-axis mean and unbiased-False std, then ``Linear(2H, 512) → ReLU → Dropout →
    Linear(512, 256)``, L2-normalize that 256-d vector, then ``Linear(256, 1)``.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        n_states = int(backbone.config.num_hidden_layers) + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_states))
        self.embedding = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, waveform: torch.Tensor):
        states = self.backbone(waveform, output_hidden_states=True).hidden_states
        if len(states) != self.layer_weights.shape[0]:
            raise ValueError(
                f"unexpected hidden state count: got {len(states)}, "
                f"layer_weights has {self.layer_weights.shape[0]}"
            )
        weights = self.layer_weights.softmax(0)
        pooled = sum(weight * state for weight, state in zip(weights, states))
        stats = torch.cat([pooled.mean(1), pooled.std(1, unbiased=False)], 1)
        embedding = F.normalize(self.embedding(stats), dim=1)
        return self.classifier(embedding).squeeze(1), embedding


def _float_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
            converted[key] = value.float()
        else:
            converted[key] = value
    return converted


def build_wavlm_classifier(config_dict: dict) -> WeightedMeanStdClassifier:
    """Build an uninitialized WavLM classifier. ``layerdrop`` is forced to 0 (no network access)."""

    from transformers import WavLMConfig, WavLMModel

    config = WavLMConfig.from_dict(dict(config_dict))
    config.layerdrop = 0.0
    backbone = WavLMModel(config)
    return WeightedMeanStdClassifier(backbone)


def load_wavlm_classifier(config_dict: dict, state_dict: dict, device: torch.device | str = "cpu") -> WeightedMeanStdClassifier:
    model = build_wavlm_classifier(config_dict)
    model.load_state_dict(_float_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def reporter_probability(model: nn.Module, signal: torch.Tensor, device: torch.device | str = "cpu") -> float:
    """P(신고자) for one utterance. Each window is cropped from ``signal`` alone."""

    if signal.numel() == 0:
        signal = torch.zeros(1)
    windows = torch.stack([fit_length(signal, start=start) for start in window_starts(int(signal.shape[-1]))])
    logits = model(windows.to(device))[0].float()
    return float(logits.mean().sigmoid().item())


def combine_probabilities(probabilities: list[float], weights: list[float], combine: str) -> float:
    if combine == COMBINE_PROB_MEAN:
        return float(sum(probabilities) / len(probabilities))
    if combine != COMBINE_WEIGHTED:
        raise ValueError(f"unknown combine mode {combine!r}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("ensemble weights must sum to a positive value")
    return float(sum(weight * probability for weight, probability in zip(weights, probabilities)) / total)


def decision_from_probability(probability: float) -> int:
    """Fixed contest threshold: class 1 iff P(신고자) >= 0.5. Not configurable."""

    return int(probability >= FIXED_THRESHOLD)
