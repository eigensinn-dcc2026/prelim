"""Mission 1 model code (team eigensinn): caller gender from WAV + JSON.

Per call, the caller (speaker=1) speech is cropped using only ``startAt`` /
``endAt`` / ``speaker`` and concatenated. Features (564-d):

* 52-d acoustic vector (MFCC/F0/energy statistics) of the whole caller audio;
* 512-d x-vector from ``microsoft/wavlm-base-plus-sv``: the caller audio is cut
  into non-overlapping 20 s windows (at most 4; a later window is kept only if
  it is at least 3 s long), each window embedding is L2-normalised, averaged
  weighted by window length, then L2-normalised again.

A frozen head (standardise, then a logistic-regression score; stored as plain
tensors) maps the 564-d vector to F/M. No fitting happens at inference time.

Everything the model needs is in ONE ``.pt`` checkpoint (format ``dcc-m1-v1``, built by
``train/build_model_pt.py``): the head and the WavLM weights together with the text of its
config files. Nothing is downloaded, so the run needs no internet.

If one call fails, the row still gets the training majority class ``F`` so the
row count equals the number of input pairs; the failure is printed on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audio_features import _crop_caller_segments, _parse_caller_intervals, acoustic_feature_vector, pair_by_stem

COLUMNS = ("audio file name", "gender")

N_ACOUSTIC_FEATURES = 52
N_XVECTOR = 512
N_FEATURES = N_ACOUSTIC_FEATURES + N_XVECTOR
LABEL_TO_GENDER = {0: "F", 1: "M"}
FALLBACK_GENDER = "F"  # Training majority class

FORMAT = "dcc-m1-v1"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 20
MAX_WINDOWS = 4
TAIL_SECONDS = 3
MIN_SAMPLES = 8000


def window_ranges(length: int, sample_rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """First window always included; later tails need at least three seconds."""
    width = WINDOW_SECONDS * sample_rate
    result = [(0, min(length, width))]
    for start in range(width, min(length, MAX_WINDOWS * width), width):
        stop = min(length, start + width)
        if stop - start >= TAIL_SECONDS * sample_rate:
            result.append((start, stop))
    return result


def average_windows(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Length-weighted mean of unit x-vectors, re-normalised (as in training)."""
    weights = np.asarray(lengths, dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    mean = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid averaged x-vector")
    return (mean / norm).astype(np.float32)


def load_model(ckpt_path: Path):
    """Read the single ``.pt`` checkpoint: ``(head, feature processor, WavLM x-vector model, torch)``.

    The WavLM model is rebuilt from the config text stored in the file and its weights are loaded
    strictly, so a file that does not match this code is refused instead of run.
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMForXVector

    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{ckpt_path} is not a {FORMAT} checkpoint")
    wavlm = payload["wavlm"]
    if (wavlm["repo_id"], wavlm["revision"]) != (WAVLM_ID, WAVLM_REVISION):
        raise ValueError(f"unexpected x-vector encoder: {wavlm['repo_id']}@{wavlm['revision']}")
    head = {name: tensor.numpy() for name, tensor in payload["head"].items()}
    if head["scaler_mean"].shape != (N_FEATURES,) or head["coef"].shape != (1, N_FEATURES):
        raise ValueError(f"head is not the {N_FEATURES}-feature binary F/M model")
    processor = Wav2Vec2FeatureExtractor.from_dict(json.loads(wavlm["preprocessor_config"]))
    model = WavLMForXVector(WavLMConfig.from_dict(json.loads(wavlm["config"])))
    model.load_state_dict(wavlm["state_dict"], strict=True)
    model.eval()
    if model.config.xvector_output_dim != N_XVECTOR:
        raise ValueError("expected a 512-dimensional x-vector")
    return head, processor, model, torch


def caller_waveform(wav_path: Path, json_path: Path) -> tuple[np.ndarray, int]:
    """Concatenated caller waveform (T,) built from startAt/endAt/speaker only."""
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    segments = _crop_caller_segments(wav_path, _parse_caller_intervals(document, json_path))
    if not segments:
        raise ValueError(f"no caller segments: {json_path}")
    rates = {segment.sample_rate for segment in segments}
    if len(rates) != 1:
        raise ValueError(f"caller segments have mixed sample rates: {sorted(rates)}")
    waveform = np.concatenate([segment.waveform for segment in segments], axis=1)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"expected mono caller waveform (1, T), got {waveform.shape}")
    return waveform[0], segments[0].sample_rate


def extract_features(wav_path: Path, json_path: Path, processor, model, torch) -> np.ndarray:
    """564-d row: 52-d acoustic + length-weighted mean of up to 4 x 20 s x-vectors."""
    wave, sr = caller_waveform(wav_path, json_path)
    acoustic = acoustic_feature_vector(wave, sr)
    if acoustic.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(f"acoustic feature vector has {acoustic.shape[0]} dims")
    vectors, lengths = [], []
    for left, right in window_ranges(wave.shape[0], sr):
        chunk = wave[left:right]
        if sr != SAMPLE_RATE:
            import librosa

            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=SAMPLE_RATE)
        chunk = np.asarray(chunk[: WINDOW_SECONDS * SAMPLE_RATE], dtype=np.float32)
        chunk = np.pad(chunk, (0, max(0, MIN_SAMPLES - len(chunk))))
        inputs = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            embedding = model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        vectors.append(embedding[0].numpy())
        lengths.append((right - left) / sr)
    mean = average_windows(np.stack(vectors), np.asarray(lengths))
    return np.concatenate([acoustic.astype(np.float64), mean.astype(np.float64)])


def classify(head: dict, features: np.ndarray) -> int:
    """Class 1 (M) if the logistic score of the standardised 564-d vector is positive, else 0 (F)."""
    standardised = (features - head["scaler_mean"]) / head["scaler_scale"]
    score = float(standardised @ head["coef"][0] + head["intercept"][0])
    return int(score > 0)


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    """Official Mission 1 rows: ``audio file name``, ``gender`` (F/M)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    pairs = pair_by_stem(Path(audio_dir), Path(label_dir))
    if not pairs:
        raise FileNotFoundError(f"no WAV/JSON stem pairs under {audio_dir} and {label_dir}")
    head, processor, model, torch = load_model(ckpt_path)

    rows, n_fallback = [], 0
    for stem, wav_path, json_path in pairs:
        try:
            x = extract_features(wav_path, json_path, processor, model, torch)
            gender = LABEL_TO_GENDER[classify(head, x)]
        except Exception as exc:  # noqa: BLE001 - keep row count stable
            gender = FALLBACK_GENDER
            n_fallback += 1
            print(
                f"[mission1] fallback majority={gender} stem={stem} "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        rows.append((wav_path.name, gender))
        if len(rows) % 100 == 0:
            print(f"[mission1] {len(rows)}/{len(pairs)} done", file=sys.stderr, flush=True)
    print(
        f"[mission1] pairs={len(pairs)} ok={len(pairs) - n_fallback} fallback={n_fallback}",
        file=sys.stderr,
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))
