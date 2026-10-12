"""Mission 1 model code (team eigensinn): caller gender from WAV + JSON.

Per call, the caller (speaker=1) speech is cropped using only ``startAt`` /
``endAt`` / ``speaker`` and concatenated. Features (564-d):

* 52-d acoustic vector (MFCC/F0/energy statistics) of the whole caller audio;
* 512-d x-vector from ``microsoft/wavlm-base-plus-sv``: the caller audio is cut
  into non-overlapping 20 s windows (at most 4; a later window is kept only if
  it is at least 3 s long), each window embedding is L2-normalised, averaged
  weighted by window length, then L2-normalised again.

A frozen sklearn Pipeline (StandardScaler + LogisticRegression) maps the 564-d
vector to F/M. No fitting happens at inference time.

The WavLM weights are read locally (``$DCC_WAVLM_DIR`` or
``<checkpoint folder>/wavlm-base-plus-sv``) so the run needs no internet. Only
if neither exists is the pinned revision downloaded from the Hugging Face Hub.

If one call fails, the row still gets the training majority class ``F`` so the
row count equals the number of input pairs; the failure is printed on stderr.
"""

from __future__ import annotations

import json
import os
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

WAVLM_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
WAVLM_DIRNAME = "wavlm-base-plus-sv"
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


def resolve_wavlm_dir(ckpt_path: Path) -> Path | None:
    env = os.environ.get("DCC_WAVLM_DIR")
    for candidate in ([Path(env)] if env else []) + [Path(ckpt_path).resolve().parent / WAVLM_DIRNAME]:
        if (candidate / "config.json").exists():
            return candidate
    return None


def load_xvector_model(ckpt_path: Path):
    """Load WavLM x-vector model offline; download the pinned revision only if absent."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    root = resolve_wavlm_dir(ckpt_path)
    source = str(root) if root is not None else WAVLM_ID
    kwargs = {"local_files_only": True} if root is not None else {"revision": WAVLM_REVISION}
    processor = Wav2Vec2FeatureExtractor.from_pretrained(source, **kwargs)
    model = WavLMForXVector.from_pretrained(source, **kwargs).eval()
    if model.config.xvector_output_dim != N_XVECTOR:
        raise ValueError("expected a 512-dimensional x-vector")
    return processor, model, torch


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


def load_pipeline(ckpt_path: Path):
    import joblib
    from sklearn.pipeline import Pipeline

    pipeline = joblib.load(ckpt_path)
    if not isinstance(pipeline, Pipeline) or getattr(pipeline, "n_features_in_", None) != N_FEATURES:
        raise ValueError(f"{ckpt_path} is not the {N_FEATURES}-feature Mission 1 Pipeline")
    return pipeline


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    """Official Mission 1 rows: ``audio file name``, ``gender`` (F/M)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    pipeline = load_pipeline(ckpt_path)
    pairs = pair_by_stem(Path(audio_dir), Path(label_dir))
    if not pairs:
        raise FileNotFoundError(f"no WAV/JSON stem pairs under {audio_dir} and {label_dir}")
    processor, model, torch = load_xvector_model(ckpt_path)

    rows, n_fallback = [], 0
    for stem, wav_path, json_path in pairs:
        try:
            x = extract_features(wav_path, json_path, processor, model, torch)
            pred = int(pipeline.predict(x.reshape(1, -1))[0])
            if pred not in LABEL_TO_GENDER:
                raise ValueError(f"unexpected class label {pred}")
            gender = LABEL_TO_GENDER[pred]
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
