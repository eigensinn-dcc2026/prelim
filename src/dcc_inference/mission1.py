"""Mission 1 module: caller gender from WAV + JSON (owner: DCC M1).

Pairs WAV/JSON by filename stem, crops caller (speaker=1) speech using only
startAt/endAt/speaker, extracts the 52-d acoustic vector used in training and
predicts F/M with a frozen LR (or GBM) checkpoint. No fitting.

Two checkpoint kinds are accepted:
* legacy ``m1_baseline_*.pkl`` (52-d acoustic features only);
* a bare sklearn Pipeline over 564 features = 52-d acoustic + 512-d multi-window
  mean WavLM x-vector (M1-XVEC-002 (c)). The WavLM model is read locally from
  ``$DCC_WAVLM_DIR`` or ``<ckpt dir>/wavlm-base-plus-sv``; the pinned revision is
  downloaded only if neither exists.

On a per-call failure the row still gets the training majority class from the
checkpoint meta so the row count matches the input pair count; the failure is
reported on stderr.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from dcc_data.audio import read_wav_frames, read_wav_info
from dcc_data.mission1 import _crop_caller_segments, _parse_caller_intervals
from dcc_data.mission1_features import ACOUSTIC_FEATURE_COLUMNS, N_ACOUSTIC_FEATURES
from dcc_data.mission1_train import LABEL_TO_GENDER, load_mission1_gbm, load_mission1_lr
from dcc_data.mission2 import milliseconds_to_frame
from dcc_data.mission2_features import acoustic_feature_vector, pair_by_stem

COLUMNS = ("audio file name", "gender")

XVEC_FEATURES = N_ACOUSTIC_FEATURES + 512
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
WAVLM_DIRNAME = "wavlm-base-plus-sv"
XVEC_SAMPLE_RATE = 16000
XVEC_WINDOW_SECONDS = 20
XVEC_MAX_WINDOWS = 4
XVEC_TAIL_SECONDS = 3
XVEC_MIN_SAMPLES = 8000
XVEC_FALLBACK = "F"  # Training majority class


def load_checkpoint(ckpt_path: Path):
    """Load the LR Pipeline; fall back to GBM if the file is a GBM checkpoint."""
    try:
        model, meta = load_mission1_lr(ckpt_path)
        return model, meta, "lr"
    except TypeError:
        model, meta = load_mission1_gbm(ckpt_path)
        return model, meta, "gbm"


def extract_call_features(wav_path: Path, json_path: Path) -> np.ndarray:
    """52-d caller acoustic vector; mirrors mission1_features crop rules."""
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")

    intervals = _parse_caller_intervals(document, json_path)
    info = read_wav_info(wav_path)
    audio = read_wav_frames(wav_path, 0, info.frame_count)
    parts: list[np.ndarray] = []
    for start_at, end_at in intervals:
        start_frame = milliseconds_to_frame(start_at, info.sample_rate)
        end_frame = milliseconds_to_frame(end_at, info.sample_rate)
        if start_frame < 0 or end_frame <= start_frame or end_frame > info.frame_count:
            raise ValueError(
                f"Invalid frame interval [{start_frame}, {end_frame}) for "
                f"{info.frame_count} frames: {wav_path}"
            )
        parts.append(audio.waveform[:, start_frame:end_frame])

    waveform = np.concatenate(parts, axis=1)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            f"expected mono caller waveform with shape (1, T), got {waveform.shape}: "
            f"{wav_path}"
        )
    vector = acoustic_feature_vector(waveform[0], audio.sample_rate)
    if vector.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(
            f"acoustic feature vector has {vector.shape[0]} dims, "
            f"expected {N_ACOUSTIC_FEATURES}"
        )
    return vector.astype(np.float64, copy=False)


def predict_gender(model, kind: str, features: np.ndarray) -> str:
    x = features.reshape(1, -1)
    if kind == "lr" and not isinstance(model, Pipeline):
        raise TypeError("LR checkpoint is not a Pipeline")
    pred = int(model.predict(x)[0])
    if pred not in LABEL_TO_GENDER:
        raise ValueError(f"unexpected class label {pred}")
    return LABEL_TO_GENDER[pred]


def window_ranges(length: int, sample_rate: int = XVEC_SAMPLE_RATE) -> list[tuple[int, int]]:
    """First window always included; later tails need at least three seconds."""
    width = XVEC_WINDOW_SECONDS * sample_rate
    result = [(0, min(length, width))]
    for start in range(width, min(length, XVEC_MAX_WINDOWS * width), width):
        stop = min(length, start + width)
        if stop - start >= XVEC_TAIL_SECONDS * sample_rate:
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
    for candidate in ([Path(env)] if env else []) + [Path(ckpt_path).parent / WAVLM_DIRNAME]:
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
    if model.config.xvector_output_dim != 512:
        raise ValueError("expected a 512-dimensional x-vector")
    return processor, model, torch


def caller_waveform(wav_path: Path, json_path: Path) -> tuple[np.ndarray, int]:
    """Concatenated caller waveform (T,) exactly as the training extractor builds it."""
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


def extract_xvec_features(wav_path: Path, json_path: Path, processor, model, torch) -> np.ndarray:
    """564-d row: 52-d acoustic + length-weighted mean of up to 4 x 20 s x-vectors."""
    wave, sr = caller_waveform(wav_path, json_path)
    acoustic = acoustic_feature_vector(wave, sr)
    if acoustic.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(f"acoustic feature vector has {acoustic.shape[0]} dims")
    vectors, lengths = [], []
    for left, right in window_ranges(wave.shape[0], sr):
        chunk = wave[left:right]
        if sr != XVEC_SAMPLE_RATE:
            import librosa

            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=XVEC_SAMPLE_RATE)
        chunk = np.asarray(chunk[: XVEC_WINDOW_SECONDS * XVEC_SAMPLE_RATE], dtype=np.float32)
        chunk = np.pad(chunk, (0, max(0, XVEC_MIN_SAMPLES - len(chunk))))
        inputs = processor(
            chunk, sampling_rate=XVEC_SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            embedding = model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        vectors.append(embedding[0].numpy())
        lengths.append((right - left) / sr)
    mean = average_windows(np.stack(vectors), np.asarray(lengths))
    return np.concatenate([acoustic.astype(np.float64), mean.astype(np.float64)])


def _load_xvec_pipeline(ckpt_path: Path):
    """Return the 564-d Pipeline if ``ckpt_path`` is one, else None."""
    import joblib

    try:
        obj = joblib.load(ckpt_path)
    except Exception:  # noqa: BLE001 - legacy checkpoints use their own loader
        return None
    if isinstance(obj, Pipeline) and getattr(obj, "n_features_in_", None) == XVEC_FEATURES:
        return obj
    return None


def predict_xvec(pipeline, pairs, ckpt_path: Path) -> pd.DataFrame:
    processor, model, torch = load_xvector_model(ckpt_path)
    rows, n_fallback = [], 0
    for stem, wav_path, json_path in pairs:
        try:
            x = extract_xvec_features(wav_path, json_path, processor, model, torch)
            pred = int(pipeline.predict(x.reshape(1, -1))[0])
            if pred not in LABEL_TO_GENDER:
                raise ValueError(f"unexpected class label {pred}")
            gender = LABEL_TO_GENDER[pred]
        except Exception as exc:  # noqa: BLE001 - keep row count stable
            gender = XVEC_FALLBACK
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
        f"[mission1] pairs={len(pairs)} ok={len(pairs) - n_fallback} "
        f"fallback={n_fallback} ckpt=xvec564",
        file=sys.stderr,
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    """Official Mission 1 rows: ``audio file name``, ``gender`` (F/M)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    xvec_pipeline = _load_xvec_pipeline(ckpt_path)
    if xvec_pipeline is not None:
        pairs = pair_by_stem(Path(audio_dir), Path(label_dir))
        if not pairs:
            raise FileNotFoundError(f"no WAV/JSON stem pairs under {audio_dir} and {label_dir}")
        return predict_xvec(xvec_pipeline, pairs, ckpt_path)
    model, meta, kind = load_checkpoint(ckpt_path)
    feature_columns = tuple(meta.get("feature_columns", ()))
    if feature_columns != ACOUSTIC_FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match ACOUSTIC_FEATURE_COLUMNS "
            f"(got {len(feature_columns)} cols)"
        )
    fallback = str(meta.get("majority_class", "F"))
    if fallback not in {"F", "M"}:
        fallback = "F"

    pairs = pair_by_stem(Path(audio_dir), Path(label_dir))
    if not pairs:
        raise FileNotFoundError(f"no WAV/JSON stem pairs under {audio_dir} and {label_dir}")

    rows = []
    n_fallback = 0
    for stem, wav_path, json_path in pairs:
        try:
            gender = predict_gender(model, kind, extract_call_features(wav_path, json_path))
        except Exception as exc:  # noqa: BLE001 - keep row count stable
            gender = fallback
            n_fallback += 1
            print(
                f"[mission1] fallback majority={fallback} stem={stem} "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        rows.append((wav_path.name, gender))
    print(
        f"[mission1] pairs={len(pairs)} ok={len(pairs) - n_fallback} "
        f"fallback={n_fallback} ckpt={kind}",
        file=sys.stderr,
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))