"""Mission 1 module: caller gender from WAV + JSON (owner: DCC M1).

Pairs WAV/JSON by filename stem, crops caller (speaker=1) speech using only
startAt/endAt/speaker, extracts the 52-d acoustic vector used in training and
predicts F/M with a frozen LR (or GBM) checkpoint. No fitting.

On a per-call failure the row still gets the training majority class from the
checkpoint meta so the row count matches the input pair count; the failure is
reported on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from dcc_data.audio import read_wav_frames, read_wav_info
from dcc_data.mission1 import _parse_caller_intervals
from dcc_data.mission1_features import ACOUSTIC_FEATURE_COLUMNS, N_ACOUSTIC_FEATURES
from dcc_data.mission1_train import LABEL_TO_GENDER, load_mission1_gbm, load_mission1_lr
from dcc_data.mission2 import milliseconds_to_frame
from dcc_data.mission2_features import acoustic_feature_vector, pair_by_stem

COLUMNS = ("audio file name", "gender")


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


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    """Official Mission 1 rows: ``audio file name``, ``gender`` (F/M)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
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