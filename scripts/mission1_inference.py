#!/usr/bin/env python3
"""Mission 1 inference entry point (contest-mandated CLI contract).

    python scripts/mission1_inference.py \
        --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

Pairs WAV/JSON by filename stem (flat grader folders; no Training/Validation
path requirement). Uses only startAt/endAt/speaker from JSON to crop caller
(speaker=1) speech, extracts the same 52-d acoustic vector as training, and
predicts gender with a frozen LR (or GBM) checkpoint. No fitting / no Val
backprop.

CSV columns follow the official PDF: ``audio file name``, ``gender``.
One output row per paired call. On feature/predict failure the row still
emits the training majority class from checkpoint meta so the row count
matches the input pair count; failures go to stderr.

Submission packaging: copy or rename this file to top-level ``inference.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.audio import read_wav_frames, read_wav_info  # noqa: E402
from dcc_data.mission1 import _parse_caller_intervals  # noqa: E402
from dcc_data.mission1_features import (  # noqa: E402
    ACOUSTIC_FEATURE_COLUMNS,
    N_ACOUSTIC_FEATURES,
)
from dcc_data.mission1_train import (  # noqa: E402
    LABEL_TO_GENDER,
    load_mission1_gbm,
    load_mission1_lr,
)
from dcc_data.mission2 import milliseconds_to_frame  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    acoustic_feature_vector,
    pair_by_stem,
)
from sklearn.pipeline import Pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(ckpt_path: Path):
    """Load LR Pipeline preferentially; fall back to GBM if meta says so."""
    # LR checkpoints are small sklearn Pipelines; try LR first, then GBM.
    try:
        model, meta = load_mission1_lr(ckpt_path)
        kind = "lr"
        return model, meta, kind
    except TypeError:
        model, meta = load_mission1_gbm(ckpt_path)
        kind = "gbm"
        return model, meta, kind


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
    if kind == "lr":
        if not isinstance(model, Pipeline):
            raise TypeError("LR checkpoint is not a Pipeline")
        pred = int(model.predict(x)[0])
    else:
        pred = int(model.predict(x)[0])
    if pred not in LABEL_TO_GENDER:
        raise ValueError(f"unexpected class label {pred}")
    return LABEL_TO_GENDER[pred]


def main() -> int:
    args = parse_args()
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")

    model, meta, kind = load_checkpoint(args.ckpt_path)
    feature_columns = tuple(meta.get("feature_columns", ()))
    if feature_columns != ACOUSTIC_FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match ACOUSTIC_FEATURE_COLUMNS "
            f"(got {len(feature_columns)} cols)"
        )
    fallback = str(meta.get("majority_class", "F"))
    if fallback not in {"F", "M"}:
        fallback = "F"

    pairs = pair_by_stem(args.audio_dir, args.label_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no WAV/JSON stem pairs under {args.audio_dir} and {args.label_dir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fallback = 0
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio file name", "gender"],
        )
        writer.writeheader()
        for stem, wav_path, json_path in pairs:
            audio_name = wav_path.name
            try:
                features = extract_call_features(wav_path, json_path)
                gender = predict_gender(model, kind, features)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - keep row count stable
                gender = fallback
                n_fallback += 1
                print(
                    f"[mission1_inference] fallback majority={fallback} "
                    f"stem={stem} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            writer.writerow({"audio file name": audio_name, "gender": gender})

    print(
        f"Mission 1 CSV saved: {args.output} "
        f"(pairs={len(pairs)}, ok={n_ok}, fallback={n_fallback}, ckpt={kind})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
