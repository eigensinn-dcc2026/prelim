"""Mission 1 baseline features: call-level 52-dim acoustic statistics.

Reuses ``dcc_data.mission2_features.acoustic_feature_vector`` instead of
re-implementing MFCC/RMS/ZCR/spectral statistics. Caller crops follow the
``dcc_data.mission1`` contract (speaker=1 intervals, ``startAt`` order,
concatenate without padding). WAV files are opened once per call; the
in-memory slice is kept bit-identical to ``Mission1Dataset`` by using the
same ``milliseconds_to_frame`` boundaries and the same
``read_wav_frames`` invalid-interval checks.

``duration_sec`` is intentionally omitted. ``startAt``/``endAt`` are used
only to crop caller speech and are never written as model features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio import read_wav_frames, read_wav_info
from .guards import TRAINING, VALIDATION, ensure_split
from .manifest import ManifestRow
from .mission1 import _parse_caller_intervals, _read_gender
from .mission2 import milliseconds_to_frame
from .mission2_features import (
    FEATURE_COLUMNS,
    N_ACOUSTIC_FEATURES,
    acoustic_feature_vector,
    is_seoul_path,
    sample_calls,
    select_seoul_training_rows,
)

VALID_GENDERS = frozenset({"F", "M"})
ACOUSTIC_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS[1:]
assert len(ACOUSTIC_FEATURE_COLUMNS) == N_ACOUSTIC_FEATURES
assert "duration_sec" not in ACOUSTIC_FEATURE_COLUMNS

OnCallDone = Callable[[int, int], None]


def select_seoul_split_rows(
    rows: Iterable[ManifestRow], split: str
) -> list[ManifestRow]:
    """Seoul-only rows for one official split. Does not mix Training and Validation."""

    if split == TRAINING:
        return select_seoul_training_rows(rows)
    split_only = [row for row in rows if row.split == split]
    split_only = ensure_split(split_only, split)
    return [
        row
        for row in split_only
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def concatenated_caller_waveform(
    row: ManifestRow, data_root: str | Path
) -> tuple[np.ndarray, int]:
    """Return ``(channels, time)`` caller audio matching ``Mission1Dataset``.

    The WAV is opened once. Interval parsing is delegated to
    ``mission1._parse_caller_intervals`` so crop rules cannot drift.
    """

    root = Path(data_root)
    json_path = root / row.json_path
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    intervals = _parse_caller_intervals(document, json_path)

    wav_path = root / row.wav_path
    info = read_wav_info(wav_path)
    audio = read_wav_frames(wav_path, 0, info.frame_count)
    parts: list[np.ndarray] = []
    for start_at, end_at in intervals:
        start_frame = milliseconds_to_frame(start_at, info.sample_rate)
        end_frame = milliseconds_to_frame(end_at, info.sample_rate)
        if start_frame < 0 or end_frame <= start_frame or end_frame > info.frame_count:
            raise ValueError(
                f"Invalid frame interval [{start_frame}, {end_frame}) for "
                f"{info.frame_count} frames"
            )
        parts.append(audio.waveform[:, start_frame:end_frame])
    return np.concatenate(parts, axis=1), audio.sample_rate


def extract_call_feature_row(row: ManifestRow, data_root: str | Path) -> dict:
    """Build one call-level feature dict. Raises on invalid labels or audio."""

    root = Path(data_root)
    json_path = root / row.json_path
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    gender = _read_gender(document, json_path)
    if gender not in VALID_GENDERS:
        raise ValueError(f"gender must be 'F' or 'M': {json_path} got {gender!r}")
    intervals = _parse_caller_intervals(document, json_path)

    waveform, sample_rate = concatenated_caller_waveform(row, root)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            f"expected mono caller waveform with shape (1, T), got {waveform.shape}: "
            f"{row.wav_path}"
        )
    vector = acoustic_feature_vector(waveform[0], sample_rate)
    if vector.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(
            f"acoustic feature vector has {vector.shape[0]} dims, "
            f"expected {N_ACOUSTIC_FEATURES}"
        )
    return {
        "stem": row.stem,
        "split": row.split,
        "gender": gender,
        "n_caller_utterances": len(intervals),
        **dict(zip(ACOUSTIC_FEATURE_COLUMNS, vector.tolist())),
    }


def build_mission1_feature_dataset(
    rows: Iterable[ManifestRow],
    data_root: str | Path,
    split: str,
    *,
    n_calls: int = 0,
    seed: int = 42,
    feature_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
) -> tuple[list[dict], list[dict]]:
    """Extract Seoul-only features for ``split``. Validation never enters Training."""

    if split not in {TRAINING, VALIDATION}:
        raise ValueError(f"unknown split: {split!r}")

    feature_sink = feature_sink if feature_sink is not None else []
    failure_sink = failure_sink if failure_sink is not None else []
    root = Path(data_root)
    selected = select_seoul_split_rows(rows, split)
    sampled = sample_calls(selected, n_calls, seed)

    for call_number, row in enumerate(sampled, start=1):
        try:
            feature_sink.append(extract_call_feature_row(row, root))
        except Exception as exc:  # noqa: BLE001 - logged, not swallowed
            failure_sink.append(
                {
                    "stage": "feature",
                    "stem": row.stem,
                    "split": row.split,
                    "wav_path": row.wav_path,
                    "json_path": row.json_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        if on_call_done:
            on_call_done(call_number, len(sampled))

    return feature_sink, failure_sink
