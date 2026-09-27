"""Mission 1 call-level Dataset: caller waveform features and gender targets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from .audio import read_wav_frames, read_wav_info
from .guards import ensure_split
from .manifest import ManifestRow
from .mission2 import SPEAKER_CALLER, VALID_SPEAKERS, milliseconds_to_frame


@dataclass(frozen=True)
class CallerSegment:
    waveform: np.ndarray
    sample_rate: int
    start_at: Real
    end_at: Real


@dataclass(frozen=True)
class _CallSample:
    row: ManifestRow
    gender: str
    caller_intervals: tuple[tuple[Real, Real], ...]


def _validate_utterance(
    utterance: dict[str, object], json_path: Path
) -> tuple[Real, Real, int]:
    start_at = utterance["startAt"]
    end_at = utterance["endAt"]
    speaker = utterance["speaker"]
    if not isinstance(start_at, Real) or isinstance(start_at, bool):
        raise TypeError(f"startAt must be numeric: {json_path}")
    if not isinstance(end_at, Real) or isinstance(end_at, bool):
        raise TypeError(f"endAt must be numeric: {json_path}")
    if not isinstance(speaker, int) or isinstance(speaker, bool):
        raise TypeError(f"speaker must be an integer: {json_path}")
    if start_at < 0:
        raise ValueError(f"startAt must be non-negative: {json_path}")
    if end_at <= start_at:
        raise ValueError(f"endAt must be greater than startAt: {json_path}")
    if speaker not in VALID_SPEAKERS:
        raise ValueError(f"speaker must be 0 or 1: {json_path}")
    return start_at, end_at, speaker


def _read_gender(document: dict[str, object], json_path: Path) -> str:
    if "gender" not in document:
        raise ValueError(f"gender is required: {json_path}")
    gender = document["gender"]
    if not isinstance(gender, str) or not gender:
        raise ValueError(f"gender must be a non-empty string: {json_path}")
    return gender


def _parse_caller_intervals(
    document: dict[str, object], json_path: Path
) -> list[tuple[Real, Real]]:
    utterances = document.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"utterances must be a list: {json_path}")

    callers: list[tuple[int, Real, Real]] = []
    for index, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            raise ValueError(f"utterance must be an object: {json_path}")
        start_at, end_at, speaker = _validate_utterance(utterance, json_path)
        if speaker == SPEAKER_CALLER:
            callers.append((index, start_at, end_at))

    if not callers:
        raise ValueError(f"no caller utterances: {json_path}")

    callers.sort(key=lambda item: (item[1], item[0]))
    return [(start_at, end_at) for _, start_at, end_at in callers]


def _crop_caller_segments(
    wav_path: Path, intervals: Sequence[tuple[Real, Real]]
) -> list[CallerSegment]:
    info = read_wav_info(wav_path)
    segments: list[CallerSegment] = []
    for start_at, end_at in intervals:
        start_frame = milliseconds_to_frame(start_at, info.sample_rate)
        end_frame = milliseconds_to_frame(end_at, info.sample_rate)
        audio = read_wav_frames(wav_path, start_frame, end_frame)
        segments.append(
            CallerSegment(
                waveform=audio.waveform,
                sample_rate=audio.sample_rate,
                start_at=start_at,
                end_at=end_at,
            )
        )
    return segments


def extract_caller_segments(
    row: ManifestRow, data_root: str | Path
) -> list[CallerSegment]:
    """Return original-length caller crops in ``startAt`` order."""

    root = Path(data_root)
    json_path = root / row.json_path
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    intervals = _parse_caller_intervals(document, json_path)
    return _crop_caller_segments(root / row.wav_path, intervals)


def concatenate_caller_waveforms(segments: Sequence[CallerSegment]) -> np.ndarray:
    """Join caller segments along time without padding or inserted silence."""

    if not segments:
        raise ValueError("no caller segments to concatenate")
    sample_rates = {segment.sample_rate for segment in segments}
    if len(sample_rates) != 1:
        raise ValueError(f"caller segments have mixed sample rates: {sorted(sample_rates)}")
    return np.concatenate([segment.waveform for segment in segments], axis=1)


class Mission1Dataset:
    """Return concatenated caller waveform features and gender targets.

    JSON ``startAt``/``endAt`` values are milliseconds used only to crop caller
    speech. They are not returned as model features. Padding and sliding
    windows are left to a later model-input stage.
    """

    def __init__(
        self,
        rows: list[ManifestRow],
        data_root: str | Path,
        split: str,
    ) -> None:
        self.rows = ensure_split(rows, split)
        self.data_root = Path(data_root)
        self.split = split
        self._samples: list[_CallSample] = []
        for row in self.rows:
            json_path = self.data_root / row.json_path
            with json_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            gender = _read_gender(document, json_path)
            intervals = _parse_caller_intervals(document, json_path)
            self._samples.append(_CallSample(row, gender, tuple(intervals)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        segments = _crop_caller_segments(
            self.data_root / sample.row.wav_path, sample.caller_intervals
        )
        return {
            "features": {"waveform": concatenate_caller_waveforms(segments)},
            "target": sample.gender,
            "metadata": {
                "split": sample.row.split,
                "stem": sample.row.stem,
                "n_caller_utterances": len(sample.caller_intervals),
            },
        }
