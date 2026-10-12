"""Mission 2 utterance-level Dataset with a strict feature whitelist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from .audio import read_wav_frames, read_wav_info
from .guards import ensure_split
from .manifest import ManifestRow


SPEAKER_OPERATOR = 0
SPEAKER_CALLER = 1
VALID_SPEAKERS = frozenset({SPEAKER_OPERATOR, SPEAKER_CALLER})


def milliseconds_to_frame(time_ms: Real, sample_rate: int) -> int:
    """Convert milliseconds to a frame index with Python ties-to-even rounding.

    ``round`` is used deliberately: exact half-frame values round to the nearest
    even integer, which makes crop boundaries deterministic across all calls.
    """

    if not isinstance(time_ms, Real) or isinstance(time_ms, bool):
        raise TypeError("time_ms must be numeric")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return int(round(time_ms / 1000.0 * sample_rate))


@dataclass(frozen=True)
class _UtteranceSample:
    row: ManifestRow
    start_at: Real
    end_at: Real
    speaker: int


class Mission2Dataset:
    """Return waveform features and speaker targets at utterance granularity.

    JSON ``startAt``/``endAt`` values are milliseconds. They define crop
    boundaries and are returned only as metadata, never as model features.
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
        self._samples: list[_UtteranceSample] = []
        for row in self.rows:
            json_path = self.data_root / row.json_path
            with json_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            utterances = document.get("utterances")
            if not isinstance(utterances, list):
                raise ValueError(f"utterances must be a list: {json_path}")
            for utterance in utterances:
                if not isinstance(utterance, dict):
                    raise ValueError(f"utterance must be an object: {json_path}")
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
                    raise ValueError(
                        f"endAt must be greater than startAt: {json_path}"
                    )
                if speaker not in VALID_SPEAKERS:
                    raise ValueError(f"speaker must be 0 or 1: {json_path}")
                self._samples.append(
                    _UtteranceSample(row, start_at, end_at, speaker)
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        wav_path = self.data_root / sample.row.wav_path
        info = read_wav_info(wav_path)
        start_frame = milliseconds_to_frame(sample.start_at, info.sample_rate)
        end_frame = milliseconds_to_frame(sample.end_at, info.sample_rate)
        audio = read_wav_frames(wav_path, start_frame, end_frame)
        return {
            "features": {"waveform": audio.waveform},
            "target": sample.speaker,
            "metadata": {
                "start_at": sample.start_at,
                "end_at": sample.end_at,
                "split": sample.row.split,
                "stem": sample.row.stem,
            },
        }
