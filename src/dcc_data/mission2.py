"""Mission 2 utterance-level Dataset with a strict feature whitelist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable

from .audio import read_wav_frames, read_wav_info
from .guards import ensure_split
from .manifest import ManifestRow


TimeToFrame = Callable[[Real, int], int]


@dataclass(frozen=True)
class _UtteranceSample:
    row: ManifestRow
    start_at: Real
    end_at: Real
    speaker: int


class Mission2Dataset:
    """Return audio/time features and speaker targets at utterance granularity.

    ``time_to_frame`` is mandatory because the unit of ``startAt``/``endAt``
    has not yet been verified. No seconds/milliseconds conversion is assumed.
    """

    def __init__(
        self,
        rows: list[ManifestRow],
        data_root: str | Path,
        split: str,
        *,
        time_to_frame: TimeToFrame,
    ) -> None:
        self.rows = ensure_split(rows, split)
        self.data_root = Path(data_root)
        self.split = split
        self.time_to_frame = time_to_frame
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
                self._samples.append(
                    _UtteranceSample(row, start_at, end_at, speaker)
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        wav_path = self.data_root / sample.row.wav_path
        info = read_wav_info(wav_path)
        start_frame = self.time_to_frame(sample.start_at, info.sample_rate)
        end_frame = self.time_to_frame(sample.end_at, info.sample_rate)
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            raise TypeError("time_to_frame must return integer frame indices")
        audio = read_wav_frames(wav_path, start_frame, end_frame)
        return {
            "features": {
                "waveform": audio.waveform,
                "start_at": sample.start_at,
                "end_at": sample.end_at,
            },
            "target": sample.speaker,
            "split": sample.row.split,
            "stem": sample.row.stem,
        }
