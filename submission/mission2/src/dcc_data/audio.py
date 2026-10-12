"""프레임 단위 경계로 PCM WAV를 읽는, 의존성이 적은 로더."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WaveInfo:
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int


@dataclass(frozen=True)
class AudioSegment:
    waveform: np.ndarray
    sample_rate: int


def read_wav_info(path: str | Path) -> WaveInfo:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"Only uncompressed PCM WAV is supported: {path}")
        info = WaveInfo(
            channels=wav_file.getnchannels(),
            sample_rate=wav_file.getframerate(),
            sample_width=wav_file.getsampwidth(),
            frame_count=wav_file.getnframes(),
        )
    if min(info.channels, info.sample_rate, info.sample_width, info.frame_count) <= 0:
        raise ValueError(f"Invalid WAV parameters: {info}")
    return info


def read_wav_frames(
    path: str | Path, start_frame: int = 0, end_frame: int | None = None
) -> AudioSegment:
    """반열린 프레임 구간 ``[start_frame, end_frame)``을 읽는다.

        이 함수는 일부러 프레임 번호만 받는다. 데이터의 ``startAt``/``endAt``이
        초 단위인지 밀리초 단위인지는 가정하지 않는다.
    """

    if not isinstance(start_frame, int):
        raise TypeError("start_frame must be an integer frame index")

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"Only uncompressed PCM WAV is supported: {path}")
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        actual_end = frame_count if end_frame is None else end_frame
        if not isinstance(actual_end, int):
            raise TypeError("end_frame must be an integer frame index")
        if start_frame < 0 or actual_end <= start_frame or actual_end > frame_count:
            raise ValueError(
                f"Invalid frame interval [{start_frame}, {actual_end}) for "
                f"{frame_count} frames"
            )
        wav_file.setpos(start_frame)
        requested_frames = actual_end - start_frame
        raw = wav_file.readframes(requested_frames)

    expected_bytes = requested_frames * channels * sample_width
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Truncated WAV segment: expected {expected_bytes} bytes, got {len(raw)}"
        )

    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    waveform = values.reshape(-1, channels).T.copy()
    return AudioSegment(waveform=waveform, sample_rate=sample_rate)
