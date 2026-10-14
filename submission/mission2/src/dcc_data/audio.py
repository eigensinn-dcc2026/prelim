"""PCM WAV 읽기 (외부 라이브러리 없이 표준 wave 모듈로). 위치는 샘플(프레임) 번호로 받는다."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WaveInfo:
    """WAV 파일 정보: 채널 수, 샘플링 주파수, 샘플 크기(바이트), 전체 샘플 수."""

    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int


@dataclass(frozen=True)
class AudioSegment:
    """읽어 들인 소리: (채널, 샘플) float32 배열과 샘플링 주파수."""

    waveform: np.ndarray
    sample_rate: int


def read_wav_info(path: str | Path) -> WaveInfo:
    """WAV 헤더만 읽어 정보를 돌려준다. 압축된 WAV 이거나 값이 이상하면 멈춘다."""

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
    """샘플 구간 ``[start_frame, end_frame)`` 을 읽어 -1~1 범위의 float32 로 돌려준다.

    일부러 샘플 번호만 받는다. 라벨의 ``startAt``/``endAt`` 이 초인지 밀리초인지는 여기서 가정하지 않는다
    (밀리초 -> 샘플 변환은 dcc_data.mission2.milliseconds_to_frame).
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

    # 샘플 크기(8/16/32비트)에 맞춰 정수를 -1~1 범위 실수로 바꾼다
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
