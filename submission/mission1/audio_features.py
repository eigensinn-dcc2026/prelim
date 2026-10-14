"""Mission 1 추론용 오디오 도구: 신고자 구간 자르기, 52차원 음향 벡터, WAV·JSON 짝짓기.

이 파일의 코드는 팀의 데이터 유틸리티(WAV 읽기, 신고자 구간 규칙, 음향 특징 추출)에서 그대로 옮겼고,
추론이 실제로 부르는 부분만 남겼다.
"""

from __future__ import annotations

import librosa
import numpy as np
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path


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


SPEAKER_OPERATOR = 0


SPEAKER_CALLER = 1


VALID_SPEAKERS = frozenset({SPEAKER_OPERATOR, SPEAKER_CALLER})


def milliseconds_to_frame(time_ms: Real, sample_rate: int) -> int:
    """밀리초를 프레임 번호로 바꾼다(파이썬의 짝수 쪽 반올림 규칙 사용).

    ``round``를 일부러 쓴다: 정확히 반 프레임인 값은 가장 가까운 짝수로 반올림되어,
    모든 통화에서 자르는 경계가 항상 같게 정해진다.
    """

    if not isinstance(time_ms, Real) or isinstance(time_ms, bool):
        raise TypeError("time_ms must be numeric")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return int(round(time_ms / 1000.0 * sample_rate))


N_FFT = 512


HOP_LENGTH = 160


N_MFCC = 20


N_ACOUSTIC_FEATURES = 52  # MFCC·RMS·ZCR·스펙트럼 통계만 센다 (acoustic_feature_vector()가 돌려주는 차원 수)


def acoustic_feature_vector(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """52차원 MFCC·RMS·ZCR·스펙트럼 통계. 호출하는 쪽에서 앞에 duration_sec를 붙인다."""

    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=segment, n_fft=N_FFT, hop_length=HOP_LENGTH)

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [rms.mean(), rms.std()],
        [zcr.mean(), zcr.std()],
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [bandwidth.mean(), bandwidth.std()],
        [flatness.mean(), flatness.std()],
    ]
    vector = np.concatenate([np.asarray(part, dtype=np.float64).ravel() for part in parts])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(f"acoustic feature vector has {vector.shape[0]} dims, expected {N_ACOUSTIC_FEATURES}")
    return vector.astype(np.float32)


def pair_by_stem(audio_dir: Path, label_dir: Path) -> list[tuple[str, Path, Path]]:
    wav_files = {p.stem: p for p in Path(audio_dir).rglob("*.wav")}
    json_files = {p.stem: p for p in Path(label_dir).rglob("*.json")}
    common = sorted(set(wav_files) & set(json_files))
    return [(stem, wav_files[stem], json_files[stem]) for stem in common]


@dataclass(frozen=True)
class CallerSegment:
    waveform: np.ndarray
    sample_rate: int
    start_at: Real
    end_at: Real


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
