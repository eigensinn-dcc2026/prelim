"""미션 2 공통 음성·라벨 처리: 서울 데이터 고르기, 발화 목록 만들기, 통화 음성 읽기, 발화 구간 자르기.

CNN 과 WavLM 의 학습·추론이 모두 이 함수들을 쓴다. 저장소의 기존 도구를 그대로 쓰고 다시 만들지 않는다.

- ``dcc_data.manifest``: wav/json 을 파일 이름으로 짝짓는다(JSON 안의 ``audioPath`` 는 믿을 수 없어서 쓰지 않음).
- ``dcc_data.guards.ensure_training_rows``: Validation 데이터가 학습 경로로 들어가지 못하게 막는다.
- ``dcc_data.audio``: WAV 읽기. 통화 WAV 는 한 번만 읽고, 발화는 메모리에서 잘라 쓴다
  (발화마다 파일을 다시 열면 그게 가장 큰 시간 비용이었다).

라벨 JSON 의 발화에서는 ``startAt``/``endAt``/``speaker`` 만 읽는다. ``text`` 는 절대 읽지 않는다.
``utt_idx``(JSON 안에서의 발화 순서)는 제출 행 순서를 되살리는 데만 쓰고, 모델 입력으로는 절대 쓰지 않는다
(대회 규정: "발화의 홀짝 등 순서 사용불가").

(제출 폴더 정리 메모: 초기 L1 모델(LightGBM)용 53차원 음향 통계 특징과 그 데이터셋 만들기 코드는 최종 모델에 쓰이지 않아 뺐다.)
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Iterable

# numpy/librosa 를 불러오기 전에 정해야 한다(BLAS 스레드 수는 라이브러리를 불러올 때 고정됨).
# 여러 프로세스로 나눠 처리할 때 프로세스마다 BLAS 가 CPU 코어 수만큼 스레드를 또 만들면 스레드가 넘쳐서,
# 12코어/16GB PC 에서 작업 프로세스가 메모리 할당 실패로 죽었다. 이 모듈을 쓰는 모든 곳이 혜택을 보도록 여기서 정한다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from .audio import read_wav_frames, read_wav_info
from .guards import TRAINING, VALIDATION, ensure_split, ensure_training_rows
from .manifest import ManifestRow
from .mission2 import milliseconds_to_frame


# ---------------------------------------------------------------------------
# 지역 고르기 (학습은 서울 데이터만, 대회 규정과 팀 EDA 기준). ManifestRow 에는 지역 정보가 없어서
# 경로의 "TS_<지역>_구급" 같은 폴더 이름에서 지역을 알아낸다.
# ---------------------------------------------------------------------------

_REGION_DIR_RE = re.compile(r"^(TS|TL|VS|VL)_([^_]+)_구급$")
SEOUL_REGION = unicodedata.normalize("NFC", "서울")


def path_region(relative_path: str) -> str | None:
    """manifest 상대 경로에 들어 있는 지역 이름을 돌려준다(없으면 None).

    zip 안의 한글 폴더 이름은 자모가 분리된 형태(NFD)로 들어올 수 있다. 화면에는 같아 보여도 문자열 비교에서 다르고,
    아래 정규식의 "_구급$"에도 안 걸린다. 그래서 경로의 각 부분을 합쳐진 형태(NFC)로 바꾼 뒤 비교한다.
    """

    for part in PurePosixPath(relative_path).parts:
        normalized_part = unicodedata.normalize("NFC", part)
        match = _REGION_DIR_RE.fullmatch(normalized_part)
        if match:
            return match.group(2)
    return None


def is_seoul_path(relative_path: str) -> bool:
    return path_region(relative_path) == SEOUL_REGION


def select_seoul_training_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Training 이면서 서울인 행만 고른다. 학습에 쓸 수 있는 유일한 데이터다.

    ``ensure_training_rows`` 는 걸러 주는 함수가 아니라 "이미 순수 Training 인지" 확인하고 섞여 있으면 멈추는 검사다.
    manifest 에는 보통 두 분할이 함께 들어 있으므로 먼저 ``row.split`` 으로 거르고, 거르기 실수에 대비해 검사를 한 번 더 한다.
    """

    training_only = [row for row in rows if row.split == TRAINING]
    training_rows = ensure_training_rows(training_only)
    return [
        row
        for row in training_rows
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def select_seoul_validation_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Validation 이면서 서울인 행만 고른다.

    대회 규정에 따라 평가·모델 선택에만 쓴다. 이 결과로 학습(역전파)을 하면 안 된다.
    ``select_seoul_training_rows`` 와 같은 방식이고, 검사만 ``VALIDATION`` 기준이다.
    """

    validation_only = [row for row in rows if row.split == VALIDATION]
    validation_rows = ensure_split(validation_only, VALIDATION)
    return [
        row
        for row in validation_rows
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def sample_calls(rows: list[ManifestRow], n_calls: int, seed: int) -> list[ManifestRow]:
    """통화(발화가 아님) 단위로 ``n_calls`` 개를 재현 가능하게 뽑는다. 0 이하이거나 전체보다 크면 전부 쓴다.

    통화 단위로 뽑아야 뒤에서 통화 단위 분할(GroupShuffleSplit)이 의미가 있다.
    """

    if n_calls <= 0 or n_calls >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n_calls)


# ---------------------------------------------------------------------------
# 발화 목록 (메타데이터만, 음성은 아직 읽지 않음)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtteranceRecord:
    call_id: str
    wav_stem: str
    wav_path: str
    json_path: str
    utt_idx: int
    start_at: float
    end_at: float
    duration_sec: float
    speaker: int


def _load_call_utterances(json_path: Path) -> tuple[str, list[dict]]:
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    utterances = document.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"utterances must be a list: {json_path}")
    call_id = document.get("_id") or json_path.stem
    return str(call_id), utterances


def build_utterance_records(row: ManifestRow, data_root: Path) -> list[UtteranceRecord]:
    """한 통화의 manifest 행을 발화 단위 목록으로 펼친다.

    각 발화에서는 ``startAt``/``endAt``/``speaker`` 만 읽고 ``text`` 는 절대 읽지 않는다(미션 2는 대사 내용을 쓸 수 없음).

    ``speaker`` 는 제공된 라벨 파일의 값을 그대로 쓴다. 운영 측 판단(2026-10-01)에 따라 ``text`` 를 근거로 라벨을 고치거나
    학습 데이터를 고르지 않는다. 의심되는 라벨 오류는 고치지 않고 따로 목록으로만 남긴다.
    """

    json_path = data_root / row.json_path
    call_id, utterances = _load_call_utterances(json_path)
    records: list[UtteranceRecord] = []
    for idx, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            raise ValueError(f"utterance must be an object: {json_path}[{idx}]")
        start_at = utterance["startAt"]
        end_at = utterance["endAt"]
        speaker = utterance["speaker"]
        if not isinstance(start_at, Real) or isinstance(start_at, bool):
            raise TypeError(f"startAt must be numeric: {json_path}[{idx}]")
        if not isinstance(end_at, Real) or isinstance(end_at, bool):
            raise TypeError(f"endAt must be numeric: {json_path}[{idx}]")
        if start_at < 0:
            raise ValueError(f"startAt must be non-negative: {json_path}[{idx}] got {start_at!r}")
        if end_at <= start_at:
            raise ValueError(
                f"endAt must be greater than startAt: {json_path}[{idx}] "
                f"got startAt={start_at!r} endAt={end_at!r}"
            )
        if speaker not in (0, 1):
            raise ValueError(f"speaker must be 0 or 1: {json_path}[{idx}] got {speaker!r}")
        records.append(
            UtteranceRecord(
                call_id=call_id,
                wav_stem=row.stem,
                wav_path=row.wav_path,
                json_path=row.json_path,
                utt_idx=idx,
                start_at=float(start_at),
                end_at=float(end_at),
                duration_sec=(float(end_at) - float(start_at)) / 1000.0,
                speaker=int(speaker),
            )
        )
    return records


# ---------------------------------------------------------------------------
# 음성 읽기: 통화 WAV 는 한 번만 열고, 발화는 메모리에서 자른다
# ---------------------------------------------------------------------------


def read_call_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    """통화 WAV 전체를 한 번 읽는다. 팀 EDA 에서 모든 WAV 가 모노임을 확인했다(아니면 멈춤)."""

    info = read_wav_info(wav_path)
    if info.channels != 1:
        raise ValueError(f"expected mono WAV, got {info.channels} channels: {wav_path}")
    segment = read_wav_frames(wav_path, 0, info.frame_count)
    return segment.waveform[0], segment.sample_rate


def utterance_frame_range(
    start_at_ms: Real, end_at_ms: Real, sample_rate: int, total_frames: int
) -> tuple[int, int]:
    """발화의 startAt/endAt(밀리초)을 샘플 위치로 바꾼다. 파일 길이를 넘으면 잘라 맞추고, 길이가 0 이하면 멈춘다."""

    start_frame = max(0, min(milliseconds_to_frame(start_at_ms, sample_rate), total_frames))
    end_frame = max(0, min(milliseconds_to_frame(end_at_ms, sample_rate), total_frames))
    if end_frame <= start_frame:
        raise ValueError(
            f"non-positive frame range after clamping: start={start_frame} end={end_frame} "
            f"(raw ms {start_at_ms}-{end_at_ms}, total_frames={total_frames})"
        )
    return start_frame, end_frame


def slice_and_pad(waveform: np.ndarray, start_frame: int, end_frame: int, min_length: int) -> np.ndarray:
    """통화 음성에서 발화 하나를 잘라 낸다. ``min_length`` 보다 짧으면 뒤를 0(무음)으로 채운다.

    짧은 발화(100ms 이하 포함)도 실제 데이터라 버리지 않는다. 대신 STFT 계산이 실패하지 않도록 최소 길이까지만 채운다.
    """

    segment = waveform[start_frame:end_frame]
    if len(segment) < min_length:
        segment = np.pad(segment, (0, min_length - len(segment)))
    return segment.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# STFT 설정 (CNN 스펙트로그램이 쓰는 값. 체크포인트에 기록된 학습 설정과 실행 때 대조한다)
# ---------------------------------------------------------------------------

N_FFT = 512
HOP_LENGTH = 160
