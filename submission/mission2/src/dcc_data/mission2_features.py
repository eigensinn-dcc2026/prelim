"""Mission 2 기준선(L1) 특징 파이프라인: 음향 통계만 쓴다.

wav<->json 짝짓기나 split 분리를 다시 구현하지 않고, 저장소에 이미 있는 manifest/guards/audio 기본 도구 위에
만든다.

- ``dcc_data.manifest``는 이미 wav/json 파일을 파일 이름(stem)으로 짝짓는다(각 JSON 안의 ``audioPath``
  필드는 믿을 수 없다고 알려져 있다. ``manifest.build_manifest`` 참고). 여기서는 그 결과(읽어 온 목록)를
  그대로 쓴다.
- ``dcc_data.guards.ensure_training_rows``는 이미 Validation 데이터가 학습 경로에 닿지 못하게 강제한다.
  ``row.split``을 손으로 다시 검사하지 않고 이 함수를 부른다.
- ``dcc_data.audio``는 이미 의존성이 적은 PCM WAV 읽기를 한다. 통화마다 WAV를 정확히 한 번만
  (``read_wav_frames(path, 0, frame_count)``) 읽고 메모리의 배열에서 발화를 잘라 낸다. 발화마다 파일을 다시
  여는 것은 통화당 약 30개 발화에서 가장 큰 비용임이 측정되었다(M2 기준선 계획 노트 참고).

라벨 JSON의 발화에서 읽는 것은 ``startAt``/``endAt``/``speaker``뿐이다. ``text``는 절대 건드리지 않는다.
``utt_idx``(JSON 안에서 그 발화의 위치)는 이후 코드가 제출 행 순서를 복원할 수 있도록 들고만 다니며, 모델
특징으로는 절대 쓰면 안 된다. 대회가 순서에서 나온 신호를 명시적으로 금지하기 때문이다
("발화의 홀짝 등 순서 사용불가").
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
from typing import Callable, Iterable

# numpy/librosa를 불러오기 전에 설정해야 한다(BLAS 스레드 수는
# 라이브러리를 불러올 때 정해진다). build_seoul_training_dataset()의 max_workers > 1
# 경로는 이미 OS 프로세스 단위로 병렬화한다. 이 설정이 없으면 각
# 프로세스의 BLAS 백엔드가 그 위에 CPU 코어마다 스레드를 또 하나씩 만들어서,
# 워커 N개 x BLAS 스레드 C개가 되어 순전한 과다 할당이 된다.
# 12코어/16GB 컴퓨터에서는 워커가 4개만 되어도 모든 워커가 어김없이 죽었다("OpenBLAS
# error: Memory allocation still failed after 10 retries" / 윈도우 "paging
# file too small"). 이 모듈을 부르는 모든 곳(CLI 스크립트, 테스트, 임시 노트북)이
# 각자 설정하는 것을 기억할 필요 없이, 여기서 고정하면 모두
# 이득을 본다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import librosa

from .audio import read_wav_frames, read_wav_info
from .guards import TRAINING, VALIDATION, ensure_split, ensure_training_rows
from .manifest import ManifestRow
from .mission2 import milliseconds_to_frame


# ---------------------------------------------------------------------------
# 지역 걸러내기(Training은 서울만 써야 한다. 대회 규정과
# 팀 EDA 참고). ManifestRow에는 지역 필드가 없으므로 경로에서 지역을 되찾는다.
# 경로에서 "TS_<지역>_구급" 형식의 폴더 규칙으로 구한다. 같은 목적으로
# scripts/audit_dcc_dataset.py도 이미 이 규칙에 의존한다.
# ---------------------------------------------------------------------------

_REGION_DIR_RE = re.compile(r"^(TS|TL|VS|VL)_([^_]+)_구급$")
SEOUL_REGION = unicodedata.normalize("NFC", "서울")


def path_region(relative_path: str) -> str | None:
    """목록 기준 상대 경로에 들어 있는 지역 토큰을 돌려준다(있다면).

        zip 압축 파일 안의 한글 이름 폴더로 출발한 경로(이 데이터셋이 그렇다)는 NFC(조합형) 대신 NFD(자모 분리형)로
        들어올 수 있다. 둘은 화면에는 똑같이 보이지만 비교하면 다르고, NFD는 아래 정규식의 "_구급$"와 글자
        그대로의 일치도 깨뜨린다. 파일 시스템이나 zip이 어떤 형태를 주었든 상관없도록, 경로의 모든 구성 요소를
        NFC로 바꾼 뒤 맞춰 본다.
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
    """Training만, 서울만 고른 행. 이 파이프라인이 쓸 수 있는 유일한 행이다.

        ``ensure_training_rows``는 배치가 이미 순수한 Training인지 확인하는 즉시 실패(fail-fast) *단언*이다
        (섞인 배치에서는 오류를 내며, 걸러 내지는 않는다). ``dcc_data.guards`` 참고.
        ``load_manifest``로 읽은 목록에는 보통 두 split이 함께 들어 있으므로, 먼저 ``row.split``으로
        거른 뒤 걸러 내는 코드의 버그에 대비한 이중 방어로 이 검사를 실행한다.
    """

    training_only = [row for row in rows if row.split == TRAINING]
    training_rows = ensure_training_rows(training_only)
    return [
        row
        for row in training_rows
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def select_seoul_validation_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Validation만, 서울만 고른 행.

        대회 규정상 평가/조기 종료/체크포인트 선택용일 뿐이다. 이 결과를 쓰는 어떤 코드도 ``.fit()``이나
        역전파를 해서는 안 된다. ``select_seoul_training_rows``와 똑같되, ``TRAINING`` 대신 ``VALIDATION``을
        기준으로 같은 ``ensure_split`` 검사를 쓴다.
    """

    validation_only = [row for row in rows if row.split == VALIDATION]
    validation_rows = ensure_split(validation_only, VALIDATION)
    return [
        row
        for row in validation_rows
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def sample_calls(rows: list[ManifestRow], n_calls: int, seed: int) -> list[ManifestRow]:
    """통화(발화가 아니라) 단위로 ``n_calls``개 통화를 재현 가능하게 뽑는다.

        이후 GroupShuffleSplit-by-call_id가 의미를 가지려면 발화 단위가 아니라 통화 단위로 뽑아야 한다.
    """

    if n_calls <= 0 or n_calls >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n_calls)


# ---------------------------------------------------------------------------
# 모듈 A: 발화 단위 인덱스(메타데이터만 다루고, 여기서는 오디오를 건드리지 않는다)
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
    """통화 하나의 목록 행을 발화 단위 인덱스 행들로 펼친다.

        각 발화에서 읽는 것은 ``startAt``/``endAt``/``speaker``뿐이다. ``text``는 절대 읽지 않는다(Mission 2는
        전사 내용을 특징으로 쓰는 것을 금지한다).

        ``speaker``는 제공된 라벨 파일에 있는 그대로 쓴다. 주최 측은 ``text``를 근거로 화자 라벨을 고치거나
        학습 데이터를 고르는 것을 금지한다고 판정했다(2026-10-01). 의심되는 라벨 오류는 고치지 않고
        reports/mission2/m2_label_noise_suspected_calls.md에 목록으로만 적어 둔다.
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
# 오디오 접근: 통화마다 WAV를 한 번만 열고, 발화는 메모리에서 잘라 낸다
# ---------------------------------------------------------------------------


def read_call_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    """통화 전체의 WAV를 한 번에 읽는다. 팀 EDA에서 모든 WAV가 모노임을 확인했다."""

    info = read_wav_info(wav_path)
    if info.channels != 1:
        raise ValueError(f"expected mono WAV, got {info.channels} channels: {wav_path}")
    segment = read_wav_frames(wav_path, 0, info.frame_count)
    return segment.waveform[0], segment.sample_rate


def utterance_frame_range(
    start_at_ms: Real, end_at_ms: Real, sample_rate: int, total_frames: int
) -> tuple[int, int]:
    start_frame = max(0, min(milliseconds_to_frame(start_at_ms, sample_rate), total_frames))
    end_frame = max(0, min(milliseconds_to_frame(end_at_ms, sample_rate), total_frames))
    if end_frame <= start_frame:
        raise ValueError(
            f"non-positive frame range after clamping: start={start_frame} end={end_frame} "
            f"(raw ms {start_at_ms}-{end_at_ms}, total_frames={total_frames})"
        )
    return start_frame, end_frame


def slice_and_pad(waveform: np.ndarray, start_frame: int, end_frame: int, min_length: int) -> np.ndarray:
    """통화 파형에서 발화 하나를 잘라 내고, 짧은 구간은 0으로 채운다.

        짧은 발화를 버려서는 안 된다(대회 자체의 100ms 이하 발화도 잡음이 아니라 실제 데이터이다).
        그래서 ``min_length``까지 0으로 채워, 이후 STFT 기반 특징이 이 발화에서 오류 없이 계산되게 한다.
    """

    segment = waveform[start_frame:end_frame]
    if len(segment) < min_length:
        segment = np.pad(segment, (0, min_length - len(segment)))
    return segment.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 모듈 B: 53차원 음향 특징 벡터(길이 + 스펙트럼 통계)
# ---------------------------------------------------------------------------

N_FFT = 512
HOP_LENGTH = 160
N_MFCC = 20
N_ACOUSTIC_FEATURES = 52  # MFCC·RMS·ZCR·스펙트럼 통계만 센다 (acoustic_feature_vector()가 돌려주는 차원 수)
N_FEATURES = 53  # duration_sec + 아래의 음향 통계 52개

FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_sec",
    *(f"mfcc_mean_{i}" for i in range(N_MFCC)),
    *(f"mfcc_std_{i}" for i in range(N_MFCC)),
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
)
assert len(FEATURE_COLUMNS) == N_FEATURES


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


def utterance_feature_vector(segment: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    """발화 하나를 다른 정보 없이 단독으로 본 53차원 벡터."""

    acoustic = acoustic_feature_vector(segment, sample_rate)
    vector = np.concatenate([np.array([duration_sec], dtype=np.float32), acoustic])
    if vector.shape[0] != N_FEATURES:
        raise ValueError(f"feature vector has {vector.shape[0]} dims, expected {N_FEATURES}")
    return vector


# ---------------------------------------------------------------------------
# 모듈 C와 전체 진행: 통화 뽑기, 인덱스 만들기, 특징 추출
# ---------------------------------------------------------------------------

OnCallDone = Callable[[int, int], None]

_CallResult = tuple[list[UtteranceRecord], list[dict], list[dict]]


def _process_one_call(row: ManifestRow, data_root: Path) -> _CallResult:
    """통화 하나를 처음부터 끝까지 처리한다: 인덱스와 특징 추출.

        ``(row, data_root)``의 순수 함수이며 공유 상태가 없다. ``build_seoul_training_dataset``이 순차 반복
        경로와 프로세스 풀 경로 모두에서 쓰는 작업 단위이고, 윈도우(spawn 시작 방식)에서 피클로 직렬화해
        워커 프로세스에 보낼 수 있도록 클로저가 아니라 최상위 함수여야 한다.
    """

    index_records: list[UtteranceRecord] = []
    feature_rows: list[dict] = []
    failures: list[dict] = []

    wav_path = data_root / row.wav_path
    try:
        records = build_utterance_records(row, data_root)
    except Exception as exc:  # noqa: BLE001 - 모듈 설명 참고: 기록하며, 삼키지 않는다
        failures.append(
            {
                "stage": "index",
                "call_id": row.stem,
                "utt_idx": None,
                "wav_path": row.wav_path,
                "json_path": row.json_path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return index_records, feature_rows, failures

    index_records.extend(records)

    try:
        waveform, sample_rate = read_call_waveform(wav_path)
    except Exception as exc:  # noqa: BLE001 - 모듈 설명 참고
        for record in records:
            failures.append(
                {
                    "stage": "audio_open",
                    "call_id": record.call_id,
                    "utt_idx": record.utt_idx,
                    "wav_path": row.wav_path,
                    "json_path": row.json_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        return index_records, feature_rows, failures

    total_frames = waveform.shape[0]
    for record in records:
        try:
            start_frame, end_frame = utterance_frame_range(
                record.start_at, record.end_at, sample_rate, total_frames
            )
            segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
            vector = utterance_feature_vector(segment, sample_rate, record.duration_sec)
        except Exception as exc:  # noqa: BLE001 - 모듈 설명 참고
            failures.append(
                {
                    "stage": "feature",
                    "call_id": record.call_id,
                    "utt_idx": record.utt_idx,
                    "wav_path": row.wav_path,
                    "json_path": row.json_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        feature_rows.append(
            {
                "call_id": record.call_id,
                "utt_idx": record.utt_idx,
                "speaker": record.speaker,
                **dict(zip(FEATURE_COLUMNS, vector.tolist())),
            }
        )

    return index_records, feature_rows, failures


def _build_seoul_dataset(
    seoul_rows: list[ManifestRow],
    data_root: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[UtteranceRecord] | None = None,
    feature_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
) -> tuple[list[UtteranceRecord], list[dict], list[dict]]:
    """아래의 Training과 Validation 빌더가 함께 쓰는, 뽑기와 추출을 하는 반복문.
        둘의 유일한 차이는 ``seoul_rows``를 만든 ``select_seoul_*_rows`` 거르개가 무엇이냐뿐이다.

        ``index_sink``/``feature_sink``/``failure_sink``를 밖에서 넘길 수 있어서, 호출하는 쪽(예: CLI 스크립트)이
        실행 도중에 진행 상황을 부분적으로 저장할 수 있다. 이 객체들은 제자리에서 바뀌고 편의상 돌려주기도 한다.

        발화 하나의 특징 추출이 실패해도 그 발화를 인덱스(모듈 A)에서 빼지 않고 통화의 나머지 처리도 멈추지
        않는다. 대신 call_id/utt_idx/이유와 함께 ``failure_sink``에 기록한다("원인 로깅 없는 광범위한 except
        금지" 규칙에 따름). 오디오/librosa의 실패 유형이 많아서 넓게 잡지만, 모든 실패를 조치할 수 있을 만큼
        자세히 기록한다.

        ``max_workers`` > 1이면 프로세스 풀로 통화를 병렬 처리한다. 통화마다 자기 wav 파일만 다루고 공유 상태가
        없으므로 순수한 속도 최적화이다: 세 sink의 전체 내용은 ``max_workers=1`` 실행과 같다(다만 끝나는 순서에
        따라 다른 순서로 섞여 들어가는데, 이후 어떤 코드도 그 순서에 기대지 않는다. 한 통화 *안에서*의 utt_idx
        순서는 어느 쪽이든 영향을 받지 않는다).
    """

    index_sink = index_sink if index_sink is not None else []
    feature_sink = feature_sink if feature_sink is not None else []
    failure_sink = failure_sink if failure_sink is not None else []

    sampled_rows = sample_calls(seoul_rows, n_calls, seed)
    total = len(sampled_rows)

    if max_workers <= 1:
        for call_number, row in enumerate(sampled_rows, start=1):
            records, features, failures = _process_one_call(row, data_root)
            index_sink.extend(records)
            feature_sink.extend(features)
            failure_sink.extend(failures)
            if on_call_done:
                on_call_done(call_number, total)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_one_call, row, data_root) for row in sampled_rows]
            for call_number, future in enumerate(as_completed(futures), start=1):
                records, features, failures = future.result()
                index_sink.extend(records)
                feature_sink.extend(features)
                failure_sink.extend(failures)
                if on_call_done:
                    on_call_done(call_number, total)

    return index_sink, feature_sink, failure_sink


def build_seoul_training_dataset(
    rows: Iterable[ManifestRow],
    data_root: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[UtteranceRecord] | None = None,
    feature_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
) -> tuple[list[UtteranceRecord], list[dict], list[dict]]:
    """서울/Training 통화를 뽑아 발화 단위 인덱스와 특징 캐시를 만든다.

        공통 추출 반복문의 약속은 ``_build_seoul_dataset``을 보라.
    """

    return _build_seoul_dataset(
        select_seoul_training_rows(rows),
        data_root,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        feature_sink=feature_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
    )


def build_seoul_validation_dataset(
    rows: Iterable[ManifestRow],
    data_root: Path,
    *,
    n_calls: int = -1,
    seed: int = 42,
    index_sink: list[UtteranceRecord] | None = None,
    feature_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
) -> tuple[list[UtteranceRecord], list[dict], list[dict]]:
    """서울/Validation 통화를 뽑아 발화 단위 인덱스와 특징 캐시를 만든다.

        ``build_seoul_training_dataset``과 같은 추출 파이프라인(같은 53개 특징 열, 제공된 그대로의 라벨)이며,
        ``select_seoul_validation_rows``를 기준으로 한다. 대회 규정상 평가 전용이다(그 함수의 설명 참고).
        ``n_calls=-1``(기본값)은 Validation 통화를 잘라 내지 않고 모두 가져온다. Validation은 이미
        Training보다 훨씬 작아서 더 줄일 이유가 없기 때문이다.
    """

    return _build_seoul_dataset(
        select_seoul_validation_rows(rows),
        data_root,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        feature_sink=feature_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# 모듈 E 보조: 추론용 평평한(flat) stem 짝짓기(채점 쪽이 주는 폴더에는 Training/Validation
# 경로 구성 요소가 있다는 보장이 없으므로,
# 바로 그것을 요구하는 dcc_data.manifest.build_manifest는
# 여기에 쓸 수 없다. 이것은 split 요구 조건을 뺀 같은 짝짓기 논리이다).
# ---------------------------------------------------------------------------


def pair_by_stem(audio_dir: Path, label_dir: Path) -> list[tuple[str, Path, Path]]:
    wav_files = {p.stem: p for p in Path(audio_dir).rglob("*.wav")}
    json_files = {p.stem: p for p in Path(label_dir).rglob("*.json")}
    common = sorted(set(wav_files) & set(json_files))
    return [(stem, wav_files[stem], json_files[stem]) for stem in common]
