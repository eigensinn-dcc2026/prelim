"""미션 2 CNN 입력: 발화별 log-mel 스펙트로그램 만들기와 그 목록(index)의 출처 검사.

초기 L1 모델은 발화를 손으로 고른 요약 통계 53~64개로 줄였는데, 통계를 세 번 더 늘려도(음높이·에너지, 통화 상대값, 겹침 신호)
가장 중요한 두 오류(발화 길이에 따른 대원/신고자 치우침, 목소리가 겹칠 때의 정확도 저하)가 줄지 않았다.
이 모듈은 시간-주파수 구조를 그대로 남긴다. 두 오류가 의존하는 정보(세밀한 음높이·에너지 *변화*, 두 목소리가 겹치는 무늬)는
평균·표준편차로 줄일 때 사라지는 정보이기 때문이다.

틀 설정(N_FFT, HOP_LENGTH)과 발화 자르기(read_call_waveform, slice_and_pad, build_utterance_records)는
dcc_data.mission2_features 의 것을 그대로 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import librosa

from .guards import TRAINING, SplitContaminationError
from .manifest import ManifestRow
from .mission2_features import (
    HOP_LENGTH,
    N_FFT,
    build_utterance_records,
    read_call_waveform,
    sample_calls,
    select_seoul_training_rows,
    select_seoul_validation_rows,
    slice_and_pad,
    utterance_frame_range,
)

if TYPE_CHECKING:
    import pandas as pd

N_MELS = 40   # mel 주파수 칸 수
# librosa.power_to_db 의 기본 잘라내기 범위(top_db=80, 그 스펙트로그램의 최댓값 기준). 채움 프레임을 정확히 이 바닥값으로 두어
# CNN 이 무음으로 읽게 한다(0.0 으로 채우면 반대로 아래 ref=np.max 정규화에서 그 발화의 *가장 큰* 소리가 된다).
SPEC_FLOOR_DB = -80.0

# 목록의 ``speaker`` 열이 제공 라벨 파일과 어떤 관계인지 나타내는 값. 운영 측 판단(2026-10-01)에 따라 미션 2 학습은 제공된 speaker
# 라벨을 그대로 써야 한다(``text`` 근거의 보정이나 데이터 선별 금지). 그래서 목록은 자기 라벨이 제공 라벨임을 표시해야 한다.
LABEL_SOURCE_PROVIDED = "provided_json"  # 목록을 만들 때 제공 라벨 JSON 에서 바로 옮겨 적음
LABEL_SOURCE_VERIFIED = "verified_against_provided_json"  # 팀 저장소의 scripts/verify_mission2_raw_labels.py 로 한 행씩 다시 대조함
ALLOWED_LABEL_SOURCES = (LABEL_SOURCE_PROVIDED, LABEL_SOURCE_VERIFIED)

# 목록의 모든 행에 스펙트로그램의 출처(그 통화의 분할), 만든 설정, 라벨 출처를 기록한다. 그래서 학습은 입력이 Training 전용이고
# 제공 라벨임을 증명할 수 있고, 체크포인트는 무엇으로 학습했는지 기록할 수 있다(assert_training_spectrogram_index 참고).
INDEX_PROVENANCE_COLUMNS = ("split", "n_mels", "hop_length", "n_fft", "sample_rate", "label_source")


class LabelProvenanceError(ValueError):
    """스펙트로그램 목록이 자기 speaker 라벨이 제공 라벨이라고 표시하지 않을 때."""


def assert_training_spectrogram_index(index_df: "pd.DataFrame") -> dict:
    """스펙트로그램 목록의 모든 행이 Training 에서, 제공 라벨로 만들어졌다고 표시돼 있지 않으면 멈춘다. (학습 입구의 안전장치)

    목록을 만든 설정 하나 ``{split, n_mels, hop_length, n_fft, sample_rate, label_source}`` 를 돌려준다.
    표시가 없는 목록은 Training 전용인지, 제공 라벨인지 증명할 수 없으므로 믿지 않고 거부한다. 그러지 않으면 같은 경로에 만든
    Validation 목록이 Training 목록과 똑같아 보이고, 라벨이 수정된 캐시로도 조용히 학습될 수 있다.
    """

    missing = [column for column in INDEX_PROVENANCE_COLUMNS if column not in index_df.columns]
    if missing:
        raise SplitContaminationError(
            f"spectrogram index has no build provenance (missing column(s): {missing}), so it cannot be shown "
            "to come from Training only with the provided speaker labels. Rebuild it with "
            "scripts/build_mission2_spectrograms.py, or, for a cache you know is a Training-only, "
            "default-settings one, run scripts/verify_mission2_raw_labels.py --index <index> --label-root "
            "<Training label folder> --output <new index> --stamp-training-defaults -- it checks every speaker "
            "label against the provided label files, restores any that differ, and writes the stamp."
        )
    if len(index_df) == 0:
        raise ValueError("spectrogram index is empty")

    # Training 이 아닌 행이 하나라도 있으면 멈춘다
    foreign = index_df.loc[index_df["split"] != TRAINING, "split"]
    if len(foreign):
        counts = foreign.astype(str).value_counts().to_dict()
        raise SplitContaminationError(
            f"Training run received {len(foreign)} non-Training spectrogram index row(s): {counts}"
        )

    # 설정 값이 행마다 같아야 한다(섞여 있으면 멈춤)
    provenance = {}
    for column in INDEX_PROVENANCE_COLUMNS:
        values = index_df[column].unique()
        if len(values) != 1:
            raise ValueError(f"spectrogram index mixes several {column} values: {sorted(map(str, values))[:5]}")
        value = values[0]
        provenance[column] = value.item() if hasattr(value, "item") else value
    if provenance["label_source"] not in ALLOWED_LABEL_SOURCES:
        raise LabelProvenanceError(
            f"spectrogram index label_source is {provenance['label_source']!r}; training needs the provided "
            f"speaker labels as they are ({', '.join(ALLOWED_LABEL_SOURCES)}). Rebuild the index, or check and "
            "restore its labels with scripts/verify_mission2_raw_labels.py."
        )
    return provenance


def mel_spectrogram(
    segment: np.ndarray, sample_rate: int, *, n_mels: int = N_MELS, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """log-mel 스펙트로그램, 모양 (n_mels, n_frames). 그 발화 자신의 최댓값 기준으로 정규화한다(최대 0 dB, 최저 SPEC_FLOOR_DB).

    발화 자신 기준의 상대적인 소리 크기를 쓰는 이유: 통화마다 녹음 음량이 달라서 절대 크기는 화자를 가리는 신호가 아니다.
    (다른 발화는 전혀 쓰지 않는다.) ``hop_length`` 를 줄이면 같은 소리에서 프레임이 더 많이 나온다
    (scripts/build_mission2_spectrograms.py 의 --hop-length). 최종 모델은 기본값(160)이다.
    """

    # STFT 창 하나보다 짧으면 0 으로 채운다(아주 짧은 발화도 계산이 실패하지 않게)
    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=-SPEC_FLOOR_DB)
    return np.nan_to_num(log_mel, nan=SPEC_FLOOR_DB, neginf=SPEC_FLOOR_DB).astype(np.float32)


# ---------------------------------------------------------------------------
# 학습 데이터 만들기: 통화를 고르고, 발화마다 잘라서 스펙트로그램 .npy 하나씩 저장하고, 저장한 것의 목록(index)을 만든다
# ---------------------------------------------------------------------------

OnCallDone = Callable[[int, int], None]


def _spectrogram_path(output_dir: Path, call_id: str, utt_idx: int) -> Path:
    return output_dir / call_id / f"{utt_idx}.npy"


def _process_one_call_spectrograms(
    row: ManifestRow, data_root: Path, output_dir: Path, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS
) -> tuple[list[dict], list[dict]]:
    """통화 하나를 처리한다: 발화마다 잘라 스펙트로그램을 저장하고 목록 행을 만든다.

    실패한 발화는 조용히 빼지 않고 실패 목록에 이유와 함께 기록한다. 여러 프로세스로 나눠 처리할 때(Windows) 보낼 수 있도록
    맨 바깥 함수로 둔다.
    """

    index_rows: list[dict] = []
    failures: list[dict] = []

    wav_path = data_root / row.wav_path
    try:
        records = build_utterance_records(row, data_root)
    except Exception as exc:  # noqa: BLE001 - 삼키지 않고 기록한다
        failures.append(
            {"stage": "index", "call_id": row.stem, "utt_idx": None, "error_type": type(exc).__name__, "error_message": str(exc)}
        )
        return index_rows, failures

    try:
        waveform, sample_rate = read_call_waveform(wav_path)
    except Exception as exc:  # noqa: BLE001
        for record in records:
            failures.append(
                {"stage": "audio_open", "call_id": record.call_id, "utt_idx": record.utt_idx, "error_type": type(exc).__name__, "error_message": str(exc)}
            )
        return index_rows, failures

    total_frames = waveform.shape[0]
    for record in records:
        try:
            start_frame, end_frame = utterance_frame_range(record.start_at, record.end_at, sample_rate, total_frames)
            segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
            spec = mel_spectrogram(segment, sample_rate, hop_length=hop_length, n_mels=n_mels)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"stage": "spectrogram", "call_id": record.call_id, "utt_idx": record.utt_idx, "error_type": type(exc).__name__, "error_message": str(exc)}
            )
            continue

        path = _spectrogram_path(output_dir, record.call_id, record.utt_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, spec)
        # 목록 행: 라벨(제공 라벨 그대로), 출처 분할, 만든 설정을 함께 기록한다
        index_rows.append(
            {
                "call_id": record.call_id,
                "utt_idx": record.utt_idx,
                "speaker": record.speaker,
                "duration_sec": record.duration_sec,
                "n_frames": spec.shape[1],
                "npy_path": str(path.relative_to(output_dir)),
                "split": row.split,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "n_fft": N_FFT,
                "sample_rate": int(sample_rate),
                "label_source": LABEL_SOURCE_PROVIDED,
            }
        )

    return index_rows, failures


def _build_seoul_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    select_rows: Callable[[Iterable[ManifestRow]], list[ManifestRow]],
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> tuple[list[dict], list[dict]]:
    """Training 용과 Validation 용 만들기의 공통 부분. 어떤 행을 쓸지(``select_rows``)만 다르다.

    ``max_workers`` > 1 이면 통화들을 여러 프로세스로 나눠 처리한다. 통화마다 자기 파일만 다루므로 결과 내용은 같다(순서만 다름).
    """

    index_sink = index_sink if index_sink is not None else []
    failure_sink = failure_sink if failure_sink is not None else []

    seoul_rows = select_rows(rows)
    sampled_rows = sample_calls(seoul_rows, n_calls, seed)
    total = len(sampled_rows)
    output_dir = Path(output_dir)

    if max_workers <= 1:
        for call_number, row in enumerate(sampled_rows, start=1):
            index_rows, failures = _process_one_call_spectrograms(row, data_root, output_dir, hop_length, n_mels)
            index_sink.extend(index_rows)
            failure_sink.extend(failures)
            if on_call_done:
                on_call_done(call_number, total)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_process_one_call_spectrograms, row, data_root, output_dir, hop_length, n_mels)
                for row in sampled_rows
            ]
            for call_number, future in enumerate(as_completed(futures), start=1):
                index_rows, failures = future.result()
                index_sink.extend(index_rows)
                failure_sink.extend(failures)
                if on_call_done:
                    on_call_done(call_number, total)

    return index_sink, failure_sink


def build_seoul_training_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> tuple[list[dict], list[dict]]:
    """서울 Training 통화를 골라 발화마다 스펙트로그램 .npy 를 저장한다(학습용).

    ``hop_length``/``n_mels`` 기본값이 최종 모델의 설정이다. 다른 값으로 만들 때는 다른 ``output_dir`` 을 쓴다.
    """

    return _build_seoul_spectrograms(
        rows,
        data_root,
        output_dir,
        select_rows=select_seoul_training_rows,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
        hop_length=hop_length,
        n_mels=n_mels,
    )


def build_seoul_validation_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> tuple[list[dict], list[dict]]:
    """build_seoul_training_spectrograms 와 같지만 공식 Validation 을 대상으로 한다. 평가 전용이며 학습에 넣으면 안 된다
    (학습 입구의 assert_training_spectrogram_index 가 Validation 목록을 거부한다).
    """

    return _build_seoul_spectrograms(
        rows,
        data_root,
        output_dir,
        select_rows=select_seoul_validation_rows,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
        hop_length=hop_length,
        n_mels=n_mels,
    )
