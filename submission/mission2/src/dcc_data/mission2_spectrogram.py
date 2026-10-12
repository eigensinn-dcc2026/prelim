"""Mission 2 L2 입력 표현: CNN을 위한 로그 멜 스펙트로그램.

L1(dcc_data.mission2_features)은 발화마다 53~64개의 직접 고른 요약 통계로 줄인다. 그런 통계를 더하는
시도를 세 번(피치/에너지, 통화 상대 z-점수, 자기상관 기반의 조화성/겹침 신호) 했는데, 각각 따로 보면
분명한 효과가 있었지만 가장 중요한 두 오류 유형, 즉 길이 구간별 대원/신고자 편향과 겹침 때문에 생기는
정확도 격차(reports/mission2/ 참고)는 움직이지 못했다. 이 모듈은 시간-주파수 구조를 그대로 보존한다.
그 두 오류 유형이 기대는 정보(세밀한 피치/에너지의 *윤곽*, 또는 겹친 두 목소리의 간섭 무늬)가 바로
평균/표준편차/백분위수로 줄이면서 버려지는 정보라는 가설에 따른 것이다.

dcc_data.mission2_features의 프레임 상수(N_FFT, HOP_LENGTH)와 목록/오디오 자르기 파이프라인
(read_call_waveform, slice_and_pad, build_utterance_records)을 다시 만들지 않고 그대로 재사용한다.
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

N_MELS = 40
# librosa.power_to_db의 기본 잘라냄 범위(top_db=80, 스펙트로그램 자신의
# 최댓값 기준)이다. 채움 프레임을 정확히 이 하한으로 두어서
# 채운 "무음" 프레임이 CNN에게 큰 신호가 아니라 무음으로 읽히게 한다
# (그냥 0.0으로 채우면 정반대가 된다: 아래의 ref=np.max 정규화에서 0 dB는 이 발화에서
# *가장 큰* 프레임이다).
SPEC_FLOOR_DB = -80.0

# 인덱스의 ``speaker`` 열이 제공된 라벨 파일과 어떤 관계인지. 주최 측은 2026-10-01에
# Mission 2 학습에는 제공된 화자 라벨을 그대로 써야 한다고 판정했다: ``text``를 근거로 한 수정도,
# 데이터 선별도 안 된다. 그래서 인덱스는 자신의 라벨이 제공된 라벨이라고 밝혀야 한다.
LABEL_SOURCE_PROVIDED = "provided_json"  # 만든 쪽이 제공된 라벨 JSON에서 곧바로 써 넣은 것
LABEL_SOURCE_VERIFIED = "verified_against_provided_json"  # scripts/verify_mission2_raw_labels.py가 행마다 다시 확인한 것
ALLOWED_LABEL_SOURCES = (LABEL_SOURCE_PROVIDED, LABEL_SOURCE_VERIFIED)

# 인덱스의 모든 행은 그 스펙트로그램이 어디서 왔는지(그 통화의 목록 split), 어떤 설정으로 만들었는지,
# 라벨이 어디서 왔는지를 기록한다. 그래서 학습 실행은 입력이 제공된 라벨의 Training뿐임을 증명할 수 있고
# 체크포인트는 무엇으로 학습했는지를 기록할 수 있다.
# assert_training_spectrogram_index() 참고.
INDEX_PROVENANCE_COLUMNS = ("split", "n_mels", "hop_length", "n_fft", "sample_rate", "label_source")


class LabelProvenanceError(ValueError):
    """스펙트로그램 인덱스가 화자 라벨이 제공된 그대로의 것이라고 밝히지 않는다."""


def assert_training_spectrogram_index(index_df: "pd.DataFrame") -> dict:
    """스펙트로그램 인덱스의 모든 행이 제공된 라벨로 Training에서 만들었다는 도장이 찍혀 있지 않으면 실패한다.

        인덱스를 만든 ``{split, n_mels, hop_length, n_fft, sample_rate, label_source}`` 하나를 돌려준다.
        도장이 없는 인덱스(도장이 생기기 전에 만든 것)는 Training만 썼는지, 제공된 화자 라벨을 담고 있는지
        보일 수 없으므로 믿지 않고 거부한다. 공식 Validation 폴더가 학습 루프에 닿으면 절대 안 되고,
        Training과 같은 경로에 만든 Validation 인덱스는 도장이 없으면 겉보기가 똑같으며, 라벨을 고친 캐시는
        조용히 그 라벨로 학습하게 되기 때문이다.
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

    foreign = index_df.loc[index_df["split"] != TRAINING, "split"]
    if len(foreign):
        counts = foreign.astype(str).value_counts().to_dict()
        raise SplitContaminationError(
            f"Training run received {len(foreign)} non-Training spectrogram index row(s): {counts}"
        )

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
    """로그 멜 스펙트로그램, 모양 (n_mels, n_frames). 이 조각 자신의 최댓값을 기준으로 정규화한다
        (상한 0 dB, 하한 SPEC_FLOOR_DB). 발화 단위의 상대적인 크기로, CALL_RELATIVE_FEATURE_COLUMNS가 절대 척도가
        아니라 각 통화 자신의 평균/표준편차로 z-점수를 내는 것과 같은 이유이다: 절대적인 녹음 이득은 통화마다
        다르고 화자를 구별하는 신호가 아니다.

        ``hop_length``의 기본값은 L1의 수작업 특징이 쓰는 HOP_LENGTH와 같아서 기존의 모든 호출부는 영향을
        받지 않는다. 더 작은 값을 주면 같은 오디오에서 프레임이 더 많아진다. SpeakerCNN은 MaxPool(stride 2)을
        3번 거치므로 그렇지 않으면 짧은(맞장구 길이의) 발화는 AdaptiveAvgPool2d 앞에 시간 단계가 1~2개만
        남을 수 있다. scripts/build_mission2_spectrograms.py의 --hop-length를 보라.
    """

    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=-SPEC_FLOOR_DB)
    return np.nan_to_num(log_mel, nan=SPEC_FLOOR_DB, neginf=SPEC_FLOOR_DB).astype(np.float32)


# ---------------------------------------------------------------------------
# 전체 진행: 통화를 뽑고, 발화마다 잘라 내고, 발화마다 .npy 하나와
# 무엇을 저장했는지 적은 인덱스를 저장한다(
# mission2_features.build_seoul_training_dataset과 같고, 출력 형식만
# 다르다. 스펙트로그램은 61차원 벡터처럼
# 평평한 표의 한 행에 들어가지 않기 때문이다).
# ---------------------------------------------------------------------------

OnCallDone = Callable[[int, int], None]


def _spectrogram_path(output_dir: Path, call_id: str, utt_idx: int) -> Path:
    return output_dir / call_id / f"{utt_idx}.npy"


def _process_one_call_spectrograms(
    row: ManifestRow, data_root: Path, output_dir: Path, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS
) -> tuple[list[dict], list[dict]]:
    """통화 하나를 처리한다: 모든 발화를 잘라 스펙트로그램을 저장한다.

        mission2_features._process_one_call과 같은 "기록은 하되 조용히 버리지는 않는다" 실패 정책이며,
        최상위 함수여야 하는 이유도 같다: 윈도우의 프로세스 풀 워커로 보내려면 피클로 직렬화할 수 있어야 한다.
    """

    index_rows: list[dict] = []
    failures: list[dict] = []

    wav_path = data_root / row.wav_path
    try:
        records = build_utterance_records(row, data_root)
    except Exception as exc:  # noqa: BLE001 - 기록하며, 삼키지 않는다
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
    """build_seoul_training_spectrograms와 build_seoul_validation_spectrograms가 함께 쓰는 구현.
        대상이 되는 행(``select_rows``)만 다르며, mission2_features._build_seoul_dataset과 같은 구조이다.
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
    """서울/Training 통화를 뽑아 발화마다 스펙트로그램 .npy 하나를 저장한다.

        mission2_features.build_seoul_training_dataset의 뽑기/병렬 처리/실패 기록과 똑같다. 발화별 출력만 다르다
        (61차원 특징 사전 대신, 저장한 .npy와 인덱스 행).

        ``hop_length``의 기본값은 HOP_LENGTH(동작은 그대로)이다. 더 작은 값을 주면 시간 해상도가 더 높은
        별도의 스펙트로그램 묶음을 다른 ``output_dir``에 만든다. mel_spectrogram()의 설명을 보라.
        ``n_mels``의 기본값은 N_MELS(동작은 그대로)이다. 더 큰 값을 주면 주파수 해상도가 더 높은 묶음이 되며,
        ``output_dir``에 대한 주의는 같다.
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
    """build_seoul_training_spectrograms와 같되, 공식 Validation split
        (dcc_data.mission2_features.select_seoul_validation_rows)를 대상으로 한다.
        평가 전용이며, 학습 루프의 ``.fit()``/역전파에는 절대 넘기지 않는다.
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
