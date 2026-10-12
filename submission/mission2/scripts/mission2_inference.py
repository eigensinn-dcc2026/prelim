#!/usr/bin/env python3
"""Mission 2 추론 진입점(대회가 정한 실행 명령 형식).

    python scripts/mission2_inference.py \\
        --audio_dir {wav folder} --label_dir {json folder} \\
        --ckpt_path {checkpoint file} --output ./outputs/mission2.csv

audio_dir/label_dir는 파일 이름(stem)으로 짝짓고 Training/Validation 하위 폴더가 있다고 가정하지 않는다.
채점 쪽이 평평한 폴더를 주기 때문이다(그래서 Training 또는 Validation 경로 구성 요소가 정확히 하나
있어야 하는 dcc_data.manifest.build_manifest는 여기서 쓰지 않는다. 아래의 pair_by_stem_strict를 보라. 이는
dcc_data.mission2_features.pair_by_stem에 제출에 필요한 검사를 더한 것이다).

WAV에 라벨 JSON이 없거나(또는 그 반대), stem이 두 번 나오거나, 완성된 출력이 라벨이 있는 발화마다 한 행이
아니면 CSV를 쓰지 않고 실패한다. 일부만 쓰인 CSV가 성공한 실행처럼 보이면 안 되기 때문이다.

행 순서는 각 라벨 파일 안의 발화 순서를 따르고, startAt/endAt은 대회의 출력 형식
([audio file name], [startAt], [endAt], [speaker])에 따라 수정하지 않고 그대로 옮긴다.

특정 발화의 특징 추출이나 예측이 실패해도 그 행은 그대로 내보내서(다수 클래스로 대체), 출력 행 수가 항상
입력 발화 수와 같게 한다. 행을 조용히 빼지 않고, 실패는 stem/utt_idx와 함께 stderr에 기록한다.

이 저장소가 만든 두 모델을 모두 지원하며, ``--ckpt_path``의 파일 확장자로 고른다. 별도 옵션이 필요 없어서
위의 실행 명령 형식은 어느 쪽이든 그대로이다.
- L1(음향 통계 특징 위의 GBM): torch가 아닌 확장자, 예: ``.pkl``(``dcc_data.mission2_train.save_model``의
  pickle 형식).
- L2(로그 멜 스펙트로그램 위의 CNN, ``dcc_data.mission2_cnn.SpeakerCNNLarge``): ``.pt``/``.pth``/``.ckpt``이며
  ``scripts/train_mission2_cnn.py``의 ``--model large`` 체크포인트 형식과 같다. 스펙트로그램은 여기서
  ``dcc_data.mission2_spectrogram``의 현재 모듈 기본값(``N_MELS``, ``mission2_features.HOP_LENGTH``)으로
  만든다. 이 값은 체크포인트를 실제로 학습한 값과 같아야 한다. 어긋나도 오류가 나지 않고(``SpeakerCNNLarge``는
  ``AdaptiveAvgPool2d`` 덕분에 어떤 크기의 스펙트로그램이든 받는다) 예측만 조용히 나빠지기 때문이다. 그래서
  체크포인트에 기록된 학습 설정(학습기가 쓴 ``checkpoint["metadata"]``, ``dcc_data.mission2_checkpoint``)을
  먼저 이 값들과 대조하고, 모델 종류·멜 설정·학습 split이 다르거나, 오디오의 샘플레이트가 학습한 것과
  다르거나, 체크포인트가 제공된 화자 라벨을 그대로 쓰고 학습한 것이 아니면(주최 측의 2026-10-01 판정:
  ``text``를 근거로 한 라벨 수정 금지) 실행에 실패한다. 설정이 기록되어 있지 않은 체크포인트도 거부한다.
  예전의 ``m2_cnn_large.pt``(M2-VAL-003)는 수정한 라벨로 학습했으므로 더는 받아들이지 않는다.

L2 ``--ckpt_path``는 ``scripts/pack_mission2_l2_ensemble.py``가 만든 *앙상블* 체크포인트일 수도 있다
(``--torch-seed``만 다른, 구조와 멜 설정이 같은 모델 여러 개를 소프트 보팅. M2-ENS-003 참고: 공식
Validation 전체에서 가장 좋은 단일 시드보다 +0.50%p). 그래도 ``--ckpt_path`` 파일은 정확히 하나이므로
실행 명령 형식은 그대로이다. 앙상블 멤버의 가중치는 그 파일 하나 안에 ``"best_model_state_dict"`` 하나
대신 ``checkpoint["ensemble_model_state_dicts"]`` 아래에 묶여 있고, 읽는 쪽이 확인하는 키는
``dcc_data.mission2_checkpoint.ENSEMBLE_STATE_DICTS_KEY``이다.

개발 메모: 이것이 최상위 inference.py가 아니라 scripts/ 아래에 있는 이유는, 이 저장소가 세 미션을 모두
담고 있고 최종 제출 패키징(미션마다 inference.py 하나, 각 제출이 요구하는 경로에)이 아직 정해지지
않았기 때문이다. dcc_data.mission2_train의 체크포인트 형식 메모를 보라.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.mission2_features import (  # noqa: E402
    FEATURE_COLUMNS,
    N_FFT,
    read_call_waveform,
    slice_and_pad,
    utterance_feature_vector,
    utterance_frame_range,
)
from dcc_data.mission2_checkpoint import (  # noqa: E402
    CheckpointConfigError,
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_train import load_model  # noqa: E402

# torch.save의 기본 형식(zip 압축 파일)이다. 이 세 가지는 이 저장소의 CNN 스크립트
# (train_mission2_cnn.py, evaluate_mission2_l2_on_
# official_validation.py)가 L2 체크포인트에 이미 쓰는 확장자이고, 그 외는 모두
# dcc_data.mission2_train.load_model로 읽는 L1의 pickle 형식으로 취급한다.
CNN_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

PredictFn = Callable[[np.ndarray, int, float], int]

# CNN은 각 배치에서 가장 긴 발화에 맞춰 채운(SPEC_FLOOR_DB로, mission2_cnn.collate_batch 참고) 배치로
# 학습했으므로 채우지 않은 짧은 발화는 거의 본 적이 없다. 추론 때 그런 발화를 넣으면
# 약 1.1%p가 떨어진다(M2-VAL-004). 모든 발화를 고정 길이까지 채우면 -- 그 발화 자신의
# 스펙트로그램만 쓰고 다른 발화는 쓰지 않는다 -- Training 내부에서 떼어 둔 통화에서 그 손실이 회복된다
# (300프레임에서 89.58% -> 90.84%, 200~500은 평평한 구간). 300프레임은 이 저장소의 8 kHz / hop 160에서 6초이다.
L2_MIN_FRAMES = 300


def pad_spectrogram_to_min_frames(spectrogram: np.ndarray, min_frames: int = L2_MIN_FRAMES) -> np.ndarray:
    from dcc_data.mission2_spectrogram import SPEC_FLOOR_DB

    n_mels, n_frames = spectrogram.shape
    if n_frames >= min_frames:
        return spectrogram
    padding = np.full((n_mels, min_frames - n_frames), SPEC_FLOOR_DB, dtype=spectrogram.dtype)
    return np.concatenate([spectrogram, padding], axis=1)


def _speaker_from_logits(logits) -> int:
    """발화 하나의 로짓 두 개에 대한 대회의 판정 규칙: P(speaker=1) >= 0.5일 때만 클래스 1.

        로짓이 둘이면 ``logit1 >= logit0``과 같다. ``argmax``는 값이 정확히 같은 경우에만 다르고 그때 클래스 0을
        돌려준다. 그래서 ``logits.argmax`` 대신 이것을 쓴다.
    """

    return int((logits[0, 1] >= logits[0, 0]).item())


def _index_by_stem(root: Path, pattern: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob(pattern)):
        found.setdefault(path.stem, []).append(path)
    return found


def _preview(items, limit: int = 5) -> str:
    items = list(items)
    shown = ", ".join(str(item) for item in items[:limit])
    return f"{shown}, ... (+{len(items) - limit} more)" if len(items) > limit else shown


def pair_by_stem_strict(audio_dir: Path, label_dir: Path) -> list[tuple[str, Path, Path]]:
    """dcc_data.mission2_features.pair_by_stem이 돌려주는 것, 또는 모든 문제를 이름으로 밝힌 ValueError.

        pair_by_stem은 두 폴더 *모두에* 있는 stem만 남기고 반복된 stem이 다른 것을 조용히 덮어쓰게 둔다.
        학습 데이터를 만들 때는 괜찮지만 제출에는 안 된다: 라벨 JSON이 없는 WAV(또는 그 반대)가 CSV에서 그냥
        사라지고 실행은 여전히 0으로 끝나기 때문이다.
    """

    wavs = _index_by_stem(audio_dir, "*.wav")
    jsons = _index_by_stem(label_dir, "*.json")

    problems = []
    if not wavs and not jsons:
        problems.append(f"no .wav files under {audio_dir} and no .json files under {label_dir}")
    duplicated_wavs = sorted(stem for stem, paths in wavs.items() if len(paths) > 1)
    if duplicated_wavs:
        problems.append(f"{len(duplicated_wavs)} WAV stem(s) appear more than once: {_preview(duplicated_wavs)}")
    duplicated_jsons = sorted(stem for stem, paths in jsons.items() if len(paths) > 1)
    if duplicated_jsons:
        problems.append(f"{len(duplicated_jsons)} label JSON stem(s) appear more than once: {_preview(duplicated_jsons)}")
    wav_only = sorted(set(wavs) - set(jsons))
    if wav_only:
        problems.append(f"{len(wav_only)} WAV stem(s) have no label JSON: {_preview(wav_only)}")
    json_only = sorted(set(jsons) - set(wavs))
    if json_only:
        problems.append(f"{len(json_only)} label JSON stem(s) have no WAV: {_preview(json_only)}")
    if problems:
        raise ValueError(
            f"audio/label inputs do not line up ({len(wavs)} WAV stems, {len(jsons)} label stems) -- "
            "refusing to write a partial CSV:\n  - " + "\n  - ".join(problems)
        )
    return [(stem, wavs[stem][0], jsons[stem][0]) for stem in sorted(wavs)]


def count_label_utterances(json_paths) -> int:
    total = 0
    for json_path in json_paths:
        with Path(json_path).open("r", encoding="utf-8") as stream:
            total += len(json.load(stream).get("utterances", []))
    return total


def verify_output_complete(*, expected_calls: int, calls_seen: int, expected_rows: int, rows: list) -> None:
    if calls_seen != expected_calls or len(rows) != expected_rows:
        raise ValueError(
            f"output is incomplete: processed {calls_seen} of {expected_calls} calls and produced "
            f"{len(rows)} rows for {expected_rows} labelled utterances -- refusing to write a partial CSV"
        )


def _load_l1_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    model, meta = load_model(ckpt_path)
    feature_columns = tuple(meta["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match the current feature extractor "
            "(dcc_data.mission2_features.FEATURE_COLUMNS) -- retrain or update the checkpoint"
        )
    fallback_speaker = int(meta.get("majority_class", 1))

    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        vector = utterance_feature_vector(segment, sample_rate, duration_sec)
        return int(model.predict(vector.reshape(1, -1).astype(np.float64))[0])

    return predict, fallback_speaker


def _load_l2_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    import torch
    import torch.nn.functional as torch_functional

    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import mel_spectrogram

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # SpeakerCNNLarge 모양의 state_dict는 무엇으로 학습했든 오류 없이 불러와지므로, 믿기 전에
    # 체크포인트가 무엇으로 학습했다고 기록했는지 확인한다(dcc_data.mission2_checkpoint 참고).
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    expected_sample_rate = int(metadata["sample_rate"])

    # 앙상블 체크포인트(scripts/pack_mission2_l2_ensemble.py)는 여러 멤버의 가중치를
    # "best_model_state_dict" 하나 대신 한 키 아래에 묶는다. 그 외의 모든 것(--ckpt_path 파일 하나,
    # 같은 멜 설정)은 그대로이므로 여기가 유일한 갈림길이다.
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    # L2 체크포인트에는 저장된 majority_class가 없다(train_mission2_cnn.py는
    # 그것을 계산하지 않는다). 1은 위에 있는 load_model 자신의 "대체의 대체"
    # 기본값과 같으므로, 더 나은 값을 모를 때 두 백엔드가
    # 일치한다. 통화의 오디오조차 읽지 못했을 때만 쓰인다.
    fallback_speaker = 1

    @torch.no_grad()
    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        if sample_rate != expected_sample_rate:
            raise CheckpointConfigError(
                f"audio is {sample_rate} Hz but this checkpoint was trained on {expected_sample_rate} Hz audio"
            )
        spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
        batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # (배치, 채널, n_mels, n_frames)
        if len(models) == 1:
            # 항상 이랬던 단일 모델 경로와 비트 단위로 같다. 로짓 대신 softmax 출력을 비교해도
            # 수학적으로는 같지만(softmax는 순서를 보존한다), 평균낼 것이 없을 때
            # 이미 시험한 동점 처리 동작을 건드릴 이유가 없다.
            return _speaker_from_logits(models[0](batch))
        average_probs = sum(torch_functional.softmax(model(batch), dim=1) for model in models) / len(models)
        return int((average_probs[0, 1] >= average_probs[0, 0]).item())

    return predict, fallback_speaker


def load_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    """``ckpt_path``가 어떤 모델이든 그에 맞는 (predict_fn, fallback_speaker)를 돌려준다."""

    if ckpt_path.suffix.lower() in CNN_CHECKPOINT_SUFFIXES:
        return _load_l2_predictor(ckpt_path)
    return _load_l1_predictor(ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict, fallback_speaker = load_predictor(args.ckpt_path)

    pairs = pair_by_stem_strict(args.audio_dir, args.label_dir)
    expected_rows = count_label_utterances(json_path for _, _, json_path in pairs)

    rows: list[dict] = []
    failure_count = 0
    calls_seen = 0

    for stem, wav_path, json_path in pairs:
        calls_seen += 1
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        utterances = document.get("utterances", [])

        waveform = None
        sample_rate = None
        try:
            waveform, sample_rate = read_call_waveform(wav_path)
        except Exception as exc:  # noqa: BLE001 - 통화 하나의 오류가 전체 실행을 멈추게 해서는 안 된다
            print(f"[warn] failed to read {wav_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

        for utt_idx, utterance in enumerate(utterances):
            start_at = utterance.get("startAt")
            end_at = utterance.get("endAt")
            speaker = fallback_speaker

            if waveform is not None:
                try:
                    start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
                    segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
                    duration_sec = (float(end_at) - float(start_at)) / 1000.0
                    speaker = predict(segment, sample_rate, duration_sec)
                except CheckpointConfigError:
                    raise  # 체크포인트와 입력이 맞지 않는 것은 설정 오류이지, 덮어 가릴 발화 하나의 문제가 아니다
                except Exception as exc:  # noqa: BLE001 - 다수 클래스로 대체하고, 기록하고, 행은 유지한다
                    failure_count += 1
                    print(
                        f"[warn] feature/predict failed for stem={stem} utt_idx={utt_idx}: "
                        f"{type(exc).__name__}: {exc} -- falling back to majority class",
                        file=sys.stderr,
                    )
            else:
                failure_count += 1

            rows.append(
                {
                    "audio_file_name": wav_path.name,
                    "startAt": start_at,
                    "endAt": end_at,
                    "speaker": speaker,
                }
            )

    verify_output_complete(expected_calls=len(pairs), calls_seen=calls_seen, expected_rows=expected_rows, rows=rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_file_name", "startAt", "endAt", "speaker"])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, args.output)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    print(f"Wrote {len(rows)} rows to {args.output} ({failure_count} fell back to majority class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
