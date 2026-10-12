#!/usr/bin/env python3
"""Mission 2 추론 진입점 (대회가 정한 CLI 규격).

    python scripts/mission2_inference.py \\
        --audio_dir {wav 폴더} --label_dir {json 폴더} \\
        --ckpt_path {체크포인트 파일} --output ./outputs/mission2.csv

audio_dir와 label_dir는 파일 이름 stem으로 짝짓는다. 채점기가 평평한 폴더를 주므로
Training/Validation 하위 폴더가 있다고 가정하지 않는다. (Training 또는 Validation 경로
구성요소가 정확히 하나 있어야 하는 dcc_data.manifest.build_manifest를 여기서 쓰지 않는
이유다. 아래 pair_by_stem_strict를 본다. 이것은 stem 짝짓기에 제출용으로 필요한 검사를
더한 것이다.)

WAV에 라벨 JSON이 없거나(또는 그 반대), 같은 stem이 두 번 나오거나, 완성된 출력의 행 수가
라벨이 달린 발화 수와 같지 않으면 CSV를 쓰지 않고 실패한다. 일부만 담긴 CSV가 성공한
실행처럼 보여서는 안 된다.

행 순서는 각 라벨 파일 안의 발화 순서를 따르고, startAt/endAt은 대회 출력 형식
([audio file name], [startAt], [endAt], [speaker])대로 고치지 않고 그대로 복사한다.

어떤 발화의 feature 추출이나 예측이 실패해도 그 행은 그대로 내보낸다(다수 클래스로 대체).
그래서 출력 행 수는 항상 입력 발화 수와 같다. 실패는 행을 조용히 빼 버리는 대신
stem/utt_idx와 함께 stderr에 기록한다.

이 저장소가 만든 두 모델을 모두 지원하며, ``--ckpt_path``의 파일 확장자로 고른다.
별도 플래그가 필요 없으므로 위의 CLI 규격은 어느 쪽이든 바뀌지 않는다.
- L1 (음향 통계 feature 위의 GBM): torch가 아닌 모든 확장자, 예: ``.pkl``
  (``dcc_data.mission2_train.save_model``의 pickle 형식).
- L2 (log-mel 스펙트로그램 위의 CNN, ``dcc_data.mission2_cnn.SpeakerCNNLarge``):
  ``.pt``/``.pth``/``.ckpt``. ``scripts/train_mission2_cnn.py``의 ``--model large``
  체크포인트 형식과 같다. 스펙트로그램은 여기서 ``dcc_data.mission2_spectrogram``의
  현재 모듈 수준 기본값(``N_MELS``, 그리고 ``mission2_features.HOP_LENGTH``)으로 만든다.
  이 값들은 체크포인트를 실제로 학습할 때 쓴 값과 같아야 한다. 다르더라도 오류가 나지
  않고(``SpeakerCNNLarge``는 ``AdaptiveAvgPool2d`` 덕분에 어떤 크기의 스펙트로그램도
  받는다) 예측 품질만 조용히 떨어지기 때문이다. 그래서 학습기가 기록한 체크포인트의
  학습 설정(``checkpoint["metadata"]``, ``dcc_data.mission2_checkpoint``)을 먼저 이 값들과
  대조해, 모델 종류, mel 설정, 학습 split이 다르거나, 오디오의 샘플링 레이트가 학습 때와
  다르거나, 체크포인트가 제공된 화자 라벨을 그대로 쓰고 학습한 것이 아니면 실행을 실패시킨다
  (주최 측의 2026-10-01 판정: ``text``를 보고 라벨을 고치는 것은 금지).
  기록된 설정이 없는 체크포인트도 거부한다. 이전의 ``m2_cnn_large.pt``(M2-VAL-003)는
  고친 라벨로 학습했으므로 이제는 일부러 받아들이지 않는다.

L2 ``--ckpt_path``는 ``scripts/pack_mission2_l2_ensemble.py``로 만든 *앙상블* 체크포인트일 수도
있다 (구조와 mel 설정이 같고 ``--torch-seed``만 다른 모델 여럿을 소프트 보팅으로 합친 것.
M2-ENS-003 참고: 공식 Validation 전체에서 가장 좋은 단일 시드보다 +0.50pp). 이때도
``--ckpt_path`` 파일은 정확히 하나라서 CLI 규격은 바뀌지 않는다. 앙상블 멤버의 가중치는
``"best_model_state_dict"`` 하나 대신 그 한 파일 안의 ``checkpoint["ensemble_model_state_dicts"]``
아래에 묶여 있고, 로더는 ``dcc_data.mission2_checkpoint.ENSEMBLE_STATE_DICTS_KEY``가 가리키는
키를 확인한다.
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

# torch.save의 기본 형식(zip 아카이브). 이 저장소의 CNN 스크립트(train_mission2_cnn.py,
# evaluate_mission2_l2_on_official_validation.py)가 L2 체크포인트에 이미 쓰는 확장자가 이 세 가지다.
# 그 외의 확장자는 dcc_data.mission2_train.load_model을 통해 L1의 pickle 형식으로 취급한다.
CNN_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

PredictFn = Callable[[np.ndarray, int, float], int]

# CNN은 각 배치에서 가장 긴 발화에 맞춰 (SPEC_FLOOR_DB로, mission2_cnn.collate_batch 참고) 채운
# 배치로 학습했다. 그래서 채우지 않은 짧은 발화는 거의 본 적이 없고, 추론 때 그런 발화를 그대로
# 넣으면 약 1.1pp 손해다 (M2-VAL-004). 모든 발화를 (다른 발화가 아닌 자기 자신의 스펙트로그램만
# 써서) 고정 길이까지 채우면, Training 내부에서 따로 떼어 둔 통화에서 이를 회복한다
# (300프레임에서 89.58% -> 90.84%, 200~500은 평탄한 구간). 300프레임 = 이 저장소의 8 kHz / hop 160
# 기준 6초.
L2_MIN_FRAMES = 300


def pad_spectrogram_to_min_frames(spectrogram: np.ndarray, min_frames: int = L2_MIN_FRAMES) -> np.ndarray:
    from dcc_data.mission2_spectrogram import SPEC_FLOOR_DB

    n_mels, n_frames = spectrogram.shape
    if n_frames >= min_frames:
        return spectrogram
    padding = np.full((n_mels, min_frames - n_frames), SPEC_FLOOR_DB, dtype=spectrogram.dtype)
    return np.concatenate([spectrogram, padding], axis=1)


def _speaker_from_logits(logits) -> int:
    """발화 하나의 logit 두 개에 대한 대회의 판정 규칙: P(speaker=1) >= 0.5 일 때만 클래스 1.

    logit이 두 개일 때 이는 ``logit1 >= logit0``과 같다. ``argmax``는 정확히 같은 값일 때만 다르게
    클래스 0을 돌려주므로, ``logits.argmax`` 대신 이 함수를 쓴다.
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
    """(stem, wav 경로, json 경로) 목록을 돌려준다. 문제가 있으면 모든 문제를 이름으로 나열한 ValueError를 낸다.

    *두* 폴더에 모두 있는 stem만 남기고 반복된 stem을 조용히 덮어쓰는 방식은 제출에는 안 된다.
    라벨 JSON이 없는 WAV(또는 그 반대)가 CSV에서 그냥 사라지는데도 실행은 종료 코드 0으로
    끝나기 때문이다.
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


def _load_prob_ensemble_predictor(checkpoint: dict) -> tuple[PredictFn, int]:
    """한 파일에 든 CNN + WavLM 멤버. 발화마다 그 발화 파형만 본다.

    반환 함수는 여전히 화자 정수다. ``--save_probs``용으로 ``member_probs``와 ``mean_prob``를 남긴다.
    판정 임계값은 0.5로 고정되어 있고 인자로 받지 않는다.
    """

    import torch

    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import ALLOWED_LABEL_SOURCES, mel_spectrogram
    from dcc_data.mission2_wavlm import (
        COMBINE_PROB_MEAN,
        COMBINE_WEIGHTED,
        FIXED_THRESHOLD,
        PROB_ENSEMBLE_FORMAT,
        combine_probabilities,
        decision_from_probability,
        load_wavlm_classifier,
        read_utterance_signal,
        reporter_probability,
    )

    if checkpoint.get("format") != PROB_ENSEMBLE_FORMAT:
        raise CheckpointConfigError(f"not a prob-ensemble checkpoint: {checkpoint.get('format')!r}")
    metadata = checkpoint.get("metadata") or {}
    if metadata.get("format") != PROB_ENSEMBLE_FORMAT:
        raise CheckpointConfigError("prob-ensemble checkpoint is missing metadata")
    combine = metadata.get("combine")
    if combine not in (COMBINE_PROB_MEAN, COMBINE_WEIGHTED):
        raise CheckpointConfigError(f"combine must be prob_mean or weighted_prob_mean, got {combine!r}")
    if float(metadata.get("threshold", -1)) != FIXED_THRESHOLD:
        raise CheckpointConfigError("the decision threshold is fixed at 0.5")
    label_source = metadata.get("label_source")
    if label_source not in ALLOWED_LABEL_SOURCES:
        raise CheckpointConfigError(
            f"label_source is {label_source!r}; members must use the provided speaker labels"
        )
    members = list(checkpoint.get("members") or [])
    recorded_members = list(metadata.get("members") or [])
    if not members or len(members) != len(recorded_members):
        raise CheckpointConfigError("prob-ensemble members do not match the recorded member list")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = []
    names: list[str] = []
    weights: list[float] = []
    for member, recorded in zip(members, recorded_members):
        if member.get("name") != recorded.get("name") or member.get("kind") != recorded.get("kind"):
            raise CheckpointConfigError("prob-ensemble member metadata does not match the stored weights")
        if member.get("sha256") != recorded.get("sha256"):
            raise CheckpointConfigError(f"member {member.get('name')!r} sha256 does not match metadata")
        if member.get("label_source") != label_source:
            raise CheckpointConfigError(f"member {member.get('name')!r} label_source disagrees with the checkpoint")
        names.append(str(member["name"]))
        weights.append(float(member.get("weight", 1.0)))
        kind = member["kind"]
        if kind == "cnn_l2":
            models = []
            for state_dict in member["state_dicts"]:
                model = SpeakerCNNLarge()
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models.append(model)
            if not models:
                raise CheckpointConfigError(f"CNN member {member['name']!r} has no weights")
            loaded.append(("cnn_l2", models, int(member["sample_rate"])))
        elif kind == "wavlm":
            model = load_wavlm_classifier(member["config"], member["state_dict"], device)
            loaded.append(("wavlm", model, int(member.get("sr", 16000))))
        else:
            raise CheckpointConfigError(f"unknown ensemble member kind {kind!r}")
    if not any(kind == "wavlm" for kind, _model, _sr in loaded):
        raise CheckpointConfigError("a prob-ensemble checkpoint needs at least one WavLM member")

    @torch.inference_mode()
    def predict(
        segment: np.ndarray,
        sample_rate: int,
        duration_sec: float,
        wav_path: Path | None = None,
        start_at=None,
        end_at=None,
    ) -> int:
        del duration_sec  # CNN 구간과 WavLM 구간 모두 그 발화 자체에서만 만든다
        probabilities: list[float] = []
        for kind, model, member_sr in loaded:
            if kind == "cnn_l2":
                if sample_rate != member_sr:
                    raise CheckpointConfigError(
                        f"audio is {sample_rate} Hz but this CNN member was trained on {member_sr} Hz audio"
                    )
                spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
                batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(device)
                member_probs = [torch.nn.functional.softmax(item(batch), dim=1)[0, 1] for item in model]
                probabilities.append(float((sum(member_probs) / len(member_probs)).item()))
            else:
                if wav_path is None or start_at is None or end_at is None:
                    raise CheckpointConfigError("WavLM members need the target utterance's own startAt/endAt")
                signal = read_utterance_signal(wav_path, start_at, end_at, target_sr=member_sr)
                probabilities.append(reporter_probability(model, signal, device))
        mean_probability = combine_probabilities(probabilities, weights, combine)
        predict.member_probs = probabilities
        predict.mean_prob = mean_probability
        return decision_from_probability(mean_probability)

    predict.needs_bounds = True
    predict.member_names = names
    predict.member_probs = None
    predict.mean_prob = None
    return predict, 1


def _load_l2_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    import torch
    import torch.nn.functional as torch_functional

    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import mel_spectrogram

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # CNN+WavLM을 한 파일로 묶은 체크포인트도 --ckpt_path 하나다. L2 CNN 전용 경로는 아니다.
    # 아래 CNN 단독 경로는 그대로다.
    if checkpoint.get("format") == "mission2_prob_ensemble_v1":
        return _load_prob_ensemble_predictor(checkpoint)
    # SpeakerCNNLarge 모양의 state_dict는 무엇으로 학습했든 오류 없이 불러와지므로, 믿기 전에
    # 체크포인트가 무엇으로 학습했다고 기록했는지 확인한다(dcc_data.mission2_checkpoint 참고).
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    expected_sample_rate = int(metadata["sample_rate"])

    # 앙상블 체크포인트(scripts/pack_mission2_l2_ensemble.py)는 "best_model_state_dict" 하나 대신
    # 한 키 아래에 여러 멤버의 가중치를 묶어 둔다. 그 외(--ckpt_path 파일 하나, 같은 mel 설정)는
    # 달라지지 않으므로, 갈라지는 지점은 여기뿐이다.
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    # L2 체크포인트에는 저장된 majority_class가 없다 (train_mission2_cnn.py가 계산하지 않는다).
    # 1은 위 load_model 자체의 최후 기본값과 같으므로, 더 나은 정보가 없을 때 두 백엔드가 일치한다.
    # 이 값은 통화의 오디오조차 읽지 못했을 때만 쓰인다.
    fallback_speaker = 1

    @torch.no_grad()
    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        if sample_rate != expected_sample_rate:
            raise CheckpointConfigError(
                f"audio is {sample_rate} Hz but this checkpoint was trained on {expected_sample_rate} Hz audio"
            )
        spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
        batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, n_frames)
        if len(models) == 1:
            # 항상 이래 왔던 단일 모델 경로와 비트 단위로 똑같다. raw logit 대신 softmax 출력을 비교해도
            # 수학적으로는 같지만(softmax는 순서를 보존한다), 평균 낼 것이 없을 때 이미 검증된
            # 동점 처리 방식을 건드릴 이유가 없다.
            return _speaker_from_logits(models[0](batch))
        average_probs = sum(torch_functional.softmax(model(batch), dim=1) for model in models) / len(models)
        return int((average_probs[0, 1] >= average_probs[0, 0]).item())

    return predict, fallback_speaker


def load_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    """``ckpt_path``가 어떤 모델이든 (predict_fn, fallback_speaker)를 돌려준다."""

    if ckpt_path.suffix.lower() in CNN_CHECKPOINT_SUFFIXES:
        return _load_l2_predictor(ckpt_path)
    return _load_l1_predictor(ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save_probs",
        type=Path,
        default=None,
        help="묶인 CNN+WavLM 체크포인트의 멤버별 P(신고자)를 따로 저장한다. 채점 CSV 형식은 바꾸지 않는다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict, fallback_speaker = load_predictor(args.ckpt_path)
    if args.save_probs is not None and not getattr(predict, "needs_bounds", False):
        raise SystemExit("--save_probs requires a packed CNN+WavLM checkpoint from scripts/pack_mission2_ensemble.py")

    pairs = pair_by_stem_strict(args.audio_dir, args.label_dir)
    expected_rows = count_label_utterances(json_path for _, _, json_path in pairs)

    rows: list[dict] = []
    prob_rows: list[dict] | None = [] if args.save_probs is not None else None
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
        except Exception as exc:  # noqa: BLE001 - 통화 하나가 잘못됐다고 전체 실행이 멈춰서는 안 된다
            print(f"[warn] failed to read {wav_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

        for utt_idx, utterance in enumerate(utterances):
            start_at = utterance.get("startAt")
            end_at = utterance.get("endAt")
            speaker = fallback_speaker
            if getattr(predict, "needs_bounds", False):
                predict.member_probs = None
                predict.mean_prob = None

            if waveform is not None:
                try:
                    start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
                    segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
                    duration_sec = (float(end_at) - float(start_at)) / 1000.0
                    if getattr(predict, "needs_bounds", False):
                        speaker = predict(
                            segment,
                            sample_rate,
                            duration_sec,
                            wav_path=wav_path,
                            start_at=start_at,
                            end_at=end_at,
                        )
                    else:
                        speaker = predict(segment, sample_rate, duration_sec)
                except CheckpointConfigError:
                    raise  # 체크포인트/입력 불일치는 설정 오류다. 발화 하나의 문제로 덮고 넘어갈 일이 아니다
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
            if prob_rows is not None:
                item = {"audio_file_name": wav_path.name, "startAt": start_at, "endAt": end_at}
                values = predict.member_probs
                if values is None:
                    for name in predict.member_names:
                        item[name] = ""
                    item["p_mean"] = ""
                else:
                    for name, value in zip(predict.member_names, values):
                        item[name] = value
                    item["p_mean"] = predict.mean_prob
                prob_rows.append(item)

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

    if prob_rows is not None:
        args.save_probs.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["audio_file_name", "startAt", "endAt", *predict.member_names, "p_mean"]
        fd, tmp_name = tempfile.mkstemp(prefix=f".{args.save_probs.name}.", suffix=".tmp", dir=args.save_probs.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(prob_rows)
            os.replace(tmp_name, args.save_probs)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        print(f"Wrote {len(prob_rows)} probability rows to {args.save_probs}")

    print(f"Wrote {len(rows)} rows to {args.output} ({failure_count} fell back to majority class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
