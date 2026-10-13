#!/usr/bin/env python3
"""미션 2 CNN 단독 추론 (대회 채점 명령과 같은 형식).

    python scripts/mission2_inference.py \\
        --audio_dir {wav folder} --label_dir {json folder} \\
        --ckpt_path {CNN 체크포인트 .pt} --output ./outputs/mission2.csv

이 파일의 쓰임은 두 가지다.
1. 최종 추론(scripts/mission2_final_inference.py)이 여기의 도우미 함수(라벨 짝짓기, 출력 검사, 스펙트로그램 채우기)를 그대로 가져다 쓴다.
2. CNN 앙상블만 따로 평가할 때 쓴다. CNN 학습 노트북(train_mission2_l2_from_raw.ipynb)의 마지막 평가 단계가 당시
   ``inference.py`` 로 실행한 것이 바로 이 코드다(지금의 ``inference.py`` 는 최종 묶음 체크포인트용이라 CNN 체크포인트를 받지 않는다).

채점 폴더에는 Training/Validation 하위 폴더가 있다는 보장이 없어서, audio_dir/label_dir 를 폴더 구조 가정 없이 파일 이름으로
짝짓는다(``pair_by_stem_strict``). WAV 에 짝 JSON 이 없거나(반대도), 같은 이름이 두 번 나오거나, 다 끝난 출력의 행 수가 라벨 발화 수와
다르면 CSV 를 쓰지 않고 멈춘다. 일부만 담긴 CSV 가 성공한 결과처럼 보이면 안 되기 때문이다.

행 순서는 각 라벨 파일의 발화 순서 그대로이고, startAt/endAt 은 고치지 않고 그대로 옮겨 적는다(대회 출력 형식:
[audio file name], [startAt], [endAt], [speaker]).

어떤 발화의 처리에 실패해도 그 행은 빼지 않고 다수 클래스로 채워 넣는다(출력 행 수 = 입력 발화 수). 실패는 stem/utt_idx 와 함께
stderr 에 남긴다.

CNN(``dcc_data.mission2_cnn.SpeakerCNNLarge``, log-mel 스펙트로그램 입력)의 스펙트로그램은 ``dcc_data.mission2_spectrogram`` 의 설정
(``N_MELS``, ``mission2_features.HOP_LENGTH``)으로 만든다. 이 값이 학습 때와 다르면 오류 없이 예측만 조용히 나빠지므로
(``SpeakerCNNLarge`` 는 ``AdaptiveAvgPool2d`` 라 어떤 크기든 받는다), 체크포인트에 기록된 학습 설정(``checkpoint["metadata"]``)과 먼저
대조한다. 모델 종류, mel 설정, 학습 분할, 음성 샘플링 주파수가 다르거나, 제공 라벨을 그대로 쓰지 않고 학습한 체크포인트면 멈춘다
(운영 측 2026-10-01 판단: ``text`` 근거의 라벨 보정 금지). 학습 설정이 기록되지 않은 체크포인트도 받지 않는다.

``--ckpt_path`` 는 여러 시드의 CNN 을 하나로 묶은 *앙상블* 체크포인트(scripts/pack_mission2_l2_ensemble.py)일 수 있다.
그때는 멤버들의 확률을 평균(soft voting)한다. 파일은 여전히 하나라 채점 명령 형식은 그대로다.

(제출 폴더 정리 메모: 초기 L1 모델(LightGBM, .pkl) 분기는 최종 모델에 쓰이지 않아 뺐다. 이제 .pt 같은 CNN 체크포인트만 받는다.)
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
    N_FFT,
    read_call_waveform,
    slice_and_pad,
    utterance_frame_range,
)
from dcc_data.mission2_checkpoint import (  # noqa: E402
    CheckpointConfigError,
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)

# torch.save 의 기본 형식(zip). 이 저장소의 CNN 스크립트가 쓰는 확장자들이다.
CNN_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

PredictFn = Callable[[np.ndarray, int, float], int]

# CNN 은 학습 때 배치 안에서 가장 긴 발화에 맞춰 바닥값(SPEC_FLOOR_DB)으로 채운 입력만 봤다(mission2_cnn.collate_batch).
# 그래서 짧은 발화를 채우지 않고 넣으면 정확도가 약 1.1%p 떨어졌다(M2-VAL-004). 모든 발화를 고정 길이까지
# 그 발화 자신의 스펙트로그램에만 바닥값을 덧붙여 채우면(다른 발화는 전혀 쓰지 않음) 떼어 둔 Training 통화에서
# 89.58% -> 90.84% 로 회복된다(300 기준, 200~500 은 거의 같음). 300프레임 = 8 kHz / hop 160 에서 6초.
L2_MIN_FRAMES = 300


def pad_spectrogram_to_min_frames(spectrogram: np.ndarray, min_frames: int = L2_MIN_FRAMES) -> np.ndarray:
    """스펙트로그램이 ``min_frames`` 보다 짧으면 뒤를 바닥값(무음)으로 채운다. 길면 그대로 둔다."""

    from dcc_data.mission2_spectrogram import SPEC_FLOOR_DB

    n_mels, n_frames = spectrogram.shape
    if n_frames >= min_frames:
        return spectrogram
    padding = np.full((n_mels, min_frames - n_frames), SPEC_FLOOR_DB, dtype=spectrogram.dtype)
    return np.concatenate([spectrogram, padding], axis=1)


def _speaker_from_logits(logits) -> int:
    """대회 판정 규칙: P(speaker=1) >= 0.5 이면 1.

    두 클래스 logit 으로는 ``logit1 >= logit0`` 과 같다. ``argmax`` 는 정확히 같을 때 0 을 돌려주므로 이 함수를 쓴다.
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
    """WAV 와 라벨 JSON 을 파일 이름(확장자 제외)으로 짝지은 목록. 문제가 하나라도 있으면 전부 적어서 ValueError 로 멈춘다.

    양쪽에 다 있는 것만 남기고 넘어가면, 라벨 없는 WAV(또는 반대)가 CSV 에서 조용히 빠진 채 정상 종료된다.
    제출물에서는 그러면 안 되므로 짝이 안 맞거나 이름이 겹치면 실행을 멈춘다. 결과는 이름순으로 정렬된다.
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
    """라벨 JSON 들의 발화 수 합계 (출력 행 수가 이것과 같아야 한다)."""

    total = 0
    for json_path in json_paths:
        with Path(json_path).open("r", encoding="utf-8") as stream:
            total += len(json.load(stream).get("utterances", []))
    return total


def verify_output_complete(*, expected_calls: int, calls_seen: int, expected_rows: int, rows: list) -> None:
    """처리한 통화 수와 출력 행 수가 기대와 다르면 멈춘다(일부만 담긴 CSV 를 쓰지 않기 위해)."""

    if calls_seen != expected_calls or len(rows) != expected_rows:
        raise ValueError(
            f"output is incomplete: processed {calls_seen} of {expected_calls} calls and produced "
            f"{len(rows)} rows for {expected_rows} labelled utterances -- refusing to write a partial CSV"
        )


def _load_l2_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    """CNN(단일 또는 앙상블) 체크포인트를 불러 발화 하나를 판정하는 함수를 만든다."""

    import torch
    import torch.nn.functional as torch_functional

    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import mel_spectrogram

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # SpeakerCNNLarge 모양의 가중치는 무엇으로 학습했든 오류 없이 불러와지므로, 믿기 전에 체크포인트에 기록된
    # 학습 설정부터 확인한다(dcc_data.mission2_checkpoint).
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    expected_sample_rate = int(metadata["sample_rate"])

    # 앙상블 체크포인트는 멤버 가중치를 "best_model_state_dict" 하나 대신 여러 개 담고 있다. 그 외(파일 하나, 같은 mel 설정)는
    # 같으므로 갈라지는 곳은 여기뿐이다.
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    # CNN 체크포인트에는 다수 클래스가 기록돼 있지 않아 1 을 쓴다. 통화 음성을 아예 읽지 못했을 때만 쓰인다.
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
            # 모델이 하나면 예전과 똑같이 logit 을 바로 비교한다(softmax 를 거쳐도 결과는 같지만, 검증된 동점 처리를 건드리지 않음).
            return _speaker_from_logits(models[0](batch))
        # 앙상블: 멤버들의 softmax 확률 평균에서 P(1) >= P(0) 이면 1 (= 평균 P(1) >= 0.5).
        average_probs = sum(torch_functional.softmax(model(batch), dim=1) for model in models) / len(models)
        return int((average_probs[0, 1] >= average_probs[0, 0]).item())

    return predict, fallback_speaker


def load_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    """(발화 하나를 판정하는 함수, 실패 시 대체 화자)를 돌려준다. CNN 체크포인트(.pt/.pth/.ckpt)만 받는다."""

    if ckpt_path.suffix.lower() not in CNN_CHECKPOINT_SUFFIXES:
        raise ValueError(f"{ckpt_path}: expected a CNN checkpoint ({', '.join(sorted(CNN_CHECKPOINT_SUFFIXES))})")
    return _load_l2_predictor(ckpt_path)


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
        except Exception as exc:  # noqa: BLE001 - 통화 하나가 잘못돼도 전체 실행은 멈추지 않는다
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
                    raise  # 체크포인트와 입력이 안 맞는 것은 설정 오류라 발화 하나의 실패로 덮지 않는다
                except Exception as exc:  # noqa: BLE001 - 다수 클래스로 채우고 기록한 뒤 행은 남긴다
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

    # 임시 파일에 다 쓴 뒤 이름을 바꿔서, 중간에 멈춰도 덜 쓴 CSV 가 남지 않게 한다.
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
