#!/usr/bin/env python3
"""미션 2 최종 추론: CNN 3개 앙상블 + 미세조정한 WavLM-large 2개의 확률을 섞어 0.5 로 판정한다.

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \\
        --ckpt_path checkpoints/m2_final_ensemble.pt --output ./outputs/mission2.csv

[규정 준수] 발화 하나하나를 그 발화 자신의 소리(라벨 JSON 의 ``startAt``..``endAt`` 구간)만으로 판정한다.
``text``, 발화 순서, 같은 통화의 다른 발화는 쓰지 않는다. WavLM 입력은 그 발화의 3초 창이다.
발화끼리 섞이는 계산도 없다: CNN 은 발화를 하나씩 넣고(BatchNorm 은 학습 때 고정된 통계만 씀, eval 모드),
WavLM 은 발화마다 따로 정규화한다(layer normalization). 그래서 한 통화의 발화들을 WavLM 에 한 묶음으로 넣는 것은 속도를 위한
것일 뿐 예측을 바꾸지 않는다(18발화로 확인: 하나씩 넣을 때와 확률 최대 차이 9e-9).

[판정 규칙] p = w_cnn * P_cnn(신고자) + sum_k w_k * P_wavlm_k(신고자),  p >= 0.5 이면 신고자(1), 아니면 대원(0).
비율(CNN 0.4, WavLM 0.3 + 0.3)은 체크포인트 안에 들어 있고, 공식 Validation 이 아니라 떼어 둔 Training 발화로 골랐다.
GPU 가 있으면 GPU 를 쓴다(WavLM 은 float16 autocast, 정확도를 잰 설정과 같음). GPU 가 없으면 전부 CPU float32 로 돈다(매우 느림).

출력 행, 행 순서, "일부만 담긴 CSV 는 쓰지 않는다" 검사는 CNN 단독 추론(scripts/mission2_inference.py)과 같고,
그 파일의 도우미 함수를 그대로 가져다 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from dcc_data.mission2_checkpoint import (  # noqa: E402
    CheckpointConfigError,
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_features import N_FFT, read_call_waveform, slice_and_pad, utterance_frame_range  # noqa: E402
from mission2_inference import (  # noqa: E402
    count_label_utterances,
    pad_spectrogram_to_min_frames,
    pair_by_stem_strict,
    verify_output_complete,
)

FINAL_FORMAT = "dcc2026-mission2-final-v1"   # 최종 묶음 체크포인트(scripts/pack_mission2_final_ensemble.py)의 형식 이름
WAVLM_FORMAT = "dcc2026-mission2-wavlm-v1"   # WavLM 체크포인트(scripts/train_mission2_wavlm.py)의 형식 이름
THRESHOLD = 0.5                              # 대회 규정: 판정 기준 0.5 고정
FALLBACK_SPEAKER = 1  # 음성을 읽거나 자르지 못한 발화에만 쓰는 대체 답 (CNN 단독 추론과 같은 약속)
OFFICIAL_HEADER = ["audio file name", "startAt", "endAt", "speaker"]  # 출제문제 11쪽의 출력 컬럼 이름 그대로


class FinalEnsemble:
    """최종 체크포인트 안의 CNN 3개, WavLM 2개, 섞는 비율을 불러 둔다."""

    def __init__(self, ckpt_path: Path, device: str, batch_size: int) -> None:
        import torch

        from dcc_data.mission2_cnn import SpeakerCNNLarge
        from dcc_data.mission2_wavlm import WavLMRoleClassifier

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.amp = device == "cuda"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != FINAL_FORMAT:
            raise CheckpointConfigError(f"{ckpt_path} is not a {FINAL_FORMAT} checkpoint")
        # 섞는 비율과 판정 기준 확인: 기준은 반드시 0.5, 비율의 합은 1
        blend = checkpoint["blend"]
        if float(blend["threshold"]) != THRESHOLD:
            raise CheckpointConfigError(f"the decision threshold must be {THRESHOLD}, the checkpoint says {blend['threshold']}")
        self.w_cnn = float(blend["weights"]["cnn"])
        self.w_wavlm = [float(weight) for weight in blend["weights"]["wavlm"]]
        if len(self.w_wavlm) != len(checkpoint["wavlm"]) or abs(self.w_cnn + sum(self.w_wavlm) - 1.0) > 1e-6:
            raise CheckpointConfigError("blend weights do not match the stored models or do not sum to 1")

        # CNN 앙상블: CNN 단독 추론과 같은 검사(학습 설정, 제공 라벨 그대로 학습했는지)를 거친 뒤 불러온다
        cnn_checkpoint = checkpoint["cnn"]
        metadata, _ = resolve_l2_checkpoint_metadata(cnn_checkpoint, ckpt_path)
        require_l2_inference_config(metadata)
        self.cnn_sample_rate = int(metadata["sample_rate"])
        self.cnn_models = []
        for state_dict in l2_member_state_dicts(cnn_checkpoint, metadata):
            model = SpeakerCNNLarge()
            model.load_state_dict(state_dict)
            self.cnn_models.append(model.to(device).eval())

        # WavLM 2개: 모델 구조는 저장된 설정으로 만들고(아무것도 내려받지 않음), 가중치는 체크포인트에서 넣는다
        self.wavlm_models = []
        for member in checkpoint["wavlm"]:
            if member.get("format") != WAVLM_FORMAT:
                raise CheckpointConfigError("a stored WavLM member has the wrong format")
            model = WavLMRoleClassifier.from_config(member["backbone_config"])
            model.load_state_dict(member["state_dict"])
            self.wavlm_models.append(model.float().to(device).eval())
        del checkpoint

    def score_call(self, waveform: np.ndarray, sample_rate: int, utterances: list, stem: str):
        """한 통화의 발화마다 (판정, P_cnn, [P_wavlm_k])를 돌려준다. 처리하지 못한 발화는 확률이 None 이다."""

        from dcc_data.mission2_spectrogram import mel_spectrogram
        from dcc_data.mission2_wavlm import prepare_window

        torch = self.torch
        if sample_rate != self.cnn_sample_rate:
            raise CheckpointConfigError(f"audio is {sample_rate} Hz but the CNN was trained on {self.cnn_sample_rate} Hz audio")

        results = [(FALLBACK_SPEAKER, None, None)] * len(utterances)
        indices, spectrograms, windows = [], [], []
        # 1단계: 발화마다 그 발화 구간만 잘라 두 모델의 입력을 만든다
        for index, utterance in enumerate(utterances):
            try:
                start_frame, end_frame = utterance_frame_range(
                    utterance.get("startAt"), utterance.get("endAt"), sample_rate, waveform.shape[0]
                )
                # CNN: 구간을 FFT 창 하나 길이까지 0 으로 채운 것 (CNN 단독 제출본과 같음)
                # WavLM: 구간 그대로 (WavLM 의 학습·검증 때와 같음). prepare_window 가 16 kHz 변환과 3초 맞춤을 한다.
                cnn_segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
                spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(cnn_segment, sample_rate))
                window = prepare_window(np.asarray(waveform[start_frame:end_frame], dtype=np.float32), sample_rate)
            except Exception as exc:  # noqa: BLE001 - 행은 남기고(다수 클래스) 실패를 기록한다
                print(f"[warn] preprocessing failed for stem={stem} utt_idx={index}: {type(exc).__name__}: {exc} "
                      "-- falling back to the majority class", file=sys.stderr)
                continue
            indices.append(index)
            spectrograms.append(spectrogram)
            windows.append(window)
        if not indices:
            return results

        with torch.no_grad():
            # 2단계: CNN 3개의 P(신고자) 평균 (발화 하나씩)
            cnn_p1 = []
            for spectrogram in spectrograms:
                batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, n_mels, frames)
                probs = sum(torch.softmax(model(batch), dim=1) for model in self.cnn_models) / len(self.cnn_models)
                cnn_p1.append(float(probs[0, 1]))

            # 3단계: WavLM 마다 P(신고자) (속도를 위해 최대 batch_size 발화씩 묶음, 결과는 하나씩과 같음)
            stacked = np.stack(windows)
            wavlm_p1 = []
            for model in self.wavlm_models:
                parts = []
                for offset in range(0, len(stacked), self.batch_size):
                    chunk = torch.from_numpy(stacked[offset : offset + self.batch_size]).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                        logits = model(chunk).float()
                    parts.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float64))
                wavlm_p1.append(np.concatenate(parts))

        # 4단계: 비율대로 섞어 0.5 이상이면 신고자(1)
        for position, index in enumerate(indices):
            members = [float(p[position]) for p in wavlm_p1]
            score = self.w_cnn * cnn_p1[position] + sum(w * p for w, p in zip(self.w_wavlm, members))
            results[index] = (int(score >= THRESHOLD), cnn_p1[position], members)
        return results


def write_csv_atomically(path: Path, header: list, rows: list) -> None:
    """임시 파일에 다 쓴 뒤 이름을 바꾼다. 중간에 멈춰도 덜 쓴 CSV 가 남지 않는다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 대회 채점 명령의 필수 인자 4개
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # 선택 인자 (채점에는 필요 없음, 기본값으로 동작)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="default: the GPU if there is one")
    parser.add_argument("--batch_size", type=int, default=64, help="WavLM batch size within one call (speed only)")
    # 검증 전용: 모델별 확률 저장(연구용 계산과 발화별 대조), 앞쪽 N 통화만 실행(빠른 시험)
    parser.add_argument("--probs_output", type=Path, default=None, help="optional: also write each member's P(speaker=1) (checks only)")
    parser.add_argument("--max_calls", type=int, default=0, help="checks only: score just the first N calls (sorted by name)")
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    started = time.time()
    ensemble = FinalEnsemble(args.ckpt_path, device, args.batch_size)
    gpu = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"device={device}{gpu}, WavLM batch_size={args.batch_size}, CNN members={len(ensemble.cnn_models)}, "
          f"WavLM members={len(ensemble.wavlm_models)}, weights cnn={ensemble.w_cnn} wavlm={ensemble.w_wavlm}", flush=True)

    # WAV 와 라벨 JSON 을 파일 이름으로 짝짓는다 (짝이 안 맞으면 멈춤). 이름순으로 처리한다.
    pairs = pair_by_stem_strict(args.audio_dir, args.label_dir)
    if args.max_calls:
        pairs = pairs[: args.max_calls]
        print(f"[check run] only the first {len(pairs)} calls", flush=True)
    expected_rows = count_label_utterances(json_path for _, _, json_path in pairs)

    rows, prob_rows, failure_count, calls_seen = [], [], 0, 0
    infer_started = time.time()
    for stem, wav_path, json_path in pairs:
        calls_seen += 1
        # 라벨 JSON 에서는 발화 목록(startAt/endAt)만 쓴다. speaker 와 text 는 읽지 않는다.
        with json_path.open("r", encoding="utf-8") as stream:
            utterances = json.load(stream).get("utterances", [])
        try:
            waveform, sample_rate = read_call_waveform(wav_path)
        except Exception as exc:  # noqa: BLE001 - 통화 하나가 잘못돼도 전체 실행은 멈추지 않는다
            print(f"[warn] failed to read {wav_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            results = [(FALLBACK_SPEAKER, None, None)] * len(utterances)
        else:
            results = ensemble.score_call(waveform, sample_rate, utterances, stem)
        # 라벨 JSON 의 발화 순서대로 한 줄씩 (startAt/endAt 은 라벨 값 그대로)
        for utterance, (speaker, cnn_p1, wavlm_p1) in zip(utterances, results):
            if cnn_p1 is None:
                failure_count += 1
            rows.append([wav_path.name, utterance.get("startAt"), utterance.get("endAt"), speaker])
            if args.probs_output is not None:
                prob_rows.append([wav_path.name, utterance.get("startAt"), utterance.get("endAt"), speaker,
                                  "" if cnn_p1 is None else f"{cnn_p1:.8f}",
                                  *(["", ""] if wavlm_p1 is None else [f"{p:.8f}" for p in wavlm_p1])])
        if calls_seen % 200 == 0:
            print(f"  {calls_seen}/{len(pairs)} calls, {len(rows)} utterances, {(time.time() - infer_started) / 60:.1f} min", flush=True)

    # 처리한 통화 수와 행 수가 라벨과 맞는지 확인한 뒤에만 CSV 를 쓴다
    verify_output_complete(expected_calls=len(pairs), calls_seen=calls_seen, expected_rows=expected_rows, rows=rows)
    write_csv_atomically(args.output, OFFICIAL_HEADER, rows)
    if args.probs_output is not None:
        write_csv_atomically(args.probs_output, OFFICIAL_HEADER + ["p1_cnn", "p1_wavlm1", "p1_wavlm2"], prob_rows)
    elapsed = time.time() - infer_started
    print(f"Wrote {len(rows)} rows to {args.output} ({failure_count} fell back to the majority class); "
          f"inference {elapsed / 60:.1f} min, {elapsed / max(len(rows), 1) * 1000:.1f} ms per utterance "
          f"(+ {infer_started - started:.0f} s model loading)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
