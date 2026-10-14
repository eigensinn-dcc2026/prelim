"""Mission 1 모델 코드 (팀 eigensinn): WAV와 JSON으로 신고자 성별을 예측한다.

통화마다 ``startAt`` / ``endAt`` / ``speaker``만 사용해 신고자(speaker=1)의 음성을 잘라 이어 붙인 뒤
564차원 특징을 만든다.

* 52차원 음향 벡터(MFCC·F0·에너지 통계): 신고자 음성 전체에서 계산한다.
* 512차원 x-vector(``microsoft/wavlm-base-plus-sv``): 신고자 음성을 겹치지 않는 20초 창으로 나눈다
  (최대 4개, 뒤쪽 창은 3초 이상일 때만 사용). 창마다 임베딩을 L2 정규화하고, 창 길이로 가중 평균한 뒤
  다시 L2 정규화한다.

학습이 끝난 헤드(표준화 후 로지스틱 회귀 점수, 일반 텐서로 저장)가 564차원 벡터를 F/M으로 분류한다.
추론 중에는 학습을 하지 않는다.

모델에 필요한 모든 것이 ``.pt`` 체크포인트 하나(형식 ``dcc-m1-v1``, ``train/build_model_pt.py``로 생성)에
들어 있다: 헤드와 WavLM 가중치, 그리고 WavLM 설정 파일 내용. 내려받는 것이 없어 인터넷 없이 실행된다.

한 통화에서 오류가 나도 그 행에는 학습 데이터의 다수 클래스 ``F``를 채워 행 수가 입력 쌍의 수와 같게 하고,
오류 내용은 stderr에 출력한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audio_features import _crop_caller_segments, _parse_caller_intervals, acoustic_feature_vector, pair_by_stem

COLUMNS = ("audio file name", "gender")

N_ACOUSTIC_FEATURES = 52
N_XVECTOR = 512
N_FEATURES = N_ACOUSTIC_FEATURES + N_XVECTOR
LABEL_TO_GENDER = {0: "F", 1: "M"}
FALLBACK_GENDER = "F"  # 학습 데이터의 다수 클래스

FORMAT = "dcc-m1-v1"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 20
MAX_WINDOWS = 4
TAIL_SECONDS = 3
MIN_SAMPLES = 8000


def window_ranges(length: int, sample_rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """첫 번째 창은 항상 포함하고, 뒤쪽 창은 길이가 3초 이상일 때만 포함한다."""
    width = WINDOW_SECONDS * sample_rate
    result = [(0, min(length, width))]
    for start in range(width, min(length, MAX_WINDOWS * width), width):
        stop = min(length, start + width)
        if stop - start >= TAIL_SECONDS * sample_rate:
            result.append((start, stop))
    return result


def average_windows(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """단위 길이 x-vector들을 창 길이로 가중 평균한 뒤 다시 정규화한다(학습 때와 같은 방식)."""
    weights = np.asarray(lengths, dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    mean = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid averaged x-vector")
    return (mean / norm).astype(np.float32)


def load_model(ckpt_path: Path):
    """``.pt`` 체크포인트 하나를 읽어 ``(헤드, 특징 추출기, WavLM x-vector 모델, torch)``를 돌려준다.

    WavLM 모델은 파일에 저장된 설정 텍스트로 다시 만들고 가중치는 엄격하게(strict) 불러온다.
    그래서 이 코드와 맞지 않는 파일은 실행하지 않고 거부한다.
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMForXVector

    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{ckpt_path} is not a {FORMAT} checkpoint")
    wavlm = payload["wavlm"]
    if (wavlm["repo_id"], wavlm["revision"]) != (WAVLM_ID, WAVLM_REVISION):
        raise ValueError(f"unexpected x-vector encoder: {wavlm['repo_id']}@{wavlm['revision']}")
    head = {name: tensor.numpy() for name, tensor in payload["head"].items()}
    if head["scaler_mean"].shape != (N_FEATURES,) or head["coef"].shape != (1, N_FEATURES):
        raise ValueError(f"head is not the {N_FEATURES}-feature binary F/M model")
    processor = Wav2Vec2FeatureExtractor.from_dict(json.loads(wavlm["preprocessor_config"]))
    model = WavLMForXVector(WavLMConfig.from_dict(json.loads(wavlm["config"])))
    model.load_state_dict(wavlm["state_dict"], strict=True)
    model.eval()
    if model.config.xvector_output_dim != N_XVECTOR:
        raise ValueError("expected a 512-dimensional x-vector")
    return head, processor, model, torch


def caller_waveform(wav_path: Path, json_path: Path) -> tuple[np.ndarray, int]:
    """startAt/endAt/speaker만으로 만든, 이어 붙인 신고자 파형 (T,)."""
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    segments = _crop_caller_segments(wav_path, _parse_caller_intervals(document, json_path))
    if not segments:
        raise ValueError(f"no caller segments: {json_path}")
    rates = {segment.sample_rate for segment in segments}
    if len(rates) != 1:
        raise ValueError(f"caller segments have mixed sample rates: {sorted(rates)}")
    waveform = np.concatenate([segment.waveform for segment in segments], axis=1)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"expected mono caller waveform (1, T), got {waveform.shape}")
    return waveform[0], segments[0].sample_rate


def extract_features(wav_path: Path, json_path: Path, processor, model, torch) -> np.ndarray:
    """564차원 한 행: 52차원 음향 벡터 + (20초 창 최대 4개의) 길이 가중 평균 x-vector."""
    wave, sr = caller_waveform(wav_path, json_path)
    acoustic = acoustic_feature_vector(wave, sr)
    if acoustic.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(f"acoustic feature vector has {acoustic.shape[0]} dims")
    vectors, lengths = [], []
    for left, right in window_ranges(wave.shape[0], sr):
        chunk = wave[left:right]
        if sr != SAMPLE_RATE:
            import librosa

            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=SAMPLE_RATE)
        chunk = np.asarray(chunk[: WINDOW_SECONDS * SAMPLE_RATE], dtype=np.float32)
        chunk = np.pad(chunk, (0, max(0, MIN_SAMPLES - len(chunk))))
        inputs = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            embedding = model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        vectors.append(embedding[0].numpy())
        lengths.append((right - left) / sr)
    mean = average_windows(np.stack(vectors), np.asarray(lengths))
    return np.concatenate([acoustic.astype(np.float64), mean.astype(np.float64)])


def classify(head: dict, features: np.ndarray) -> int:
    """표준화한 564차원 벡터의 로지스틱 점수가 양수이면 1(M), 아니면 0(F)을 돌려준다."""
    standardised = (features - head["scaler_mean"]) / head["scaler_scale"]
    score = float(standardised @ head["coef"][0] + head["intercept"][0])
    return int(score > 0)


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    """공식 Mission 1 출력 행: ``audio file name``, ``gender``(F/M)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    pairs = pair_by_stem(Path(audio_dir), Path(label_dir))
    if not pairs:
        raise FileNotFoundError(f"no WAV/JSON stem pairs under {audio_dir} and {label_dir}")
    head, processor, model, torch = load_model(ckpt_path)

    rows, n_fallback = [], 0
    for stem, wav_path, json_path in pairs:
        try:
            x = extract_features(wav_path, json_path, processor, model, torch)
            gender = LABEL_TO_GENDER[classify(head, x)]
        except Exception as exc:  # noqa: BLE001 - 행 수를 유지한다
            gender = FALLBACK_GENDER
            n_fallback += 1
            print(
                f"[mission1] fallback majority={gender} stem={stem} "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        rows.append((wav_path.name, gender))
        if len(rows) % 100 == 0:
            print(f"[mission1] {len(rows)}/{len(pairs)} done", file=sys.stderr, flush=True)
    print(
        f"[mission1] pairs={len(pairs)} ok={len(pairs) - n_fallback} fallback={n_fallback}",
        file=sys.stderr,
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))
