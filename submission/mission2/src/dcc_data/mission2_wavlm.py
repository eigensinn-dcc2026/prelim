"""미션 2 WavLM 모델: 공개 사전학습 음성 모델 WavLM-large 를 미세조정해 대원(0)/신고자(1)를 가린다.

log-mel CNN(dcc_data.mission2_cnn)은 Training 발화 약 89만 개로 처음부터 배운다. 큰 공개 음성 데이터로 미리 학습된 음성 모델은
이미 목소리의 특성을 담고 있어서, 적은 학습으로 더 높은 정확도에 닿는다. 대회 규정상 공개 오픈소스 사전학습 모델을 쓸 수 있고,
가중치는 우리 체크포인트 안에 (설정 + state_dict) 저장하므로 추론 때 인터넷이 필요 없다.

[규정 준수] 입력 규칙은 다른 미션 2 모델과 같다: 그 발화 자신의 소리(``startAt``..``endAt`` 구간)만 쓴다.
``prepare_window`` 가 유일한 음성 전처리이고 학습 스크립트와 추론 스크립트가 모두 이 함수를 불러서, 창이 두 곳에서 똑같이 만들어진다.
이 발화만 본다: 8 kHz -> 16 kHz 로 바꾸고, 3초 창으로 자르거나(길 때) 반복해 채운 뒤(짧을 때), 그 창 안에서만 평균 0·분산 1 로 정규화한다.
추론 때는 발화의 가운데 3초를, 학습 때는 발화 안의 무작위 3초를 쓴다(데이터 증강, 두 클래스에 똑같이 적용).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

TARGET_SAMPLE_RATE = 16000                              # WavLM 이 사전학습된 샘플링 주파수
WINDOW_SECONDS = 3.0                                    # 모델 입력 길이(초)
WINDOW_SAMPLES = int(TARGET_SAMPLE_RATE * WINDOW_SECONDS)  # = 48,000 샘플
HEAD_DIM = 256                                          # 분류 head 의 중간 차원


def resample_to_16k(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """발화 소리를 16 kHz 로 바꾼다(대회 음성은 8 kHz). 샘플이 2개 미만이면 무음 2샘플을 돌려준다."""

    segment = np.asarray(segment, dtype=np.float32)
    if segment.size < 2:
        return np.zeros(2, dtype=np.float32)
    if sample_rate == TARGET_SAMPLE_RATE:
        return segment
    import librosa

    return librosa.resample(segment, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE).astype(np.float32, copy=False)


def fit_window(waveform: np.ndarray, *, train: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    """이 발화의 고정 길이(3초) 창: 길면 자르고(추론은 가운데, 학습은 무작위 위치), 짧으면 그 발화를 반복해 채운다."""

    count = waveform.size
    if count == 0:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if count >= WINDOW_SAMPLES:
        if train:
            if rng is None:
                raise ValueError("train=True needs an rng for the random crop")
            offset = int(rng.integers(0, count - WINDOW_SAMPLES + 1))
        else:
            offset = (count - WINDOW_SAMPLES) // 2
        return waveform[offset : offset + WINDOW_SAMPLES].copy()
    repeats = math.ceil(WINDOW_SAMPLES / count)
    return np.tile(waveform, repeats)[:WINDOW_SAMPLES].copy()


def normalize_window(window: np.ndarray) -> np.ndarray:
    """창 하나를 그 창 자신의 평균·표준편차로 정규화한다(다른 발화는 쓰지 않음). 거의 무음이면 0 으로 채운 창을 돌려준다."""

    deviation = float(window.std())
    if not np.isfinite(deviation) or deviation < 1e-5:
        return np.zeros_like(window)
    return ((window - window.mean()) / deviation).astype(np.float32)


def prepare_window(
    segment: np.ndarray, sample_rate: int, *, train: bool = False, rng: np.random.Generator | None = None
) -> np.ndarray:
    """발화 하나의 소리 -> WavLMRoleClassifier 의 입력 (WINDOW_SAMPLES,) float32. 학습과 추론이 모두 이 함수를 쓴다."""

    return normalize_window(fit_window(resample_to_16k(segment, sample_rate), train=train, rng=rng))


class WavLMRoleClassifier(nn.Module):
    """WavLM 본체 -> 모든 층 출력의 학습되는 가중합 -> 시간축 평균·표준편차 -> 작은 MLP -> 2개 logit [대원, 신고자].

    화자·역할 정보가 어느 층에 가장 많은지 미리 알 수 없어서(화자 특성은 아래·중간 층, 말의 내용은 위층에 많다)
    25개 층(입력 임베딩 + 트랜스포머 24층)의 가중치를 함께 학습한다.
    """

    def __init__(self, backbone: nn.Module, *, dropout: float = 0.2) -> None:
        super().__init__()
        config = backbone.config
        self.backbone = backbone
        # 층별 가중치(softmax 전 값). 처음엔 모든 층을 똑같이 섞는다.
        self.layer_logits = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        # 분류 head: (평균 + 표준편차) 2 x 1024 -> 256 -> 2
        self.head = nn.Sequential(
            nn.Linear(2 * config.hidden_size, HEAD_DIM), nn.ReLU(), nn.Dropout(dropout), nn.Linear(HEAD_DIM, 2)
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str, **head_options: Any) -> "WavLMRoleClassifier":
        """학습 시작용: Hugging Face 의 공개 사전학습 가중치(예: microsoft/wavlm-large)를 내려받아 만든다."""

        from transformers import AutoModel

        # LayerDrop 과 시간 마스킹은 긴 사전학습용 규제 기법이다. 돌려받는 층 수가 바뀌고 학습/추론 동작이 달라지므로 끈다.
        # AutoModel 은 WavLM 이름이면 WavLMModel 을 돌려준다.
        backbone = AutoModel.from_pretrained(name_or_path, layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0)
        return cls(backbone, **head_options)

    @classmethod
    def from_config(cls, config_dict: dict, **head_options: Any) -> "WavLMRoleClassifier":
        """추론용: 저장해 둔 설정만으로 같은 구조를 만든다. 아무것도 내려받지 않고, 가중치는 이후 우리 체크포인트에서 넣는다."""

        from transformers import CONFIG_MAPPING, AutoModel

        config = CONFIG_MAPPING[config_dict["model_type"]].from_dict(config_dict)
        return cls(AutoModel.from_config(config), **head_options)

    def backbone_config_dict(self) -> dict:
        """체크포인트에 저장할 WavLM 구조 설정."""

        return self.backbone.config.to_dict()

    def configure_trainable(self, freeze_layers: int = 0) -> None:
        """파형을 처리하는 앞쪽 CNN 은 항상 고정하고, ``freeze_layers`` > 0 이면 그 아래 트랜스포머 층도 고정한다.

        아래층을 고정하면 계산이 줄어든다(첫 학습 층 아래로는 기울기가 필요 없어 역전파가 거기서 멈춤).
        최종 모델은 ``freeze_layers=0`` (파형 CNN 만 고정, 나머지 전부 미세조정)으로 학습했다.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        self.backbone.freeze_feature_encoder()
        if freeze_layers > 0:
            frozen = [self.backbone.feature_projection, self.backbone.encoder.pos_conv_embed]
            frozen += list(self.backbone.encoder.layers[:freeze_layers])
            for module in frozen:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # 모든 층의 출력(25개)을 받아 층별 가중치(softmax)로 섞는다
        hidden = self.backbone(waveforms, output_hidden_states=True).hidden_states
        weights = torch.softmax(self.layer_logits, dim=0)
        mixed = sum(weight * state for weight, state in zip(weights, hidden))
        mixed = mixed.float()
        # 시간축으로 평균과 표준편차를 내어 발화 하나를 2048차원으로 요약한 뒤 분류
        pooled = torch.cat([mixed.mean(dim=1), mixed.std(dim=1)], dim=-1)
        return self.head(pooled)


def predicted_speaker(logits: torch.Tensor) -> torch.Tensor:
    """대회 판정 규칙: P(speaker=1) >= 0.5 이면 1, 즉 logit1 >= logit0."""

    return (logits[:, 1] >= logits[:, 0]).long()
