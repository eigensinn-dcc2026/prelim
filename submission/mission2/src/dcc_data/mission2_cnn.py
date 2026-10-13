"""미션 2 CNN 모델: log-mel 스펙트로그램을 입력으로 받는 CNN(``SpeakerCNNLarge``)과 학습·평가 루프.

최종 제출 모델의 CNN 3개(torch seed 1/2/3)가 모두 ``SpeakerCNNLarge`` 이다. 발화 하나의 스펙트로그램만 보고
대원(0)/신고자(1) 두 클래스의 logit 을 낸다.

(제출 폴더 정리 메모: 실험했지만 최종 모델에 쓰지 않은 작은 모델 ``SpeakerCNN``, 발화 길이를 추가 입력으로 넣은
``SpeakerCNNLargeWithDuration`` 과 그 전용 데이터셋·묶음 함수, 쓰이지 않는 ``predict_with_confidence`` 는 뺐다.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


class SpectrogramDataset(torch.utils.data.Dataset):
    """발화별로 미리 계산해 둔 .npy 스펙트로그램을 읽는다(index 표의 ``npy_path`` 열, ``spectrogram_dir`` 기준 상대 경로).

    이 index 는 dcc_data.mission2_spectrogram.build_seoul_training_spectrograms 가 만든다. 저장된 배열을 읽는 것이
    매 에폭 음성을 다시 읽고 스펙트로그램을 다시 계산하는 것보다 훨씬 빠르다.
    """

    def __init__(self, index_df: pd.DataFrame, spectrogram_dir: str | Path) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.spectrogram_dir = Path(spectrogram_dir)

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.index_df.iloc[idx]
        spec = np.load(self.spectrogram_dir / row["npy_path"])
        return torch.from_numpy(spec), int(row["speaker"])


class PackedSpectrogramDataset(torch.utils.data.Dataset):
    """SpectrogramDataset 과 같은 항목을, 발화마다 .npy 하나 대신 큰 배열 하나(묶음 파일)에서 읽는다. (최종 학습이 쓴 방식)

    ``packed_path`` 는 scripts/pack_mission2_spectrograms.py 가 만든 .npy 다. 모든 발화의 스펙트로그램을 (n_frames, n_mels) 로
    돌려서 프레임 축으로 이어 붙이고 float16 으로 저장한 것이다. ``index_df`` 의 i 번째 발화는
    ``packed[frame_offset : frame_offset + n_frames].T`` 이다. 파일 하나를 메모리 매핑으로 열면 에폭마다 약 70만 개의 작은 파일을
    여는 비용이 사라지고, float16 이라 크기가 절반(전체 Training 약 7GB)이라 Colab 의 페이지 캐시에 들어간다.
    --amp 학습에서는 첫 합성곱이 입력을 어차피 float16 으로 바꾸므로 정밀도 손해가 없다.

    돌려주는 값은 SpectrogramDataset 과 똑같은 (n_mels, n_frames) float32 텐서라 collate_batch 와 모델은 그대로 쓴다.
    파일은 처음 읽는 (작업) 프로세스에서 열린다.
    """

    def __init__(self, index_df: pd.DataFrame, packed_path: str | Path) -> None:
        index_df = index_df.reset_index(drop=True)
        self.frame_offsets = index_df["frame_offset"].to_numpy(dtype=np.int64)
        self.n_frames = index_df["n_frames"].to_numpy(dtype=np.int64)
        self.speakers = index_df["speaker"].to_numpy(dtype=np.int64)
        self.packed_path = Path(packed_path)
        self._packed: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.speakers)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._packed is None:
            self._packed = np.load(self.packed_path, mmap_mode="r")
        start = self.frame_offsets[idx]
        frames = self._packed[start : start + self.n_frames[idx]]
        return torch.from_numpy(np.ascontiguousarray(frames.T, dtype=np.float32)), int(self.speakers[idx])


def collate_batch(
    batch: list[tuple[torch.Tensor, int]], pad_to_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """배치 안의 스펙트로그램을 그 배치에서 가장 긴 프레임 수에 맞춰 채운다(학습용 묶음 함수).

    채우는 값은 0.0 이 아니라 SPEC_FLOOR_DB(mel_spectrogram 정규화 기준의 무음)다. 0 dB 는 그 발화의 *가장 큰* 소리 크기라서
    채움 값으로 쓰면 안 된다.

    ``pad_to_multiple`` > 1 이면 그 길이를 그 배수로 올린다(최대 ``pad_to_multiple - 1`` 프레임 더). 입력 모양 종류가 크게 줄어
    cudnn.benchmark 가 모양별로 빠른 커널을 골라 재사용할 수 있다. 1 이면 배치 최댓값 그대로다.
    """

    specs, labels = zip(*batch)
    n_mels = specs[0].shape[0]
    max_frames = max(spec.shape[1] for spec in specs)
    max_frames = -(-max_frames // pad_to_multiple) * pad_to_multiple

    padded = torch.full((len(specs), 1, n_mels, max_frames), SPEC_FLOOR_DB, dtype=torch.float32)
    for i, spec in enumerate(specs):
        padded[i, 0, :, : spec.shape[1]] = spec
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, labels_tensor


class SpeakerCNNLarge(nn.Module):
    """최종 제출 모델의 CNN. 합성곱 블록 4개(채널 32 -> 64 -> 128 -> 256), 파라미터 421,954개.

    입력: (배치, 1, 40 mel, 프레임 수). 프레임 수는 자유롭다(AdaptiveAvgPool2d 가 시간·주파수 축을 평균 내어 256차원으로 줄임).
    출력: (배치, 2) logit [대원, 신고자].

    처음에 쓴 더 작은 3블록 모델(16 -> 32 -> 64)은 전체 29,200통화로 학습해도 검증 정확도가 89.6~89.7% 에서 멈춰서
    (학습률이 6.25e-05 까지 내려가도 10에폭 동안 약 0.3%p 안에 머묾, 즉 그 크기로는 수렴) 이 큰 모델로 바꿨다.
    """

    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        # 블록마다 합성곱(3x3) -> BatchNorm -> ReLU -> 2x2 최대 풀링(마지막 블록은 풀링 없음)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 발화 길이와 상관없이 256개 값으로 요약
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 분류기: 256 -> 128 -> 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    """``optimizer.state`` 안의 모든 텐서(Adam 의 모멘텀 버퍼 등)를 ``device`` 로 옮긴다.

    CPU 로 불러온 체크포인트에서 ``optimizer.load_state_dict()`` 를 하면 필요하다. 옵티마이저 상태는 모델 파라미터와 별개라
    ``model.to(device)`` 로는 옮겨지지 않고, 이어서 학습할 때 첫 ``step()`` 에서 "같은 장치에 있어야 한다" 오류가 난다.
    """

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _autocast(device: str, amp: bool):
    """``amp`` 이고 cuda 일 때 float16 autocast, 아니면 아무것도 하지 않는 컨텍스트."""

    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda")


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, scaler=None) -> float:
    """한 에폭 학습하고 평균 loss 를 돌려준다. ``loader`` 는 ``(입력, 라벨)`` 묶음을 낸다.

    ``scaler``(torch.amp.GradScaler)를 주면 혼합 정밀도로 학습한다. 순전파는 float16 autocast 로 하고, 작은 float16 기울기가
    0 으로 사라지지 않도록 loss 를 키워서 역전파한다. None 이면 float32 그대로 학습한다.
    """

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp = scaler is not None
    # loss 합계를 GPU 에 둔다. 배치마다 .item() 을 부르면 CPU 가 매번 GPU 를 기다려야 해서 느려진다.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    n = 0
    for *inputs, labels in loader:
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with _autocast(device, amp):
            logits = model(*inputs)
            loss = criterion(logits, labels)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().double() * len(labels)
        n += len(labels)
    return total_loss.item() / n


def predicted_classes(logits: torch.Tensor) -> torch.Tensor:
    """대회 판정 규칙을 (배치, 2) logit 에 적용: P(speaker=1) >= 0.5 이면 1, 즉 ``logit1 >= logit0``.

    ``argmax`` 는 정확히 같을 때 0 을 돌려준다. 제출 추론은 동점을 1 로 정하므로, 최고 에폭을 고르는 정확도도 같은 규칙으로 계산한다.
    """

    return (logits[:, 1] >= logits[:, 0]).long()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, amp: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """검증 데이터의 (정확도, 예측, 라벨)을 돌려준다. ``amp`` 면 순전파를 float16 autocast 로 한다(학습과 같은 조건)."""

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for *inputs, labels in loader:
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        with _autocast(device, amp):
            logits = model(*inputs).float()
        all_preds.append(predicted_classes(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    accuracy = float((preds == labels_arr).mean())
    return accuracy, preds, labels_arr


def train_with_best_checkpoint(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    *,
    epochs: int,
    device: str,
    scheduler=None,
    on_epoch_end=None,
    scaler=None,
    should_stop=None,
) -> dict:
    """``epochs`` 에폭 학습하면서, 마지막 에폭이 아니라 검증 정확도가 가장 높았던 에폭을 기억한다.

    실제 학습에서 후반에 검증 정확도가 크게 출렁일 수 있다(학습률이 높을 때 train_loss 는 매끄럽게 줄어도 검증 정확도가
    인접 에폭 사이에 20%p 이상 흔들림). 마지막 에폭의 성적을 보고하면 이미 도달했던 최고점보다 훨씬 나쁠 수 있으므로
    최고 에폭의 가중치를 따로 저장한다.

    ``scheduler``(예: ReduceLROnPlateau)를 주면 에폭마다 검증 정확도로 한 번 진행한다. ``on_epoch_end`` 를 주면 에폭마다
    그 에폭 기록(dict)으로 불러 진행 상황을 출력하게 한다(이 함수 자체는 부수 효과가 없다).
    ``scaler``(GradScaler)를 주면 혼합 정밀도로 학습·평가한다. ``should_stop`` 을 주면 에폭마다(on_epoch_end 다음) 불러서
    True 면 학습을 끝낸다(조기 종료: 이전 실행에서 이어 온 에폭까지 아는 호출자가 판단).

    돌려주는 값: best_val_acc, best_val_preds, best_val_labels, best_epoch, best_model_state(CPU 텐서 state_dict),
    history({epoch, train_loss, val_acc} 목록).
    """

    best_val_acc = -1.0
    best_val_preds: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None
    best_epoch = 0
    best_model_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device, scaler=scaler)
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device=device, amp=scaler is not None)
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
        history.append(epoch_record)

        if scheduler is not None:
            scheduler.step(val_acc)
        if on_epoch_end is not None:
            on_epoch_end(epoch_record)

        # 검증 정확도가 지금까지 최고면 그 에폭의 가중치를 복사해 둔다
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if should_stop is not None and should_stop(epoch_record):
            break

    return {
        "best_val_acc": best_val_acc,
        "best_val_preds": best_val_preds,
        "best_val_labels": best_val_labels,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "history": history,
    }
