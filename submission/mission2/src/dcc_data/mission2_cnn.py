"""Mission 2 L2 모델: log-mel 스펙트로그램 위의 작은 CNN.

왜 이 모델을 쓰는지는 dcc_data.mission2_spectrogram의 모듈 docstring을 본다
(L1의 발화별 요약 통계는, 중요한 두 오류 유형인 길이 구간 편향과 겹침 격차를 줄이지 못한 채
정체됐다).

모델은 SpeakerCNNLarge(conv 블록 4개, 채널 32->64->128->256)이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


class SpectrogramDataset(torch.utils.data.Dataset):
    """미리 계산해 둔 .npy 스펙트로그램을 읽는다. 어떤 파일인지는 인덱스 DataFrame의
    ``npy_path`` 열(``spectrogram_dir`` 기준 상대 경로)이 알려 준다. 이 인덱스를 정확히 만드는 것이
    dcc_data.mission2_spectrogram.build_seoul_training_spectrograms다. 저장된 배열을 읽는 편이
    매 에폭마다 오디오를 다시 디코딩하고 스펙트로그램을 다시 계산하는 것보다 훨씬 싸다.
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
    """SpectrogramDataset과 같은 항목을, 발화마다 .npy 파일을 따로 읽지 않고 묶어 둔 배열 하나에서 읽는다.

    ``packed_path``는 scripts/pack_mission2_spectrograms.py가 쓴 .npy다. 모든 발화의 스펙트로그램을
    (n_frames, n_mels)로 뒤집어 프레임 축으로 이어 붙인 float16 배열이다.
    ``index_df``의 i번째 행의 스펙트로그램은 ``packed[frame_offset : frame_offset + n_frames].T``다.
    메모리 매핑된 파일 하나라서 매 에폭마다 약 70만 개의 작은 파일을 열지 않아도 되고, float16이라
    크기가 절반(Training 전체 약 7 GB)이어서 Colab의 page cache에 들어간다. --amp에서는 첫 conv가
    어차피 입력을 float16으로 반올림하므로 float16으로 둬도 손해가 없다.

    항목은 float32로 돌려주며, SpectrogramDataset이 돌려주는 (n_mels, n_frames) 텐서와 정확히 같다.
    그래서 collate_batch와 모델은 바뀌지 않는다. 파일은 처음 읽는 (worker) 프로세스에서 필요할 때 연다.
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
    """배치의 모든 스펙트로그램을 그 배치의 최대 프레임 수에 맞춰 채운다.

    채우는 값은 SPEC_FLOOR_DB(mel_spectrogram 자체 정규화에서의 무음)이고, 0.0은 절대 쓰지 않는다.
    0 dB는 채움이 아니라 그 발화의 *가장 큰* 프레임으로 읽히기 때문이다.

    ``pad_to_multiple``이 1보다 크면 그 길이를 그 배수로 올림한다 (이미 평균 약 400프레임까지 채워진
    배치에서 무음 프레임이 최대 ``pad_to_multiple - 1``개 늘어날 뿐이다). 입력 모양의 종류가 훨씬
    줄어들어야 cudnn.benchmark가 모양마다 빠른 커널을 골라 캐시할 수 있다. 1이면 배치 최대값을 그대로 쓴다.
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
    """conv 블록 4개, 채널 32->64->128->256, 파라미터 421,954개.
    scripts/train_mission2_cnn.py의 ``--model large``로 학습한다.
    """

    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
    """``optimizer.state``의 모든 텐서(예: Adam의 exp_avg/exp_avg_sq 모멘텀 버퍼)를 제자리에서
    ``device``로 옮긴다.

    ``map_location="cpu"``로 불러온 체크포인트에서 ``optimizer.load_state_dict()``를 한 뒤에 필요하다.
    optimizer의 내부 상태 텐서는 모델 파라미터와 따로 있어서 ``model.to(device)``로 모델을 옮겨도
    *옮겨지지 않는다*. 그러면 이어서 학습할 때 첫 ``step()``이, 모델 파라미터는 cuda에 있는데
    불러온 모멘텀 버퍼는 아직 cpu에 있어서 "Expected all tensors to be on the same device"로 실패한다.
    """

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _autocast(device: str, amp: bool):
    """``amp``이면 cuda에서 float16 autocast, 아니면 아무 일도 하지 않는 컨텍스트 (cpu에서도 그렇다. cpu에서는 쓰이지 않는다)."""

    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda")


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, scaler=None) -> float:
    """``loader``는 ``(*모델 입력, labels)`` 튜플을 돌려준다 (SpeakerCNNLarge는 ``(specs, labels)``).

    ``scaler``(torch.amp.GradScaler)를 주면 혼합 정밀도를 켠다. forward는 float16 autocast로
    돌리고, loss를 스케일해서 작은 float16 gradient가 0으로 반올림되지 않게 한다.
    None이면 일반 float32를 쓴다.
    """

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp = scaler is not None
    # 장치 위에서 합산한다. 배치마다 .item()을 부르면 매 step마다 CPU가 GPU를 기다리게 된다.
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
    """(batch, 2) logit 텐서에 대한 대회의 판정 규칙: P(speaker=1) >= 0.5 일 때만 클래스 1.

    이는 ``logit1 >= logit0``과 같다. ``argmax``는 정확히 같은 값일 때만 클래스 0을 돌려주어 다르다.
    제출 스크립트(scripts/mission2_inference.py)는 동점을 이미 클래스 1로 정하므로, 최고 에폭을
    고르는 정확도도 똑같은 방식으로 계산해야 한다.
    """

    return (logits[:, 1] >= logits[:, 0]).long()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, amp: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """``loader``의 튜플 모양은 train_one_epoch()의 docstring을 본다. ``amp``이면 forward를
    float16 autocast(cuda 전용)로 돌리며, GradScaler로 학습할 때와 같다."""

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
    """``epochs`` 에폭 동안 학습하면서, 마지막 에폭이 아니라 val 정확도가 가장 좋았던 에폭을 따로 기록한다.

    실제 실행(reports/mission2/, 이 방식을 만들게 한 Colab 스모크 테스트 참고)은 학습 후반에
    불안정해질 수 있다. train_loss는 계속 부드럽게 줄어드는데 val_acc가 인접한 에폭 사이에서
    20포인트 넘게 출렁이는 전형적인 "학습률이 너무 높다"는 증상이다. 그렇다고 ``epochs``를 정확히
    정점에서 멈추도록 맞춰 둔 것도 아니다. 마지막으로 돈 에폭의 지표를 그대로 보고하면, 그 에폭이
    출렁이는 중간이라 이미 도달해서 저장해 둔 최고 에폭보다 훨씬 나쁠 때 오해를 부른다.

    ``scheduler``(예: ReduceLROnPlateau)를 주면 에폭마다 그 에폭의 val 정확도로 한 번씩 step한다.
    ``on_epoch_end``를 주면 에폭마다 그 에폭의 history dict와 함께 한 번씩 호출된다
    (진행 상황을 실시간으로 출력하려는 호출자용. 그 외에는 테스트하기 쉽도록 이 함수에
    부작용이 없게 둔다).

    ``scaler``(torch.amp.GradScaler)를 주면 혼합 정밀도로 학습하고 평가한다 (train_one_epoch 참고).
    ``should_stop``을 주면 매 에폭이 끝난 뒤(on_epoch_end 다음) 그 에폭의 history dict와 함께
    호출되고, True를 돌려주면 ``epochs``에 닿기 전에 거기서 학습을 끝낸다. 조기 종료용인데,
    이 함수가 보지 못하는 이전에 이어서 학습한 실행의 에폭도 알아야 하기 때문에 따로 둔다.

    돌려주는 dict: best_val_acc, best_val_preds, best_val_labels, best_epoch,
    best_model_state (state_dict, CPU 텐서), history ({epoch, train_loss, val_acc}의 리스트).
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
