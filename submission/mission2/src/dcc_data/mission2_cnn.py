"""Mission 2 L2 모델: 로그 멜 스펙트로그램 위의 작은 CNN.

왜 이 모델을 쓰는지는 dcc_data.mission2_spectrogram의 모듈 설명을 보라(L1의 발화별 요약 통계는
중요한 두 오류 유형, 즉 길이 구간별 편향과 겹침 격차를 해결하지 못한 채 한계에 부딪혔다).

일부러 작게 만들었다(합성곱 블록 3개). 정확성을 확인하는 첫 단계이지 튜닝한 구조가 아니다. Colab GPU에서
파이프라인이 동작함을 확인한 뒤에 팀이 넓히거나 깊게 만들 수 있다(이 모듈은 여기서 CPU 전용의 아주 작은
스모크 테스트로만 확인했다. tests/test_mission2_cnn.py 참고).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


class SpectrogramDataset(torch.utils.data.Dataset):
    """인덱스 DataFrame의 ``npy_path`` 열(``spectrogram_dir`` 기준 상대 경로)이 가리키는 미리 계산한 .npy
    스펙트로그램을 읽는다. 정확히 이런 인덱스를 만드는 dcc_data.mission2_spectrogram.
    build_seoul_training_spectrograms를 보라. 저장된 배열을 읽는 것이 에폭마다 오디오를 다시 디코딩하고
    스펙트로그램을 다시 계산하는 것보다 훨씬 싸다.
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
    """SpectrogramDataset과 같은 항목을 발화마다 .npy 파일 하나가 아니라 묶은 배열 하나에서 읽는다.

    ``packed_path``는 scripts/pack_mission2_spectrograms.py가 쓴 .npy이다: 모든 발화의 스펙트로그램을
    (n_frames, n_mels)로 뒤집어 프레임 축으로 이어 붙이고 float16으로 저장한 것이다.
    ``index_df``의 i번째 행의 스펙트로그램은 ``packed[frame_offset : frame_offset + n_frames].T``이다.
    메모리 매핑 파일 하나를 쓰면 에폭마다 작은 파일 약 70만 개를 여는 일을 피하고, float16이면 크기가 절반이
    되어(Training 전체에서 약 7GB) Colab의 페이지 캐시에 들어간다. --amp에서는 첫 합성곱이 어차피 입력을
    float16으로 반올림하므로 float16으로 저장해도 손해가 없다.

    항목은 float32로 돌려주며, SpectrogramDataset이 돌려주는 (n_mels, n_frames) 텐서와 똑같다. 그래서
    collate_batch와 모델은 바꿀 필요가 없다. 파일은 처음 읽는 (워커) 프로세스에서 늦게 연다.
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

        채우는 값은 SPEC_FLOOR_DB(mel_spectrogram 자체 정규화에서의 무음)이며, 0.0은 절대 쓰지 않는다.
        0 dB는 채움이 아니라 그 발화에서 *가장 큰* 프레임으로 읽히기 때문이다.

        ``pad_to_multiple``이 1보다 크면 그 길이를 그 값의 배수로 올림한다(이미 평균 약 400프레임으로 채운
        배치에 무음 프레임이 최대 ``pad_to_multiple - 1``개 더 붙는다). 입력 모양의 종류가 훨씬 줄어들면
        cudnn.benchmark가 모양마다 빠른 커널을 골라 캐시할 수 있다. 1이면 배치의 최대 길이를 그대로 쓴다.
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


class SpectrogramWithDurationDataset(SpectrogramDataset):
    """SpectrogramDataset과 같지만, 스펙트로그램과 라벨 외에 발화 자신의 ``duration_sec``
    (그 발화 자신의 startAt/endAt 구간. 사용 가능, SpeakerCNNLargeWithDuration 참고)도 함께 돌려준다.

        SpectrogramDataset 자체에 길이를 더하지 않고 별도 클래스로 둔 이유는, 기존의 모든
        SpectrogramDataset/collate_batch 호출부(특히 제출에 쓴 91.22% 파이프라인)를 건드리지 않기 위해서이다.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, int]:
        row = self.index_df.iloc[idx]
        spec = np.load(self.spectrogram_dir / row["npy_path"])
        return torch.from_numpy(spec), float(row["duration_sec"]), int(row["speaker"])


def collate_batch_with_duration(
    batch: list[tuple[torch.Tensor, float, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """collate_batch와 같되, 각 항목 자신의 duration_sec를 (batch,) 모양의 float 텐서로
    쌓아서 함께 돌려준다.
    """

    specs, durations, labels = zip(*batch)
    n_mels = specs[0].shape[0]
    max_frames = max(spec.shape[1] for spec in specs)

    padded = torch.full((len(specs), 1, n_mels, max_frames), SPEC_FLOOR_DB, dtype=torch.float32)
    for i, spec in enumerate(specs):
        padded[i, 0, :, : spec.shape[1]] = spec
    durations_tensor = torch.tensor(durations, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, durations_tensor, labels_tensor


class SpeakerCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class SpeakerCNNLarge(nn.Module):
    """합성곱 블록 4개, 채널 32->64->128->256. (SpeakerCNN은 블록 3개, 16->32->64였는데 전체 29,200통화
    데이터에서 검증 정확도 89.6~89.7%에서 정체했다. 학습률이 6.25e-05까지 줄었고 val_acc가 10에폭 연속으로
    약 0.3%p 범위에 머물렀다. 즉 학습이 부족해서가 아니라 그 용량에서는 수렴한 것이다.) SpeakerCNN의 크기를
    바꾸지 않고 별도 클래스로 둔 이유는, 기존 SpeakerCNN 체크포인트를 그대로 불러오고 이어 학습할 수 있게
    하기 위해서이다. 이 모델을 쓰려면 scripts/train_mission2_cnn.py에 --model large를 주고, state_dict
    모양이 SpeakerCNN과 맞지 않으므로 --model-output도 새로 정한다.
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


class SpeakerCNNLargeWithDuration(nn.Module):
    """SpeakerCNNLarge의 합성곱 층에 더해, 발화 자신의 duration_sec를 명시적인 스칼라 입력으로
    분류 헤드에 이어 붙인 모델.

    이유: L2 오류 분석에서 남은 대원/신고자 격차가 내용이 없는 짧은 맞장구 발화("네"/"예")에 몰려 있었다.
    SpeakerCNNLarge는 짧은 발화의 실제 프레임 수를 이미 보지만 암묵적으로만 보고, 그마저도 배치 자체의 0 채움에
    희석된다. 아주 짧은 발화는 MaxPool(stride 2)을 3번 거치면 AdaptiveAvgPool2d 앞에 시간 단계가 1~2개만
    남을 수 있다. 길이를 직접 주면(원래 초 단위 값이 심하게 오른쪽으로 치우쳐 있으므로 log1p로 압축) 그 약하고
    간접적인 신호에 기대지 않고도 "이것은 아주 짧은 발화이다"를 조건으로 삼을 수 있다. duration_sec는
    현재 발화 자신의 endAt-startAt 구간이며 통화의 다른 발화에서 구한 것이 아니므로, 발화 하나만 보고 판단해야
    한다는 규칙 안에 머문다([[project-dcc2026-contest-rules]] 참고).

    SpeakerCNNLarge를 고치지 않고 별도 클래스로 둔 이유는, 이미 학습한 91.22% 체크포인트와
    scripts/train_mission2_cnn.py의 기본 경로를 건드리지 않기 위해서이다. 이 모델은 분류기의 첫 Linear 입력 폭이
    다르므로(256이 아니라 257) 옛 체크포인트의 state_dict는 어차피 불러와지지 않는다. 이 모델을 학습하려면
    scripts/train_mission2_cnn.py에 --model large-duration을 준다.
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
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        duration_feature = torch.log1p(duration.to(x.dtype)).unsqueeze(1)
        x = torch.cat([x, duration_feature], dim=1)
        return self.classifier(x)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    """``optimizer.state``의 모든 텐서(예: Adam의 exp_avg/exp_avg_sq 모멘텀 버퍼)를 ``device``로 제자리에서 옮긴다.

        ``map_location="cpu"``로 불러온 체크포인트에서 ``optimizer.load_state_dict()``를 한 뒤에 필요하다.
        옵티마이저 내부 상태 텐서는 모델 파라미터와 따로 있어서 ``model.to(device)``로 모델을 옮겨도 그것들은
        옮겨지지 않는다. 그러면 이어 학습한 뒤 첫 ``step()``이, 모델 파라미터는 cuda에 있는데 불러온 모멘텀
        버퍼는 아직 cpu에 있어서 "Expected all tensors to be on the same device"로 실패한다.
    """

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _autocast(device: str, amp: bool):
    """``amp``이면 cuda에서 float16 autocast를, 아니면 아무 일도 하지 않는 컨텍스트를 쓴다(cpu에서도 쓰이지 않는다)."""

    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda")


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, scaler=None) -> float:
    """``loader``는 ``(*모델 입력, 라벨)`` 튜플을 돌려준다. SpeakerCNN/SpeakerCNNLarge는 ``(specs, labels)``,
        SpeakerCNNLargeWithDuration은 ``(specs, duration, labels)``이고, ``model(*모델 입력)``을 그에 맞게
        호출한다. 그래서 이 반복문 하나가 둘 다 처리한다.

        ``scaler``(torch.amp.GradScaler)를 주면 혼합 정밀도를 켠다: 순전파는 float16 autocast에서 실행하고,
        작은 float16 기울기가 0으로 반올림되지 않도록 손실에 배율을 곱한다. None이면 일반 float32이다.
    """

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp = scaler is not None
    # 장치(device) 위에서 합산한다: 배치마다 .item()을 부르면 CPU가 매 단계 GPU를 기다리게 된다.
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
    """(batch, 2) 로짓 텐서에 대한 대회의 판정 규칙: P(speaker=1) >= 0.5일 때만 클래스 1.

        이것은 ``logit1 >= logit0``과 같다. ``argmax``는 값이 정확히 같은 경우에만 다르고, 그때 클래스 0을
        돌려준다. 제출 스크립트(scripts/mission2_inference.py)는 이미 값이 같으면 클래스 1로 판정하므로, 최고
        에폭을 고르는 정확도도 같은 방식으로 계산해야 한다.
    """

    return (logits[:, 1] >= logits[:, 0]).long()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, amp: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """``loader``의 튜플 모양은 train_one_epoch()의 설명을 보라. ``amp``이면 순전파를 float16 autocast에서
        실행한다(cuda만). GradScaler로 학습할 때와 같다.
    """

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


@torch.no_grad()
def predict_with_confidence(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """evaluate()와 같되, 예측마다 신뢰도 점수를 더한다: 모델이 실제로 예측한 클래스에 준 softmax 확률이다.
        scripts/analyze_mission2_errors.py에서 L1의 predict_proba 기반 신뢰도 열과 같은 용도(오류 분석)이다
        (틀린 예측이 확신을 갖고 틀린 것인지, 아슬아슬한 것인지). 2클래스 판정에서는 항상 [0.5, 1.0]이다
        (값이 정확히 같으면 클래스 1이고 0.5이다).

        ``loader``의 튜플 모양은 train_one_epoch()의 설명을 보라.
    """

    model.to(device)
    model.eval()
    all_preds, all_labels, all_confidence = [], [], []
    for *inputs, labels in loader:
        inputs = [t.to(device) for t in inputs]
        logits = model(*inputs)
        probs = torch.softmax(logits, dim=1)
        preds = predicted_classes(logits)
        confidence = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_confidence.append(confidence.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_confidence)


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
    """``epochs``만큼 학습하되, 마지막 에폭만이 아니라 검증 정확도가 가장 좋았던 에폭을 추적한다.

        실제 실행(reports/mission2/ 참고, 이 함수를 만들게 한 Colab 스모크 테스트)은 학습 후반에 불안정해질 수 있다
        (train_loss는 계속 매끄럽게 줄어드는데 val_acc가 인접한 에폭 사이에서 20%p 이상 출렁이는, 전형적인
        "학습률이 너무 높다" 증상). ``epochs``를 정점에서 정확히 멈추도록 맞추지 않아도 된다. 마지막에 돌아간
        에폭이 마침 진동 중이어서 이미 도달해 저장해 둔 최고 에폭보다 훨씬 나쁠 때, 그 에폭의 지표를 보고하는
        것은 오해를 부른다.

        ``scheduler``(예: ReduceLROnPlateau)를 주면 에폭마다 그 에폭의 검증 정확도로 한 번 호출한다.
        ``on_epoch_end``를 주면 에폭마다 그 에폭의 history 사전으로 한 번 호출한다(진행 상황을 실시간으로
        출력하려는 호출자용이며, 그 외에는 테스트하기 쉽도록 이 함수가 부작용 없는 순수 함수로 남는다).

        ``scaler``(torch.amp.GradScaler)를 주면 혼합 정밀도로 학습하고 평가한다(train_one_epoch 참고).
        ``should_stop``을 주면 에폭이 끝날 때마다(on_epoch_end 뒤에) 그 에폭의 history 사전으로 호출하고,
        True를 돌려주면 ``epochs``에 이르기 전에 거기서 학습을 끝낸다. 조기 종료용이며, 이 함수가 보지 못하는
        이전에 이어 받은 실행의 에폭까지 알아야 하기 때문이다.

        돌려주는 사전: best_val_acc, best_val_preds, best_val_labels, best_epoch, best_model_state
        (state_dict, CPU 텐서), history({epoch, train_loss, val_acc}의 목록).
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
