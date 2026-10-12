#!/usr/bin/env python3
"""Mission 2 L2 모델을 학습한다: log-mel 스펙트로그램 위의 작은 CNN.

scripts/build_mission2_spectrograms.py가 만든 스펙트로그램 인덱스를 읽고
(Training으로 만들었다고 표시되지 않은 인덱스는 거부한다. dcc_data.mission2_spectrogram.
assert_training_spectrogram_index 참고), 정확도를 비교할 수 있도록 L1 baseline과 똑같은
call_id 단위 80/20 분할(dcc_data.mission2_train.make_group_split)을 다시 쓰고,
dcc_data.mission2_cnn.SpeakerCNNLarge를 학습한다.

ReduceLROnPlateau를 쓰고(val 정확도가 더 오르지 않으면 학습률을 절반으로), 마지막 에폭이
아니라 항상 *가장 좋았던* 에폭의 체크포인트와 지표를 보고/저장한다. 첫 Colab 실행에서
학습 후반에 val_acc가 인접한 에폭 사이에서 20포인트 넘게 출렁였고(그동안 train_loss는 계속
부드럽게 줄었다. 전형적인 "학습률이 너무 높다"는 증상), 마지막 에폭이 출렁이는 중간에 걸려서
이미 도달한 정점보다 훨씬 나쁠 수 있기 때문이다.

--resume은 처음부터 다시 하지 않고 --model-output에서 이어서 학습한다. 두 번째 Colab 실행
(같은 3,000통화 샘플)은 20에폭 중 맨 마지막에 최고(86.11%)를 찍었고, 이는 아직 수렴하지
않았다는 뜻이다. 학습이 이미 도달한 지점까지 가려고 새로 20에폭을 다시 돌리면 GPU에서 약
20분이 낭비된다. 체크포인트는 *마지막* 에폭의 model/optimizer/scheduler 상태(학습률 스케줄과
Adam의 모멘텀을 그 시점 그대로 이어가려는 것)를 *최고* 에폭의 모델 상태(추론용)와 따로 담는다.
그래서 이어서 학습한 실행이 기존 최고를 넘지 못해도 최고 기록을 덮어쓰지 않는다.
이어서 학습한 실행은 *자신의* 인덱스 출처(provenance)를 도장으로 찍은 체크포인트를 저장하므로,
--resume은 먼저 이어갈 파일이 정확히 같은 설정(모델, mel 설정, 라벨 출처, split, 시드.
dcc_data.mission2_checkpoint.require_resumable_checkpoint)을 기록하고 있는지 확인하고,
아무것도 기록하지 않은 파일(설정을 저장하기 전에 쓴 모든 파일)은 거부한다.

작은 스모크 실행은 CPU로도 되지만, 전체 데이터에 가까운 규모는 Colab GPU(--device cuda)에서
돌린다. 팀이 scripts/build_mission2_dataset.py를 거기서 돌리는 것과 같은 방식이다.

사용법:
    python scripts/train_mission2_cnn.py \\
        --index data/m2_spectrogram_index.parquet \\
        --spectrogram-dir data/m2_spectrograms \\
        --epochs 20 --device cuda \\
        --model-output checkpoints/mission2/m2_cnn.pt

    # 같은 체크포인트를 20에폭 더 이어서 학습한다:
    python scripts/train_mission2_cnn.py ... --epochs 20 --resume

    # 전체 데이터셋용 Colab 방식 (--packed-spectrograms, --amp, --pad-to-multiple, --early-stop-patience).
    # 호출 한 번에 1에폭씩 돌려서, 연결이 끊겨도 최대 1에폭만 잃는다:
    python scripts/train_mission2_cnn.py --index data/m2_spectrogram_index_packed.parquet \\
        --packed-spectrograms data/m2_spectrograms_packed.npy --model large --device cuda \\
        --amp --pad-to-multiple 8 --cudnn-benchmark --num-workers 2 \\
        --epochs 1 --early-stop-patience 5 --resume
"""

from __future__ import annotations

import argparse
import functools
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, recall_score  # noqa: E402

from dcc_data.mission2_cnn import (  # noqa: E402
    SpeakerCNNLarge,
    PackedSpectrogramDataset,
    SpectrogramDataset,
    collate_batch,
    evaluate,
    move_optimizer_state_to_device,
    train_with_best_checkpoint,
)
from dcc_data.mission2_checkpoint import build_checkpoint_metadata, require_resumable_checkpoint  # noqa: E402
from dcc_data.mission2_spectrogram import assert_training_spectrogram_index  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402

MODEL_CLASSES = {"large": SpeakerCNNLarge}
DATASET_CLASSES = {"large": SpectrogramDataset}
COLLATE_FNS = {"large": collate_batch}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path("data/m2_spectrogram_index.parquet"))
    parser.add_argument("--spectrogram-dir", type=Path, default=Path("data/m2_spectrograms"))
    parser.add_argument("--seed", type=int, default=42, help="seed for the call_id-grouped train/val split only (make_group_split)")
    parser.add_argument(
        "--torch-seed", type=int, default=None,
        help="seed for model init + DataLoader shuffle order (defaults to --seed). Kept separate from --seed "
        "so repeated runs can keep the same held-out val split while varying only these two sources of "
        "run-to-run noise -- collate_batch pads each batch to its own longest utterance, so an uncontrolled "
        "shuffle order changes what a short utterance is batched with, not just its weight-init luck.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20, help="how many MORE epochs to run this invocation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=2, help="epochs with no val_acc improvement before halving lr")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model", type=str, default="large", choices=list(MODEL_CLASSES),
        help="large=SpeakerCNNLarge (422k params)",
    )
    parser.add_argument("--model-output", type=Path, default=Path("checkpoints/mission2/m2_cnn.pt"))
    parser.add_argument("--resume", action="store_true", help="continue from --model-output instead of a fresh model")
    parser.add_argument(
        "--packed-spectrograms", type=Path, default=None,
        help="read spectrograms from this one packed float16 array (scripts/pack_mission2_spectrograms.py) instead "
        "of --spectrogram-dir's per-utterance files; --index must then be the packed index it wrote",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="mixed precision on cuda (float16 autocast + gradient scaling) -- several times faster on a T4",
    )
    parser.add_argument(
        "--pad-to-multiple", type=int, default=1,
        help="round each batch's padded length up to a multiple of this (see collate_batch); 1 = exact batch max",
    )
    parser.add_argument(
        "--cudnn-benchmark", action="store_true",
        help="let cudnn time and cache the fastest conv kernel per input shape (pair with --pad-to-multiple)",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="stop once this many epochs in a row (counted across --resume runs) have not beaten the best "
        "val_acc; a --resume run that finds the checkpoint already stopped trains nothing and leaves it as is",
    )
    args = parser.parse_args()
    if args.amp and args.device != "cuda":
        parser.error("--amp needs --device cuda")
    if args.pad_to_multiple < 1:
        parser.error("--pad-to-multiple must be >= 1")
    return args


def _capture_rng_state(train_shuffle_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_shuffle": train_shuffle_generator.get_state(),
    }


def _restore_rng_state(state: dict, train_shuffle_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_shuffle_generator.set_state(state["train_shuffle"])


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    torch_seed = args.torch_seed if args.torch_seed is not None else args.seed
    random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    train_shuffle_generator = torch.Generator()
    train_shuffle_generator.manual_seed(torch_seed)
    print(f"torch_seed={torch_seed} (model init + train shuffle order); split seed={args.seed}")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    df = pd.read_parquet(args.index)
    # Training으로 만들었다고 표시되지 않은 것은 거부한다. --index를 엉뚱한 캐시로 줘도
    # 공식 Validation 폴더가 loss/역전파에 들어가는 일은 절대 없어야 한다.
    index_provenance = assert_training_spectrogram_index(df)
    print(f"index provenance: {index_provenance}")
    train_idx, val_idx = make_group_split(df, seed=args.seed, test_size=args.test_size)
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"train={len(train_df)} val={len(val_df)} train_calls={train_df['call_id'].nunique()} val_calls={val_df['call_id'].nunique()}")

    if args.packed_spectrograms is not None:
        def make_dataset(part: pd.DataFrame) -> torch.utils.data.Dataset:
            return PackedSpectrogramDataset(part, args.packed_spectrograms)
    else:
        def make_dataset(part: pd.DataFrame) -> torch.utils.data.Dataset:
            return DATASET_CLASSES[args.model](part, args.spectrogram_dir)
    collate_fn = COLLATE_FNS[args.model]
    if args.pad_to_multiple > 1:
        collate_fn = functools.partial(collate_batch, pad_to_multiple=args.pad_to_multiple)
    loader_options = {
        "batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda", "persistent_workers": args.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        make_dataset(train_df), shuffle=True, generator=train_shuffle_generator, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(make_dataset(val_df), shuffle=False, **loader_options)

    model_cls = MODEL_CLASSES[args.model]
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # 이번 실행이 저장할 체크포인트에 찍을 내용. --resume은 이어갈 파일이 정확히 이것으로 학습됐음을
    # 먼저 증명해야 하므로 미리 만들어 둔다 (require_resumable_checkpoint 참고). 그렇지 않으면
    # 다른 라벨/설정으로 학습한 옛 가중치가 오늘의 출처 도장을 달고 나오게 된다.
    checkpoint_metadata = build_checkpoint_metadata(
        model_name=args.model,
        model_class=model_cls.__name__,
        index_provenance=index_provenance,
        split_seed=args.seed,
        test_size=args.test_size,
        torch_seed=torch_seed,
    )

    prev_best_val_acc = -1.0
    prev_best_model_state = None
    prev_best_epoch = 0
    epochs_already_trained = 0

    if args.resume:
        if not args.model_output.exists():
            raise SystemExit(f"--resume given but {args.model_output} does not exist")
        checkpoint = torch.load(args.model_output, map_location="cpu")
        require_resumable_checkpoint(checkpoint, checkpoint_metadata, args.model_output)
        model.load_state_dict(checkpoint["last_model_state_dict"])
        model.to(args.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, args.device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        prev_best_val_acc = checkpoint["best_val_acc"]
        prev_best_model_state = checkpoint["best_model_state_dict"]
        prev_best_epoch = checkpoint["best_epoch"]
        epochs_already_trained = checkpoint["total_epochs_trained"]
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # 난수열(셔플 순서, dropout)을 torch_seed에서 다시 시작하지 않고 이어간다. 다시 시작하면
        # 이어서 학습하는 모든 구간마다 첫 실행의 데이터 순서가 되풀이된다.
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"], train_shuffle_generator)
        else:  # 난수 상태를 기록하기 전에 저장된 파일: 적어도 1에폭째의 순서를 되풀이하지는 않는다
            resumed_seed = torch_seed + epochs_already_trained
            random.seed(resumed_seed)
            torch.manual_seed(resumed_seed)
            torch.cuda.manual_seed_all(resumed_seed)
            train_shuffle_generator.manual_seed(resumed_seed)
            print(f"NOTE: {args.model_output} has no saved random state -- reseeding with {resumed_seed}")
        print(
            f"Resumed from {args.model_output}: {epochs_already_trained} epochs already trained, "
            f"standing best val_acc={prev_best_val_acc:.4f} (epoch {prev_best_epoch})"
        )
        if args.early_stop_patience is not None and epochs_already_trained - prev_best_epoch >= args.early_stop_patience:
            print(
                f"EARLY STOPPED: no val_acc improvement for {epochs_already_trained - prev_best_epoch} epochs "
                f"(patience {args.early_stop_patience}) -- not training, {args.model_output} left unchanged"
            )
            return 0

    epoch_start = time.time()

    def on_epoch_end(record: dict) -> None:
        nonlocal epoch_start
        lr = optimizer.param_groups[0]["lr"]
        absolute_epoch = epochs_already_trained + record["epoch"]
        print(
            f"epoch {absolute_epoch:3d}  train_loss={record['train_loss']:.4f}  val_acc={record['val_acc']:.4f}  "
            f"lr={lr:.2e}  time={time.time() - epoch_start:.0f}s"
        )
        epoch_start = time.time()

    # 조기 종료는 이번 실행을 포함해 지금까지 모든 실행에서의 최고 에폭부터 센다.
    running_best = {"val_acc": prev_best_val_acc, "epoch": prev_best_epoch}

    def should_stop(record: dict) -> bool:
        absolute_epoch = epochs_already_trained + record["epoch"]
        if record["val_acc"] > running_best["val_acc"]:
            running_best.update(val_acc=record["val_acc"], epoch=absolute_epoch)
        if args.early_stop_patience is None or absolute_epoch - running_best["epoch"] < args.early_stop_patience:
            return False
        print(f"EARLY STOPPED after epoch {absolute_epoch}: best val_acc is still epoch {running_best['epoch']}")
        return True

    start = time.time()
    result = train_with_best_checkpoint(
        model, train_loader, val_loader, optimizer,
        epochs=args.epochs, device=args.device, scheduler=scheduler, on_epoch_end=on_epoch_end,
        scaler=scaler, should_stop=should_stop,
    )
    elapsed = time.time() - start
    total_epochs_trained = epochs_already_trained + len(result["history"])

    improved = result["best_val_acc"] > prev_best_val_acc
    if improved:
        best_val_acc = result["best_val_acc"]
        best_model_state = result["best_model_state"]
        best_epoch = epochs_already_trained + result["best_epoch"]
        best_preds, best_labels = result["best_val_preds"], result["best_val_labels"]
    else:
        best_val_acc = prev_best_val_acc
        best_model_state = prev_best_model_state
        best_epoch = prev_best_epoch
        best_preds = best_labels = None

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_model_state_dict": best_model_state,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "last_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "total_epochs_trained": total_epochs_trained,
            "rng_state": _capture_rng_state(train_shuffle_generator),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            # 이번 실행이 어떻게 학습했는지 (속도 설정이며 모델이 무엇인지와는 무관). 기록용일 뿐 확인하지 않는다.
            "training_options": {
                "amp": args.amp, "pad_to_multiple": args.pad_to_multiple, "batch_size": args.batch_size,
                "lr": args.lr, "lr_patience": args.lr_patience, "early_stop_patience": args.early_stop_patience,
                "packed_spectrograms": args.packed_spectrograms is not None,
            },
            # 이 체크포인트가 무엇으로 학습됐는지. scripts/mission2_inference.py는 이것이 추론에 쓸
            # 설정과 다르면 실행을 거부한다 (dcc_data.mission2_checkpoint 참고).
            "metadata": checkpoint_metadata,
        },
        args.model_output,
    )

    print(f"\nBest epoch so far: {best_epoch}/{total_epochs_trained}  val_acc={best_val_acc:.4f}  (this run: {elapsed:.1f}s)")
    if not improved:
        print("(this run did not beat the standing best -- checkpoint's best/resume state both updated, recall/confusion matrix below are the standing best's)")
        eval_model = model_cls()
        eval_model.load_state_dict(best_model_state)
        _, best_preds, best_labels = evaluate(eval_model, val_loader, device=args.device, amp=args.amp)
    recall_0 = recall_score(best_labels, best_preds, pos_label=0)
    recall_1 = recall_score(best_labels, best_preds, pos_label=1)
    print(f"Recall 대원(0)={recall_0:.4f} 신고자(1)={recall_1:.4f}")
    print(f"Confusion matrix: {confusion_matrix(best_labels, best_preds, labels=[0, 1]).tolist()}")
    print(f"Model saved: {args.model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
