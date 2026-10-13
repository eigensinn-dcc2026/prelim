#!/usr/bin/env python3
"""미션 2 WavLM 미세조정: 공개 사전학습 음성 모델(microsoft/wavlm-large)을 서울 Training 발화로 미세조정한다.

[규정 준수] 데이터 규칙은 scripts/train_mission2_cnn.py 와 같다.
* 서울 Training 통화만 (``select_seoul_training_rows``. 공식 Validation 폴더는 이 스크립트의 학습에 절대 들어오지 않는다)
* 제공된 ``speaker`` 라벨 그대로 (``build_utterance_records`` 는 startAt/endAt/speaker 만 읽고 text 는 읽지 않는다)
* 모델은 발화 하나 자신의 소리만 본다 (dcc_data.mission2_wavlm.prepare_window)
* CNN 과 같은 통화 단위 80/20 분할 (``make_group_split``, ``--split-seed 42``). 두 모델이 같은 통화를 떼어 두므로 내부 검증 정확도를
  비교하고 섞는 비율을 정할 수 있다.

최고 step 은 그 떼어 둔 통화의 정확도(기준 0.5 고정)로 고르고, 공식 Validation 으로는 고르지 않는다.

Colab 이 끊긴 뒤 이어가는 데 필요한 것(모델, 옵티마이저, 스케줄러, 스케일러, step)을 ``--save-every`` step 마다 ``<out-dir>/last.pt`` 에 쓰고,
``--resume`` 이 거기서 이어간다. ``<out-dir>/best.pt`` 에는 최고 step 의 float16 가중치와 모델 구조 설정이 들어 있고, 추론에는 이것만 있으면 된다.

사용법 (최종 모델의 실제 명령, WavLM 학습 노트북과 같음. seed 1 과 seed 2 를 각각 학습):
    python scripts/train_mission2_wavlm.py --manifest /content/dcc_manifest.csv --data-root /content/dcc_data_full \
        --pretrained microsoft/wavlm-large --out-dir /content/wavlm_local/model --seed 1 --freeze-layers 0 --amp \
        --num-workers 2 --batch-size 32 --max-steps 42500 --eval-every 1000 --save-every 1000 --resume
    python scripts/train_mission2_wavlm.py ... --benchmark-steps 20     # step 당 시간만 재고 끝남
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dcc_data.audio import read_wav_frames, read_wav_info  # noqa: E402
from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    build_utterance_records,
    select_seoul_training_rows,
    select_seoul_validation_rows,
    utterance_frame_range,
)
from dcc_data.mission2_train import make_group_split  # noqa: E402
from dcc_data.mission2_wavlm import WavLMRoleClassifier, predicted_speaker, prepare_window  # noqa: E402

BEST_NAME = "best.pt"
LAST_NAME = "last.pt"
CHECKPOINT_FORMAT = "dcc2026-mission2-wavlm-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV from scripts/build_manifest.py")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--head-only-steps", type=int, default=300, help="backbone stays untouched for these first steps")
    parser.add_argument("--freeze-layers", type=int, default=0, help="also freeze the lowest N transformer layers")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-utts", type=int, default=12000, help="held-out utterances scored at every evaluation")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1, help="model head init, data order, random crops")
    parser.add_argument("--split-seed", type=int, default=42, help="call-grouped split seed (same as the CNN)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true", help="float16 autocast + gradient scaling (cuda only)")
    parser.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing: less memory, ~25%% slower")
    parser.add_argument("--no-save-optimizer", action="store_true", help="smaller last.pt, optimizer restarts on resume")
    parser.add_argument("--benchmark-steps", type=int, default=0, help=">0: time this many steps, print, exit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="best.pt to start from when there is no last.pt (weights only): step and best accuracy are taken from it, "
        "the optimizer starts fresh with a short learning-rate re-warm-up",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.amp and args.device == "cpu":
        parser.error("--amp needs a cuda device")
    return args


# ---------------------------------------------------------------------------
# 데이터: 발화 목록과 발화별 3초 창
# ---------------------------------------------------------------------------


def build_items(manifest: Path, data_root: Path, *, validation: bool = False) -> pd.DataFrame:
    """발화마다 한 행. ``validation=True`` (확률 계산 스크립트 전용, 학습에는 절대 쓰지 않음)면 Training 대신
    공식 Validation 통화를 쓴다."""

    select = select_seoul_validation_rows if validation else select_seoul_training_rows
    rows = select(load_manifest(manifest, data_root=data_root))
    records = []
    for row in rows:
        wav_path = str(data_root / row.wav_path)
        for record in build_utterance_records(row, data_root):
            records.append(
                (record.call_id, record.utt_idx, wav_path, float(record.start_at), float(record.end_at), int(record.speaker))
            )
    frame = pd.DataFrame(records, columns=["call_id", "utt_idx", "wav_path", "start_at", "end_at", "speaker"])
    if frame.empty:
        raise SystemExit("no Seoul Training utterances found -- check --manifest / --data-root")
    return frame


class UtteranceWindows(torch.utils.data.Dataset):
    def __init__(self, items: pd.DataFrame, *, train: bool, seed: int) -> None:
        self.wav_path = items["wav_path"].to_numpy()
        self.start_at = items["start_at"].to_numpy()
        self.end_at = items["end_at"].to_numpy()
        self.speaker = items["speaker"].to_numpy(dtype=np.int64)
        self.train = train
        self.seed = seed
        self._wave_info: dict = {}
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.speaker)

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng([self.seed, 0 if worker is None else worker.id + 1])
        return self._rng

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        wav_path = str(self.wav_path[index])
        info = self._wave_info.get(wav_path)
        if info is None:
            info = self._wave_info[wav_path] = read_wav_info(wav_path)
        start_frame, end_frame = utterance_frame_range(
            self.start_at[index], self.end_at[index], info.sample_rate, info.frame_count
        )
        segment = read_wav_frames(wav_path, start_frame, end_frame).waveform[0]
        window = prepare_window(
            segment, info.sample_rate, train=self.train, rng=self._worker_rng() if self.train else None
        )
        return torch.from_numpy(window), int(self.speaker[index])


# ---------------------------------------------------------------------------
# 학습률 일정 / 평가 / 체크포인트
# ---------------------------------------------------------------------------


def lr_factor(step: int, *, warmup: int, total: int) -> float:
    """학습률 배율: 처음 ``warmup`` step 동안 0 에서 1 로 직선으로 올리고, 이후 ``total`` 에서 0 이 되도록 코사인으로 줄인다."""

    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, device: str, amp: bool) -> dict:
    model.eval()
    predictions, labels = [], []
    for waveforms, speakers in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(waveforms.to(device, non_blocking=True)).float()
        predictions.append(predicted_speaker(logits).cpu().numpy())
        labels.append(speakers.numpy())
    model.train()
    predicted, truth = np.concatenate(predictions), np.concatenate(labels)
    return {
        "accuracy": float((predicted == truth).mean()),
        "recall_0": float((predicted[truth == 0] == 0).mean()),
        "recall_1": float((predicted[truth == 1] == 1).mean()),
        "n": int(truth.size),
    }


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def held_out_hash(call_ids) -> str:
    return hashlib.sha256("\n".join(sorted(set(map(str, call_ids)))).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 메인: 분할 -> 모델 -> 옵티마이저 -> (이어가기) -> 학습 반복 -> 평가·저장
# ---------------------------------------------------------------------------


REWARM_STEPS = 200


def main() -> int:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    amp = args.amp and device == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import transformers

    print(f"torch {torch.__version__} transformers {transformers.__version__} device={device} amp={amp}", flush=True)
    items = build_items(args.manifest, args.data_root)
    train_index, val_index = make_group_split(items, seed=args.split_seed, test_size=args.test_size)
    train_items, val_items = items.iloc[train_index], items.iloc[val_index]
    print(
        f"train={len(train_items)} utterances / {train_items['call_id'].nunique()} calls, "
        f"held-out={len(val_items)} / {val_items['call_id'].nunique()} calls "
        f"(held-out calls sha256 {held_out_hash(val_items['call_id'])[:16]})",
        flush=True,
    )

    eval_rng = np.random.default_rng(args.split_seed)
    eval_pick = np.sort(eval_rng.choice(len(val_items), size=min(args.eval_utts, len(val_items)), replace=False))
    eval_loader = torch.utils.data.DataLoader(
        UtteranceWindows(val_items.iloc[eval_pick], train=False, seed=args.seed),
        batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=device == "cuda",
    )

    model = WavLMRoleClassifier.from_pretrained(args.pretrained)
    commit = getattr(model.backbone.config, "_commit_hash", None)
    print(
        f"pretrained={args.pretrained} (commit {commit}) parameters: total "
        f"{sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    model.configure_trainable(args.freeze_layers)
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
    model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,} (freeze_layers={args.freeze_layers})", flush=True)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters()) + [model.layer_logits]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay,
    )

    init_step = 0  # best.pt 에서 시작한 실행이면 >0: Adam 의 평균값이 비어 있으므로 학습률을 다시 천천히 올린다

    def rewarm(step: int) -> float:
        return min(1.0, (step - init_step + 1) / REWARM_STEPS) if init_step and step < init_step + REWARM_STEPS else 1.0

    def backbone_lambda(step: int) -> float:
        if step < args.head_only_steps:
            return 0.0
        return lr_factor(step, warmup=args.warmup_steps, total=args.max_steps) * rewarm(step)

    def head_lambda(step: int) -> float:
        return lr_factor(step, warmup=args.warmup_steps, total=args.max_steps) * rewarm(step)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [backbone_lambda, head_lambda])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, best_accuracy, best_step = 0, -1.0, 0
    last_path, best_path = args.out_dir / LAST_NAME, args.out_dir / BEST_NAME
    if args.resume and last_path.exists() and not args.benchmark_steps:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise SystemExit(f"{last_path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if saved["pretrained"] != args.pretrained or saved["split_seed"] != args.split_seed:
            raise SystemExit(f"{last_path} was trained with another --pretrained/--split-seed; refusing to resume")
        model.load_state_dict(saved["model"])
        if saved.get("optimizer") is not None:
            optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, best_accuracy, best_step = saved["step"], saved["best_accuracy"], saved["best_step"]
        print(f"RESUMED from {last_path}: step {step}/{args.max_steps}, best {best_accuracy:.4f}@{best_step}", flush=True)
        del saved

    if step == 0 and args.init_from is not None and args.init_from.exists() and not args.benchmark_steps:
        saved = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise SystemExit(f"{args.init_from} is not a {CHECKPOINT_FORMAT} checkpoint")
        if saved["pretrained"] != args.pretrained or saved["split_seed"] != args.split_seed:
            raise SystemExit(f"{args.init_from} was trained with another --pretrained/--split-seed; refusing to start from it")
        model.load_state_dict(saved["state_dict"])  # float16 가중치가 float32 파라미터로 복사된다
        # 최신 가중치 스냅숏(최고 step 이 아님)에는 정확도가 없다: 첫 평가가 최고 기준이 된다.
        accuracy = saved["internal_accuracy"]
        step, best_accuracy, best_step = saved["step"], -1.0 if accuracy is None else accuracy, saved["step"]
        init_step = step
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [backbone_lambda, head_lambda], last_epoch=step - 1)
        print(
            f"INIT from {args.init_from}: step {step}/{args.max_steps}, best {'none yet' if best_accuracy < 0 else f'{best_accuracy:.4f}'}@{best_step}; "
            f"optimizer starts fresh ({REWARM_STEPS}-step learning-rate re-warm-up)",
            flush=True,
        )
        del saved

    remaining = (args.benchmark_steps or args.max_steps) - step
    if remaining <= 0:
        print(f"nothing to train: step {step} >= {args.max_steps}", flush=True)
        return 0
    train_dataset = UtteranceWindows(train_items, train=True, seed=args.seed + step)
    sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=False, num_samples=remaining * args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + step),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        pin_memory=device == "cuda", drop_last=True, persistent_workers=args.num_workers > 0,
    )

    criterion = torch.nn.CrossEntropyLoss()
    window_loss, window_hits, window_count = 0.0, 0, 0
    started = time.time()
    window_started = started
    for waveforms, speakers in train_loader:
        waveforms, speakers = waveforms.to(device, non_blocking=True), speakers.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(waveforms)
        loss = criterion(logits.float(), speakers)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        window_loss += float(loss.detach()) * len(speakers)
        window_hits += int((predicted_speaker(logits.detach().float()) == speakers).sum())
        window_count += len(speakers)

        if args.benchmark_steps:
            if step == 3:
                torch.cuda.synchronize() if device == "cuda" else None
                window_started = time.time()
            if step == args.benchmark_steps:
                torch.cuda.synchronize() if device == "cuda" else None
                per_step = (time.time() - window_started) / max(1, step - 3)
                peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
                print(f"BENCH sec_per_step={per_step:.3f} peak_gpu_gb={peak:.1f} batch={args.batch_size}", flush=True)
                return 0
            continue

        if step % 50 == 0 or step == args.max_steps:
            now = time.time()
            print(
                f"step {step}/{args.max_steps} loss={window_loss / window_count:.4f} "
                f"train_acc={window_hits / window_count:.4f} lr_backbone={optimizer.param_groups[0]['lr']:.2e} "
                f"sec/step={(now - window_started) / 50:.2f}",
                flush=True,
            )
            window_loss, window_hits, window_count, window_started = 0.0, 0, 0, now

        if step % args.eval_every == 0 or step == args.max_steps:
            result = evaluate(model, eval_loader, device, amp)
            improved = result["accuracy"] > best_accuracy
            if improved:
                best_accuracy, best_step = result["accuracy"], step
                state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
                atomic_save(
                    {
                        "format": CHECKPOINT_FORMAT, "state_dict": state, "backbone_config": model.backbone_config_dict(),
                        "pretrained": args.pretrained, "pretrained_commit": commit, "step": step,
                        "internal_accuracy": best_accuracy, "split_seed": args.split_seed, "test_size": args.test_size,
                        "seed": args.seed, "window_samples": int(48000),
                    },
                    best_path,
                )
            print(
                f"EVAL step {step} internal_acc={result['accuracy']:.4f} recall_0={result['recall_0']:.4f} "
                f"recall_1={result['recall_1']:.4f} (n={result['n']}) best={best_accuracy:.4f}@{best_step}"
                f"{' <- saved best.pt' if improved else ''}",
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            atomic_save(
                {
                    "format": CHECKPOINT_FORMAT, "model": model.state_dict(),
                    "optimizer": None if args.no_save_optimizer else optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "best_accuracy": best_accuracy, "best_step": best_step, "pretrained": args.pretrained,
                    "split_seed": args.split_seed,
                },
                last_path,
            )
            print(f"saved {last_path} at step {step}", flush=True)

    print(
        f"TRAINING DONE: {step} steps in {(time.time() - started) / 60:.1f} min; best internal accuracy "
        f"{best_accuracy:.4f} at step {best_step} -> {best_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
