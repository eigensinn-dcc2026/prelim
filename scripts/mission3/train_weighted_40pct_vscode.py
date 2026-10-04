"""VS Code에서 실행하는 Mission 3 Training 40% 완화 가중치 학습.

기본 설정:
- 전체 Training 중 seed 42로 40% 선택
- 선택한 데이터의 90% 학습, 10% 내부 검증
- 물리 batch 4 x gradient accumulation 4 = 실효 batch 16
- 양성 가중치 scale 0.25, 최대 3.0
- 3 epochs, 내부 검증 Macro F1 최고 체크포인트 저장

VS Code에서 이 파일을 열고 오른쪽 위 Run Python File을 누르거나:
  .venv/Scripts/python.exe train_weighted_40pct_vscode.py

설정만 확인:
  .venv/Scripts/python.exe train_weighted_40pct_vscode.py --dry_run
"""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

DATA_ROOT = None  # None이면 config.py 또는 DCC_DATA_ROOT 환경 변수 사용
SAMPLE_RATIO = 0.40
EPOCHS = 3
PHYSICAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
SEED = 42
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
LONG_TURN_OVERLAP = 64
POS_WEIGHT_SCALE = 0.25
POS_WEIGHT_MAX = 3.0
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "mission3_kcbert_turn_chunked_40pct_weight025_effbatch16_3epochs.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VS Code용 Mission3 Training 40% 완화 가중치 학습"
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--sample_ratio", type=float, default=SAMPLE_RATIO)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PHYSICAL_BATCH_SIZE)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--pos_weight_scale", type=float, default=POS_WEIGHT_SCALE)
    parser.add_argument("--pos_weight_max", type=float, default=POS_WEIGHT_MAX)
    parser.add_argument("--checkpoint_output", type=Path, default=CHECKPOINT)
    parser.add_argument("--dry_run", action="store_true", help="학습하지 않고 설정만 확인")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch_size, gradient_accumulation_steps는 1 이상이어야 합니다.")
    if not math.isfinite(args.sample_ratio) or not 0 < args.sample_ratio <= 1:
        parser.error("sample_ratio는 유한한 0보다 크고 1 이하인 값이어야 합니다.")
    if not math.isfinite(args.pos_weight_scale) or args.pos_weight_scale < 0:
        parser.error("pos_weight_scale은 유한한 0 이상의 값이어야 합니다.")
    if args.pos_weight_max is not None and (
        not math.isfinite(args.pos_weight_max) or args.pos_weight_max < 1
    ):
        parser.error("pos_weight_max는 유한한 1 이상의 값이어야 합니다.")

    python = VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable)
    checkpoint = args.checkpoint_output.resolve()
    command = [
        str(python),
        str(ROOT / "train.py"),
        "--sample_ratio", str(args.sample_ratio),
        "--seed", str(SEED),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--lr", str(LEARNING_RATE),
        "--max_length", str(MAX_LENGTH),
        "--chunk_overlap", str(LONG_TURN_OVERLAP),
        "--pos_weight_scale", str(args.pos_weight_scale),
        "--pos_weight_max", str(args.pos_weight_max),
        "--checkpoint_output", str(checkpoint),
    ]
    if args.data_root is not None:
        command.extend(["--data_root", str(args.data_root.resolve())])

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Python: {python}", flush=True)
    print(
        f"Training {args.sample_ratio:.0%} 선택 / {args.epochs} epochs / seed {SEED}",
        flush=True,
    )
    print("선택한 데이터의 90%는 학습, 10%는 내부 검증입니다.", flush=True)
    print(
        f"물리 batch {args.batch_size} x 누적 {args.gradient_accumulation_steps} "
        f"= 실효 batch {effective_batch_size}",
        flush=True,
    )
    print(
        f"양성 가중치: scale={args.pos_weight_scale}, max={args.pos_weight_max}",
        flush=True,
    )
    print(f"체크포인트: {checkpoint}", flush=True)
    if args.dry_run:
        print("실행 명령:", subprocess.list2cmdline(command), flush=True)
        return

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"40% 가중치 학습 완료: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
