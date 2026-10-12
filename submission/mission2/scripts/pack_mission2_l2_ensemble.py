#!/usr/bin/env python3
"""Mission 2 L2(CNN) 체크포인트 여러 개를 앙상블 체크포인트 하나로 묶는다.

멤버는 구조, 멜 설정, 라벨 출처가 같아야 하고 --torch-seed만 달라야 한다(시드 앙상블의 핵심이다.
M2-ENS-002/003 참고: 3,000통화 파일럿에서 +0.48%p를 준 같은 3개 모델 소프트 보팅이 전체 29,200통화
원본 라벨 모델에서는 +0.50%p를 주었다). scripts/mission2_inference.py는 --ckpt_path를 정확히 하나만
받으므로, 모든 멤버의 가중치를 "best_model_state_dict" 하나 대신 "ensemble_model_state_dicts"
아래 한 파일에 담는다. 그래서 대회의 실행 명령 형식은 바꿀 필요가 없다.

사용법:
    python scripts/pack_mission2_l2_ensemble.py \\
        --ckpt-path checkpoints/mission2/m2_cnn_large_raw_ts1.pt \\
        --ckpt-path checkpoints/mission2/m2_cnn_large_raw_ts2.pt \\
        --ckpt-path checkpoints/mission2/m2_cnn_large_raw_ts3.pt \\
        --output checkpoints/mission2/m2_cnn_large_raw_ensemble.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from dcc_data.mission2_checkpoint import (  # noqa: E402
    ENSEMBLE_STATE_DICTS_KEY,
    build_ensemble_metadata,
    resolve_l2_checkpoint_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ckpt-path", type=Path, action="append", required=True, dest="ckpt_paths",
        help="pass once per member checkpoint (at least 2)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ckpt_paths) < 2:
        parser.error(f"need at least 2 --ckpt-path values to build an ensemble, got {len(args.ckpt_paths)}")
    return args


def main() -> int:
    args = parse_args()

    member_metadata = []
    state_dicts = []
    best_val_accs = []
    for ckpt_path in args.ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        metadata, source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
        print(f"{ckpt_path}: torch_seed={metadata['torch_seed']}  best_val_acc={checkpoint.get('best_val_acc')}"
              f"  ({source})")
        member_metadata.append(metadata)
        state_dicts.append(checkpoint["best_model_state_dict"])
        best_val_accs.append(checkpoint.get("best_val_acc"))

    merged_metadata = build_ensemble_metadata(member_metadata)  # 같은 torch_seed가 반복되면 거부한다
    for i in range(len(state_dicts)):
        for j in range(i + 1, len(state_dicts)):
            if all(torch.equal(state_dicts[i][name], state_dicts[j][name]) for name in state_dicts[i]):
                raise SystemExit(
                    f"{args.ckpt_paths[i]} and {args.ckpt_paths[j]} hold identical weights -- "
                    "refusing to pack a copy as a second member"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            ENSEMBLE_STATE_DICTS_KEY: state_dicts,
            "metadata": merged_metadata,
            "ensemble_member_paths": [str(p) for p in args.ckpt_paths],
            "ensemble_member_best_val_acc": best_val_accs,
        },
        args.output,
    )
    print(f"\nwrote ensemble of {len(args.ckpt_paths)} member(s) -> {args.output}")
    print(f"ensemble_member_torch_seeds: {merged_metadata['ensemble_member_torch_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
