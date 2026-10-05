#!/usr/bin/env python3
"""Bundle several Mission 2 L2 (CNN) checkpoints into one ensemble checkpoint.

The members must be the same architecture/mel settings/label source, varied only by
--torch-seed (that is the whole point of a seed ensemble -- see M2-ENS-002/003:
the same 3-model soft vote that gave +0.48pp on a 3,000-call pilot gave +0.50pp
on the full 29,200-call raw-label models). scripts/mission2_inference.py takes
exactly one --ckpt_path, so this packs every member's weights into a single
file under "ensemble_model_state_dicts" instead of one "best_model_state_dict"
-- the contest's CLI contract never needs to change.

Usage:
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

    merged_metadata = build_ensemble_metadata(member_metadata)  # also refuses a repeated torch_seed
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
