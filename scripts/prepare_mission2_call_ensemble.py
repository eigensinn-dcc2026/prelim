#!/usr/bin/env python3
"""Write Mission 2 call-id lists for the ensemble gate. Does not train or read audio.

Reads ``call_id`` (plus the index provenance columns) from a Training spectrogram
index, checks ``assert_training_spectrogram_index``, and writes one sorted call-id
file per list plus ``assignments.json`` (counts and sha256).

``--mode gate`` is the frozen 1,500 / 6,000 / 1,200 split. ``--mode full`` shuffles
every Training call id on its own and writes the 3-fold member lists. Torch seed
for every full-fold member is 1.

The commands printed at the end are for a later Colab run. This script does not
run them and does not score official Validation.

Usage:
    python scripts/prepare_mission2_call_ensemble.py \\
        --index data/m2_spectrogram_index_packed.parquet \\
        --out-dir outputs/mission2_call_ensemble/gate \\
        --mode gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd  # noqa: E402

from dcc_data.mission2_call_ensemble import (  # noqa: E402
    FULL_FOLD_TORCH_SEED,
    GATE_MCNEMAR_ALPHA,
    GATE_MIN_ACCURACY_GAIN,
    assert_not_official_validation_path,
    full_fold_assignments,
    gate_assignments,
)
from dcc_data.mission2_spectrogram import INDEX_PROVENANCE_COLUMNS, assert_training_spectrogram_index  # noqa: E402

GATE_ROLES = {
    "arm_a_seed1": {"train": "arm_a_train", "val": "arm_a_val", "torch_seed": 1},
    "arm_a_seed2": {"train": "arm_a_train", "val": "arm_a_val", "torch_seed": 2},
    "b0": {"train": "b0_train", "val": "b0_val", "torch_seed": 1},
    "b1": {"train": "b1_train", "val": "b1_val", "torch_seed": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Training spectrogram index parquet")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gate", "full"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_sorted_ids(path: Path, call_ids: list[str]) -> dict:
    ordered = sorted(call_ids)
    text = "".join(f"{call_id}\n" for call_id in ordered)
    path.write_text(text, encoding="utf-8")
    return {"count": len(ordered), "sha256": _sha256_text(text), "file": path.name}


def _read_training_call_ids(index_path: Path) -> list[str]:
    assert_not_official_validation_path(index_path, label="spectrogram index")
    columns = ["call_id", *INDEX_PROVENANCE_COLUMNS]
    index_df = pd.read_parquet(index_path, columns=columns)
    assert_training_spectrogram_index(index_df)
    return index_df["call_id"].astype(str).tolist()


def _gate_payload(call_ids: list[str], seed: int) -> tuple[dict[str, list[str]], dict]:
    lists = gate_assignments(call_ids, seed=seed)
    extra = {
        "roles": GATE_ROLES,
        "score_on": "holdout",
        "gate": {"min_accuracy_gain": GATE_MIN_ACCURACY_GAIN, "mcnemar_p_less_than": GATE_MCNEMAR_ALPHA},
    }
    return lists, extra


def _full_payload(call_ids: list[str], seed: int) -> tuple[dict[str, list[str]], dict]:
    assigned = full_fold_assignments(call_ids, seed=seed, n_folds=3)
    lists = {}
    roles = {}
    for member in assigned["members"]:
        fold = member["fold"]
        train_name = f"member_{fold}_train"
        val_name = f"member_{fold}_val"
        lists[train_name] = member["train"]
        lists[val_name] = member["val"]
        roles[f"member_{fold}"] = {
            "train": train_name,
            "val": val_name,
            "torch_seed": FULL_FOLD_TORCH_SEED,
            "val_fold": fold,
        }
    extra = {"n_folds": 3, "torch_seed": FULL_FOLD_TORCH_SEED, "roles": roles}
    return lists, extra


def _colab_commands(args: argparse.Namespace) -> str:
    index = args.index
    out = args.out_dir
    packed_index = "data/m2_spectrogram_index_packed.parquet"
    packed_specs = "data/m2_spectrograms_packed.npy"
    common = (
        f"python scripts/train_mission2_cnn.py --index {packed_index} "
        f"--packed-spectrograms {packed_specs} --model large --device cuda "
        f"--amp --pad-to-multiple 8 --cudnn-benchmark --num-workers 2 "
        f"--epochs 1 --early-stop-patience 5 --seed {args.seed}"
    )
    if args.mode == "gate":
        lines = [
            "Prepare (no training, no audio):",
            f"python scripts/prepare_mission2_call_ensemble.py --index {index} --out-dir {out} --mode gate",
            "",
            "Arm A, same call ids, torch seeds 1 and 2. First epoch omits --resume; later epochs add it.",
            f"{common} --torch-seed 1 --train-call-ids {out / 'arm_a_train.txt'} "
            f"--val-call-ids {out / 'arm_a_val.txt'} --model-output checkpoints/mission2/call_ensemble/arm_a_torch1.pt",
            f"{common} --torch-seed 2 --train-call-ids {out / 'arm_a_train.txt'} "
            f"--val-call-ids {out / 'arm_a_val.txt'} --model-output checkpoints/mission2/call_ensemble/arm_a_torch2.pt",
            "",
            "Arm B, both torch seed 1, different train calls.",
            f"{common} --torch-seed 1 --train-call-ids {out / 'b0_train.txt'} "
            f"--val-call-ids {out / 'b0_val.txt'} --model-output checkpoints/mission2/call_ensemble/b0_torch1.pt",
            f"{common} --torch-seed 1 --train-call-ids {out / 'b1_train.txt'} "
            f"--val-call-ids {out / 'b1_val.txt'} --model-output checkpoints/mission2/call_ensemble/b1_torch1.pt",
            "",
            "Continue a member for one more epoch (same flags, plus --resume):",
            f"{common} --resume --torch-seed 1 --train-call-ids {out / 'arm_a_train.txt'} "
            f"--val-call-ids {out / 'arm_a_val.txt'} --model-output checkpoints/mission2/call_ensemble/arm_a_torch1.pt",
            "",
            "Score the holdout only, batch size 1, each utterance padded on its own to 300 frames.",
            "python scripts/score_mission2_call_ensemble.py "
            f"--index {packed_index} --packed-spectrograms {packed_specs} "
            f"--holdout-call-ids {out / 'holdout.txt'} "
            "--group a=checkpoints/mission2/call_ensemble/arm_a_torch1.pt,checkpoints/mission2/call_ensemble/arm_a_torch2.pt "
            "--group b=checkpoints/mission2/call_ensemble/b0_torch1.pt,checkpoints/mission2/call_ensemble/b1_torch1.pt "
            "--device cuda "
            f"--metrics {out / 'metrics.json'} --save-probs {out / 'probs'}",
            "",
            "Re-aggregate saved probabilities without loading checkpoints:",
            "python scripts/score_mission2_call_ensemble.py "
            f"--index {packed_index} --holdout-call-ids {out / 'holdout.txt'} "
            "--group a=checkpoints/mission2/call_ensemble/arm_a_torch1.pt,checkpoints/mission2/call_ensemble/arm_a_torch2.pt "
            "--group b=checkpoints/mission2/call_ensemble/b0_torch1.pt,checkpoints/mission2/call_ensemble/b1_torch1.pt "
            f"--probs-dir {out / 'probs'} --metrics {out / 'metrics.json'}",
        ]
        return "\n".join(lines)
    lines = [
        "Prepare the full 3-fold lists (no training, no audio):",
        f"python scripts/prepare_mission2_call_ensemble.py --index {index} --out-dir {out} --mode full",
        "",
        "Each member uses torch seed 1, early-stops on its own fold, and trains on the other folds.",
    ]
    for fold in range(3):
        lines.append(
            f"{common} --torch-seed 1 --train-call-ids {out / f'member_{fold}_train.txt'} "
            f"--val-call-ids {out / f'member_{fold}_val.txt'} "
            f"--model-output checkpoints/mission2/call_ensemble/full_member{fold}_torch1.pt"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    assert_not_official_validation_path(args.out_dir, label="output directory")
    call_ids = _read_training_call_ids(args.index)
    if args.mode == "gate":
        lists, extra = _gate_payload(call_ids, args.seed)
    else:
        lists, extra = _full_payload(call_ids, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {name: _write_sorted_ids(args.out_dir / f"{name}.txt", ids) for name, ids in lists.items()}
    payload = {
        "mode": args.mode,
        "seed": args.seed,
        "n_index_calls": len(set(call_ids)),
        "lists": written,
        **extra,
    }
    (args.out_dir / "assignments.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(written)} call-id lists to {args.out_dir}")
    for name, record in written.items():
        print(f"  {name}: {record['count']}  sha256={record['sha256']}")
    print()
    print(_colab_commands(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
