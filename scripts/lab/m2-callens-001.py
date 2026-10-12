#!/usr/bin/env python3
"""M2-CALLENS-001: call-subset ensemble gate. Resume-safe. No official Validation.

Run from the repository root. One Colab invocation trains whichever members are
not finished, one epoch at a time, then scores the 1,500-call holdout.

    python scripts/lab/m2-callens-001.py --out "$LAB_OUT"

--smoke only writes the call-id lists and exits before training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXP_ID = "M2-CALLENS-001"
MAX_EPOCHS = 30
PATIENCE = 5
REPO = Path(__file__).resolve().parents[2]

MEMBERS = (
    ("arm_a_torch1", "arm_a_train.txt", "arm_a_val.txt", 1),
    ("arm_a_torch2", "arm_a_train.txt", "arm_a_val.txt", 2),
    ("b0_torch1", "b0_train.txt", "b0_val.txt", 1),
    ("b1_torch1", "b1_train.txt", "b1_val.txt", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="defaults to $LAB_OUT")
    parser.add_argument("--smoke", action="store_true", help="write call-id lists only")
    args = parser.parse_args()
    if args.out is None:
        raw = os.environ.get("LAB_OUT")
        if not raw:
            parser.error("--out or $LAB_OUT is required")
        args.out = Path(raw)
    return args


def packed_cache() -> tuple[Path, Path]:
    run_root = os.environ.get("DCC_RUN_ROOT")
    if not run_root:
        raise SystemExit("DCC_RUN_ROOT is not set")
    root = Path(run_root) / "m2_l2_spec"
    index = root / "m2_spectrogram_index_packed.parquet"
    specs = root / "m2_spectrograms_packed.npy"
    if not index.is_file() or not specs.is_file():
        raise SystemExit(f"packed Training cache missing under {root}")
    return index, specs


def run_streaming(command: list[str]) -> str:
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    code = process.wait()
    if code != 0:
        raise SystemExit(code)
    return "".join(chunks)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_lists(index: Path, gate_dir: Path) -> None:
    if (gate_dir / "assignments.json").is_file() and (gate_dir / "holdout.txt").is_file():
        print(f"reusing call-id lists in {gate_dir}", flush=True)
        return
    run_streaming(
        [
            sys.executable,
            "scripts/prepare_mission2_call_ensemble.py",
            "--index",
            str(index),
            "--out-dir",
            str(gate_dir),
            "--mode",
            "gate",
        ]
    )


def train_member(name: str, train_ids: Path, val_ids: Path, torch_seed: int, index: Path, specs: Path, ckpt: Path) -> None:
    done = ckpt.with_suffix(".done.json")
    if done.is_file():
        print(f"{name} already finished", flush=True)
        return
    for _ in range(MAX_EPOCHS):
        command = [
            sys.executable,
            "scripts/train_mission2_cnn.py",
            "--index",
            str(index),
            "--packed-spectrograms",
            str(specs),
            "--model",
            "large",
            "--device",
            "cuda",
            "--amp",
            "--pad-to-multiple",
            "8",
            "--cudnn-benchmark",
            "--num-workers",
            "2",
            "--epochs",
            "1",
            "--early-stop-patience",
            str(PATIENCE),
            "--seed",
            "42",
            "--torch-seed",
            str(torch_seed),
            "--train-call-ids",
            str(train_ids),
            "--val-call-ids",
            str(val_ids),
            "--model-output",
            str(ckpt),
        ]
        if ckpt.is_file():
            command.append("--resume")
        text = run_streaming(command)
        if "not training" in text or "EARLY STOPPED after epoch" in text:
            write_json(done, {"member": name, "reason": "early_stop", "torch_seed": torch_seed})
            return
    write_json(done, {"member": name, "reason": "max_epochs", "torch_seed": torch_seed, "max_epochs": MAX_EPOCHS})


def score(index: Path, specs: Path, gate_dir: Path, lab_out: Path) -> None:
    metrics_path = lab_out / "metrics.json"
    if metrics_path.is_file():
        print(f"reusing {metrics_path}", flush=True)
        return
    checkpoints = lab_out / "checkpoints"
    score_path = lab_out / "score_metrics.json"
    run_streaming(
        [
            sys.executable,
            "scripts/score_mission2_call_ensemble.py",
            "--index",
            str(index),
            "--packed-spectrograms",
            str(specs),
            "--holdout-call-ids",
            str(gate_dir / "holdout.txt"),
            "--group",
            "a="
            + ",".join(
                str(checkpoints / name)
                for name in ("arm_a_torch1.pt", "arm_a_torch2.pt")
            ),
            "--group",
            "b=" + ",".join(str(checkpoints / name) for name in ("b0_torch1.pt", "b1_torch1.pt")),
            "--device",
            "cuda",
            "--metrics",
            str(score_path),
            "--save-probs",
            str(lab_out / "probs"),
        ]
    )
    scored = json.loads(score_path.read_text(encoding="utf-8"))
    write_json(
        metrics_path,
        {
            "exp_id": EXP_ID,
            "primary_metric": {
                "name": "accuracy_gain_b_minus_a",
                "value": scored["acc_gain_b_minus_a"],
                "split": "internal_holdout",
            },
            "metrics": scored,
            "seed": 42,
            "train_rows": None,
            "eval_rows": scored.get("n_utterances"),
            "official_val": None,
            "ckpt": "checkpoints/",
            "notes": (
                "Pass iff accuracy gain >= 0.003 and exact McNemar p < 0.05. "
                "Official Validation was not opened."
            ),
        },
    )


def main() -> int:
    args = parse_args()
    index, specs = packed_cache()
    gate_dir = args.out / "gate"
    prepare_lists(index, gate_dir)
    if args.smoke:
        print("smoke: call-id lists written, training skipped", flush=True)
        return 0
    if not __import__("torch").cuda.is_available():
        raise SystemExit("cuda is required for training and scoring")
    for name, train_name, val_name, torch_seed in MEMBERS:
        train_member(
            name,
            gate_dir / train_name,
            gate_dir / val_name,
            torch_seed,
            index,
            specs,
            args.out / "checkpoints" / f"{name}.pt",
        )
    score(index, specs, gate_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
