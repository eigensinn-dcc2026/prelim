#!/usr/bin/env python3
"""Does picking the best epoch with batch-padded validation hold up under the submission script's padding?

scripts/train_mission2_cnn.py scores every epoch on the Training-internal held-out calls with
``collate_batch``, which pads each batch to its own longest utterance, and keeps the best epoch. The submission
script (scripts/mission2_inference.py) pads each utterance, on its own, up to ``L2_MIN_FRAMES`` frames. The
network averages over the whole time axis, so the padding changes what it sees: the epoch picked under one kind
of padding is not guaranteed to be the best under the other.

This script measures how much that matters for finished checkpoints. A training checkpoint stores two sets of
weights -- the best epoch's and the last epoch's -- so it compares those two, on the same held-out utterances, under
both paddings:

* ``batch_padding``: what the training-time selection saw (batches of ``--batch-size`` consecutive held-out
  utterances, padded to the batch's longest);
* ``inference_padding``: what the submission script does (each utterance padded on its own to ``L2_MIN_FRAMES``;
  utterances that share a padded length are run in one batch, which gives the same predictions as running them one
  at a time -- checked here on a random subset).

It reports only; it never changes a checkpoint. It uses Training data alone (the internal held-out split rebuilt
exactly as the trainer built it) -- never the official Validation folder. Spectrograms are rebuilt from the audio
with the same code as the spectrogram builder and the submission script, and every rebuilt spectrogram's frame
count and speaker label must match the index row, or the run stops.

Usage:
    python scripts/check_mission2_selection_padding.py \\
        --index m2_spectrogram_index_full_provided_labels.parquet --data-root C:/dcc_data_local_full \\
        --ckpt-path m2_cnn_large_raw_ts1.pt --ckpt-path m2_cnn_large_raw_ts2.pt --ckpt-path m2_cnn_large_raw_ts3.pt \\
        --output reports/mission2/m2_l2_selection_padding_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dcc_data.manifest import build_manifest  # noqa: E402
from dcc_data.mission2_checkpoint import require_l2_inference_config, resolve_l2_checkpoint_metadata  # noqa: E402
from dcc_data.mission2_cnn import SpeakerCNNLarge, collate_batch, predicted_classes  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    N_FFT,
    build_utterance_records,
    read_call_waveform,
    select_seoul_training_rows,
    slice_and_pad,
    utterance_frame_range,
)
from dcc_data.mission2_spectrogram import assert_training_spectrogram_index, mel_spectrogram  # noqa: E402
from dcc_data.mission2_train import make_group_split  # noqa: E402
from mission2_inference import L2_MIN_FRAMES, pad_spectrogram_to_min_frames  # noqa: E402

STATES = ("best", "last")
MODES = ("batch_padding", "inference_padding")
EQUIVALENCE_SAMPLE = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="the provided-labels spectrogram index (Training)")
    parser.add_argument("--data-root", type=Path, required=True, help="folder holding Training/ (the audio and label JSON)")
    parser.add_argument("--ckpt-path", type=Path, action="append", required=True, dest="ckpt_paths",
                        help="a training checkpoint (needs best and last weights); pass once per seed")
    parser.add_argument("--n-calls", type=int, default=1000, help="held-out calls to measure on (0 = all of them)")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64, help="the trainer's default; used for batch_padding")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample_held_out_rows(index_df: pd.DataFrame, *, split_seed: int, test_size: float, n_calls: int, sample_seed: int):
    """The trainer's held-out rows (same split call), reduced to ``n_calls`` calls, in their original order."""

    _, val_idx = make_group_split(index_df, seed=split_seed, test_size=test_size)
    held_out = index_df.iloc[val_idx]
    call_ids = held_out["call_id"].drop_duplicates().to_numpy()
    if n_calls and n_calls < len(call_ids):
        chosen = set(np.random.default_rng(sample_seed).choice(call_ids, size=n_calls, replace=False))
        held_out = held_out[held_out["call_id"].isin(chosen)]
    return held_out.reset_index(drop=True), len(call_ids)


def rebuild_spectrograms(sample_df: pd.DataFrame, data_root: Path) -> list[np.ndarray]:
    """One spectrogram per row of ``sample_df``, rebuilt from the audio the way the builder and the submission do."""

    candidates: dict[str, list] = {}
    for row in select_seoul_training_rows(build_manifest(data_root)):
        for key in {row.stem, row.stem.rsplit("_", 1)[0]}:  # files are named <call id>_<date>
            candidates.setdefault(key, []).append(row)

    positions_by_call: dict[str, dict[int, int]] = {}
    for position, (call_id, utt_idx) in enumerate(zip(sample_df["call_id"], sample_df["utt_idx"])):
        positions_by_call.setdefault(call_id, {})[int(utt_idx)] = position
    spectrograms: list[np.ndarray | None] = [None] * len(sample_df)
    mismatches = []

    for call_id, positions in positions_by_call.items():
        rows = candidates.get(call_id, [])
        if len(rows) != 1:
            raise ValueError(f"call {call_id}: expected exactly one Training audio/label pair, found {len(rows)}")
        row = rows[0]
        waveform, sample_rate = read_call_waveform(data_root / row.wav_path)
        for record in build_utterance_records(row, data_root):
            position = positions.get(record.utt_idx)
            if position is None or record.call_id != call_id:
                continue
            start_frame, end_frame = utterance_frame_range(record.start_at, record.end_at, sample_rate, waveform.shape[0])
            spectrogram = mel_spectrogram(slice_and_pad(waveform, start_frame, end_frame, N_FFT), sample_rate)
            expected = sample_df.iloc[position]
            if spectrogram.shape[1] != int(expected["n_frames"]) or record.speaker != int(expected["speaker"]):
                mismatches.append((call_id, record.utt_idx))
            spectrograms[position] = spectrogram

    missing = [i for i, spectrogram in enumerate(spectrograms) if spectrogram is None]
    if missing or mismatches:
        raise ValueError(
            f"the rebuilt inputs do not match the index: {len(missing)} row(s) not rebuilt, {len(mismatches)} with a "
            f"different frame count or speaker (first: {mismatches[:3]}) -- refusing to measure on other inputs"
        )
    return spectrograms


def _logits_to_classes(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    return predicted_classes(model(batch)).cpu().numpy()


@torch.no_grad()
def predict_batch_padding(model: torch.nn.Module, spectrograms: list[np.ndarray], batch_size: int) -> np.ndarray:
    """What the trainer's validation loop does: consecutive batches, each padded to its own longest utterance."""

    predictions = []
    for start in range(0, len(spectrograms), batch_size):
        chunk = [(torch.from_numpy(s), 0) for s in spectrograms[start:start + batch_size]]
        padded, _ = collate_batch(chunk)
        predictions.append(_logits_to_classes(model, padded))
    return np.concatenate(predictions)


@torch.no_grad()
def predict_inference_padding(model: torch.nn.Module, spectrograms: list[np.ndarray], chunk_size: int = 256) -> np.ndarray:
    """What the submission script does: each utterance padded on its own; equal padded lengths share a batch."""

    padded = [pad_spectrogram_to_min_frames(s) for s in spectrograms]
    by_length: dict[int, list[int]] = {}
    for position, spectrogram in enumerate(padded):
        by_length.setdefault(spectrogram.shape[1], []).append(position)
    predictions = np.zeros(len(spectrograms), dtype=np.int64)
    for positions in by_length.values():
        for start in range(0, len(positions), chunk_size):
            members = positions[start:start + chunk_size]
            batch = torch.from_numpy(np.stack([padded[p] for p in members])).unsqueeze(1)
            predictions[members] = _logits_to_classes(model, batch)
    return predictions


@torch.no_grad()
def predict_one_at_a_time(model: torch.nn.Module, spectrograms: list[np.ndarray], positions: list[int]) -> np.ndarray:
    """The submission script's own way: one utterance per forward pass."""

    return np.array([
        int(predicted_classes(model(torch.from_numpy(pad_spectrogram_to_min_frames(spectrograms[p])).unsqueeze(0).unsqueeze(0)))[0])
        for p in positions
    ])


def accuracy_entry(predictions: np.ndarray, labels: np.ndarray) -> dict:
    correct = int((predictions == labels).sum())
    return {"correct": correct, "n": int(len(labels)), "accuracy": correct / len(labels)}


def measure_checkpoint(path: Path, spectrograms: list[np.ndarray], labels: np.ndarray, batch_size: int) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    metadata, _ = resolve_l2_checkpoint_metadata(checkpoint, path)
    require_l2_inference_config(metadata)
    for key in ("best_model_state_dict", "last_model_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"{path} has no {key!r}: this check needs a training checkpoint (best and last weights)")

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for state, key in (("best", "best_model_state_dict"), ("last", "last_model_state_dict")):
        model = SpeakerCNNLarge()
        model.load_state_dict(checkpoint[key])
        model.eval()
        predictions[state] = {
            "batch_padding": predict_batch_padding(model, spectrograms, batch_size),
            "inference_padding": predict_inference_padding(model, spectrograms),
        }
        if state == "best":
            best_model = model

    result = {
        "torch_seed": metadata["torch_seed"],
        "label_source": metadata["label_source"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "total_epochs_trained": int(checkpoint["total_epochs_trained"]),
        "reported_best_val_acc_full_held_out_set": float(checkpoint["best_val_acc"]),
        "states": {state: {mode: accuracy_entry(predictions[state][mode], labels) for mode in MODES} for state in STATES},
        "paired_best_vs_last": {},
    }
    for mode in MODES:
        best_right = predictions["best"][mode] == labels
        last_right = predictions["last"][mode] == labels
        result["paired_best_vs_last"][mode] = {
            "best_right_last_wrong": int((best_right & ~last_right).sum()),
            "best_wrong_last_right": int((~best_right & last_right).sum()),
        }
    return result, {"model": best_model, "inference_predictions": predictions["best"]["inference_padding"],
                    "metadata": metadata}


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    index_df = pd.read_parquet(args.index)
    assert_training_spectrogram_index(index_df)  # Training only, with the provided labels

    first = torch.load(args.ckpt_paths[0], map_location="cpu")
    first_metadata, _ = resolve_l2_checkpoint_metadata(first, args.ckpt_paths[0])
    split_seed, test_size = first_metadata["split_seed"], first_metadata["test_size"]
    sample_df, n_held_out_calls = sample_held_out_rows(
        index_df, split_seed=split_seed, test_size=test_size, n_calls=args.n_calls, sample_seed=args.sample_seed
    )
    labels = sample_df["speaker"].to_numpy(dtype=np.int64)
    print(f"held-out calls: {n_held_out_calls} total, measuring on {sample_df['call_id'].nunique()} "
          f"({len(sample_df)} utterances)", flush=True)

    spectrograms = rebuild_spectrograms(sample_df, args.data_root)
    print(f"rebuilt {len(spectrograms)} spectrograms from audio; frame counts and speakers match the index "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    members, equivalence = [], None
    for path in args.ckpt_paths:
        metadata_check = torch.load(path, map_location="cpu")
        metadata, _ = resolve_l2_checkpoint_metadata(metadata_check, path)
        if (metadata["split_seed"], metadata["test_size"]) != (split_seed, test_size):
            raise ValueError(f"{path} was trained with another train/held-out split than {args.ckpt_paths[0]}")
        del metadata_check
        member, extras = measure_checkpoint(path, spectrograms, labels, args.batch_size)
        member["checkpoint"] = path.name
        members.append(member)
        print(f"  {path.name}: " + "  ".join(
            f"{state}/{mode}={member['states'][state][mode]['accuracy']:.4f}" for state in STATES for mode in MODES
        ), flush=True)
        if equivalence is None:  # once: batching equal-length utterances == one utterance per forward pass
            rng = np.random.default_rng(args.sample_seed)
            positions = sorted(rng.choice(len(spectrograms), size=min(EQUIVALENCE_SAMPLE, len(spectrograms)), replace=False).tolist())
            single = predict_one_at_a_time(extras["model"], spectrograms, positions)
            batched = extras["inference_predictions"][positions]
            equivalence = {"n_checked": len(positions), "n_prediction_differences": int((single != batched).sum()),
                           "checkpoint": path.name, "state": "best"}

    summary = {
        "purpose": "does selecting the best epoch with batch-padded validation agree with the submission script's padding?",
        "data_used": "Training only: the trainer's internal held-out split, rebuilt from the audio. Official Validation not used.",
        "reports_only": "no checkpoint was changed",
        "index": args.index.name,
        "split_seed": split_seed,
        "test_size": test_size,
        "n_held_out_calls_total": n_held_out_calls,
        "n_calls_measured": int(sample_df["call_id"].nunique()),
        "n_utterances_measured": int(len(sample_df)),
        "sample_seed": args.sample_seed,
        "batch_size_for_batch_padding": args.batch_size,
        "inference_min_frames": L2_MIN_FRAMES,
        "rebuilt_inputs_match_index": True,
        "batched_vs_one_at_a_time": equivalence,
        "checkpoints": members,
        "seconds": round(time.monotonic() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
