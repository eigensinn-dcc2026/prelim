#!/usr/bin/env python3
"""Score the Mission 2 call-ensemble gate on holdout calls only.

Each utterance is padded with its own spectrogram, using SPEC_FLOOR_DB, up to
L2_MIN_FRAMES (300). The forward pass uses batch size 1, so collate never pads
one utterance out to a neighbour's length. A group prediction is class 1 when
the members' mean softmax P(class=1) is at least 0.5.

``--group`` may be repeated: ``name=ckpt1,ckpt2``. Groups named ``a`` and ``b``
are the frozen gate (B accuracy - A accuracy >= 0.003 and exact McNemar p < 0.05).
``--probs-dir`` skips inference and aggregates probabilities already saved.
An official Validation index or path is refused. This script does not copy the
submission tree and does not train.

Usage:
    python scripts/score_mission2_call_ensemble.py \\
        --index data/m2_spectrogram_index_packed.parquet \\
        --packed-spectrograms data/m2_spectrograms_packed.npy \\
        --holdout-call-ids outputs/mission2_call_ensemble/gate/holdout.txt \\
        --group a=checkpoints/mission2/call_ensemble/arm_a_torch1.pt,checkpoints/mission2/call_ensemble/arm_a_torch2.pt \\
        --group b=checkpoints/mission2/call_ensemble/b0_torch1.pt,checkpoints/mission2/call_ensemble/b1_torch1.pt \\
        --device cuda \\
        --metrics outputs/mission2_call_ensemble/gate/metrics.json \\
        --save-probs outputs/mission2_call_ensemble/gate/probs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for _root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dcc_data.mission2_call_ensemble import (  # noqa: E402
    assert_not_official_validation_path,
    build_gate_metrics,
    load_call_id_list,
)
from dcc_data.mission2_spectrogram import INDEX_PROVENANCE_COLUMNS, SPEC_FLOOR_DB, assert_training_spectrogram_index  # noqa: E402
from mission2_inference import L2_MIN_FRAMES  # noqa: E402

INDEX_COLUMNS = ("call_id", "utt_idx", "speaker", "npy_path", "frame_offset", "n_frames", *INDEX_PROVENANCE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--holdout-call-ids", type=Path, required=True)
    parser.add_argument("--group", action="append", required=True, dest="groups", help="name=ckpt1,ckpt2")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--packed-spectrograms", type=Path, default=None)
    parser.add_argument("--spectrogram-dir", type=Path, default=None)
    parser.add_argument("--probs-dir", type=Path, default=None, help="aggregate these saved probabilities; do not infer")
    parser.add_argument("--save-probs", type=Path, default=None, help="where to write per-model probabilities")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.probs_dir is not None and args.save_probs is not None:
        parser.error("pass --probs-dir or --save-probs, not both")
    if args.probs_dir is None and (args.packed_spectrograms is None) == (args.spectrogram_dir is None):
        parser.error("inference needs exactly one of --packed-spectrograms or --spectrogram-dir")
    return args


def _parse_groups(raw_groups: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for raw in raw_groups:
        if "=" not in raw:
            raise SystemExit(f"--group must look like name=ckpt1,ckpt2, got {raw!r}")
        name, rest = raw.split("=", 1)
        name = name.strip()
        paths = [Path(part.strip()) for part in rest.split(",") if part.strip()]
        if not name or not paths:
            raise SystemExit(f"--group must name at least one checkpoint, got {raw!r}")
        if name in groups:
            raise SystemExit(f"duplicate --group name {name!r}")
        groups[name] = paths
    return groups


def _reject_paths(paths) -> None:
    for label, path in paths:
        if path is not None:
            assert_not_official_validation_path(path, label=label)


def _prob_file(probs_dir: Path, ckpt_path: Path) -> Path:
    key = hashlib.sha256(str(ckpt_path).encode("utf-8")).hexdigest()[:16]
    return probs_dir / f"{ckpt_path.stem}__{key}.npz"


def _holdout_frame(index_path: Path, holdout_ids: list[str]) -> pd.DataFrame:
    assert_not_official_validation_path(index_path, label="spectrogram index")
    index_df = pd.read_parquet(index_path)
    assert_training_spectrogram_index(index_df)
    keep = [column for column in INDEX_COLUMNS if column in index_df.columns]
    frame = index_df[keep]
    if "npy_path" in frame.columns:
        foreign = [path for path in frame["npy_path"].astype(str) if "Validation" in Path(path).parts]
        if foreign:
            raise ValueError(f"{len(foreign)} spectrogram path(s) point at official Validation")
    holdout = set(holdout_ids)
    frame = frame.loc[frame["call_id"].astype(str).isin(holdout)].copy()
    present = set(frame["call_id"].astype(str))
    missing = holdout - present
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} holdout call id(s) are not in the index: {preview}")
    frame["_call_id"] = frame["call_id"].astype(str)
    frame = frame.sort_values(["_call_id", "utt_idx"], kind="mergesort").reset_index(drop=True)
    return frame


def _load_saved_prob(path: Path, frame: pd.DataFrame) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"missing saved probabilities: {path}")
    saved = np.load(path, allow_pickle=False)
    call_ids = saved["call_id"].astype(str)
    utt_idx = saved["utt_idx"].astype(np.int64)
    if len(call_ids) != len(frame) or not np.array_equal(call_ids, frame["_call_id"].to_numpy()):
        raise SystemExit(f"{path} does not match this holdout's call ids")
    if not np.array_equal(utt_idx, frame["utt_idx"].to_numpy(dtype=np.int64)):
        raise SystemExit(f"{path} does not match this holdout's utterance order")
    return np.asarray(saved["p_class_1"], dtype=np.float64)


def _save_prob(path: Path, frame: pd.DataFrame, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        call_id=frame["_call_id"].to_numpy(),
        utt_idx=frame["utt_idx"].to_numpy(dtype=np.int64),
        speaker=frame["speaker"].to_numpy(dtype=np.int64),
        p_class_1=np.asarray(probs, dtype=np.float64),
    )


def _pad_own(spec: np.ndarray) -> np.ndarray:
    """Pad this utterance alone up to L2_MIN_FRAMES with SPEC_FLOOR_DB. No other utterance is involved."""

    spec = np.asarray(spec, dtype=np.float32)
    n_mels, n_frames = spec.shape
    if n_frames >= L2_MIN_FRAMES:
        return spec
    padding = np.full((n_mels, L2_MIN_FRAMES - n_frames), SPEC_FLOOR_DB, dtype=np.float32)
    return np.concatenate([spec, padding], axis=1)


def _load_spectrogram(row, packed, spectrogram_dir: Path | None) -> np.ndarray:
    if packed is not None:
        start = int(row["frame_offset"])
        n_frames = int(row["n_frames"])
        spec = np.ascontiguousarray(packed[start : start + n_frames].T, dtype=np.float32)
    else:
        spec = np.load(spectrogram_dir / row["npy_path"])
        spec = np.asarray(spec, dtype=np.float32)
    return _pad_own(spec)


def _load_model(ckpt_path: Path, device: str):
    import torch

    from dcc_data.mission2_checkpoint import (
        ENSEMBLE_STATE_DICTS_KEY,
        require_l2_inference_config,
        resolve_l2_checkpoint_metadata,
    )
    from dcc_data.mission2_cnn import SpeakerCNNLarge

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if ENSEMBLE_STATE_DICTS_KEY in checkpoint or "best_model_state_dict" not in checkpoint:
        raise SystemExit(f"{ckpt_path} must be one member checkpoint, not a packed ensemble")
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    if metadata.get("index_split") != "Training":
        raise SystemExit(f"{ckpt_path} was not trained on Training")
    model = SpeakerCNNLarge()
    model.load_state_dict(checkpoint["best_model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _infer_members(frame: pd.DataFrame, ckpt_paths: list[Path], args: argparse.Namespace, probs_dir: Path) -> dict[Path, np.ndarray]:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    packed = None
    if args.packed_spectrograms is not None:
        packed = np.load(args.packed_spectrograms, mmap_mode="r")
    models = [(path, _load_model(path, args.device)) for path in ckpt_paths]
    n_rows = len(frame)
    stored = {path: np.empty(n_rows, dtype=np.float64) for path in ckpt_paths}
    for row_index in range(n_rows):
        row = frame.iloc[row_index]
        spec = _load_spectrogram(row, packed, args.spectrogram_dir)
        batch = torch.from_numpy(np.ascontiguousarray(spec)).unsqueeze(0).unsqueeze(0).to(args.device)
        if batch.shape[0] != 1:
            raise RuntimeError("holdout inference must use batch size 1")
        with torch.no_grad():
            for path, model in models:
                logits = model(batch).float()
                stored[path][row_index] = float(torch.softmax(logits, dim=1)[0, 1].item())
        if row_index and row_index % 1000 == 0:
            print(f"inferred {row_index}/{n_rows}")
    for path, probs in stored.items():
        _save_prob(_prob_file(probs_dir, path), frame, probs)
    print(f"inferred {n_rows}/{n_rows}")
    return stored


def main() -> int:
    args = parse_args()
    groups = _parse_groups(args.groups)
    _reject_paths(
        [
            ("spectrogram index", args.index),
            ("holdout call ids", args.holdout_call_ids),
            ("packed spectrograms", args.packed_spectrograms),
            ("spectrogram dir", args.spectrogram_dir),
            ("probs dir", args.probs_dir),
            ("metrics", args.metrics),
            ("save probs", args.save_probs),
            *[("checkpoint", path) for paths in groups.values() for path in paths],
        ]
    )
    holdout_ids = load_call_id_list(args.holdout_call_ids)
    frame = _holdout_frame(args.index, holdout_ids)
    unique_ckpts = list(dict.fromkeys(path for paths in groups.values() for path in paths))
    if args.probs_dir is not None:
        by_ckpt = {path: _load_saved_prob(_prob_file(args.probs_dir, path), frame) for path in unique_ckpts}
    else:
        probs_dir = args.save_probs if args.save_probs is not None else args.metrics.parent / "probs"
        by_ckpt = _infer_members(frame, unique_ckpts, args, probs_dir)
    group_probs = {name: [by_ckpt[path] for path in paths] for name, paths in groups.items()}
    metrics = build_gate_metrics(frame["speaker"].to_numpy(dtype=np.int64), group_probs)
    metrics["holdout_calls"] = len(holdout_ids)
    metrics["min_frames"] = int(L2_MIN_FRAMES)
    metrics["pad_value_db"] = float(SPEC_FLOOR_DB)
    metrics["batch_size"] = 1
    metrics["decision"] = "mean P(class=1) >= 0.5"
    for name, paths in groups.items():
        metrics["groups"][name]["checkpoints"] = [str(path) for path in paths]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.metrics}")
    if metrics.get("mcnemar") is not None:
        print(
            f"acc_a={metrics['acc_a']:.4f} acc_b={metrics['acc_b']:.4f} "
            f"gain={metrics['acc_gain_b_minus_a']:.4f} "
            f"mcnemar_p={metrics['mcnemar']['p_value']:.6g} gate_pass={metrics['gate_pass']}"
        )
    else:
        print("groups a and b were not both present; gate_pass=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
