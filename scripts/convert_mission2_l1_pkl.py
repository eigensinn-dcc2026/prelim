#!/usr/bin/env python3
"""Pack a Mission 2 L1 (GBM) ``.pkl`` into a ``.ckpt`` the official rules accept.

The contest only takes ``.pt``/``.pth``/``.ckpt`` weight files, while the L1
model is ``dcc_data.mission2_train.save_model``'s pickle. This stores that
pickle's exact bytes (plus its SHA-256) inside a ``torch.save`` file that
``dcc_inference.mission2`` recognises by ``format`` -- no retraining, no
re-serialisation of the model, so predictions are identical by construction.
``--verify`` re-checks that anyway: the model loaded from both files must
give identical predictions on random feature vectors and, with
``--audio_dir/--label_dir``, identical Mission 2 frames.

    python scripts/convert_mission2_l1_pkl.py \\
        --pkl checkpoints/mission2/m2_baseline_lgbm_9k.pkl \\
        --output checkpoints/mission2/m2_baseline_lgbm_9k.ckpt --verify

Note: the frozen ``scripts/mission2_inference.py`` routes ``.ckpt`` to its L2
loader, so keep using the ``.pkl`` there; the root ``inference.py`` takes either.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import load_model  # noqa: E402
from dcc_inference import mission2  # noqa: E402


def verify(pkl_path: Path, ckpt_path: Path, audio_dir: Path | None = None, label_dir: Path | None = None) -> dict:
    import torch

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != mission2.L1_CKPT_FORMAT:
        raise ValueError(f"{ckpt_path} is not an L1 ckpt (format={checkpoint.get('format')!r})")
    pkl_sha = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    if checkpoint["l1_pickle"] != pkl_path.read_bytes() or checkpoint["l1_pickle_sha256"] != pkl_sha:
        raise ValueError("embedded pickle bytes differ from the source .pkl")

    model_pkl, meta_pkl = load_model(pkl_path)
    model_ckpt, meta_ckpt = mission2._l1_from_torch_checkpoint(checkpoint)
    if repr(meta_pkl) != repr(meta_ckpt):
        raise ValueError("meta differs")
    x = np.random.default_rng(0).standard_normal((2000, len(FEATURE_COLUMNS))) * 3.0
    same_pred = np.array_equal(model_pkl.predict(x), model_ckpt.predict(x))
    same_proba = (
        np.array_equal(model_pkl.predict_proba(x), model_ckpt.predict_proba(x))
        if hasattr(model_pkl, "predict_proba")
        else None
    )
    if not same_pred or same_proba is False:
        raise ValueError(f"predictions differ (predict equal={same_pred}, predict_proba equal={same_proba})")
    result = {"sha256": pkl_sha, "random_vectors": len(x), "predict_equal": same_pred, "proba_equal": same_proba}

    if audio_dir is not None and label_dir is not None:
        frame_pkl = mission2.predict(audio_dir, label_dir, pkl_path)
        frame_ckpt = mission2.predict(audio_dir, label_dir, ckpt_path)
        if not frame_pkl.equals(frame_ckpt):
            raise ValueError("Mission 2 frames differ between .pkl and .ckpt")
        result["frame_rows"] = len(frame_pkl)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help=".ckpt (or .pt/.pth) path to write")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--audio_dir", type=Path, default=None, help="optional, with --verify")
    parser.add_argument("--label_dir", type=Path, default=None, help="optional, with --verify")
    args = parser.parse_args(argv)

    if args.output.suffix.lower() not in mission2.TORCH_CHECKPOINT_SUFFIXES:
        parser.error(f"--output must end in one of {sorted(mission2.TORCH_CHECKPOINT_SUFFIXES)}")
    if args.output.resolve() == args.pkl.resolve():
        parser.error("--output must differ from --pkl")
    header = mission2.save_l1_torch_ckpt(args.pkl, args.output)
    print(f"wrote {args.output} {header}")
    if args.verify:
        print(f"verified: {verify(args.pkl, args.output, args.audio_dir, args.label_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
