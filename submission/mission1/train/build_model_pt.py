"""Pack the Mission 1 model into ONE offline ``.pt`` file (format ``dcc-m1-v1``).

The submission rules ask for weights as ``.pt`` / ``.pth`` / ``.ckpt``. The model has two parts:

* the frozen head (``StandardScaler`` + ``LogisticRegression`` over 564 features), stored here as
  plain tensors so no sklearn pickle is needed at inference time;
* the pinned x-vector encoder ``microsoft/wavlm-base-plus-sv``: its weights plus the text of
  ``config.json`` / ``preprocessor_config.json``, so nothing is downloaded.

    python train/build_model_pt.py --head m1_xvec002_refit_full.ckpt \
        --wavlm_dir wavlm-base-plus-sv --out m1_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "dcc-m1-v1"
WAVLM_ID = "microsoft/wavlm-base-plus-sv"
WAVLM_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
N_FEATURES = 564  # 52 acoustic + 512 x-vector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True, help="sklearn Pipeline (StandardScaler + LogisticRegression), joblib")
    parser.add_argument("--wavlm_dir", type=Path, required=True, help="local snapshot of microsoft/wavlm-base-plus-sv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import joblib
    import torch
    from transformers import WavLMForXVector

    pipeline = joblib.load(args.head)
    scaler, clf = pipeline[0], pipeline[-1]
    if scaler.mean_.shape != (N_FEATURES,) or clf.coef_.shape != (1, N_FEATURES) or list(clf.classes_) != [0, 1]:
        raise ValueError("head is not the 564-feature binary F/M pipeline")

    marker = json.loads((args.wavlm_dir / "dcc_revision.json").read_text(encoding="utf-8"))
    if marker != {"repo_id": WAVLM_ID, "revision": WAVLM_REVISION}:
        raise ValueError(f"WavLM snapshot is not the pinned revision: {marker}")
    model = WavLMForXVector.from_pretrained(args.wavlm_dir, local_files_only=True).eval()

    payload = {
        "format": FORMAT,
        "labels": ["F", "M"],  # class 0 -> F, class 1 -> M
        "head": {
            "scaler_mean": torch.from_numpy(scaler.mean_.copy()),
            "scaler_scale": torch.from_numpy(scaler.scale_.copy()),
            "coef": torch.from_numpy(clf.coef_.copy()),
            "intercept": torch.from_numpy(clf.intercept_.copy()),
        },
        "wavlm": {
            "repo_id": WAVLM_ID,
            "revision": WAVLM_REVISION,
            "config": (args.wavlm_dir / "config.json").read_text(encoding="utf-8"),
            "preprocessor_config": (args.wavlm_dir / "preprocessor_config.json").read_text(encoding="utf-8"),
            "state_dict": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        },
        "source": {"head_file": args.head.name, "head_sha256": sha256(args.head)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MB) sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
