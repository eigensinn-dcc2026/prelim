"""Export a Mission 3 lab ``best.ckpt`` into one offline inference checkpoint.

    python scripts/mission3_export_bert_ckpt.py \
        --lab_ckpt  <dcc_runs>/lab/M3-KCBERT-005/best.ckpt \
        --model_dir <dcc_runs>/models/kcbert-base \
        --output    checkpoints/mission3/m3_kcbert005_seed42.pt

``--model_dir`` is the Hugging Face folder the lab run trained from (its config
and tokenizer files are embedded; its weights are not, the trained state dict
replaces them). The result is what ``inference.py --ckpt_path`` takes for
Mission 3. A ``<output>.json`` next to it records the source and SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dcc_inference import mission3_bert  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab_ckpt", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        parser.error("--output must end with .pt, .pth or .ckpt (official checkpoint extensions)")

    meta = mission3_bert.export_checkpoint(args.lab_ckpt, args.model_dir, args.output)
    record = dict(meta, output=args.output.name, output_bytes=args.output.stat().st_size,
                  output_sha256=sha256(args.output))
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
