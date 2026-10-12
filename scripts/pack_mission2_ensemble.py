#!/usr/bin/env python3
"""Pack the Mission 2 CNN ensemble and one or more WavLM checkpoints into one file.

The grader still receives a single ``--ckpt_path``. This file records each member,
the sha256 of the file it came from, a weight (equal by default, or a JSON map),
and the combine mode ``prob_mean`` or ``weighted_prob_mean``. Inference averages
P(신고자) and applies the fixed 0.5 threshold; this script does not add a threshold flag.

WavLM backbone weights already live in ``best.ckpt``. The WavLMConfig JSON is stored
beside them so inference can build ``WavLMModel(config)`` and load the state dict
with ``layerdrop=0`` and no network access.

Usage:
    python scripts/pack_mission2_ensemble.py \\
        --cnn-ckpt checkpoints/m2_cnn_large_rerun_ensemble.pt \\
        --wavlm-ckpt best.ckpt --wavlm-config config.json \\
        --output checkpoints/m2_cnn_wavlm_ensemble.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from dcc_data.mission2_checkpoint import (  # noqa: E402
    checkpoint_sha256,
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_spectrogram import ALLOWED_LABEL_SOURCES  # noqa: E402
from dcc_data.mission2_wavlm import (  # noqa: E402
    COMBINE_PROB_MEAN,
    COMBINE_WEIGHTED,
    FIXED_THRESHOLD,
    PROB_ENSEMBLE_FORMAT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnn-ckpt", type=Path, required=True)
    parser.add_argument("--wavlm-ckpt", type=Path, action="append", required=True, dest="wavlm_ckpts")
    parser.add_argument(
        "--wavlm-config",
        type=Path,
        action="append",
        default=None,
        dest="wavlm_configs",
        help="config.json for each --wavlm-ckpt, in the same order. Optional when the ckpt or a sibling file already has it.",
    )
    parser.add_argument("--weights", type=Path, default=None, help="JSON object of member name -> positive weight")
    parser.add_argument("--combine", choices=(COMBINE_PROB_MEAN, COMBINE_WEIGHTED), default=COMBINE_PROB_MEAN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.wavlm_configs is not None and len(args.wavlm_configs) != len(args.wavlm_ckpts):
        parser.error("--wavlm-config must be given once per --wavlm-ckpt")
    if args.weights is not None and args.combine != COMBINE_WEIGHTED:
        parser.error("--weights is used only with --combine weighted_prob_mean")
    return args


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return value


def resolve_wavlm_config(checkpoint: dict, ckpt_path: Path, explicit: Path | None) -> dict:
    if explicit is not None:
        return _load_json(explicit)
    for key in ("wavlm_config", "config"):
        embedded = checkpoint.get(key)
        if isinstance(embedded, dict) and "hidden_size" in embedded and embedded.get("model_type") == "wavlm":
            return embedded
    candidates = [ckpt_path.parent / "config.json", ckpt_path.parent.parent / "config.json"]
    run_root = Path(os.environ["DCC_RUN_ROOT"]) if os.environ.get("DCC_RUN_ROOT") else None
    architecture = str(checkpoint.get("architecture") or "")
    if run_root is not None:
        if "large" in architecture:
            candidates.append(run_root / "models" / "wavlm-large" / "config.json")
        candidates.append(run_root / "models" / "wavlm-base" / "config.json")
        candidates.append(run_root / "models" / "wavlm-large" / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            config = _load_json(candidate)
            if config.get("model_type") == "wavlm":
                return config
    raise SystemExit(
        f"{ckpt_path} has no WavLMConfig. Pass --wavlm-config pointing at the model's config.json "
        "(inference must not download it)."
    )


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    index = 2
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name


def _require_provided_label(label_source: object, where: str) -> None:
    if label_source not in ALLOWED_LABEL_SOURCES:
        raise SystemExit(
            f"{where} label_source is {label_source!r}; Mission 2 members must use the provided speaker labels "
            f"({', '.join(ALLOWED_LABEL_SOURCES)})"
        )


def build_prob_ensemble(
    cnn_ckpt_path: Path,
    wavlm_ckpt_paths: list[Path],
    wavlm_config_paths: list[Path] | None,
    *,
    weights: dict[str, float] | None,
    combine: str,
) -> dict:
    if not wavlm_ckpt_paths:
        raise SystemExit("at least one WavLM best.ckpt is required")
    if combine not in (COMBINE_PROB_MEAN, COMBINE_WEIGHTED):
        raise SystemExit(f"combine must be prob_mean or weighted_prob_mean, got {combine!r}")

    cnn_checkpoint = torch.load(cnn_ckpt_path, map_location="cpu")
    cnn_metadata, _source = resolve_l2_checkpoint_metadata(cnn_checkpoint, cnn_ckpt_path)
    require_l2_inference_config(cnn_metadata)
    _require_provided_label(cnn_metadata["label_source"], str(cnn_ckpt_path))
    cnn_states = l2_member_state_dicts(cnn_checkpoint, cnn_metadata)

    used_names: set[str] = set()
    members: list[dict] = []
    summary: list[dict] = []

    cnn_name = _unique_name("cnn", used_names)
    cnn_sha = checkpoint_sha256(cnn_ckpt_path)
    members.append(
        {
            "name": cnn_name,
            "kind": "cnn_l2",
            "weight": 1.0,
            "sha256": cnn_sha,
            "source": str(cnn_ckpt_path),
            "label_source": cnn_metadata["label_source"],
            "sample_rate": int(cnn_metadata["sample_rate"]),
            "state_dicts": cnn_states,
            "metadata": cnn_metadata,
        }
    )
    summary.append(
        {
            "name": cnn_name,
            "kind": "cnn_l2",
            "sha256": cnn_sha,
            "weight": 1.0,
            "source": str(cnn_ckpt_path),
            "label_source": cnn_metadata["label_source"],
            "n_state_dicts": len(cnn_states),
        }
    )

    config_paths = wavlm_config_paths or [None] * len(wavlm_ckpt_paths)
    for index, (ckpt_path, config_path) in enumerate(zip(wavlm_ckpt_paths, config_paths)):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise SystemExit(f"{ckpt_path} has no model_state_dict (expected a WavLM best.ckpt)")
        label_source = checkpoint.get("label_source")
        _require_provided_label(label_source, str(ckpt_path))
        config = resolve_wavlm_config(checkpoint, ckpt_path, config_path)
        if config.get("model_type") != "wavlm":
            raise SystemExit(f"{ckpt_path} config model_type is {config.get('model_type')!r}, expected 'wavlm'")
        architecture = str(checkpoint.get("architecture") or f"wavlm_{index}")
        name = _unique_name(architecture, used_names)
        sha = checkpoint_sha256(ckpt_path)
        inference = checkpoint.get("inference") or {
            "stride_sec": 1.5,
            "max_windows": 8,
            "aggregation": "mean_logit",
            "threshold": FIXED_THRESHOLD,
        }
        if float(inference.get("threshold", FIXED_THRESHOLD)) != FIXED_THRESHOLD:
            raise SystemExit(f"{ckpt_path} records threshold {inference.get('threshold')!r}; the contest threshold is 0.5")
        member = {
            "name": name,
            "kind": "wavlm",
            "weight": 1.0,
            "sha256": sha,
            "source": str(ckpt_path),
            "label_source": label_source,
            "state_dict": checkpoint["model_state_dict"],
            "config": config,
            "sr": int(checkpoint.get("sr", 16000)),
            "win_sec": float(checkpoint.get("win_sec", 3.0)),
            "layers": int(checkpoint.get("layers", int(config["num_hidden_layers"]) + 1)),
            "hidden_size": int(checkpoint.get("hidden_size", config["hidden_size"])),
            "architecture": architecture,
            "revision": checkpoint.get("revision"),
            "inference": inference,
        }
        members.append(member)
        summary.append(
            {
                "name": name,
                "kind": "wavlm",
                "sha256": sha,
                "weight": 1.0,
                "source": str(ckpt_path),
                "label_source": label_source,
                "architecture": architecture,
                "revision": checkpoint.get("revision"),
                "layers": member["layers"],
                "hidden_size": member["hidden_size"],
                "sr": member["sr"],
            }
        )

    if weights is not None:
        unknown = sorted(set(weights) - {member["name"] for member in members})
        missing = sorted({member["name"] for member in members} - set(weights))
        if unknown or missing:
            raise SystemExit(f"weights JSON names do not match members (unknown={unknown}, missing={missing})")
        for member, item in zip(members, summary):
            weight = float(weights[member["name"]])
            if weight <= 0:
                raise SystemExit(f"weight for {member['name']} must be positive, got {weight}")
            member["weight"] = weight
            item["weight"] = weight

    label_sources = {member["label_source"] for member in members}
    if len(label_sources) != 1:
        raise SystemExit(f"members disagree on label_source: {sorted(map(str, label_sources))}")

    return {
        "format": PROB_ENSEMBLE_FORMAT,
        "metadata": {
            "format": PROB_ENSEMBLE_FORMAT,
            "combine": combine,
            "label_source": members[0]["label_source"],
            "threshold": FIXED_THRESHOLD,
            "members": summary,
        },
        "members": members,
    }


def main() -> int:
    args = parse_args()
    weights = _load_json(args.weights) if args.weights is not None else None
    if weights is not None:
        weights = {str(key): float(value) for key, value in weights.items()}
    packed = build_prob_ensemble(
        args.cnn_ckpt,
        list(args.wavlm_ckpts),
        list(args.wavlm_configs) if args.wavlm_configs is not None else None,
        weights=weights,
        combine=args.combine,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, args.output)
    print(f"wrote {args.output}")
    for member in packed["metadata"]["members"]:
        print(f"  {member['name']}: kind={member['kind']} sha256={member['sha256']} weight={member['weight']}")
    print(f"combine={packed['metadata']['combine']} threshold={packed['metadata']['threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
