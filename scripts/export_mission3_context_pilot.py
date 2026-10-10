"""Preserve complete fine-pilot checkpoints without qualifying them for release."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dcc_data import mission3_context_experiment as exp
from dcc_inference import mission3_context as context


def restore_state(model, state, method, rows, train, signature):
    expected = dict(signature=signature, seed=42, recipe=exp.RECIPE, method=method)
    if state.get("specification") != expected or state.get("epoch") != 1:
        raise ValueError("Only a matching completed seed-42 encoder epoch may be exported")
    model.head = context.ResidualHead(model.base.classifier.in_features, method, context.DEFAULTS)
    if method == "memory":
        weights = state["model"]
        labels = torch.tensor([rows[i]["gold"] for i in train], dtype=torch.float32)
        if not torch.equal(weights["bank.labels"].cpu(), labels):
            raise ValueError("Saved memory labels do not match the Training split")
        model.bank = context.MemoryBank(weights["bank.vectors"], labels,
                                       [rows[i]["call"] for i in train],
                                       [rows[i]["text_hash"] for i in train])
    else:
        model.bank = None
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def export(args):
    summary = json.loads((args.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary.get("smoke") or summary.get("status") not in ("fine_pilots_failed", "fine_pilot_complete_repeat_pending"):
        raise ValueError("A completed full-data fine pilot is required")
    rows, train, holdout, _, _ = exp.prepare_data(args.train_label_dir, args.reference_probs_dir)
    original = summary["source"]
    if (original["base_checkpoint_sha256"] != exp.file_hash(args.base_ckpt)
            or original["rows_sha256"] != exp.digest([{k: row[k] for k in ("name", "call", "text_hash", "gold")} for row in rows])
            or original["split_sha256"] != exp.digest(dict(train=train, holdout=holdout))
            or original["context_config"] != context.DEFAULTS or original["recipe"] != exp.RECIPE):
        raise ValueError("The pilot export dataset/checkpoint/configuration changed")
    signature = exp.digest(original)
    torch.set_num_threads(4)
    model, _, payload = exp.initialize(args.base_ckpt, "control", "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(experimental=True, qualified_for_submission=False, checkpoints={})
    for method in ("control", "memory", "evidence"):
        state = torch.load(args.experiment_dir / f"{method}_42/encoder_last.pt", map_location="cpu", weights_only=True)
        restore_state(model, state, method, rows, train, signature)
        source = dict(experimental=True, smoke=False, training_rows=len(train), internal_holdout_rows=len(holdout),
                      representative_seed=42, official_validation_used=False,
                      internal_gate=dict(passed=False, reason="Single-seed pilot; no three-seed adoption gate"),
                      parent_source=original, fine_metrics=summary["pilot"][method + "_fine"])
        checkpoint = args.out / f"{method}_seed42_experimental.pt"
        exp.save_torch(checkpoint, context.export_payload(model, payload, source))
        manifest["checkpoints"][checkpoint.name] = dict(bytes=checkpoint.stat().st_size,
                                                       sha256=exp.file_hash(checkpoint), method=method)
        print(f"Exported experimental {method}: {checkpoint.stat().st_size} bytes", flush=True)
        del state
    exp.write_json(args.out / "checkpoint_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-dir", "base-ckpt", "train-label-dir", "reference-probs-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(export(parser.parse_args()), indent=2))
