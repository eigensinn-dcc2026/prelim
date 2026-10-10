"""Explicit extension: one encoder epoch for all three failed frozen pilots.

The original pilot remains intact. This is a seed-42 screening experiment, not
an adoption decision; official Validation is never read by this command.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from dcc_data import mission3_context_experiment as exp
from dcc_inference import mission3_bert as baseline, mission3_context as context


def validate_source(saved, current, summary):
    if saved != current or summary.get("source") != saved:
        raise ValueError("Frozen experiment dataset/checkpoint/implementation changed")
    if saved.get("smoke") or summary.get("smoke"):
        raise ValueError("A smoke experiment cannot enter the full fine pilot")
    if summary.get("status") != "both_pilots_failed":
        raise ValueError("This extension requires completed failed frozen pilots")


def validate_head(state, signature, method):
    expected = dict(signature=signature, method=method, seed=42, recipe=exp.RECIPE)
    if state.get("specification") != expected or state.get("epoch") != 10:
        raise ValueError(f"{method}: requires the fixed epoch-10 seed-42 frozen head")


def run(args):
    started = time.monotonic()
    run_dir, output = args.run_dir, args.out
    if run_dir.resolve() == output.resolve():
        raise ValueError("Fine output must be separate from the frozen experiment")
    output.mkdir(parents=True, exist_ok=True)
    budget = exp.RunBudget(run_dir, args.max_hours)
    summary = None
    try:
        device = torch.device(args.device)
        torch.set_num_threads(args.cpu_threads)
        rows, train, holdout, base_probs, ensemble = exp.prepare_data(
            args.train_label_dir, args.reference_probs_dir)
        source = dict(
            base_checkpoint_sha256=exp.file_hash(args.base_ckpt),
            rows_sha256=exp.digest([{k: row[k] for k in ("name", "call", "text_hash", "gold")} for row in rows]),
            split_sha256=exp.digest(dict(train=train, holdout=holdout)),
            recipe=exp.RECIPE, context_config=context.DEFAULTS, smoke=False,
            feature_precision="cuda_fp16" if device.type == "cuda" else "cpu_fp32",
            implementation_sha256=exp.digest({module.__name__: exp.file_hash(module.__file__)
                                             for module in (context, baseline, exp)}))
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        frozen = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        validate_source(saved, source, frozen)
        signature = exp.digest(source)
        manifest = dict(source=source, extension="user_requested_failed_pilot_encoder_epoch",
                        seed=42, epochs=1, threshold=.5, methods=["control", "memory", "evidence"],
                        runner_sha256=exp.file_hash(__file__))
        marker = output / "run.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Fine pilot resume configuration changed")
        exp.write_json(marker, manifest)
        exp.seed_all(42)
        model, tokenizer, payload = exp.initialize(args.base_ckpt, "control", device)
        if payload.get("source", {}).get("exp_id") != "M3-ROBERTA-003" or payload["source"].get("seed") != 42:
            raise ValueError("The fine pilot requires ROBERTA-003 seed42")
        # Read the existing cache without silently recomputing missing shards.
        feature_dir = run_dir / "features"
        indices = sorted(set(train + holdout))
        expected_index = dict(signature=signature, indices=indices, batch_size=32)
        if json.loads((feature_dir / "index.json").read_text(encoding="utf-8")) != expected_index:
            raise ValueError("Frozen feature cache differs")
        if any(not (feature_dir / f"shard_{i:06d}.pt").is_file() for i in range(0, len(indices), 32)):
            raise ValueError("Frozen feature cache is incomplete")
        features = exp.extract_features(model, tokenizer, rows, indices, feature_dir, signature, 32, budget)
        bank = exp.make_bank(features, rows, train).to(device)
        del features
        initial_base = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
        gold = np.asarray([rows[i]["gold"] for i in holdout])
        docs = [rows[i]["doc"] for i in holdout]
        summary = dict(status="fine_pilot_running", smoke=False, source=source,
                       extension=manifest, selected_method=None, seed=42,
                       official_validation_used=False, adoption_eligible=False,
                       pilot={"roberta003": frozen["pilot"]["roberta003"],
                              "existing_ensemble": frozen["pilot"]["existing_ensemble"]})
        for method in manifest["methods"]:
            summary["pilot"][method + "_frozen"] = frozen["pilot"][method]
        exp.report(output, summary)
        fine_scores = {}
        for method in manifest["methods"]:
            exp.seed_all(42)
            model.base.load_state_dict(initial_base)
            model.head = context.ResidualHead(model.base.classifier.in_features, method, context.DEFAULTS).to(device)
            model.bank = bank if method == "memory" else None
            head = torch.load(run_dir / f"{method}_42" / "head_last.pt", map_location=device, weights_only=True)
            validate_head(head, signature, method)
            model.head.load_state_dict(head["head"])
            del head
            directory = output / f"{method}_42"
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Starting {method}: seed=42, encoder_lr=1e-6, head_lr=1e-4, epoch=1", flush=True)
            probability = exp.fit_encoder(model, tokenizer, rows, train, holdout, 42,
                                          directory, signature, exp.RECIPE, budget)
            measured = exp.metrics(gold, probability, docs)
            fine_scores[method] = measured["macro_f1"]
            summary["pilot"][method + "_fine"] = measured
            exp.write_json(directory / "fine_metrics.json", measured)
            summary["elapsed_seconds"] = time.monotonic() - started
            exp.report(output, summary)
            print(f"Completed {method}: macro={measured['macro_f1']:.6f}, micro={measured['micro_f1']:.6f}", flush=True)
        winner = exp.choose_pilot(fine_scores, fine_scores["control"],
                                 [summary["pilot"][name]["macro_f1"] for name in ("roberta003", "existing_ensemble")])
        summary.update(selected_method=winner,
                       status="fine_pilot_complete_repeat_pending" if winner else "fine_pilots_failed",
                       elapsed_seconds=time.monotonic() - started)
        exp.report(output, summary)
        return summary
    except exp.BudgetReached:
        if summary is None:
            raise
        summary.update(status="budget_reached_resume_pending", elapsed_seconds=time.monotonic() - started)
        exp.report(output, summary)
        return summary
    finally:
        try:
            budget.check(force=True)
        except exp.BudgetReached:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-label-dir", "base-ckpt", "reference-probs-dir", "run-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=9.)
    args = parser.parse_args()
    if args.cpu_threads < 1 or not 0 < args.max_hours <= 10:
        parser.error("positive threads and a cumulative budget of at most 10 hours are required")
    summary = run(args)
    print(json.dumps({key: summary[key] for key in ("status", "selected_method", "elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
