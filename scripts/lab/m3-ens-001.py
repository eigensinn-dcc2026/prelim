"""M3-ENS-001: fixed same-seed KcBERT/Roberta probability ensemble.

Requires the unchanged m3-kcbert-012.py and m3-roberta-003.py sources.
No substitute training recipe is allowed. Official Validation is unused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

EXP_ID = "M3-ENS-001"
SEEDS = [42, 43, 44]
SOURCES = {"kcbert": "m3-kcbert-012.py", "roberta": "m3-roberta-003.py"}
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def load_sources():
    modules = {}
    for name, filename in SOURCES.items():
        path = Path(__file__).with_name(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Required unchanged baseline source missing: {path}; "
                                    "restore M3-ROBERTA-003 before running; no recipe substitution permitted")
        spec = importlib.util.spec_from_file_location(f"ensemble_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.LABELS != LABELS or module.SEEDS != SEEDS:
            raise ValueError(f"Incompatible labels/seeds: {filename}")
        cfg = module.CONFIG
        expected = dict(epochs=3, max_length=256, threshold=0.5, split_seed=42, holdout_ratio=0.2,
                        soft_f1_weight=1.0)
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Baseline does not match specified soft-F1/80:20 configuration: {filename}")
        modules[name] = module
    return modules


def score(truth, probabilities):
    # Pure Python also makes --smoke independent of the GPU/package environment.
    pred = [[int(p >= .5) for p in row] for row in probabilities]
    counts = []
    for j in range(len(LABELS)):
        tp = sum(t[j] == 1 and p[j] == 1 for t, p in zip(truth, pred))
        fp = sum(t[j] == 0 and p[j] == 1 for t, p in zip(truth, pred))
        fn = sum(t[j] == 1 and p[j] == 0 for t, p in zip(truth, pred))
        counts.append((tp, fp, fn))
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    per = [f1(*c) for c in counts]
    return dict(macro_f1=sum(per) / len(per),
                micro_f1=f1(*(sum(c[j] for c in counts) for j in range(3))),
                per_symptom_f1=dict(zip(LABELS, per)),
                empty_predictions=sum(not any(row) for row in pred))


def evaluate(truth, left, right):
    average = [[(a + b) * .5 for a, b in zip(x, y)] for x, y in zip(left, right)]
    both = [sum((a[j] >= .5) != bool(t[j]) and (b[j] >= .5) != bool(t[j])
                for t, a, b in zip(truth, left, right)) for j in range(len(LABELS))]
    return dict(kcbert=score(truth, left), roberta=score(truth, right), ensemble=score(truth, average),
                both_wrong_label_ratio=sum(both) / (len(truth) * len(LABELS)),
                both_wrong_per_symptom_ratio=dict(zip(LABELS, [n / len(truth) for n in both])))


def probabilities(module, seed, directory, model_path, tokens, targets, train, holdout, pos_weight, out):
    import numpy as np
    import torch
    from transformers import AutoModel
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probabilities.npy"
    checkpoint = directory / f"seed_{seed}" / "best.ckpt"
    if path.exists():
        if not checkpoint.exists():
            raise ValueError(f"Probability cache without final checkpoint: {directory}")
        probs = np.load(path, allow_pickle=False)
    else:
        # Each source owns an independent model×seed resume checkpoint. Root alias
        # exposes the currently active checkpoint under the runner contract.
        alias = out / "ckpt_last.pt"
        temporary = out / "ckpt_last.link.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to((directory / "ckpt_last.pt").resolve())
        os.replace(temporary, alias)
        module.run_seed(seed, directory, model_path, tokens, targets, train, holdout, pos_weight)
        module.seed_all(seed, torch, np)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != module.CONFIG or state["seed"] != seed or state["epoch"] != 3:
            raise ValueError(f"Final checkpoint mismatch: {checkpoint}")
        encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        encoder.load_state_dict({k[len("bert."):]: v for k, v in state["model"].items() if k.startswith("bert.")})
        head = torch.nn.Linear(encoder.config.hidden_size, len(LABELS))
        head.load_state_dict({k[len("classifier."):]: v for k, v in state["model"].items() if k.startswith("classifier.")})
        del state
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device).eval()
        head.to(device).eval()
        batches = []
        with torch.no_grad():
            for offset in range(0, len(holdout), 8):
                idx = holdout[offset:offset + 8]
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    hidden = encoder(input_ids=tokens["input_ids"][idx].to(device),
                                     attention_mask=tokens["attention_mask"][idx].to(device)).last_hidden_state[:, 0]
                    logits = head(hidden)
                batches.append(logits.float().sigmoid().cpu().numpy())
        probs = np.concatenate(batches)
        del encoder, head
        if device.type == "cuda":
            torch.cuda.empty_cache()
        validate_probs(probs, len(holdout))
        def save(target):
            with target.open("wb") as handle:
                np.save(handle, probs, allow_pickle=False)
        module.atomic_write(path, save)
    validate_probs(probs, len(holdout))
    return probs.tolist()


def validate_probs(probs, rows):
    import numpy as np
    if probs.shape != (rows, len(LABELS)) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Invalid cached probabilities")


def smoke():
    truth = [[1] * 9, [0] * 9]
    left = [[.5] * 9, [.49] * 9]
    right = [[.49] * 9, [.5] * 9]
    result = evaluate(truth, left, right)
    assert result["kcbert"]["macro_f1"] == 1.0
    assert result["ensemble"]["empty_predictions"] == 2
    assert result["ensemble"]["macro_f1"] == 0.0
    assert result["both_wrong_label_ratio"] == 0.0
    assert evaluate(truth, right, right)["both_wrong_label_ratio"] == 1.0
    assert score(truth, [[.5] * 9] * 2)["empty_predictions"] == 0
    print("smoke OK: fixed averaging, >=0.5 boundary, empty predictions, F1, error overlap; no training/download")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.out:
        parser.error("--out or LAB_OUT required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    modules = load_sources()
    prepared = {name: module.prepare_model() for name, module in modules.items()}
    import torch
    out = Path(args.out)
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    base = modules["kcbert"]
    config = dict(exp_id=EXP_ID, weights=[.5, .5], threshold=.5, seeds=SEEDS,
                  sources={name: dict(config=m.CONFIG, sha256=hashlib.sha256(
                      Path(m.__file__).read_bytes()).hexdigest()) for name, m in modules.items()})
    cfg_path = out / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError("Output configuration/source mismatch")
    base.write_json(cfg_path, config)
    if (out / "metrics.json").exists():
        if not (out / "best.ckpt").exists():
            raise ValueError("Completed metrics without ensemble checkpoint")
        print("All seeds complete; reusing metrics.json")
        return
    rows = base.prepare_rows(out, work)
    train, holdout = base.prepare_split(out, rows)
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    truth = targets[holdout].tolist()
    predictions = {seed: {} for seed in SEEDS}
    for name, module in modules.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        if module.prepare_split(directory, rows) != (train, holdout):
            raise ValueError("Baseline split differs")
        model_path, tokenizer = prepared[name]
        cache = directory / "tokens.pt"
        if cache.exists():
            tokens = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                    max_length=256, return_tensors="pt"))
            base.atomic_write(cache, lambda p: torch.save(tokens, p))
        pos_weight = module.weights([rows[i]["labels"] for i in train])
        for seed in SEEDS:
            predictions[seed][name] = probabilities(module, seed, directory / f"seed_{seed}",
                model_path, tokens, targets, train, holdout, pos_weight, out)
        del tokens
    results = [dict(seed=seed, **evaluate(truth, predictions[seed]["kcbert"], predictions[seed]["roberta"]))
               for seed in SEEDS]
    mean = sum(r["ensemble"]["macro_f1"] for r in results) / 3
    exceeding = sum(r["ensemble"]["macro_f1"] > .6427 for r in results)
    # Fixed representative seed 42: bundle both complete model states for inference.
    final = dict(config=config, seed=42, threshold=.5, weights=[.5, .5], models={})
    for name in modules:
        final["models"][name] = torch.load(out / name / "seed_42" / "seed_42" / "best.ckpt",
                                           map_location="cpu", weights_only=False)
    base.atomic_write(out / "best.ckpt", lambda p: torch.save(final, p))
    means = {name: dict(macro_f1=sum(r[name]["macro_f1"] for r in results) / 3,
                       micro_f1=sum(r[name]["micro_f1"] for r in results) / 3,
                       per_symptom_f1={label: sum(r[name]["per_symptom_f1"][label] for r in results) / 3
                                       for label in LABELS}) for name in ["kcbert", "roberta", "ensemble"]}
    base.write_json(out / "metrics.json", dict(exp_id=EXP_ID,
        primary_metric=dict(name="macro_f1", value=mean, split="internal_holdout"),
        metrics=dict(seeds=results, means=means, baseline_macro_f1=.6427,
                     seeds_exceeding_baseline=exceeding, success_criteria_met=mean >= .6527 and exceeding >= 2),
        seed=42, seeds=SEEDS, train_rows=len(train), eval_rows=len(holdout), official_val=None,
        ckpt="best.ckpt", notes="Same-seed final-epoch probabilities averaged 0.5/0.5; threshold >=0.5; "
        "empty predictions unchanged; mean of three seed scores; seed42 representative ensemble; official Val unused."))


if __name__ == "__main__":
    main()
