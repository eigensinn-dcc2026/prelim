"""M3-KCBERT-016: fixed lambda 0.1/0.3 training and 0/1 checkpoint rescoring.
Training-only 80:20 call holdout; final epoch, fixed >=0.5 threshold.
Calibration uses 15 equal-width bins per label, then averages labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import unicodedata
import zipfile

EXP_ID = "M3-KCBERT-016"
MODEL_ID = "beomi/kcbert-base"
# Immutable safetensors addition commit, rather than the moving main ref:
# https://huggingface.co/beomi/kcbert-base/tree/0f2f3f8
REVISION = "0f2f3f8"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEEDS = [42, 43, 44]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=2e-5, dropout=0.1, batch_size=16, threshold=0.5,
              split_seed=42, holdout_ratio=0.2, labels=LABELS,
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)",
              baseline_source="M3-KCBERT-012; only loss weight changed",
              soft_f1_weight=[0.1, 0.3], soft_f1_epsilon=1e-8, threshold_operator=">=")


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, obj):
    atomic_write(path, lambda p: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"))


def weights(labels):
    if not labels:
        raise ValueError("Empty training split")
    positive = [sum(row[j] for row in labels) for j in range(len(LABELS))]
    return [min(6.0, max(1.0, math.sqrt((len(labels) - p) / max(p, 1)) * .5)) for p in positive]


def transcript(doc):
    return " ".join(text for u in doc.get("utterances", []) if isinstance(u, dict)
                    for text in [re.sub(r"\s+", " ", u.get("text", "")).strip()
                                 if isinstance(u.get("text", ""), str) else ""] if text)


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "kcbert-base"
    marker = root / "dcc_revision.json"
    if marker.exists():
        if json.loads(marker.read_text()) != {"repo_id": MODEL_ID, "revision": REVISION}:
            raise ValueError(f"Model cache revision differs: {root}")
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["config.json", "tokenizer*", "vocab.txt", "model.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    # Fail before touching the data if the cached model cannot actually be loaded.
    probe = AutoModel.from_pretrained(root, local_files_only=True)
    del probe
    write_json(marker, {"repo_id": MODEL_ID, "revision": REVISION})
    return root, tokenizer


def prepare_rows(out, work):
    cache = out / "rows.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from dcc_data.colab_env import find_data_root
    source = find_data_root() / "Training" / "TL_서울_구급.zip"
    local = work / source.name
    atomic_write(local, lambda p: shutil.copyfile(source, p))
    extracted = work / "labels"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local) as archive:
        for item in archive.infolist():
            target = (extracted / item.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError(f"Unsafe zip member: {item.filename}")
        archive.extractall(extracted)
    rows = []
    for path in sorted((extracted / "TL_서울_구급").rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        symptoms = doc.get("symptom", [])
        if isinstance(symptoms, str):
            symptoms = [symptoms]
        symptoms = {unicodedata.normalize("NFC", s.strip()) for s in symptoms if isinstance(s, str)}
        # Repository convention: _id, otherwise the paired WAV/JSON stem.
        call = str(doc.get("call_id") or doc.get("_id") or path.stem)
        rows.append(dict(file=unicodedata.normalize("NFC", path.name), call_id=call,
                         text=transcript(doc), labels=[int(s in symptoms) for s in LABELS]))
    if not rows:
        raise ValueError("No Training label JSONs in TL_서울_구급.zip")
    if len({r["call_id"] for r in rows}) != len(rows):
        raise ValueError("Multiple rows per call: cannot preserve P0 row split without call leakage")
    write_json(cache, rows)
    return rows


def prepare_split(out, rows):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = out / "split.json"
    if path.exists():
        split = json.loads(path.read_text())
        if split["rows_sha256"] != digest:
            raise ValueError("Cached split does not match rows")
    else:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train, holdout = next(splitter.split(np.arange(len(rows)), np.array([r["labels"] for r in rows])))
        split = dict(train=train.tolist(), holdout=holdout.tolist(), rows_sha256=digest)
        write_json(path, split)
    train, holdout = split["train"], split["holdout"]
    if not train or not holdout or sorted(train + holdout) != list(range(len(rows))):
        raise ValueError("Invalid split partition")
    if {rows[i]["call_id"] for i in train} & {rows[i]["call_id"] for i in holdout}:
        raise ValueError("call_id leakage")
    atomic_write(out / "val_files.txt", lambda p: p.write_text(
        "\n".join(sorted(rows[i]["file"] for i in holdout)) + "\n", encoding="utf-8"))
    return train, sorted(holdout, key=lambda i: rows[i]["file"])


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_terms(logits, targets, criterion, weight=1.0):
    """Keep both terms in fp32, including under the encoder's AMP context."""
    import torch
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, targets = logits.float(), targets.float()
        bce = criterion(logits, targets)
        probabilities = logits.sigmoid()
        soft_f1 = (2 * (probabilities * targets).sum(dim=0) /
                   (probabilities.sum(dim=0) + targets.sum(dim=0) + 1e-8))
        soft_f1_loss = 1 - soft_f1.mean()
        return bce + weight * soft_f1_loss, bce, soft_f1_loss


def run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight, weight, last):
    config = dict(CONFIG, soft_f1_weight=weight)
    import numpy as np
    import torch
    from transformers import AutoModel
    from sklearn.metrics import f1_score
    result_path = out / f"seed_{seed}" / "metrics.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["seed"] != seed or result["epoch"] != 3 or not result_path.with_name("best.ckpt").exists():
            raise ValueError(f"Incomplete seed artifacts: {result_path}")
        return result
    seed_all(seed, torch, np)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.dropout = torch.nn.Dropout(.1)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(LABELS))

        def forward(self, ids, mask):
            return self.classifier(self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]))

    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # P1 uses constant LR: no scheduler is introduced for this loss-only experiment.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    start, history = 0, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        same_run = state["seed"] == seed and state["config"].get("soft_f1_weight") == weight
        if same_run and state["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        if state["seed"] == seed and state["config"] == config:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start, history = state["epoch"], state["history"]
            random.setstate(state["rng"]["random"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if amp and state["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            generator.set_state(state["rng"]["generator"])
    ids, masks = tokens["input_ids"], tokens["attention_mask"]
    train = torch.tensor(train)
    for epoch in range(start + 1, 4):
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        total = total_bce = total_soft_f1_loss = 0.0
        for step, offset in enumerate(range(0, len(order), 16)):
            batch = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(ids[batch].to(device), masks[batch].to(device))
            loss, bce, soft_f1_loss = loss_terms(logits, targets[batch].to(device), criterion, weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss seed={seed} epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            total_bce += bce.item()
            total_soft_f1_loss += soft_f1_loss.item()
            if step % 200 == 0:
                print(f"seed={seed} epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for offset in range(0, len(holdout), 64):
                batch = holdout[offset:offset + 64]
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(ids[batch].to(device), masks[batch].to(device))
                probabilities.append(logits.float().sigmoid().cpu())
        probs = torch.cat(probabilities).numpy()
        truth = targets[holdout].numpy()
        pred = (probs >= .5).astype(int)
        result = dict(seed=seed, epoch=epoch, macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                      micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                      per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                      empty_predictions=int((pred.sum(axis=1) == 0).sum()),
                      train_loss=total / (step + 1),
                      train_bce=total_bce / (step + 1),
                      train_soft_f1_loss=total_soft_f1_loss / (step + 1),
                      train_soft_f1=1 - total_soft_f1_loss / (step + 1))
        history.append(result)
        state = dict(config=config, seed=seed, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=None, scaler=scaler.state_dict(), history=history,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if amp else [], generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        write_json(result_path.with_name("history.json"), history)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    # Final epoch is the experiment score; no epoch or threshold selection.
    final = dict(config=config, seed=seed, epoch=3, model=model.state_dict(),
                 model_local_path=str(model_path), pos_weight=pos_weight)
    atomic_write(result_path.with_name("best.ckpt"), lambda p: torch.save(final, p))
    write_json(result_path.with_name("history.json"), history)
    write_json(result_path, history[-1])
    return history[-1]


def smoke():
    # Synthetic loss checks only; no model download or training.
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("smoke: torch unavailable; tensor loss/gradient checks skipped", flush=True)
    else:
        logits = torch.tensor([[0., 2., -2.]] * 2, requires_grad=True)
        targets = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1., 2., 3.]))
        loss, bce, soft_loss = loss_terms(logits, targets, criterion)
        probabilities = logits.detach().sigmoid()
        expected = 1 - (2 * (probabilities * targets).sum(0) /
                        (probabilities.sum(0) + targets.sum(0) + 1e-8)).mean()
        assert loss.dtype == torch.float32
        assert torch.allclose(soft_loss, expected)
        assert torch.allclose(loss, criterion(logits, targets) + expected)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        empty = torch.zeros_like(targets)
        assert torch.isfinite(loss_terms(logits.detach(), empty, criterion)[0])
        assert (torch.tensor([.49, .5, .51]) >= .5).tolist() == [False, True, True]
    rows = [[1] * 9] + [[0] * 9] * 99
    assert weights(rows) == [math.sqrt(99) * .5] * 9
    assert weights([[0] * 9] * 1000) == [6.] * 9
    assert weights([[1] * 9] * 2) == [1.] * 9
    assert transcript({"utterances": [{"text": " 가\n 나 "}, {"text": "다", "Speaker": "ignored"}]}) == "가 나 다"
    with tempfile.TemporaryDirectory() as name:
        p = Path(name) / "check.json"
        write_json(p, {"exp_id": EXP_ID})
        assert json.loads(p.read_text())["exp_id"] == EXP_ID
        assert list(p.parent.iterdir()) == [p]
    print("smoke OK: available loss checks, weights, text whitelist, atomic writes; no training")


def calibration_metrics(truth, probs):
    import numpy as np
    from sklearn.metrics import f1_score
    truth, probs = np.asarray(truth), np.asarray(probs)
    if probs.shape != truth.shape or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid holdout probabilities")
    pred = probs >= .5
    per_ece = []
    bins = np.minimum((probs * 15).astype(int), 14)
    for j in range(len(LABELS)):
        value = 0.
        for b in range(15):
            mask = bins[:, j] == b
            if mask.any():
                value += mask.mean() * abs(probs[mask, j].mean() - truth[mask, j].mean())
        per_ece.append(float(value))
    return dict(macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                ece=float(np.mean(per_ece)), per_symptom_ece=dict(zip(LABELS, per_ece)),
                brier=float(np.mean((probs - truth) ** 2)),
                per_symptom_brier=dict(zip(LABELS, np.mean((probs - truth) ** 2, axis=0).tolist())),
                probability_minus_prevalence=float(np.mean(probs - truth)),
                per_symptom_probability_minus_prevalence=dict(zip(LABELS, np.mean(probs - truth, axis=0).tolist())),
                positive_prediction_rate=float(pred.mean()),
                per_symptom_positive_prediction_rate=dict(zip(LABELS, pred.mean(axis=0).tolist())),
                empty_predictions=int((pred.sum(axis=1) == 0).sum()))


def rescore(ckpt, directory, seed, weight, model_path, tokens, targets, holdout):
    import numpy as np
    import torch
    from transformers import AutoModel
    directory.mkdir(parents=True, exist_ok=True)
    probability_path = directory / "holdout_probs.npy"
    if probability_path.exists():
        probs = np.load(probability_path, allow_pickle=False)
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        if state["seed"] != seed or any(cfg.get(k) != CONFIG[k] for k in
                ("model", "revision", "max_length", "labels", "split_seed", "holdout_ratio", "threshold")):
            raise ValueError(f"Incompatible baseline checkpoint: {ckpt}")
        if cfg.get("soft_f1_weight", 0.) != weight:
            raise ValueError(f"Unexpected lambda in checkpoint: {ckpt}")

        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
                self.dropout = torch.nn.Dropout(.1)
                self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(LABELS))

            def forward(self, ids, mask):
                return self.classifier(self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]))

        seed_all(seed, torch, np)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Classifier()
        model.load_state_dict(state["model"], strict=True)
        del state
        model.to(device).eval()
        batches = []
        with torch.inference_mode():
            for offset in range(0, len(holdout), 64):
                batch = holdout[offset:offset + 64]
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(tokens["input_ids"][batch].to(device), tokens["attention_mask"][batch].to(device))
                batches.append(logits.float().sigmoid().cpu().numpy())
        probs = np.concatenate(batches)
        atomic_write(probability_path, lambda p: save_probs(p, probs))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(seed=seed, lambda_value=weight, **calibration_metrics(targets[holdout].numpy(), probs))
    history_path = ckpt.with_name("history.json")
    result["epoch_holdout_macro_f1"] = ([r["macro_f1"] for r in json.loads(history_path.read_text())]
                                           if history_path.exists() else None)
    write_json(directory / "rescore_metrics.json", result)
    return result


def save_probs(path, probs):
    import numpy as np
    with path.open("wb") as handle:
        np.save(handle, probs, allow_pickle=False)


def comparison_table(results):
    import numpy as np
    table = []
    for weight in (0., .1, .3, 1.):
        group = [r for r in results if r["lambda_value"] == weight]
        table.extend(group)
        if not group:
            continue
        mean = dict(lambda_value=weight, seed="mean", seed_count=len(group))
        std = dict(lambda_value=weight, seed="std", seed_count=len(group))
        for key, value in group[0].items():
            if key in ("seed", "lambda_value"):
                continue
            if isinstance(value, dict):
                mean[key] = {k: float(np.mean([r[key][k] for r in group])) for k in value}
                std[key] = {k: float(np.std([r[key][k] for r in group], ddof=0)) for k in value}
            elif isinstance(value, (int, float)):
                mean[key] = float(np.mean([r[key] for r in group]))
                std[key] = float(np.std([r[key] for r in group], ddof=0))
            elif key == "epoch_holdout_macro_f1":
                available = [r[key] for r in group if r[key] is not None]
                mean[key] = np.mean(available, axis=0).tolist() if available else None
                std[key] = np.std(available, axis=0).tolist() if available else None
        table.extend([mean, std])
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        try:
            import numpy as np
            import sklearn
        except ModuleNotFoundError as exc:
            if exc.name not in {"numpy", "sklearn"}:
                raise
            print(f"calibration smoke skipped: {exc.name} unavailable")
            return
        truth = np.array([[0] * 9, [1] * 9])
        result = calibration_metrics(truth, truth.astype(float))
        assert result["ece"] == result["brier"] == 0 and result["macro_f1"] == 1
        result = calibration_metrics(truth, np.full((2, 9), .5))
        assert result["positive_prediction_rate"] == 1 and result["brier"] == .25
        table = comparison_table([dict(result, seed=s, lambda_value=.1, epoch_holdout_macro_f1=[.2, .3, .4]) for s in SEEDS])
        assert np.isclose(table[-1]["macro_f1"], 0)
        print("calibration/aggregation smoke OK")
        return
    if not args.out:
        parser.error("--out or LAB_OUT required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    model_path, tokenizer = prepare_model()
    import numpy as np
    import torch
    out = Path(args.out)
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != CONFIG:
        raise ValueError("Output configuration mismatch")
    write_json(out / "config.json", CONFIG)
    if (out / "metrics.json").exists() and (out / "best.ckpt").exists():
        print("All stages complete; reusing metrics.json")
        return
    rows = prepare_rows(out, work)
    train, holdout = prepare_split(out, rows)
    if len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Duplicate filenames cannot define probability row order")
    pos_weight = weights([rows[i]["labels"] for i in train])
    write_json(out / "pos_weight.json", dict(weights=dict(zip(LABELS, pos_weight)), formula=CONFIG["weight_formula"]))
    cache = out / "tokens.pt"
    if cache.exists():
        tokens = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                max_length=256, return_tensors="pt"))
        atomic_write(cache, lambda p: torch.save(tokens, p))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results, skipped = [], []
    for weight, source_exp in ((0., "M3-KCBERT-005"), (1., "M3-KCBERT-012"), (.1, None), (.3, None)):
        destination = out / f"lambda_{weight:g}"
        source = Path(os.environ["DCC_RUN_ROOT"]) / "lab" / source_exp if source_exp else destination
        for seed in SEEDS:
            ckpt = source / f"seed_{seed}" / "best.ckpt"
            if source_exp:
                if not ckpt.exists():
                    skipped.append(dict(lambda_value=weight, seed=seed, reason="checkpoint missing", path=str(ckpt)))
                    continue
                # Verify training membership as well as holdout identity before any inference.
                split_path = source / "split.json"
                if not split_path.exists():
                    raise ValueError(f"Cannot verify baseline split: {split_path}")
                old = json.loads(split_path.read_text())
                current = json.loads((out / "split.json").read_text())
                if old["rows_sha256"] != current["rows_sha256"] or set(old["train"]) != set(train) or set(old["holdout"]) != set(holdout):
                    raise ValueError(f"Baseline split differs: {source_exp}")
            else:
                run_seed(seed, destination, model_path, tokens, targets, train, holdout, pos_weight, weight, out / "ckpt_last.pt")
            seed_out = destination / f"seed_{seed}"
            atomic_write(seed_out / "val_files.txt", lambda p: shutil.copyfile(out / "val_files.txt", p))
            results.append(rescore(ckpt, seed_out, seed, weight, model_path, tokens, targets, holdout))
    table = comparison_table(results)
    write_json(out / "comparison_table.json", table)
    import csv
    def write_table(path):
        columns = list(dict.fromkeys(k for row in table for k in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in table)
    atomic_write(out / "comparison_table.csv", write_table)
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "lambda_0.1" / "seed_42" / "best.ckpt", p))
    representative = next(r for r in table if r["lambda_value"] == .1 and r["seed"] == "mean")
    write_json(out / "metrics.json", dict(exp_id=EXP_ID,
        primary_metric=dict(name="macro_f1", value=representative["macro_f1"], split="internal_holdout"),
        metrics=dict(comparison_table=table, skipped=skipped), seed=42, seeds=SEEDS,
        train_rows=len(train), eval_rows=len(holdout), official_val=None, ckpt="best.ckpt",
        notes="Fixed representative lambda=0.1 seed=42; final epoch 3, no selection/tuning. "
              "Lambda 0/1 inference only; missing checkpoints recorded. Population seed std (ddof=0). "
              "ECE: mean of nine label-wise 15 equal-width bin ECEs; Brier: mean over rows and labels. "
              "Threshold >=0.5; empty predictions unchanged; official Validation unused."))


if __name__ == "__main__":
    main()
