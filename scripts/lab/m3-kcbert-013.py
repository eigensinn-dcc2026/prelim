"""M3-KCBERT-013: train-only TAPT, then KCBERT-005 weighted BCE.

Fixed P0 call-level 80:20 split; two MLM epochs shared by seeds 42/43/44.
Holdout MLM losses are diagnostic only. No official Validation is read.
Fine-tuning reports final epoch 3 with probability >= 0.5; seed 42 is exported.
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

EXP_ID = "M3-KCBERT-013"
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
              baseline_source="M3-KCBERT-005",
              tapt=dict(epochs=2, batch_size=32, lr=5e-5, warmup_ratio=.06,
                        mlm_probability=.15, seed=42, train_rows=23408, holdout_rows=5792))


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
    from transformers import BertForMaskedLM, AutoTokenizer
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
    probe = BertForMaskedLM.from_pretrained(root, local_files_only=True)
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
    return train, holdout


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_tapt(out, base_path, tokenizer, tokens, train, holdout):
    """One shared MLM run; durable epoch checkpoints and encoder completion marker."""
    import numpy as np
    import torch
    from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

    encoder = Path(os.environ["DCC_RUN_ROOT"]) / "lab" / EXP_ID / "tapt_encoder"
    marker = encoder / "dcc_tapt_complete.json"
    split_digest = json.loads((out / "split.json").read_text())["rows_sha256"]
    identity = dict(config=CONFIG, rows_sha256=split_digest,
                    split_sha256=hashlib.sha256((out / "split.json").read_bytes()).hexdigest())
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["identity"] != identity:
            raise ValueError("TAPT encoder belongs to a different configuration or split")
        # Check the completed encoder, including its tokenizer, before skipping TAPT.
        from transformers import AutoModel, AutoTokenizer
        probe = AutoModel.from_pretrained(encoder, local_files_only=True)
        AutoTokenizer.from_pretrained(encoder, local_files_only=True)
        del probe
        write_json(out / "tapt_metrics.json", record["metrics"])
        return encoder, record["metrics"]

    seed_all(42, torch, np)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = BertForMaskedLM.from_pretrained(base_path, local_files_only=True).to(device)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    steps = math.ceil(len(train) / 32) * 2
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(steps * .06), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(42)

    def batch_for(indices):
        examples = [{key: value[i] for key, value in tokens.items()} for i in indices]
        return {key: value.to(device) for key, value in collator(examples).items()}

    def holdout_loss():
        # The same random masks before/after, without perturbing training RNG.
        # Only used for reporting: never for gradient updates or model selection.
        model.eval()
        total, count = 0., 0
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(1042)
            torch.cuda.manual_seed_all(1042)
            for offset in range(0, len(holdout), 32):
                batch = batch_for(holdout[offset:offset + 32])
                masked = int((batch["labels"] != -100).sum().item())
                if not masked:
                    continue
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite diagnostic MLM loss")
                total += loss.item() * masked
                count += masked
        if not count:
            raise ValueError("No masked holdout tokens")
        return total / count

    last = out / "ckpt_last.pt"
    state = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    start, history = 0, []
    if state is not None:
        if state["config"] != CONFIG or state.get("stage") != "tapt" or state["identity"] != identity:
            raise ValueError("TAPT checkpoint mismatch or missing completed encoder")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, history, before = state["epoch"], state["history"], state["holdout_mlm_loss_before"]
        random.setstate(state["rng"]["random"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if amp:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["rng"]["generator"])
    else:
        diagnostics = out / "tapt_before.json"
        if diagnostics.exists():
            record = json.loads(diagnostics.read_text())
            if record["identity"] != identity:
                raise ValueError("TAPT diagnostic identity mismatch")
            before = record["loss"]
        else:
            before = holdout_loss()
            write_json(diagnostics, dict(identity=identity, loss=before))
        print(f"TAPT holdout MLM loss before={before:.6f} (diagnostic only)", flush=True)

    for epoch in range(start + 1, 3):
        model.train()
        order = torch.tensor(train)[torch.randperm(len(train), generator=generator)].tolist()
        total, masked_total = 0., 0
        for step, offset in enumerate(range(0, len(order), 32)):
            batch = batch_for(order[offset:offset + 32])
            masked = int((batch["labels"] != -100).sum().item())
            if not masked:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite TAPT loss epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            total += loss.item() * masked
            masked_total += masked
            if step % 200 == 0:
                print(f"TAPT epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        if not masked_total:
            raise ValueError("No masked Training tokens")
        history.append(dict(epoch=epoch, train_mlm_loss=total / masked_total))
        state = dict(config=CONFIG, identity=identity, stage="tapt", seed=42, epoch=epoch,
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                     holdout_mlm_loss_before=before,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if amp else [], generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        write_json(out / "tapt_history.json", history)
    after = holdout_loss()
    metrics = dict(seed=42, epochs=2, train_rows=len(train), eval_rows=len(holdout),
                   holdout_mlm_loss_before=before, holdout_mlm_loss_after=after,
                   history=history, encoder=str(encoder), diagnostic_only=True)
    encoder.mkdir(parents=True, exist_ok=True)
    # Completion marker is written last; partial save_pretrained is rebuilt from epoch 2.
    model.bert.save_pretrained(encoder)
    tokenizer.save_pretrained(encoder)
    write_json(marker, dict(identity=identity, metrics=metrics))
    write_json(out / "tapt_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return encoder, metrics


def run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight):
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
    last = out / "ckpt_last.pt"
    start, history = 0, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["config"] != CONFIG:
            raise ValueError("Checkpoint configuration mismatch")
        if state.get("stage") == "finetune" and state.get("seed") == seed:
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
        total = 0.0
        for step, offset in enumerate(range(0, len(order), 16)):
            batch = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(ids[batch].to(device), masks[batch].to(device)), targets[batch].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss seed={seed} epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
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
                      train_loss=total / (step + 1))
        history.append(result)
        state = dict(config=CONFIG, stage="finetune", seed=seed, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=None, scaler=scaler.state_dict(), history=history,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if amp else [], generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        write_json(result_path.with_name("history.json"), history)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    # Final epoch is the experiment score; no epoch or threshold selection.
    final = dict(config=CONFIG, seed=seed, epoch=3, model=model.state_dict(),
                 model_local_path=str(model_path), pos_weight=pos_weight)
    atomic_write(result_path.with_name("best.ckpt"), lambda p: torch.save(final, p))
    write_json(result_path, history[-1])
    return history[-1]


def smoke():
    # Standard-library checks only; no torch import, model download, or training.
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
    print("smoke OK: weight formula, clipping, text whitelist, atomic artifact writes; no training")


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
    model_path, tokenizer = prepare_model()  # Before all data preparation.
    import torch
    import numpy as np
    out = Path(args.out)
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != CONFIG:
        raise ValueError("Output directory belongs to a different configuration")
    write_json(out / "config.json", CONFIG)
    if (out / "metrics.json").exists():
        if not (out / "best.ckpt").exists():
            raise ValueError("Completed metrics exist without best.ckpt")
        print("All seeds complete; reusing metrics.json", flush=True)
        return
    rows = prepare_rows(out, work)
    train, holdout = prepare_split(out, rows)
    if (len(train), len(holdout)) != (23408, 5792):
        raise ValueError(f"Expected 005 split 23408/5792, got {len(train)}/{len(holdout)}")
    pos_weight = weights([rows[i]["labels"] for i in train])
    write_json(out / "pos_weight.json", dict(formula=CONFIG["weight_formula"], weights=dict(zip(LABELS, pos_weight)),
                                              train_positive_counts=np.array([rows[i]["labels"] for i in train]).sum(0).tolist()))
    cache = out / "tokens.pt"
    if cache.exists():
        tokens = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                max_length=256, return_tensors="pt"))
        atomic_write(cache, lambda p: torch.save(tokens, p))
    model_path, tapt_metrics = run_tapt(out, model_path, tokenizer, tokens, train, holdout)
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results = [run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight) for seed in SEEDS]
    # Fixed representative seed 42, not a seed selected using holdout performance.
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "seed_42" / "best.ckpt", p))
    mean_macro = sum(r["macro_f1"] for r in results) / 3
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name="macro_f1", value=mean_macro, split="internal_holdout"),
                   metrics=dict(tapt=tapt_metrics, seeds=results,
                                exceeds_baseline_seeds=sum(r["macro_f1"] > .6092 for r in results),
                                success=mean_macro >= .6192 and sum(r["macro_f1"] > .6092 for r in results) >= 2,
                                mean_micro_f1=sum(r["micro_f1"] for r in results) / 3,
                                mean_per_symptom_f1={s: sum(r["per_symptom_f1"][s] for r in results) / 3 for s in LABELS}),
                   seed=42, seeds=SEEDS, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3; threshold >=0.5 fixed; seed42 representative checkpoint. "
                   "M3-KCBERT-005 settings and split reused; train-only MLM 2 epochs shared across seeds; "
                   "pos_weight scale 0.5, cap 6.0; "
                   "weights computed from Training partition only; official Validation unused.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
