"""M3-QWEN-001: Qwen2.5-1.5B LoRA, Training-only 80:20 seed42.

Same transcript, split and fp32 microbatch loss as M3-ROBERTA-003.
Fixed final epoch, threshold >= .5, no official Validation or postprocessing.
Only --smoke is intended for local validation; full execution requires Colab CUDA.
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
import time
import unicodedata
import zipfile

EXP_ID = "M3-QWEN-001"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
# Immutable commit from https://huggingface.co/Qwen/Qwen2.5-1.5B/commits/main
REVISION = "8faed76"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=2, seed=42, lr=1e-4, batch_size=16, micro_batch_size=2,
              accumulation_steps=8, weight_decay=0.0, warmup_ratio=0.05,
              fp16=True, threshold=0.5, split_seed=42, holdout_ratio=0.2,
              labels=LABELS, lora_r=16, lora_alpha=32, lora_dropout=0.05,
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
              modules_to_save=["score"], peft_version="0.13.2",
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)",
              soft_f1_weight=1.0, soft_f1_epsilon=1e-8,
              baseline_macro_f1=0.6441, baseline_seed_variation=0.002)


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
    import torch
    import peft
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if peft.__version__ != "0.13.2":
        raise RuntimeError("Install peft==0.13.2 before running this experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "qwen2.5-1.5b"
    marker = root / "dcc_revision.json"
    expected = {"repo_id": MODEL_ID, "revision": REVISION}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError(f"Model cache revision differs: {root}")
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        root, local_files_only=True, use_safetensors=True, num_labels=len(LABELS),
        torch_dtype=torch.float16, problem_type="multi_label_classification")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    write_json(marker, expected)
    return root, tokenizer, model


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
    train, holdout = split["train"], sorted(split["holdout"], key=lambda i: rows[i]["file"])
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


def loss_terms(logits, targets, criterion):
    """Keep both terms in fp32, including under the encoder's AMP context."""
    import torch
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, targets = logits.float(), targets.float()
        bce = criterion(logits, targets)
        probabilities = logits.sigmoid()
        soft_f1 = (2 * (probabilities * targets).sum(dim=0) /
                   (probabilities.sum(dim=0) + targets.sum(dim=0) + 1e-8))
        soft_f1_loss = 1 - soft_f1.mean()
        return bce + soft_f1_loss, bce, soft_f1_loss


def evaluate(model, tokens, targets, holdout, device):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    probabilities = []
    with torch.no_grad():
        for offset in range(0, len(holdout), CONFIG["micro_batch_size"]):
            batch = holdout[offset:offset + CONFIG["micro_batch_size"]]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: v[batch].to(device) for k, v in tokens.items()}).logits
            probabilities.append(logits.float().sigmoid().cpu())
    probs = torch.cat(probabilities).numpy().astype(np.float32)
    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite holdout probabilities")
    truth = targets[holdout].numpy()
    pred = (probs >= .5).astype(int)
    result = dict(macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                  micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                  per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                  empty_predictions=int((pred.sum(axis=1) == 0).sum()),
                  predicted_positive_counts=pred.sum(0).tolist())
    result["training_collapse"] = result["macro_f1"] < .3 or not pred.any()
    return probs, result


def publish_epoch(out, model_path, model, tokens, targets, train, holdout, history, pos_weight):
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict
    probs, evaluated = evaluate(model, tokens, targets, holdout, torch.device("cuda"))
    history[-1].update(evaluated)
    seed_out = out / "seed_42"
    seed_out.mkdir(exist_ok=True)
    atomic_write(seed_out / "holdout_probs.npy", lambda p: save_array(p, probs, np))
    checkpoint = dict(config=CONFIG, seed=42, epoch=history[-1]["epoch"],
                      adapter=get_peft_model_state_dict(model), model_local_path=str(model_path),
                      pos_weight=pos_weight, labels=LABELS)
    atomic_write(out / "best.ckpt", lambda p: torch.save(checkpoint, p))
    write_json(seed_out / "history.json", history)
    write_json(seed_out / "metrics.json", dict(history[-1], history=history))
    metrics = dict(exp_id=EXP_ID,
                   primary_metric=dict(name="macro_f1", value=evaluated["macro_f1"], split="internal_holdout"),
                   metrics=dict(history[-1], history=history,
                                success_criteria_met=evaluated["macro_f1"] >= .6541,
                                baseline_macro_f1=.6441, baseline_seed_variation=.002,
                                collapse_epochs=[r["epoch"] for r in history if r.get("training_collapse")]),
                   seed=42, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", completed=history[-1]["epoch"] == CONFIG["epochs"],
                   notes="Fixed last completed epoch; threshold >=0.5, empty predictions unchanged. "
                   "Training-only weights and call-disjoint split; official Validation unused. "
                   "Soft F1 computed fp32 per microbatch of 2, effective batch 16. "
                   "Collapse flag: macro F1 <0.3 or all predictions empty. "
                   "Single seed; ROBERTA-003 seed variation ±0.002 is reference only.")
    write_json(out / "metrics.json", metrics)


def save_array(path, array, np):
    with open(path, "wb") as handle:
        np.save(handle, array)


def run(out, model_path, tokenizer, base_model, tokens, targets, train, holdout, pos_weight):
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_linear_schedule_with_warmup
    device = torch.device("cuda")
    model = get_peft_model(base_model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=CONFIG["target_modules"], modules_to_save=["score"], bias="none"))
    # Trainable tensors remain fp32 so GradScaler can unscale gradients safely.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-4, weight_decay=0.0)
    updates = math.ceil(len(train) / 16) * 2
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(updates * .05), updates)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(42)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    start, history = 0, []
    last = out / "ckpt_last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["config"] != CONFIG or state["seed"] != 42:
            raise ValueError("Checkpoint configuration mismatch")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, history = state["epoch"], state["history"]
        random.setstate(state["rng"]["random"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["rng"]["generator"])
        del state
    if start:
        publish_epoch(out, model_path, model, tokens, targets, train, holdout, history, pos_weight)
    for epoch in range(start + 1, CONFIG["epochs"] + 1):
        torch.cuda.synchronize()
        started = time.monotonic()
        model.train()
        order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
        total = total_bce = total_soft = 0.0
        for step, offset in enumerate(range(0, len(order), 16)):
            group = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            for micro_offset in range(0, len(group), CONFIG["micro_batch_size"]):
                batch = group[micro_offset:micro_offset + CONFIG["micro_batch_size"]]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**{k: v[batch].to(device) for k, v in tokens.items()}).logits
                loss, bce, soft = loss_terms(logits, targets[batch].to(device), criterion)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss epoch={epoch} step={step}")
                scaler.scale(loss * (len(batch) / len(group))).backward()
                total += loss.item() * len(batch)
                total_bce += bce.item() * len(batch)
                total_soft += soft.item() * len(batch)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        torch.cuda.synchronize()
        history.append(dict(epoch=epoch, seed=42, train_loss=total / len(train),
                            train_bce=total_bce / len(train), train_soft_f1_loss=total_soft / len(train),
                            epoch_train_seconds=time.monotonic() - started))
        # Commit the resumable state before evaluation/export, so interruptions there
        # never force an already completed training epoch to run again.
        state = dict(config=CONFIG, seed=42, epoch=epoch, adapter=get_peft_model_state_dict(model),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), history=history,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                              torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                              generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        del state
        publish_epoch(out, model_path, model, tokens, targets, train, holdout, history, pos_weight)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    # Also reconstruct final outputs if interruption occurred after ckpt_last commit.
    publish_epoch(out, model_path, model, tokens, targets, train, holdout, history, pos_weight)
    merged = model.merge_and_unload().half()
    merged.config.use_cache = True
    export = out / "merged"
    export.mkdir(exist_ok=True)
    merged.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    write_json(export / "complete.json", dict(config=CONFIG, epoch=2))


def smoke():
    assert CONFIG["micro_batch_size"] * CONFIG["accumulation_steps"] == 16
    assert weights([[0] * 9] * 1000) == [6.] * 9
    assert weights([[1] * 9] * 2) == [1.] * 9
    assert transcript({"utterances": [{"text": " 가\n 나 "}, {"text": "다", "Speaker": "ignored"}]}) == "가 나 다"
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "check.json"
        write_json(path, {"exp_id": EXP_ID})
        assert json.loads(path.read_text())["exp_id"] == EXP_ID
        assert list(path.parent.iterdir()) == [path]
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("torch unavailable: tensor loss checks skipped", flush=True)
    else:
        logits = torch.tensor([[0., 2., -2.], [1., -1., 0.]], requires_grad=True)
        targets = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1., 2., 3.]))
        loss, bce, soft = loss_terms(logits, targets, criterion)
        p = logits.sigmoid()
        expected = 1 - (2 * (p * targets).sum(0) / (p.sum(0) + targets.sum(0) + 1e-8)).mean()
        assert loss.dtype == torch.float32 and torch.allclose(soft, expected)
        assert torch.allclose(loss, bce + expected)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert (torch.tensor([.49, .5, .51]) >= .5).tolist() == [False, True, True]
    print("smoke OK: loss/weights/transcript/atomic writes; no downloads or training")


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
    import numpy as np
    import torch
    # Fix RNG before randomly initializing the new classification head.
    seed_all(42, torch, np)
    model_path, tokenizer, model = prepare_model()  # Fail before data preparation.
    out = Path(args.out)
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise ValueError("Output directory belongs to another configuration")
    write_json(config_path, CONFIG)
    if (out / "merged" / "complete.json").exists():
        if not (out / "best.ckpt").exists() or not (out / "metrics.json").exists():
            raise ValueError("Incomplete final artifacts")
        print("Experiment complete; reusing outputs", flush=True)
        return
    rows = prepare_rows(out, work)
    train, holdout = prepare_split(out, rows)
    seed_out = out / "seed_42"
    seed_out.mkdir(exist_ok=True)
    atomic_write(seed_out / "val_files.txt", lambda p: shutil.copyfile(out / "val_files.txt", p))
    pos_weight = weights([rows[i]["labels"] for i in train])
    write_json(out / "pos_weight.json", dict(formula=CONFIG["weight_formula"],
                                              weights=dict(zip(LABELS, pos_weight)),
                                              train_positive_counts=np.array([rows[i]["labels"] for i in train]).sum(0).tolist()))
    cache = out / "tokens.pt"
    if cache.exists():
        tokens = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        tokens = dict(tokenizer([r["text"] or tokenizer.eos_token for r in rows], padding="max_length", truncation=True,
                                max_length=256, return_token_type_ids=False, return_tensors="pt"))
        atomic_write(cache, lambda p: torch.save(tokens, p))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    run(out, model_path, tokenizer, model, tokens, targets, train, holdout, pos_weight)


if __name__ == "__main__":
    main()
