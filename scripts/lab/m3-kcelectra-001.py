"""M3-KCELECTRA-001: P0 call split, KcELECTRA front-256, seed 42.
Weighted BCE + microbatch fp32 soft-F1; final epoch 3, threshold >= .5.
Only Training Seoul is read. Interrupted epochs restart from the last epoch checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

EXP_ID = "M3-KCELECTRA-001"
MODEL_ID = "beomi/KcELECTRA-base-v2022"
# Immutable model repository commit (short Git SHA), never moving main:
# https://huggingface.co/beomi/KcELECTRA-base-v2022/commits/main
REVISION = "8728557"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEEDS = [42]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=2e-5, dropout=0.1, batch_size=16, micro_batch_size=8,
              accumulation_steps=2, weight_decay=0.01, warmup_ratio=0.1, fp16=True, threshold=0.5,
              split_seed=42, holdout_ratio=0.2, labels=LABELS,
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)",
              baseline_source="M3-ROBERTA-003", baseline_macro_f1=0.6441,
              soft_f1_weight=1.0, soft_f1_epsilon=1e-8, threshold_operator=">=",
              soft_f1_batch_size=8)


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
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "KcELECTRA-base-v2022"
    marker = root / "dcc_revision.json"
    if marker.exists():
        if json.loads(marker.read_text()) != {"repo_id": MODEL_ID, "revision": REVISION}:
            raise ValueError(f"Model cache revision differs: {root}")
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["config.json", "tokenizer*", "vocab.txt", "model.safetensors", "pytorch_model.bin"])
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
    spec = importlib.util.spec_from_file_location(
        "m3_p0", Path(__file__).resolve().parents[1] / "mission3_baseline 최종 수정본.py")
    p0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0)
    df = p0.load_training_dataframe(extracted)
    if len(df) != 29200 or df.attrs["load_failures"]:
        raise ValueError("Expected all 29,200 P0 samples without load failures")
    if list(p0.TARGET_SYMPTOMS) != LABELS:
        raise ValueError("P0 label order mismatch")
    rows = [dict(file=r.label_file_name,
                 call_id=unicodedata.normalize("NFC", Path(r.label_file_name).stem).casefold(),
                 text=r.text, labels=[int(label in r.symptoms) for label in LABELS])
            for r in df.itertuples()]
    if len({r["call_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate P0 call IDs")
    write_json(cache, rows)
    return rows


def prepare_split(out, rows):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = out / "split_indices.json"
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
    if (len(train), len(holdout)) != (23408, 5792):
        raise ValueError("P0 split size mismatch")
    if sorted(train + holdout) != list(range(len(rows))):
        raise ValueError("Invalid split partition")
    if {rows[i]["call_id"] for i in train} & {rows[i]["call_id"] for i in holdout}:
        raise ValueError("call_id leakage")
    # Byte-identical manifest convention to M3-KCBERT-002/P0 experiments.
    manifest = dict(train=[rows[i]["file"] for i in train],
                    holdout=[rows[i]["file"] for i in holdout],
                    seed=42, call_id_rule="JSON filename stem")
    write_json(out / "split.json", manifest)
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


def run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight):
    import numpy as np
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    result_path = out / f"seed_{seed}" / "metrics.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["seed"] != seed or result["epoch"] != 3 or not result_path.with_name("best.ckpt").exists() or not (out / "holdout_probs.npy").exists():
            raise ValueError(f"Incomplete seed artifacts: {result_path}")
        return result
    seed_all(seed, torch, np)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.bert.gradient_checkpointing_enable()
            self.dropout = torch.nn.Dropout(.1)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(LABELS))

        def forward(self, ids, mask):
            return self.classifier(self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]))

    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    updates_per_epoch = math.ceil(len(train) / CONFIG["batch_size"])
    total_updates = updates_per_epoch * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total_updates * CONFIG["warmup_ratio"]),
        num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    last = out / "ckpt_last.pt"
    start, history, probs = 0, [], None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["config"] != CONFIG:
            raise ValueError("Checkpoint configuration mismatch")
        if state["seed"] == seed:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            scheduler.load_state_dict(state["scheduler"])
            start, history = state["epoch"], state["history"]
            probs = state["holdout_probs"]
            random.setstate(state["rng"]["random"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if amp and state["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            generator.set_state(state["rng"]["generator"])
        del state  # Release the large CPU checkpoint before starting training.
    ids, masks = tokens["input_ids"], tokens["attention_mask"]
    train = torch.tensor(train)
    for epoch in range(start + 1, 4):
        epoch_started = time.monotonic()
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        total = total_bce = total_soft_f1_loss = 0.0
        # Accumulate sample-weighted microbatches, including the final partial group.
        for step, offset in enumerate(range(0, len(order), CONFIG["batch_size"])):
            group = order[offset:offset + CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            for micro_offset in range(0, len(group), CONFIG["micro_batch_size"]):
                batch = group[micro_offset:micro_offset + CONFIG["micro_batch_size"]]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(ids[batch].to(device), masks[batch].to(device))
                loss, bce, soft_f1_loss = loss_terms(logits, targets[batch].to(device), criterion)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss seed={seed} epoch={epoch} step={step}")
                scaler.scale(loss * (len(batch) / len(group))).backward()
                total += loss.item() * len(batch)
                total_bce += bce.item() * len(batch)
                total_soft_f1_loss += soft_f1_loss.item() * len(batch)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            if step % 200 == 0:
                print(f"seed={seed} epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for offset in range(0, len(holdout), CONFIG["micro_batch_size"]):
                batch = holdout[offset:offset + CONFIG["micro_batch_size"]]
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(ids[batch].to(device), masks[batch].to(device))
                probabilities.append(logits.float().sigmoid().cpu())
        probs = torch.cat(probabilities).numpy()
        truth = targets[holdout].numpy()
        pred = (probs >= CONFIG["threshold"]).astype(int)
        result = dict(seed=seed, epoch=epoch, macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                      micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                      per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                      empty_predictions=int((pred.sum(axis=1) == 0).sum()),
                      train_loss=total / len(train), train_bce=total_bce / len(train),
                      train_soft_f1_loss=total_soft_f1_loss / len(train),
                      train_soft_f1=1 - total_soft_f1_loss / len(train), epoch_seconds=time.monotonic() - epoch_started)
        result["training_collapse"] = result["macro_f1"] < 0.3
        history.append(result)
        state = dict(config=CONFIG, seed=seed, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history, holdout_probs=probs,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if amp else [], generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        write_json(result_path.with_name("history.json"), history)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    # Final epoch is the experiment score; no epoch or threshold selection.
    final = dict(config=CONFIG, seed=seed, epoch=3, model=model.state_dict(),
                 model_local_path=str(model_path), pos_weight=pos_weight)
    atomic_write(result_path.with_name("best.ckpt"), lambda p: torch.save(final, p))
    atomic_write(out / "holdout_probs.npy", lambda p: save_probs(p, probs))
    summary = dict(history[-1], history=history, calibration=calibration(probs, targets[holdout].numpy()),
                   training_collapse=any(r["training_collapse"] for r in history),
                   collapse_epochs=[r["epoch"] for r in history if r["training_collapse"]])
    write_json(result_path, summary)
    return summary


def save_probs(path, probabilities):
    import numpy as np
    with open(path, "wb") as stream:
        np.save(stream, probabilities, allow_pickle=False)


def calibration(probs, truth):
    import numpy as np
    flat, labels = probs.ravel(), truth.ravel()
    bins = np.minimum((flat * 15).astype(int), 14)
    ece = sum(float((bins == b).mean()) *
              abs(float(flat[bins == b].mean() - labels[bins == b].mean()))
              for b in range(15) if (bins == b).any())
    return dict(ece=ece, brier=float(np.mean((probs - truth) ** 2)), ece_bins=15)


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
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        perfect = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
        assert calibration(perfect, perfect)["brier"] == 0
        assert calibration(perfect, perfect)["ece"] == 0
    assert len(REVISION) >= 7 and MODEL_ID == "beomi/KcELECTRA-base-v2022"
    assert CONFIG["micro_batch_size"] * CONFIG["accumulation_steps"] == CONFIG["batch_size"] == 16
    assert CONFIG["soft_f1_weight"] == 1.0 and CONFIG["soft_f1_epsilon"] == 1e-8
    assert CONFIG["lr"] == 2e-5 and CONFIG["warmup_ratio"] == .1
    assert CONFIG["weight_decay"] == .01 and CONFIG["max_length"] == 256
    assert CONFIG["epochs"] == 3 and SEEDS == [42]
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
        if not all((out / name).exists() for name in
                   ["best.ckpt", "holdout_probs.npy", "holdout_ids.json", "split.json"]):
            raise ValueError("Completed metrics exist without required artifacts")
        print("Seed 42 complete; reusing metrics.json", flush=True)
        return
    rows = prepare_rows(out, work)
    train, holdout = prepare_split(out, rows)
    pos_weight = weights([rows[i]["labels"] for i in train])
    write_json(out / "pos_weight.json", dict(formula=CONFIG["weight_formula"], weights=dict(zip(LABELS, pos_weight)),
                                              train_positive_counts=np.array([rows[i]["labels"] for i in train]).sum(0).tolist()))
    cache = out / "tokens.pt"
    if cache.exists():
        tokens = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                max_length=256, return_token_type_ids=False, return_tensors="pt"))
        atomic_write(cache, lambda p: torch.save(tokens, p))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results = [run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight) for seed in SEEDS]
    # Fixed representative seed 42, not a seed selected using holdout performance.
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "seed_42" / "best.ckpt", p))
    result = results[0]
    write_json(out / "holdout_ids.json", [rows[i]["file"] for i in holdout])
    split_sha = hashlib.sha256((out / "split.json").read_bytes()).hexdigest()
    metrics = dict(exp_id=EXP_ID,
                   primary_metric=dict(name="macro_f1", value=result["macro_f1"], split="internal_holdout"),
                   metrics=dict(result, success_criteria_met=result["macro_f1"] >= .6541,
                                training_seconds=sum(r["epoch_seconds"] for r in result["history"])),
                   model_revision=REVISION, model=MODEL_ID, split_sha256=split_sha,
                   seed=42, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3; fixed threshold >=0.5; no postprocessing. "
                   "Training-only pos weights; fp32 soft-F1 microbatch 8, effective batch 16. "
                   "ECE: 15 equal-width bins of flattened label probabilities; Brier: mean squared label error.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
