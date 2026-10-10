"""M3-ROBERTA-006: joint auxiliary probabilities feed the nine-symptom head.

Requires the frozen ROBERTA-005 auxiliary label list and P0 split manifest.
No label selection, hyperparameter search, official Validation or extra seeds.
Final epoch 3 is exported; best.ckpt is a fixed artifact name, not selection.
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

EXP_ID = "M3-ROBERTA-006"
MODEL_ID = "klue/roberta-large"
# Immutable safetensors addition commit, rather than the moving main ref:
# https://huggingface.co/klue/roberta-large/commits/main
REVISION = "28d9112"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEEDS = [42]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=1e-5, dropout=0.1, batch_size=16, micro_batch_size=8,
              accumulation_steps=2, weight_decay=0.01, warmup_ratio=0.1, fp16=True, threshold=0.5,
              split_seed=42, holdout_ratio=0.2, labels=LABELS,
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)",
              baseline_source="M3-ROBERTA-003", baseline_macro_f1=0.6441,
              aux_loss_weight=0.5, aux_labels_count=20, aux_detach=False,
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
    positive = [sum(row[j] for row in labels) for j in range(len(labels[0]))]
    return [min(6.0, max(1.0, math.sqrt((len(labels) - p) / max(p, 1)) * .5)) for p in positive]


def transcript(doc):
    return " ".join(text for u in doc.get("utterances", []) if isinstance(u, dict)
                    for text in [re.sub(r"\s+", " ", u.get("text", "")).strip()
                                 if isinstance(u.get("text", ""), str) else ""] if text)


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "roberta-large"
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
                         text=transcript(doc), symptoms=sorted(symptoms), labels=[int(s in symptoms) for s in LABELS]))
    if not rows:
        raise ValueError("No Training label JSONs in TL_서울_구급.zip")
    if len({r["call_id"] for r in rows}) != len(rows):
        raise ValueError("Multiple rows per call: cannot preserve P0 row split without call leakage")
    write_json(cache, rows)
    return rows


def load_contract(aux_path, split_path, expected_sha):
    """Read frozen upstream artifacts; never rank labels using holdout data."""
    aux_raw = Path(aux_path).read_bytes()
    obj = json.loads(aux_raw)
    labels = obj if isinstance(obj, list) else obj.get("aux_labels", obj.get("labels"))
    if not isinstance(labels, list) or len(labels) != 20 or any(not isinstance(x, str) for x in labels):
        raise ValueError("005 aux_labels.json must contain an ordered list of 20 label names")
    if len(set(labels)) != 20 or set(labels) & set(LABELS):
        raise ValueError("Auxiliary labels must be unique and outside the nine main symptoms")
    raw = Path(split_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha.lower():
        raise ValueError("P0 split_sha256 mismatch")
    return labels, raw, hashlib.sha256(aux_raw).hexdigest()


def prepare_split(out, rows, raw):
    split = json.loads(raw)
    by_file = {r["file"]: i for i, r in enumerate(rows)}
    if len(by_file) != len(rows):
        raise ValueError("Duplicate filenames")
    groups = []
    for key in ("train", "holdout"):
        values = split[key]
        if all(isinstance(v, str) for v in values):
            groups.append([by_file[unicodedata.normalize("NFC", Path(v).name)] for v in values])
        elif all(type(v) is int for v in values):
            digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            baseline_rows = [{k: v for k, v in r.items() if k != "symptoms"} for r in rows]
            baseline_digest = hashlib.sha256(json.dumps(baseline_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if split.get("rows_sha256") not in (digest, baseline_digest):
                raise ValueError("Index split requires a matching rows_sha256")
            groups.append(values)
        else:
            raise ValueError("Unsupported split manifest: expected filenames or checked row indices")
    train, holdout = groups
    if not train or not holdout or sorted(train + holdout) != list(range(len(rows))):
        raise ValueError("Invalid split partition")
    if abs(len(holdout) / len(rows) - .2) > .02:
        raise ValueError("Expected P0 80:20 split")
    if {rows[i]["call_id"] for i in train} & {rows[i]["call_id"] for i in holdout}:
        raise ValueError("call_id leakage")
    atomic_write(out / "split.json", lambda p: p.write_bytes(raw))
    atomic_write(out / "val_files.txt", lambda p: p.write_text(
        "\n".join(sorted(rows[i]["file"] for i in holdout)) + "\n", encoding="utf-8"))
    return train, holdout


def hierarchical_logits(h, aux_head, main_head):
    import torch
    aux_logits = aux_head(h)
    return main_head(torch.cat((h, aux_logits.sigmoid()), dim=-1)), aux_logits


def calibration(probs, truth):
    import numpy as np
    p, y = probs.ravel(), truth.ravel()
    bins = np.minimum((p * 15).astype(int), 14)
    ece = sum(float((bins == i).mean()) * abs(float(p[bins == i].mean() - y[bins == i].mean()))
              for i in range(15) if (bins == i).any())
    return dict(ece=ece, brier=float(np.mean((p - y) ** 2)),
                ece_definition="15 equal-width bins; flattened label-positive probabilities")


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


def run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight, aux_targets, aux_weight):
    import numpy as np
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup
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
            self.bert.gradient_checkpointing_enable()
            self.dropout = torch.nn.Dropout(.1)
            self.aux_head = torch.nn.Linear(self.bert.config.hidden_size, 20)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size + 20, len(LABELS))

        def forward(self, ids, mask):
            h = self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
            return hierarchical_logits(h, self.aux_head, self.classifier)

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
    aux_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(aux_weight, device=device))
    last = out / "ckpt_last.pt"
    start, history = 0, []
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
                    logits, aux_logits = model(ids[batch].to(device), masks[batch].to(device))
                loss, bce, soft_f1_loss = loss_terms(logits, targets[batch].to(device), criterion)
                aux_loss, _, _ = loss_terms(aux_logits, aux_targets[batch].to(device), aux_criterion)
                loss = loss + CONFIG["aux_loss_weight"] * aux_loss
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
        probabilities, aux_probabilities = [], []
        with torch.no_grad():
            for offset in range(0, len(holdout), CONFIG["micro_batch_size"]):
                batch = holdout[offset:offset + CONFIG["micro_batch_size"]]
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits, aux_logits = model(ids[batch].to(device), masks[batch].to(device))
                probabilities.append(logits.float().sigmoid().cpu())
                aux_probabilities.append(aux_logits.float().sigmoid().cpu())
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
        result.update(calibration(probs, truth))
        result["aux_macro_f1"] = float(f1_score(aux_targets[holdout].numpy(),
            (torch.cat(aux_probabilities).numpy() >= .5).astype(int), average="macro", zero_division=0))
        w = model.classifier.weight.detach().float()
        result["head_h_mean_abs_weight"] = float(w[:, :-20].abs().mean().item())
        result["head_p_aux_mean_abs_weight"] = float(w[:, -20:].abs().mean().item())
        result["training_collapse"] = result["macro_f1"] < 0.3
        history.append(result)
        state = dict(config=CONFIG, seed=seed, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if amp else [], generator=generator.get_state()))
        atomic_write(last, lambda p: torch.save(state, p))
        write_json(result_path.with_name("history.json"), history)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    # Final epoch is the experiment score; no epoch or threshold selection.
    final = dict(config=CONFIG, seed=seed, epoch=3, model=model.state_dict(),
                 model_local_path=str(model_path), pos_weight=pos_weight, aux_pos_weight=aux_weight,
                 aux_labels=CONFIG["aux_labels"], architecture="concat_cls_sigmoid_aux")
    atomic_write(result_path.with_name("best.ckpt"), lambda p: torch.save(final, p))
    summary = dict(history[-1], history=history,
                   training_collapse=any(r["training_collapse"] for r in history),
                   collapse_epochs=[r["epoch"] for r in history if r["training_collapse"]])
    write_json(result_path, summary)
    return summary


def smoke():
    """Synthetic checks; no encoder load, download, data access or training."""
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.set_num_threads(1)
        h = torch.ones(2, 4, requires_grad=True)
        aux = torch.nn.Linear(4, 20)
        main = torch.nn.Linear(24, 9)
        logits, aux_logits = hierarchical_logits(h, aux, main)
        loss, _, _ = loss_terms(logits, torch.zeros(2, 9), torch.nn.BCEWithLogitsLoss())
        loss.backward()
        assert aux.weight.grad is not None and aux.weight.grad.abs().sum() > 0
        assert logits.shape == (2, 9) and aux_logits.shape == (2, 20)
        assert torch.isfinite(h.grad).all()
        aux_loss, _, _ = loss_terms(aux_logits.detach(), torch.zeros(2, 20), torch.nn.BCEWithLogitsLoss())
        assert torch.isfinite(aux_loss)
    else:
        print("torch unavailable: tensor gradient checks skipped")
    assert weights([[1] * 20, [0] * 20]) == [1.] * 20
    assert weights([[0] * 20] * 1000) == [6.] * 20
    assert SEEDS == [42] and CONFIG["epochs"] == 3
    assert CONFIG["micro_batch_size"] * CONFIG["accumulation_steps"] == 16
    assert transcript({"utterances": [{"text": " 가\n나 ", "Speaker": "ignored"}]}) == "가 나"
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        aux = root / "aux_labels.json"
        write_json(aux, [f"aux{i}" for i in range(20)])
        split = root / "p0.json"
        write_json(split, {"train": [f"{i}.json" for i in range(4)], "holdout": ["4.json"]})
        sha = hashlib.sha256(split.read_bytes()).hexdigest()
        labels, raw, _ = load_contract(aux, split, sha)
        rows = [dict(file=f"{i}.json", call_id=str(i)) for i in range(5)]
        train, holdout = prepare_split(root / "out", rows, raw)
        assert train == [0, 1, 2, 3] and holdout == [4] and len(labels) == 20
        assert (root / "out" / "split.json").read_bytes() == raw
        rows[4]["call_id"] = "0"
        try:
            prepare_split(root / "out", rows, raw)
        except ValueError:
            pass
        else:
            raise AssertionError("Leakage not rejected")
    print("smoke OK: frozen artifacts, call split, atomic writes, hierarchical gradient when torch available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aux-labels", default=os.environ.get("LAB_AUX_LABELS"), help="ROBERTA-005 frozen aux_labels.json")
    parser.add_argument("--split", default=os.environ.get("LAB_P0_SPLIT"), help="P0 split.json with filenames or verified row indices")
    parser.add_argument("--split-sha256", default=os.environ.get("LAB_P0_SPLIT_SHA256"))
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.out:
        parser.error("--out or LAB_OUT required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    if not args.aux_labels or not args.split or not args.split_sha256:
        parser.error("--aux-labels, --split, --split-sha256 (or LAB_AUX_LABELS/LAB_P0_SPLIT/LAB_P0_SPLIT_SHA256) required")
    aux_labels, split_raw, aux_sha = load_contract(args.aux_labels, args.split, args.split_sha256)
    CONFIG.update(aux_labels=aux_labels, aux_labels_sha256=aux_sha, split_sha256=args.split_sha256.lower())
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
    train, holdout = prepare_split(out, rows, split_raw)
    write_json(out / "aux_labels.json", aux_labels)
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
    aux_targets = torch.tensor([[int(label in r["symptoms"]) for label in aux_labels] for r in rows], dtype=torch.float32)
    aux_weight = weights(aux_targets[train].tolist())
    write_json(out / "aux_pos_weight.json", dict(labels=aux_labels, weights=aux_weight,
               train_positive_counts=aux_targets[train].sum(0).tolist()))
    result = run_seed(42, out, model_path, tokens, targets, train, holdout, pos_weight, aux_targets, aux_weight)
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "seed_42" / "best.ckpt", p))
    result["training_seconds"] = sum(r["epoch_seconds"] for r in result["history"])
    result["success_criteria_met"] = result["macro_f1"] >= .6541
    result["split_sha256"] = CONFIG["split_sha256"]
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name="macro_f1", value=result["macro_f1"], split="internal_holdout"),
                   metrics=result, seed=42, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3, threshold >=0.5; text-only inference; fixed 005 aux labels; "
                   "joint concat(h,sigmoid(aux_logits)); main loss + 0.5 auxiliary loss; official Validation unused.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
