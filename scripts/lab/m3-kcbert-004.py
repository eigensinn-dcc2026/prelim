"""M3-KCBERT-004: head+tail-256 KcBERT, softened weighted BCE, three seeds.

Reuses origin/sooyah/lab-m3-kcbert-003:scripts/lab/m3-kcbert-003.py.
Only token selection changes: preserve short inputs, otherwise take the first
127 and last 127 content tokens, with CLS/SEP giving 256 tokens total.
Split, softened BCE, threshold, final epoch scoring and seeds stay identical.
Only Training/TL_서울_구급.zip is read. Official Validation is never opened.
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

EXP_ID = "M3-KCBERT-004"
MODEL_ID = "beomi/kcbert-base"
# Immutable safetensors addition commit, rather than the moving main ref:
# https://huggingface.co/beomi/kcbert-base/tree/0f2f3f8
REVISION = "0f2f3f8"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEEDS = [42, 43, 44]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=2e-5, dropout=0.1, batch_size=16, threshold=0.5,
              split_seed=42, holdout_ratio=0.2, labels=LABELS,
              weight_formula="clip(0.25 * sqrt(neg / max(pos, 1)), 1, 3)",
              baseline_source="M3-KCBERT-003 (origin/sooyah/lab-m3-kcbert-003)",
              token_selection="head127+tail127+CLS/SEP; short inputs unchanged")


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
    return [min(3.0, max(1.0, math.sqrt((len(labels) - p) / max(p, 1)) * .25)) for p in positive]


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
    return train, holdout


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # Preserve baseline constant LR; scheduler state is explicitly None.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
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
        pred = (probs > .5).astype(int)
        result = dict(seed=seed, epoch=epoch, macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                      micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                      per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                      train_loss=total / (step + 1))
        history.append(result)
        state = dict(config=CONFIG, seed=seed, epoch=epoch, model=model.state_dict(),
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


def select_content_tokens(ids, special_count):
    """Keep all short inputs; split the available content budget across both ends."""
    budget = CONFIG["max_length"] - special_count
    if budget < 2:
        raise ValueError("Tokenizer leaves insufficient room for head+tail")
    if len(ids) <= budget:
        return ids
    head = (budget + 1) // 2
    tail = budget - head
    return ids[:head] + ids[-tail:]


def build_head_tail_input(tokenizer, content):
    """Build BERT inputs using token IDs supported by tokenizer wrappers."""
    if any(token_id is None for token_id in
           (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)):
        raise ValueError("KcBERT tokenizer must define CLS, SEP and PAD token IDs")
    ids = [tokenizer.cls_token_id] + select_content_tokens(content, 2) + [tokenizer.sep_token_id]
    padding = CONFIG["max_length"] - len(ids)
    return ids + [tokenizer.pad_token_id] * padding, [1] * len(ids) + [0] * padding


def tokenize_head_tail(tokenizer, texts):
    import torch
    input_ids, attention_mask = [], []
    # Tokenize in bounded batches without truncating away the tail.
    for offset in range(0, len(texts), 256):
        encoded = tokenizer(texts[offset:offset + 256], add_special_tokens=False,
                            truncation=False, padding=False, return_attention_mask=False,
                            return_token_type_ids=False)
        for content in encoded["input_ids"]:
            ids, mask = build_head_tail_input(tokenizer, content)
            input_ids.append(ids)
            attention_mask.append(mask)
    return dict(input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long))


def smoke():
    # Standard-library checks only; no torch import, model download, or training.
    from types import SimpleNamespace
    # The Colab wrapper exposes token IDs but no special-token builder method.
    tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102, pad_token_id=0)
    for length in (0, 1, 127, 128, 253, 254, 255, 256, 1000):
        ids = list(range(length))
        selected = select_content_tokens(ids, 2)
        assert len(selected) + 2 <= 256
        if length <= 254:
            assert selected == ids
        else:
            assert selected == ids[:127] + ids[-127:]
            assert len(set(selected)) == 254
        built, mask = build_head_tail_input(tokenizer, ids)
        active = len(selected) + 2
        assert built[:active] == [101] + selected + [102]
        assert built[active:] == [0] * (256 - active)
        assert mask == [1] * active + [0] * (256 - active)
        assert len(built) == len(mask) == 256
    rows = [[1] * 9] + [[0] * 9] * 99
    assert weights(rows) == [math.sqrt(99) * .25] * 9
    assert weights([[0] * 9] * 1000) == [3.] * 9
    assert weights([[1] * 9] * 2) == [1.] * 9
    assert transcript({"utterances": [{"text": " 가\n 나 "}, {"text": "다", "Speaker": "ignored"}]}) == "가 나 다"
    with tempfile.TemporaryDirectory() as name:
        p = Path(name) / "check.json"
        write_json(p, {"exp_id": EXP_ID})
        assert json.loads(p.read_text())["exp_id"] == EXP_ID
        assert list(p.parent.iterdir()) == [p]
    print("smoke OK: head/tail boundary selection, weight formula, clipping, text whitelist, atomic artifact writes; no training")


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
    pos_weight = weights([rows[i]["labels"] for i in train])
    write_json(out / "pos_weight.json", dict(formula=CONFIG["weight_formula"], weights=dict(zip(LABELS, pos_weight)),
                                              train_positive_counts=np.array([rows[i]["labels"] for i in train]).sum(0).tolist()))
    cache = out / "tokens.pt"
    if cache.exists():
        tokens = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        tokens = tokenize_head_tail(tokenizer, [r["text"] for r in rows])
        atomic_write(cache, lambda p: torch.save(tokens, p))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results = [run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight) for seed in SEEDS]
    # Fixed representative seed 42, not a seed selected using holdout performance.
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "seed_42" / "best.ckpt", p))
    mean_macro = sum(r["macro_f1"] for r in results) / 3
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name="macro_f1", value=mean_macro, split="internal_holdout"),
                   metrics=dict(seeds=results, mean_micro_f1=sum(r["micro_f1"] for r in results) / 3,
                                mean_per_symptom_f1={s: sum(r["per_symptom_f1"][s] for r in results) / 3 for s in LABELS}),
                   seed=42, seeds=SEEDS, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3; threshold >0.5 fixed; seed42 representative checkpoint. "
                   "Head127+tail127 content tokens plus CLS/SEP; KCBERT-003 training settings preserved.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
