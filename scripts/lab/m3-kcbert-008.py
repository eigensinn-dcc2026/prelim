"""M3-KCBERT-008: overlapping KcBERT windows, mean call logits, three seeds.

Preserves 005's split, loss, optimizer, threshold and final-epoch scoring.
Windows contain at most 256 tokens including special tokens, start every
128 content tokens, and are capped at three per call. Only Training text
is used. Chunk micro-batches use activation checkpointing to accumulate
exact gradients through the call-level mean while bounding encoder memory.
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

EXP_ID = "M3-KCBERT-008"
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
              stride=128, max_chunks=3, chunk_micro_batch=16,
              baseline_source="M3-KCBERT-005; overlapping windows and mean logits only")


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
    return train, holdout


def chunk_windows(content_ids, special_tokens=2):
    """Stride is the advance in content tokens; special tokens fit in 256."""
    capacity = CONFIG["max_length"] - special_tokens
    if capacity < CONFIG["stride"]:
        raise ValueError("Tokenizer special tokens leave insufficient window capacity")
    windows = []
    for start in range(0, max(1, len(content_ids)), CONFIG["stride"]):
        windows.append(content_ids[start:start + capacity])
        if start + capacity >= len(content_ids) or len(windows) == CONFIG["max_chunks"]:
            break
    return windows


def encode_call(text, tokenizer):
    """Use only the wrapper's callable interface and special-token IDs."""
    cls_id, sep_id, pad_id = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                              tokenizer.pad_token_id)
    if any(token_id is None for token_id in (cls_id, sep_id, pad_id)):
        raise ValueError("KcBERT tokenizer requires CLS, SEP and PAD token IDs")
    content = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    for window in chunk_windows(content):
        ids = [cls_id] + list(window) + [sep_id]
        padding = CONFIG["max_length"] - len(ids)
        yield ids + [pad_id] * padding, [1] * len(ids) + [0] * padding


def tokenize_calls(rows, tokenizer):
    import torch
    ids, masks, offsets = [], [], [0]
    for row in rows:
        for input_ids, attention_mask in encode_call(row["text"], tokenizer):
            ids.append(input_ids)
            masks.append(attention_mask)
        offsets.append(len(ids))
    return dict(input_ids=torch.tensor(ids, dtype=torch.long),
                attention_mask=torch.tensor(masks, dtype=torch.long),
                offsets=torch.tensor(offsets, dtype=torch.long))


def call_logits(model, tokens, batch, device, checkpoint_chunks=False):
    import torch
    from torch.utils.checkpoint import checkpoint
    indices, owners, counts = [], [], []
    offsets = tokens["offsets"].tolist()
    for owner, row in enumerate(batch):
        row = int(row)
        start, end = offsets[row:row + 2]
        indices.extend(range(start, end))
        owners.extend([owner] * (end - start))
        counts.append(end - start)
    outputs = []
    for start in range(0, len(indices), CONFIG["chunk_micro_batch"]):
        chunk = indices[start:start + CONFIG["chunk_micro_batch"]]
        ids = tokens["input_ids"][chunk].to(device)
        masks = tokens["attention_mask"][chunk].to(device)
        # Backward recomputes each micro-batch, preserving dropout RNG state.
        outputs.append(checkpoint(model, ids, masks, use_reentrant=False,
                                  preserve_rng_state=True) if checkpoint_chunks else model(ids, masks))
    logits = torch.cat(outputs).float()
    sums = logits.new_zeros((len(batch), len(LABELS))).index_add(
        0, torch.tensor(owners, device=device), logits)
    return sums / torch.tensor(counts, device=device).unsqueeze(1)


def chunk_distribution(tokens, indices):
    offsets = tokens["offsets"].tolist()
    counts = [offsets[i + 1] - offsets[i] for i in indices]
    return {str(n): counts.count(n) for n in range(1, CONFIG["max_chunks"] + 1)}


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
    # Preserve 005 constant LR; scheduler state is explicitly None.
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
    train = torch.tensor(train)
    for epoch in range(start + 1, 4):
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        total = 0.0
        for step, offset in enumerate(range(0, len(order), 16)):
            batch = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(call_logits(model, tokens, batch, device, checkpoint_chunks=True), targets[batch].to(device))
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
                    logits = call_logits(model, tokens, batch, device)
                probabilities.append(logits.float().sigmoid().cpu())
        probs = torch.cat(probabilities).numpy()
        truth = targets[holdout].numpy()
        pred = (probs > .5).astype(int)
        result = dict(seed=seed, epoch=epoch, macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
                      micro_f1=float(f1_score(truth, pred, average="micro", zero_division=0)),
                      per_symptom_f1=dict(zip(LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                      train_loss=total / (step + 1),
                      chunk_count_distribution=dict(train=chunk_distribution(tokens, train.tolist()),
                                                    holdout=chunk_distribution(tokens, holdout)))
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


def smoke():
    # Standard-library checks only; no torch import, model download, or training.
    rows = [[1] * 9] + [[0] * 9] * 99
    assert weights(rows) == [math.sqrt(99) * .5] * 9
    assert weights([[0] * 9] * 1000) == [6.] * 9
    assert weights([[1] * 9] * 2) == [1.] * 9
    assert transcript({"utterances": [{"text": " 가\n 나 "}, {"text": "다", "Speaker": "ignored"}]}) == "가 나 다"
    assert chunk_windows([]) == [[]]
    assert [len(w) for w in chunk_windows(list(range(254)))] == [254]
    assert [len(w) for w in chunk_windows(list(range(255)))] == [254, 127]
    windows = chunk_windows(list(range(1000)))
    assert len(windows) == 3
    assert [w[0] for w in windows] == [0, 128, 256]
    assert windows[0][128:] == windows[1][:126]
    assert windows[-1][-1] == 509
    # Colab wrappers need not expose prepare_for_model, pad, or special-token
    # builder methods. Exercise the same encoding path used by token caching.
    class WrapperTokenizer:
        cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

        def __call__(self, text, *, add_special_tokens, truncation):
            assert not add_special_tokens and not truncation
            return {"input_ids": list(range(1000, 1000 + int(text)))}

    tokenizer = WrapperTokenizer()
    for length in (0, 1, 254, 255, 382, 383, 1000):
        encoded = list(encode_call(str(length), tokenizer))
        expected = chunk_windows(list(range(1000, 1000 + length)))
        assert len(encoded) == len(expected)
        for (ids, mask), window in zip(encoded, expected):
            active = len(window) + 2
            assert len(ids) == len(mask) == 256
            assert ids[:active] == [101] + window + [102]
            assert ids[active:] == [0] * (256 - active)
            assert mask == [1] * active + [0] * (256 - active)
    with tempfile.TemporaryDirectory() as name:
        p = Path(name) / "check.json"
        write_json(p, {"exp_id": EXP_ID})
        assert json.loads(p.read_text())["exp_id"] == EXP_ID
        assert list(p.parent.iterdir()) == [p]
    print("smoke OK: wrapper-compatible special tokens, padding/masks, window boundaries, overlap, cap, weight formula, text whitelist, atomic writes; no training")


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
        tokens = tokenize_calls(rows, tokenizer)
        atomic_write(cache, lambda p: torch.save(tokens, p))
    if len(tokens["offsets"]) != len(rows) + 1:
        raise ValueError("Token cache does not match call rows")
    write_json(out / "chunk_counts.json", dict(all=chunk_distribution(tokens, range(len(rows))),
                                               train=chunk_distribution(tokens, train),
                                               holdout=chunk_distribution(tokens, holdout)))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results = [run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight) for seed in SEEDS]
    # Fixed representative seed 42, not a seed selected using holdout performance.
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / "seed_42" / "best.ckpt", p))
    mean_macro = sum(r["macro_f1"] for r in results) / 3
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name="macro_f1", value=mean_macro, split="internal_holdout"),
                   metrics=dict(seeds=results, chunk_count_distribution=json.loads((out / "chunk_counts.json").read_text()),
                                baseline_macro_f1=0.6092,
                                success=mean_macro >= 0.6192 and sum(r["macro_f1"] > 0.6092 for r in results) >= 2,
                                mean_micro_f1=sum(r["micro_f1"] for r in results) / 3,
                                mean_per_symptom_f1={s: sum(r["per_symptom_f1"][s] for r in results) / 3 for s in LABELS}),
                   seed=42, seeds=SEEDS, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3; threshold >0.5 fixed; seed42 representative checkpoint. "
                   "M3-KCBERT-005 settings and split reused; 256-token windows, stride 128, max 3; mean logits; "
                   "weights computed from Training partition only; official Validation unused.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
