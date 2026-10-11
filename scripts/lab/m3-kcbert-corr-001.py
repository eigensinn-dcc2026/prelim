"""M3-KCBERT-CORR-001: label-correlation training pilot based on KCBERT-012.

M3-KCBERT-012 split, seeds, epochs, tokenizer, LR and output head are unchanged.
Loss = weighted BCE + soft F1 + 0.2*pair BCE + 0.2*masked-label BCE.\nThis is PLCP/CLCP-inspired, not an exact reproduction of the ACL paper.
Three final-epoch seed scores use probability >= .5 without postprocessing.
Official Validation is never opened.
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

EXP_ID = "M3-KCBERT-CORR-001"
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
              baseline_source="M3-KCBERT-012; auxiliary pair/conditional label loss only",
              soft_f1_weight=1.0, soft_f1_epsilon=1e-8, threshold_operator=">=",
              plcp_weight=0.2, clcp_weight=0.2, aux_training_only=True)


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



def observed_label_mask(targets):
    """Expose a random nonempty proper subset of labels per training example."""
    import torch
    assert targets.ndim == 2 and targets.shape[1] >= 2
    observed = torch.rand(targets.shape, device=targets.device) < 0.5
    for row in range(targets.shape[0]):
        seen = int(observed[row].sum().item())
        if seen == 0:
            idx = int(torch.randint(targets.shape[1], (1,), device=targets.device).item())
            observed[row, idx] = True
        elif seen == targets.shape[1]:
            idx = int(torch.randint(targets.shape[1], (1,), device=targets.device).item())
            observed[row, idx] = False
    return observed


def observed_context(targets, observed):
    """Two channels per label: observed-positive and observed-negative."""
    import torch
    shown = observed.to(targets.dtype)
    return torch.cat((shown * targets, shown * (1.0 - targets)), dim=1)


def correlation_losses(representation, targets, model):
    """PLCP/CLCP-inspired auxiliary tasks. Never called during inference.

    Revealed gold labels are fed ONLY to this auxiliary head during training.
    The main nine-logit classifier never receives gold labels at inference.
    """
    import torch
    import torch.nn.functional as F
    with torch.autocast(device_type=representation.device.type, enabled=False):
        h, y = representation.float(), targets.float()
        i, j = torch.triu_indices(y.shape[1], y.shape[1], offset=1, device=y.device)
        pair_truth = y[:, i] * y[:, j]
        pair_loss = F.binary_cross_entropy_with_logits(model.plcp_head(h), pair_truth)
        observed = observed_label_mask(y)
        context = observed_context(y, observed)
        conditional_logits = model.clcp_head(torch.tanh(h + model.clcp_context(context)))
        withheld = (~observed).to(y.dtype)
        pointwise = F.binary_cross_entropy_with_logits(conditional_logits, y, reduction="none")
        conditional_loss = (pointwise * withheld).sum() / withheld.sum().clamp_min(1.0)
        return pair_loss, conditional_loss

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
            hidden = self.bert.config.hidden_size
            self.classifier = torch.nn.Linear(hidden, len(LABELS))
            # Training-only relationship heads; main nine-label output unchanged.
            self.plcp_head = torch.nn.Linear(hidden, len(LABELS) * (len(LABELS) - 1) // 2)
            self.clcp_context = torch.nn.Linear(2 * len(LABELS), hidden, bias=False)
            self.clcp_head = torch.nn.Linear(hidden, len(LABELS))

        def forward(self, ids, mask, return_representation=False):
            h = self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
            logits = self.classifier(h)
            return (logits, h) if return_representation else logits

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
        total = total_bce = total_soft_f1_loss = total_pair_loss = total_conditional_loss = 0.0
        for step, offset in enumerate(range(0, len(order), 16)):
            batch = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, representation = model(
                    ids[batch].to(device), masks[batch].to(device), return_representation=True)
            batch_targets = targets[batch].to(device)
            main_loss, bce, soft_f1_loss = loss_terms(logits, batch_targets, criterion)
            pair_loss, conditional_loss = correlation_losses(representation, batch_targets, model)
            loss = (main_loss + CONFIG["plcp_weight"] * pair_loss
                    + CONFIG["clcp_weight"] * conditional_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss seed={seed} epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            total_bce += bce.item()
            total_soft_f1_loss += soft_f1_loss.item()
            total_pair_loss += pair_loss.item()
            total_conditional_loss += conditional_loss.item()
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
                      train_soft_f1=1 - total_soft_f1_loss / (step + 1),
                      train_plcp_loss=total_pair_loss / (step + 1),
                      train_clcp_loss=total_conditional_loss / (step + 1))
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
        # Synthetic checks: hidden gold labels must never enter conditional context.
        torch.manual_seed(42)
        y = torch.tensor([[1., 0., 1., 0., 0., 1., 0., 1., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0., 1.]])
        observed = observed_label_mask(y)
        assert observed.any(dim=1).all() and (~observed).any(dim=1).all()
        changed = y.clone()
        changed[~observed] = 1 - changed[~observed]
        assert torch.equal(observed_context(y, observed), observed_context(changed, observed))
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.plcp_head = torch.nn.Linear(5, 36)
                self.clcp_context = torch.nn.Linear(18, 5, bias=False)
                self.clcp_head = torch.nn.Linear(5, 9)
        heads = Heads()
        embedding = torch.randn(2, 5, requires_grad=True)
        plcp, clcp = correlation_losses(embedding, y, heads)
        assert plcp.isfinite() and clcp.isfinite()
        (plcp + clcp).backward()
        assert embedding.grad is not None and embedding.grad.abs().sum() > 0
        assert all(p.grad is not None and p.grad.isfinite().all()
                   for p in heads.parameters())
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
    print("smoke OK: base and correlation loss, masked-label no-leakage, gradients, text only; no training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Run a single seed (42) for a pilot before a full 3-seed run.")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.out:
        parser.error("--out or LAB_OUT required")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed not in SEEDS for seed in args.seeds):
        parser.error("--seeds must be unique values from 42, 43, 44.")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    model_path, tokenizer = prepare_model()  # Before all data preparation.
    import torch
    import numpy as np
    out = Path(args.out)
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    run_config = {**CONFIG, "training_seeds": args.seeds}
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != run_config:
        raise ValueError("Output directory belongs to a different configuration")
    write_json(out / "config.json", run_config)
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
        tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                                max_length=256, return_tensors="pt"))
        atomic_write(cache, lambda p: torch.save(tokens, p))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    results = [run_seed(seed, out, model_path, tokens, targets, train, holdout, pos_weight) for seed in args.seeds]
    # Deterministic representative (first selected seed), not chosen by holdout score.
    representative_seed = args.seeds[0]
    atomic_write(out / "best.ckpt", lambda p: shutil.copyfile(out / f"seed_{representative_seed}" / "best.ckpt", p))
    mean_macro = sum(r["macro_f1"] for r in results) / len(results)
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name="macro_f1", value=mean_macro, split="internal_holdout"),
                   metrics=dict(seeds=results, mean_micro_f1=sum(r["micro_f1"] for r in results) / len(results),
                                mean_per_symptom_f1={s: sum(r["per_symptom_f1"][s] for r in results) / len(results) for s in LABELS}),
                   seed=representative_seed, seeds=args.seeds, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt="best.ckpt", notes="Final epoch 3; threshold >=0.5 fixed; empty predictions unchanged; first-seed representative checkpoint. "
                   "M3-KCBERT-012 split reused; BCE + soft macro-F1 + training-only 0.2 PLCP and 0.2 CLCP; "
                   "weights computed from Training partition only; official Validation unused.")
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
