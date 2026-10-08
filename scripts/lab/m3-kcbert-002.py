"""M3-KCBERT-002: P0 split, front-256 KcBERT, three fixed training seeds.

Run from repository root with PYTHONPATH=src and LAB_OUT/LAB_WORK/DCC_RUN_ROOT.
Only initial model preparation uses the network. All model loads are local.
Epoch-boundary recovery repeats an interrupted epoch from its preceding checkpoint.
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
import shutil
import tempfile
import time
import unicodedata
import zipfile

EXP_ID = "M3-KCBERT-002"
MODEL = "beomi/kcbert-base"
# Fixed commit from https://huggingface.co/beomi/kcbert-base/commits/main
REVISION = "0f2f3f8"
SEEDS = (42, 43, 44)
LABEL_DIR = "TL_서울_구급"
REPO = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_save(path, value):
    import torch
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def p0_module():
    path = REPO / "scripts" / "mission3_baseline 최종 수정본.py"
    spec = importlib.util.spec_from_file_location("m3_p0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_model(run_root):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    path = run_root / "models" / "kcbert-base"
    marker = path / "dcc_revision.json"
    if marker.exists():
        if json.loads(marker.read_text())["revision"] != REVISION:
            raise ValueError(f"Model revision mismatch: {marker}")
    else:
        snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(path),
                          allow_patterns=["*.json", "vocab.txt", "model.safetensors"])
    AutoConfig.from_pretrained(path, local_files_only=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing local model weights: {path}")
    atomic_json(marker, {"repo_id": MODEL, "revision": REVISION})
    return path


def stage_labels(data_root, work):
    """Copy one large Training archive before reading any individual JSONs."""
    training = data_root / "Training"
    archives = [p for p in training.rglob("*.zip")
                if unicodedata.normalize("NFC", p.stem) == LABEL_DIR]
    if len(archives) != 1:
        raise ValueError(f"Need exactly one Training/{LABEL_DIR}.zip, found {archives}. "
                         "Package the Seoul labels as a zip; loose Drive JSON reads are disabled.")
    source = archives[0]
    local = work / "seoul_labels.zip"
    if not local.exists() or local.stat().st_size != source.stat().st_size:
        temporary = local.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, local)
    destination = work / "labels"
    if not (destination / ".complete").exists():
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        with zipfile.ZipFile(local) as archive:
            for entry in archive.infolist():
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(destination)
        (destination / ".complete").write_text("ok")
    return destination


def prepare_tokens(out, work, model_path):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from dcc_data.colab_env import find_data_root
    p0 = p0_module()
    cache = out / "tokens.pt"
    if cache.exists():
        local = work / "tokens.pt"
        shutil.copyfile(cache, local)
        data = torch.load(local, map_location="cpu", weights_only=False)
        if data["contract"] != [EXP_ID, REVISION, 256, 42]:
            raise ValueError("Token cache contract mismatch")
        return data
    labels = stage_labels(find_data_root(), work)
    df = p0.load_training_dataframe(labels)
    if len(df) != 29200 or df.attrs["load_failures"]:
        raise ValueError("Expected all 29,200 P0 JSON samples without load failures")
    mlb = p0.MultiLabelBinarizer(classes=p0.TARGET_SYMPTOMS)
    mlb.fit([p0.TARGET_SYMPTOMS])
    train, valid = p0._make_split(df, 0.2, 42, mlb)
    if (len(train), len(valid)) != (23408, 5792):
        raise ValueError(f"P0 split size mismatch: {len(train)}, {len(valid)}")
    # Repository manifest defines call IDs by JSON/audio stem: one JSON = one call.
    calls = [unicodedata.normalize("NFC", Path(name).stem).casefold()
             for name in df.label_file_name]
    if len(set(calls)) != len(calls):
        raise ValueError("Duplicate call IDs; cannot reproduce P0 safely")
    if {calls[i] for i in train.index} & {calls[i] for i in valid.index}:
        raise ValueError("Train/holdout call leakage")
    split = {"train": train.label_file_name.tolist(), "holdout": valid.label_file_name.tolist(),
             "seed": 42, "call_id_rule": "JSON filename stem"}
    atomic_json(out / "split.json", split)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    enc = tokenizer(df.text.tolist(), padding="max_length", truncation=True,
                    max_length=256, return_tensors="pt")
    data = {"contract": [EXP_ID, REVISION, 256, 42], "enc": dict(enc),
            "labels": torch.tensor(np.asarray(mlb.transform(df.symptoms)), dtype=torch.float32),
            "train": torch.tensor(train.index.to_numpy()),
            "valid": torch.tensor(valid.index.to_numpy()), "classes": p0.TARGET_SYMPTOMS,
            "split_sha256": hashlib.sha256((out / "split.json").read_bytes()).hexdigest()}
    atomic_save(cache, data)
    return data


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    import numpy as np
    import torch
    return {"random": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def train_seed(seed, data, out, model_path, batch, accum):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModel
    result_path = out / f"seed_{seed}.json"
    if result_path.exists():
        if not (out / f"seed_{seed}.ckpt").exists():
            raise FileNotFoundError("Completed seed is missing checkpoint")
        return json.loads(result_path.read_text())
    seed_all(seed)
    device = torch.device("cuda")

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 9)

        def forward(self, inputs):
            return self.classifier(self.dropout(self.bert(**inputs).last_hidden_state[:, 0]))

    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # P1 has constant LR: persist an explicit identity scheduler for recovery.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    last = out / "ckpt_last.pt"
    history, start_epoch, elapsed = [], 0, 0.0
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["seed"] == seed:
            if (saved["exp_id"] != EXP_ID or saved["revision"] != REVISION
                    or saved["split_sha256"] != data["split_sha256"]
                    or saved["batch"] != batch or saved["accum"] != accum):
                raise ValueError("Resume split/batch configuration mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            history, start_epoch, elapsed = saved["history"], saved["epoch"], saved["training_seconds"]
            restore_rng(saved["rng"])

    def checkpoint(epoch):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "rng": rng_state(), "seed": seed, "history": history,
                "training_seconds": elapsed, "split_sha256": data["split_sha256"],
                "batch": batch, "accum": accum, "model_name": MODEL,
                "model_path": str(model_path), "revision": REVISION, "classes": data["classes"],
                "max_length": 256, "threshold": 0.5, "exp_id": EXP_ID}

    if start_epoch == 0:
        atomic_save(last, checkpoint(0))
    for epoch in range(start_epoch + 1, 4):
        started = time.monotonic()
        model.train()
        order = data["train"][torch.randperm(len(data["train"]))]
        batches = math.ceil(len(order) / batch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step in range(batches):
            ids = order[step * batch:(step + 1) * batch]
            inputs = {key: value[ids].to(device) for key, value in data["enc"].items()}
            # Correctly scale the last, possibly shorter accumulation window.
            window = min(accum, batches - (step // accum) * accum)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(inputs), data["labels"][ids].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            scaler.scale(loss / window).backward()
            if (step + 1) % accum == 0 or step + 1 == batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += loss.item()
            if step % 200 == 0:
                print(f"seed={seed} epoch={epoch} step={step}/{batches} loss={loss.item():.4f}", flush=True)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for ids in data["valid"].split(batch):
                inputs = {key: value[ids].to(device) for key, value in data["enc"].items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs)
                probabilities.append(logits.float().sigmoid().cpu().numpy())
        pred = np.concatenate(probabilities) >= 0.5
        truth = data["labels"][data["valid"]].numpy()
        seconds = time.monotonic() - started
        elapsed += seconds
        record = {"epoch": epoch, "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                  "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
                  "per_symptom_f1": dict(zip(data["classes"], f1_score(truth, pred, average=None, zero_division=0).tolist())),
                  "loss": loss_sum / batches, "epoch_seconds": seconds}
        history.append(record)
        atomic_save(last, checkpoint(epoch))
        print(f"seed={seed} epoch={epoch} macro_f1={record['macro_f1']:.6f}", flush=True)
    # Fixed epoch 3, no epoch/threshold tuning. Reconstruct after a crash at this boundary.
    atomic_save(out / f"seed_{seed}.ckpt", checkpoint(3))
    result = {"seed": seed, **history[-1], "training_seconds": elapsed, "history": history}
    atomic_json(result_path, result)
    return result


def smoke():
    """Stdlib-only contract checks: no dataset, model download, or training."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        atomic_json(root / "metrics.json", {"exp_id": EXP_ID})
        assert json.loads((root / "metrics.json").read_text())["exp_id"] == EXP_ID
        training = root / "Training"
        training.mkdir()
        with zipfile.ZipFile(training / f"{LABEL_DIR}.zip", "w") as archive:
            archive.writestr(f"{LABEL_DIR}/call.json", '{"utterances": []}')
        work = root / "work"
        work.mkdir()
        staged = stage_labels(root, work)
        assert len(list(staged.rglob("*.json"))) == 1
        assert stage_labels(root, work) == staged
    print("smoke passed (atomic JSON, local archive staging and reuse; no training)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("LAB_OUT"))
    parser.add_argument("--batch-size", type=int, choices=(8, 16), default=16)
    parser.add_argument("--grad-accum", type=int, choices=(1, 2), default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        parser.error("LAB_EXP_ID mismatch")
    if not args.out or not os.environ.get("LAB_WORK") or not os.environ.get("DCC_RUN_ROOT"):
        parser.error("LAB_OUT/--out, LAB_WORK and DCC_RUN_ROOT are required")
    if args.batch_size * args.grad_accum != 16:
        parser.error("Effective batch size must be 16 (16x1 or 8x2)")
    model_path = prepare_model(Path(os.environ["DCC_RUN_ROOT"]))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA fp16")
    out, work = Path(args.out), Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if out.resolve() == work.resolve():
        raise ValueError("LAB_WORK must be separate from permanent LAB_OUT")
    data = prepare_tokens(out, work, model_path)
    results = [train_seed(seed, data, out, model_path, args.batch_size, args.grad_accum) for seed in SEEDS]
    mean = sum(row["macro_f1"] for row in results) / len(results)
    # Submission artifact is seed 42 epoch 3; no seed selection or ensemble tuning.
    temporary = out / "best.ckpt.tmp"
    shutil.copyfile(out / "seed_42.ckpt", temporary)
    os.replace(temporary, out / "best.ckpt")
    atomic_json(out / "metrics.json", {
        "exp_id": EXP_ID, "primary_metric": {"name": "macro_f1", "value": mean, "split": "internal_holdout"},
        "metrics": {"seeds": results, "mean_macro_f1": mean,
                    "mean_micro_f1": sum(row["micro_f1"] for row in results) / 3,
                    "threshold": 0.5, "split_seed": 42, "split_sha256": data["split_sha256"],
                    "baseline_macro_f1": 0.5676,
                    "training_seconds": sum(row["training_seconds"] for row in results)},
        "seed": 42, "train_rows": len(data["train"]), "eval_rows": len(data["valid"]),
        "official_val": None, "ckpt": "best.ckpt",
        "notes": "Training Seoul only; fixed epoch 3 and threshold >=0.5; seeds 42/43/44; best.ckpt is seed 42."
    })


if __name__ == "__main__":
    main()
