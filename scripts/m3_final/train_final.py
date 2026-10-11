"""Mission 3 final training: M3-ROBERTA-003 settings on 100% of Training, then export and score.

Same model, loss and optimizer as scripts/lab/m3-roberta-003.py (klue/roberta-large rev 28d9112,
front 256 tokens, BCE with pos_weight clip(0.5*sqrt(neg/pos), 1, 6) + 1.0 * (1 - soft macro-F1),
AdamW lr 1e-5, wd 0.01, linear warmup 10%, effective batch 16 = micro 8 x 2, fp16 autocast,
3 epochs, final epoch kept). The only difference: every Training label JSON is used for training
(no internal holdout), and pos_weight is computed on all of them. The official Validation is used
only for the final score, never for training, weights or selection.

    python scripts/m3_final/train_final.py --train_dir <Training label folder> --out runs/m3_final \
        --seeds 42 43 44 [--val_dir <Validation label folder>]
    python scripts/m3_final/train_final.py ... --quick      # 512 rows, 1 epoch: speed/plumbing check

Outputs per seed: runs/.../seed_<s>/best.ckpt (lab format), m3_roberta003_full_seed<s>.pt (offline
inference checkpoint for inference.py / submission), history.json, and val_score.json if --val_dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "klue/roberta-large"
REVISION = "28d9112"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
CONFIG = dict(exp_id="M3-FINAL-ROBERTA-003", model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=1e-5, dropout=0.1, batch_size=16, micro_batch_size=8, accumulation_steps=2,
              weight_decay=0.01, warmup_ratio=0.1, fp16=True, threshold=0.5, labels=LABELS,
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)", soft_f1_weight=1.0,
              soft_f1_epsilon=1e-8, threshold_operator=">=", train_scope="all Training label JSONs")


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def transcript(doc):
    return " ".join(text for u in doc.get("utterances", []) if isinstance(u, dict)
                    for text in [re.sub(r"\s+", " ", u.get("text", "")).strip()
                                 if isinstance(u.get("text", ""), str) else ""] if text)


def weights(labels):
    positive = [sum(row[j] for row in labels) for j in range(len(LABELS))]
    return [min(6.0, max(1.0, math.sqrt((len(labels) - p) / max(p, 1)) * .5)) for p in positive]


def load_rows(train_dir):
    rows = []
    for path in sorted(Path(train_dir).rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        symptoms = doc.get("symptom", [])
        symptoms = [symptoms] if isinstance(symptoms, str) else symptoms
        symptoms = {unicodedata.normalize("NFC", s.strip()) for s in symptoms if isinstance(s, str)}
        rows.append(dict(file=path.name, text=transcript(doc), labels=[int(s in symptoms) for s in LABELS]))
    if not rows:
        raise FileNotFoundError(f"no label JSON under {train_dir}")
    return rows


def prepare_model(cache_dir):
    from huggingface_hub import snapshot_download
    root = Path(cache_dir) / "roberta-large"
    marker = root / "dcc_revision.json"
    if not marker.exists():
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["config.json", "tokenizer*", "vocab.txt", "model.safetensors"])
        marker.write_text(json.dumps({"repo_id": MODEL_ID, "revision": REVISION}), encoding="utf-8")
    elif json.loads(marker.read_text(encoding="utf-8")).get("revision") != REVISION:
        raise ValueError(f"model cache revision differs: {root}")
    return root


def loss_terms(logits, targets, criterion):
    import torch
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, targets = logits.float(), targets.float()
        bce = criterion(logits, targets)
        p = logits.sigmoid()
        soft_f1 = (2 * (p * targets).sum(0) / (p.sum(0) + targets.sum(0) + 1e-8))
        return bce + (1 - soft_f1.mean()), bce, 1 - soft_f1.mean()


def train_seed(seed, out, model_path, tokens, targets, pos_weight, epochs):
    import numpy as np
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    n = len(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_updates = math.ceil(n / CONFIG["batch_size"]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_updates * CONFIG["warmup_ratio"]), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    ids, masks = tokens["input_ids"], tokens["attention_mask"]
    index = torch.arange(n)
    history = []
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        order = index[torch.randperm(n, generator=generator)]
        total = total_bce = total_sf = 0.0
        for step, offset in enumerate(range(0, n, CONFIG["batch_size"])):
            group = order[offset:offset + CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            for mo in range(0, len(group), CONFIG["micro_batch_size"]):
                batch = group[mo:mo + CONFIG["micro_batch_size"]]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(ids[batch].to(device), masks[batch].to(device))
                loss, bce, sf = loss_terms(logits, targets[batch].to(device), criterion)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss seed={seed} epoch={epoch} step={step}")
                scaler.scale(loss * (len(batch) / len(group))).backward()
                total += loss.item() * len(batch); total_bce += bce.item() * len(batch); total_sf += sf.item() * len(batch)
            prev = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev:
                scheduler.step()
            if step % 200 == 0:
                log(f"seed={seed} epoch={epoch} step={step}/{math.ceil(n / CONFIG['batch_size'])} loss={loss.item():.4f}")
        rec = dict(seed=seed, epoch=epoch, train_loss=total / n, train_bce=total_bce / n,
                   train_soft_f1=1 - total_sf / n, epoch_seconds=round(time.monotonic() - started, 1))
        history.append(rec)
        (seed_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        log("epoch done " + json.dumps(rec, ensure_ascii=False))
    ckpt = seed_dir / "best.ckpt"
    torch.save(dict(config=CONFIG, seed=seed, epoch=epochs, model=model.state_dict(),
                    model_local_path=str(model_path), pos_weight=pos_weight), ckpt)
    del model, optimizer
    if amp:
        torch.cuda.empty_cache()
    return ckpt, history


ENSEMBLE_WEIGHTS = dict(bert=0.55, tfidf=0.45)  # chosen on the 80:20 holdout (ROBERTA-003 seed average + P0), never on the official Val


def train_tfidf(rows, out):
    """P0 TF-IDF + OvR logistic regression with the P0 script's parameters, on all Training rows."""
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    random.seed(42); np.random.seed(42)
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    x = vectorizer.fit_transform([r["text"] for r in rows])
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear",
                                                        random_state=42), n_jobs=1)
    classifier.fit(x, np.array([r["labels"] for r in rows]))
    path = out / "mission3_tfidf_logreg_full.joblib"
    joblib.dump(dict(vectorizer=vectorizer, classifier=classifier, target_symptoms=LABELS, threshold=0.5,
                     seed=42, model_name="tfidf_ovr_logreg (P0 parameters, all Training)"), path)
    log(f"TF-IDF trained on {len(rows)} rows, vocabulary {len(vectorizer.vocabulary_)} -> {path.name}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--val_dir", type=Path, help="official Validation label folder (scoring only)")
    ap.add_argument("--model_cache", type=Path, default=Path("model_cache"))
    ap.add_argument("--quick", action="store_true", help="512 rows, 1 epoch: check speed and plumbing")
    ap.add_argument("--limit", type=int, help="use only the first N rows (smoke test; implies 1 epoch)")
    ap.add_argument("--val_limit", type=int, help="score only the first N Validation files (smoke test)")
    ap.add_argument("--drop_lab_ckpt", action="store_true",
                    help="delete seed_<s>/best.ckpt right after it is exported to the .pt (saves 1.35GB per seed)")
    ap.add_argument("--ensemble", action="store_true",
                    help="also train TF-IDF on all rows and pack seeds + TF-IDF into m3_ensemble_full.pt "
                         f"(weights {ENSEMBLE_WEIGHTS}, fixed in advance)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from dcc_inference import mission3_bert

    log(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
        torch.zeros(1, device="cuda").add_(1)  # fails here, not mid-training, if the build lacks this GPU's kernels
    else:
        log("WARNING: no CUDA GPU - training will be extremely slow")
    model_path = prepare_model(args.model_cache)
    rows = load_rows(args.train_dir)
    if args.quick:
        rows = rows[:512]
    if args.limit:
        rows = rows[:args.limit]
    epochs = 1 if (args.quick or args.limit) else CONFIG["epochs"]
    log(f"training rows {len(rows)} (all Training label JSONs{' - SUBSET, not the final model' if (args.quick or args.limit) else ''}), epochs {epochs}")
    pos_weight = weights([r["labels"] for r in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(dict(CONFIG, epochs=epochs, rows=len(rows), quick=bool(args.quick or args.limit),
                                                          pos_weight=dict(zip(LABELS, pos_weight))), ensure_ascii=False, indent=1), encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokens = dict(tokenizer([r["text"] for r in rows], padding="max_length", truncation=True, max_length=256,
                            return_token_type_ids=False, return_tensors="pt"))
    targets = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)

    for seed in args.seeds:
        log(f"=== seed {seed} ===")
        ckpt, history = train_seed(seed, args.out, model_path, tokens, targets, pos_weight, epochs)
        pt = args.out / f"m3_roberta003_full_seed{seed}.pt"
        meta = mission3_bert.export_checkpoint(ckpt, model_path, pt)
        record = dict(meta, output=pt.name, output_bytes=pt.stat().st_size, output_sha256=sha256(pt), history=history)
        pt.with_suffix(pt.suffix + ".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"exported {pt.name} ({record['output_bytes']:,} bytes, sha256 {record['output_sha256'][:12]}...)")
        if args.drop_lab_ckpt:
            ckpt.unlink()
            log(f"deleted {ckpt.relative_to(args.out)} (the exported .pt holds the same weights)")
        if args.val_dir:
            score_val(pt, args.val_dir, args.out / f"seed_{seed}", args.val_limit)
    if args.ensemble:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import build_ensemble
        tfidf = train_tfidf(rows, args.out)
        ens = args.out / "m3_ensemble_full.pt"
        build_ensemble.main(["--bert", *[str(args.out / f"m3_roberta003_full_seed{s}.pt") for s in args.seeds],
                             "--bert_weight", str(ENSEMBLE_WEIGHTS["bert"]), "--tfidf", str(tfidf),
                             "--tfidf_weight", str(ENSEMBLE_WEIGHTS["tfidf"]), "--out", str(ens)])
        log(f"ensemble {ens.name}: {ens.stat().st_size:,} bytes")
        if args.val_dir:
            ens_dir = args.out / "ensemble"
            ens_dir.mkdir(exist_ok=True)
            score_val(ens, args.val_dir, ens_dir, args.val_limit)


def score_val(pt, val_dir, seed_dir, limit=None):
    """Official Validation: evaluation only (same path as inference.py, >= 0.5, no post-processing)."""
    import numpy as np
    from sklearn.metrics import f1_score
    from dcc_inference import mission3_bert

    paths = sorted(Path(val_dir).rglob("*.json"))[:limit]
    texts = [mission3_bert.read_text(p) for p in paths]
    gold = []
    for p in paths:
        s = json.loads(p.read_text(encoding="utf-8-sig")).get("symptom", [])
        s = {unicodedata.normalize("NFC", x.strip()) for x in ([s] if isinstance(s, str) else s) if isinstance(x, str)}
        gold.append([int(lab in s) for lab in LABELS])
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / "submission" / "mission3"))
    import mission3_model  # the submission code path: handles single and ensemble checkpoints
    probs = mission3_model.predict_probabilities(pt, texts, 32)
    seconds = time.monotonic() - started
    gold, pred = np.array(gold), (probs >= 0.5).astype(int)
    res = dict(rows=len(paths), macro_f1=float(f1_score(gold, pred, average="macro", zero_division=0)),
               micro_f1=float(f1_score(gold, pred, average="micro", zero_division=0)),
               per_symptom_f1=dict(zip(LABELS, f1_score(gold, pred, average=None, zero_division=0).round(4).tolist())),
               empty_predictions=int((pred.sum(1) == 0).sum()), inference_seconds=round(seconds, 1), batch_size=32)
    (seed_dir / "val_score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(seed_dir / "val_probs.npy", probs.astype(np.float32))
    log("official Val " + json.dumps({k: v for k, v in res.items() if k != "per_symptom_f1"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
