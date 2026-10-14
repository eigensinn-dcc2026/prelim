"""Four M3 experiments. GPU training is opt-in and restricted to Google Colab.

All model choices use the existing Training-only 23408/5792 split. Official
Validation is intentionally not an argument. A finished pilot is not submission
qualification. Human annotations and final GPU runs are independent prerequisites.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dcc_data import mission3_approved_core as core
from dcc_data import mission3_context_experiment as exp
from dcc_inference import mission3_bert as base


def token_pack(tokenizer, rows, indices, length, device, offsets=False):
    packed = tokenizer([base.transcript(rows[i]["doc"]) for i in indices],
                       padding="max_length", truncation=True, max_length=length,
                       return_tensors="pt", return_offsets_mapping=offsets)
    mapping = packed.pop("offset_mapping").tolist() if offsets else None
    return packed["input_ids"].to(device), packed["attention_mask"].to(device), mapping


def tokenizer_from_files(files):
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder)/Path(name).name).write_text(text, encoding="utf-8")
        return AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)


def checkpoint_model(payload):
    model = base._build(payload["files"], payload)
    if payload["format"] == "dcc-m3-human-evidence-v1":
        model = core.EvidenceClassifier(model)
    model.load_state_dict(payload["model"], strict=True)
    return model


def pretrained_model(model_id, revision):
    """Only public pretrained weights: never reuse a task-trained OOF encoder."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = Path(snapshot_download(model_id, revision=revision,
        allow_patterns=["config.json", "tokenizer*", "vocab*", "special_tokens_map.json", "*.safetensors"]))
    encoder = AutoModel.from_pretrained(path, local_files_only=True)
    model = base._classifier(encoder.config, 9)
    model.bert = encoder
    meta = dict(format=base.FORMAT, labels=list(base.LABELS), max_length=256,
                dropout=.1, files=base._text_files(path),
                source=dict(initialization="public_pretrained", model_id=model_id,
                            requested_revision=revision, resolved_revision=path.name))
    return model, AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True), meta


def predict(model, tokenizer, rows, indices, length, device, budget=None, batch=8):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            if budget is not None:
                budget.check()
            ids, mask, _ = token_pack(tokenizer, rows, indices[start:start+batch], length, device)
            # Final exported CPU/GPU path is fp32; AMP is only used for training.
            probabilities.append(model(ids, mask).float().sigmoid().cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 9), np.float32)


def fit(model, tokenizer, payload, rows, train, evaluate, output, signature, budget,
        *, device, seed=42, epochs=1, lr=1e-6, effective=16, micro=8,
        aggregation="micro", length=256, teacher=None, annotations=None,
        auxiliary_weight=.1):
    """Fixed final epoch, equal examples/steps across each aggregation pair.

    Completed jobs resume via exact signatures. Interrupted jobs resume from the
    last completed epoch, restoring optimizer/scheduler/scaler/RNG state. A crash
    in the first epoch restarts that epoch deterministically.
    """
    from transformers import get_linear_schedule_with_warmup
    core.assert_disjoint(rows, train, evaluate)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import transformers
    spec = dict(signature=signature, train=train, evaluate=evaluate, seed=seed,
                runtime=dict(torch=str(torch.__version__), transformers=transformers.__version__,
                             numpy=np.__version__, device=str(device),
                             precision="bf16" if device.type=="cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type=="cuda" else "fp32")),
                epochs=epochs, lr=lr, effective=effective, micro=micro,
                aggregation=aggregation, length=length, auxiliary_weight=auxiliary_weight,
                model_source=payload.get("source", {}),
                teacher_sha256=exp.digest(teacher.tolist()) if teacher is not None else None,
                annotations_sha256=exp.digest(sorted((str(k), v) for k,v in annotations.items())) if annotations else None)
    if (output/"spec.json").exists() and json.loads((output/"spec.json").read_text()) != spec:
        raise ValueError("Resume configuration changed; use a new output directory")
    exp.write_json(output/"spec.json", spec)
    if (output/"result.json").exists():
        return np.load(output/"predictions.npz", allow_pickle=False)["probs"].copy()
    if teacher is not None and teacher.shape != (len(rows), 9):
        raise ValueError("Teacher array must use the canonical row order")
    exp.seed_all(seed)
    model.to(device)
    encoder = model.base.bert if isinstance(model, core.EvidenceClassifier) else model.bert
    # This is enabled in all arms, including controls, with identical microbatch size.
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if isinstance(model, core.EvidenceClassifier):
        parameters = [dict(params=model.base.parameters(), lr=lr),
                      dict(params=[model.queries, model.residual, *model.states.parameters()], lr=1e-4)]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
    steps = math.ceil(len(train)/effective)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(steps*.1), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    weights = exp.positive_weights(rows, train, device)
    gold = np.asarray([row["gold"] for row in rows], dtype=np.float32)
    history, start_epoch = [], 0
    if (output/"resume.pt").exists():
        saved = torch.load(output/"resume.pt", map_location=device, weights_only=True)
        if saved["spec_hash"] != exp.digest(spec):
            raise ValueError("Resume state provenance mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        start_epoch, history = saved["epoch"], saved["history"]
        del saved
    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(train).tolist()
        started, total = time.monotonic(), 0.
        for start in range(0, len(order), effective):
            budget.check()
            group = order[start:start+effective]
            optimizer.zero_grad(set_to_none=True)
            logits_parts, aux_parts = [], []
            for offset in range(0, len(group), micro):
                chunk = group[offset:offset+micro]
                ids, mask, mapping = token_pack(tokenizer, rows, chunk, length, device,
                                                offsets=annotations is not None)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    if isinstance(model, core.EvidenceClassifier):
                        logits, state, attention = model(ids, mask, detail=True)
                        if annotations is not None:
                            aux_parts.append(core.annotation_objective(state, attention, mapping, chunk, annotations)*len(chunk)/len(group))
                    else:
                        logits = model(ids, mask)
                logits_parts.append(logits)
            logits = torch.cat(logits_parts)
            target = torch.tensor(gold[group], device=device)
            soft_target = torch.tensor(teacher[group], device=device) if teacher is not None else None
            value = core.grouped_objective(logits, target, weights, aggregation, micro, soft_target)
            if aux_parts:
                value = value + auxiliary_weight*sum(aux_parts)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite training loss")
            scaler.scale(value).backward()
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous:
                scheduler.step()
            total += value.item()*len(group)
            if start % (effective*100) == 0:
                print(f"{output.name} epoch={epoch+1}/{epochs} rows={start}/{len(order)} loss={value.item():.5f}", flush=True)
        history.append(dict(epoch=epoch+1, loss=total/len(train), seconds=time.monotonic()-started))
        exp.save_torch(output/"resume.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch+1, history=history,
            spec_hash=exp.digest(spec), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        exp.write_json(output/"history.json", history)
    probabilities = predict(model, tokenizer, rows, evaluate, length, device, budget)
    exported = dict(payload, max_length=length,
                    format="dcc-m3-human-evidence-v1" if isinstance(model, core.EvidenceClassifier) else base.FORMAT,
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    experiment=dict(spec_hash=exp.digest(spec), official_validation_used=False,
                                    qualified_for_submission=False))
    exp.save_torch(output/"model.pt", exported)
    np.savez_compressed(output/"predictions.npz", names=[rows[i]["name"] for i in evaluate],
                        gold=gold[evaluate], probs=probabilities)
    result = exp.metrics(gold[evaluate], probabilities)
    result.update(history=history, status="pilot_complete_not_qualified", checkpoint_sha256=exp.file_hash(output/"model.pt"))
    exp.write_json(output/"result.json", result)
    return probabilities


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loss_experiment(args, rows, train, holdout, signature, budget):
    original = torch.load(args.bundle/"baseline.pt", map_location="cpu", weights_only=True)
    tokenizer = tokenizer_from_files(original["files"])
    for effective in (16, 64):
        for mode in ("micro", "global"):
            exp.seed_all(args.seed)
            model = checkpoint_model(original)
            fit(model, tokenizer, original, rows, train, holdout, args.out/f"loss_b{effective}_{mode}_s{args.seed}",
                signature, budget, device=torch.device("cuda"), seed=args.seed,
                aggregation=mode, effective=effective)
            del model
            release()


def teacher_experiment(args, rows, train, holdout, signature, budget):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    output = args.out/"teachers"
    output.mkdir(parents=True, exist_ok=True)
    gold = np.asarray([row["gold"] for row in rows])
    probabilities = np.full((len(train), 9), np.nan, np.float32)
    holdout_parts, folds = [], []
    # These NEW folds are nested inside 23,408 fitting rows. Existing 29,200-row
    # ROB5F teachers must not be used to evaluate the 5,792-row outer holdout.
    for fold, (a, b) in enumerate(MultilabelStratifiedKFold(3, shuffle=True, random_state=42).split(np.zeros(len(train)), gold[train])):
        fit_rows, predict_rows = [train[i] for i in a], [train[i] for i in b]
        core.assert_disjoint(rows, fit_rows, predict_rows)
        core.assert_disjoint(rows, fit_rows, holdout)
        exp.seed_all(args.seed)
        model, tokenizer, payload = pretrained_model("klue/roberta-large", "28d9112")
        score_rows = predict_rows + holdout
        neural = fit(model, tokenizer, payload, rows, fit_rows, score_rows, output/f"fold_{fold}",
                     signature, budget, device=torch.device("cuda"), seed=args.seed,
                     epochs=3, lr=1e-5)
        del model
        release()
        budget.check()
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=150000,
                                    sublinear_tf=True, min_df=2)
        x = vectorizer.fit_transform([base.transcript(rows[i]["doc"]) for i in fit_rows])
        classifier = OneVsRestClassifier(LogisticRegression(C=4., class_weight="balanced", max_iter=1000))
        classifier.fit(x, gold[fit_rows])
        linear = classifier.predict_proba(vectorizer.transform([base.transcript(rows[i]["doc"]) for i in score_rows]))
        import joblib
        joblib.dump(dict(vectorizer=vectorizer,classifier=classifier),output/f"fold_{fold}/linear.joblib")
        # Fixed mixture, no weight search on outer holdout. This is a new teacher
        # recipe; do not equate its score with the historical three-model ensemble.
        mixture = .6*neural + .4*linear
        probabilities[b] = mixture[:len(b)]
        holdout_parts.append(mixture[len(b):])
        folds.append(dict(fit=fit_rows, predict=predict_rows, initialization="public_pretrained",
                          source=payload["source"], neural_weight=.6,
                          linear_sha256=exp.file_hash(output/f"fold_{fold}/linear.joblib"),
                          checkpoint_sha256=exp.file_hash(output/f"fold_{fold}/model.pt")))
    core.verify_oof(rows, train, holdout, folds, probabilities)
    teacher_val = np.mean(holdout_parts, axis=0)
    np.savez_compressed(output/"oof.npz", names=[rows[i]["name"] for i in train], gold=gold[train], probs=probabilities,
                        holdout_names=[rows[i]["name"] for i in holdout], holdout_gold=gold[holdout], holdout_probs=teacher_val)
    exp.write_json(output/"provenance.json", dict(signature=signature, folds=folds,
                    archive_sha256=exp.file_hash(output/"oof.npz")))
    exp.write_json(output/"result.json", exp.metrics(gold[holdout], teacher_val))


def student_experiment(args, rows, train, holdout, signature, budget):
    source = args.out/"teachers"
    provenance = json.loads((source/"provenance.json").read_text())
    if provenance["signature"] != signature or exp.file_hash(source/"oof.npz") != provenance["archive_sha256"]:
        raise ValueError("OOF provenance mismatch")
    with np.load(source/"oof.npz", allow_pickle=False) as archive:
        if archive["names"].tolist() != [rows[i]["name"] for i in train]:
            raise ValueError("OOF row order changed")
        values = archive["probs"].copy()
    core.verify_oof(rows, train, holdout, provenance["folds"], values)
    teacher = np.full((len(rows),9), np.nan, np.float32)
    teacher[train] = values
    for distilled in (False, True):
        exp.seed_all(args.seed)
        model, tokenizer, payload = pretrained_model("beomi/KcELECTRA-base-v2022", "8728557")
        fit(model, tokenizer, payload, rows, train, holdout,
            args.out/f"student_{'kd' if distilled else 'hard'}_s{args.seed}", signature, budget,
            device=torch.device("cuda"), seed=args.seed, epochs=3, lr=2e-5,
            teacher=teacher if distilled else None)
        del model
        release()


def annotation_template(args, rows, train, signature):
    output = args.out/"human_annotation"
    output.mkdir(parents=True, exist_ok=True)
    selected = np.random.default_rng(42).choice(train, min(300,len(train)), replace=False).tolist()
    json_path = output/"template.jsonl"
    if json_path.exists():
        raise FileExistsError("Annotation template exists; refusing to overwrite possible human edits")
    with json_path.open("w", encoding="utf-8") as stream, (output/"texts.csv").open("w", encoding="utf-8-sig", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=["name", "text"])
        writer.writeheader()
        for i in selected:
            writer.writerow(dict(name=rows[i]["name"], text=base.transcript(rows[i]["doc"])))
            for label in base.LABELS:
                stream.write(json.dumps(dict(name=rows[i]["name"], label=label, state="", span=None,
                                             reviewer="", provenance="unreviewed"), ensure_ascii=False)+"\n")
    exp.write_json(output/"manifest.json", dict(signature=signature, sampled_calls=len(selected),
                    train_only=True, human_annotations_completed=False, labels_hidden_from_reviewers=True))


def evidence_experiment(args, rows, train, holdout, signature, budget):
    if not args.annotations or not args.human_annotation_rules_confirmed:
        raise ValueError("Reviewed annotations and an explicit rules confirmation flag are required")
    annotations = core.read_annotations(args.annotations, rows, train)
    payload = torch.load(args.bundle/"baseline.pt", map_location="cpu", weights_only=True)
    tokenizer = tokenizer_from_files(payload["files"])
    for supervised in (False, True):
        exp.seed_all(args.seed)
        model = core.EvidenceClassifier(checkpoint_model(payload))
        fit(model, tokenizer, payload, rows, train, holdout,
            args.out/f"evidence_{'human' if supervised else 'control'}_s{args.seed}", signature,
            budget, device=torch.device("cuda"), seed=args.seed,
            annotations=annotations if supervised else None)
        del model
        release()


def cascade_experiment(args, rows, train, holdout, signature, budget):
    payload = torch.load(args.bundle/"baseline.pt", map_location="cpu", weights_only=True)
    tokenizer = tokenizer_from_files(payload["files"])
    predictions = {}
    # Both lengths get the same one-epoch continuation, so extra training is not
    # confused with access to more text. Fixed-length control is new, not R007.
    for length in (256,512):
        model = checkpoint_model(payload)
        predictions[length] = fit(model, tokenizer, payload, rows, train, holdout,
            args.out/f"cascade_{length}_s{args.seed}", signature, budget,
            device=torch.device("cuda"), seed=args.seed, length=length)
        del model
        release()
    lengths = [len(tokenizer(base.transcript(rows[i]["doc"]), add_special_tokens=False)["input_ids"]) for i in holdout]
    report, probabilities = core.crossfit_cascade(np.asarray([rows[i]["gold"] for i in holdout]),
                                                 predictions[256], predictions[512], lengths)
    report.update(seed=args.seed,timing_status="not_measured", official_validation_used=False, qualified_for_submission=False)
    exp.write_json(args.out/f"cascade_crossfit_s{args.seed}.json", report)
    np.savez_compressed(args.out/f"cascade_crossfit_s{args.seed}.npz", names=[rows[i]["name"] for i in holdout],
                        gold=[rows[i]["gold"] for i in holdout], probs=probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight","loss","teachers","student","annotations","evidence","cascade"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--human-annotation-rules-confirmed", action="store_true")
    args = parser.parse_args()
    if args.phase == "teachers" and args.seed != 42:
        parser.error("OOF teachers use fixed seed 42; student replication varies the student seed only")
    if not 0 < args.max_hours <= 3:
        parser.error("Maximum cumulative training budget is 3 hours; no automatic purchases")
    if args.phase not in ("preflight", "annotations"):
        if not args.execute:
            parser.error("GPU execution is paused; explicit --execute is required")
        if not os.environ.get("COLAB_RELEASE_TAG") or not torch.cuda.is_available():
            parser.error("GPU training must run in Google Colab")
    rows, train, holdout, _, reference = exp.prepare_data(args.bundle/"training", args.bundle/"reference")
    core.assert_disjoint(rows, train, holdout)
    meta = json.loads((args.bundle/"bundle.json").read_text())
    if meta.get("official_validation_included") is not False or exp.file_hash(args.bundle/"baseline.pt") != meta["checkpoint_sha256"]:
        raise ValueError("Input bundle provenance mismatch")
    source_hash = exp.digest({p.name: exp.file_hash(p) for p in (Path(__file__), Path(core.__file__), Path(exp.__file__), Path(base.__file__))})
    signature = exp.digest(dict(source=source_hash, train=train, holdout=holdout,
                                rows=[{k:r[k] for k in ("name","call","text_hash","gold")} for r in rows],
                                checkpoint=meta["checkpoint_sha256"]))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"preflight.json").exists() and json.loads((args.out/"preflight.json").read_text())["signature"] != signature:
        raise ValueError("Run source/data changed; use a new output directory")
    preflight = dict(signature=signature, train_rows=len(train), holdout_rows=len(holdout),
                     reference_macro_f1=exp.metrics(np.asarray([rows[i]["gold"] for i in holdout]),reference)["macro_f1"],
                     official_validation_used=False, gpu_training_started=False)
    exp.write_json(args.out/"preflight.json", preflight)
    (args.out/"reference").mkdir(exist_ok=True)
    np.savez_compressed(args.out/"reference/predictions.npz",names=[rows[i]["name"] for i in holdout],
                        gold=[rows[i]["gold"] for i in holdout],probs=reference)
    if args.phase == "preflight":
        print(json.dumps(preflight, indent=2)); return
    if args.phase == "annotations":
        annotation_template(args, rows, train, signature); return
    budget = exp.RunBudget(args.out, args.max_hours)
    try:
        operations = dict(loss=loss_experiment, teachers=teacher_experiment, student=student_experiment,
                          evidence=evidence_experiment, cascade=cascade_experiment)
        operations[args.phase](args, rows, train, holdout, signature, budget)
    finally:
        budget.check(force=True)


if __name__ == "__main__":
    main()
