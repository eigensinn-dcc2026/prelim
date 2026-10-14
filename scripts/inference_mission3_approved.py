"""Offline experimental inference and CPU timing; never qualifies a submission."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from train_mission3_approved import checkpoint_model, tokenizer_from_files, predict, base, core, exp


def infer_teacher(teacher_dir, label_dir, limit=0, batch=8):
    """CPU end-to-end teacher, including all three neural and linear members."""
    import joblib
    import gc
    teacher_dir=Path(teacher_dir)
    provenance=json.loads((teacher_dir/"provenance.json").read_text())
    paths=sorted(Path(label_dir).rglob("*.json"))
    if limit: paths=paths[:limit]
    if not paths or len({p.name for p in paths})!=len(paths): raise ValueError("Invalid inference files")
    rows=[dict(name=p.name,doc={"utterances":[dict(text=base.read_text(p))]}) for p in paths]
    outputs=[]
    started=time.perf_counter()
    for number,fold in enumerate(provenance["folds"]):
        directory=teacher_dir/f"fold_{number}"
        if exp.file_hash(directory/"model.pt")!=fold["checkpoint_sha256"] or exp.file_hash(directory/"linear.joblib")!=fold["linear_sha256"]:
            raise ValueError("Teacher member checksum mismatch")
        payload=torch.load(directory/"model.pt",map_location="cpu",weights_only=True)
        model=checkpoint_model(payload).eval()
        tokenizer=tokenizer_from_files(payload["files"])
        neural=predict(model,tokenizer,rows,list(range(len(rows))),payload["max_length"],torch.device("cpu"),batch=batch)
        linear=joblib.load(directory/"linear.joblib")
        features=linear["vectorizer"].transform([base.transcript(row["doc"]) for row in rows])
        probability=linear["classifier"].predict_proba(features)
        outputs.append(fold["neural_weight"]*neural+(1-fold["neural_weight"])*probability)
        del model,payload,linear,features
        gc.collect()
    seconds=time.perf_counter()-started
    return rows,np.mean(outputs,axis=0),dict(rows=len(rows),device="cpu",torch_threads=torch.get_num_threads(),
        total_seconds_including_load=seconds,seconds_per_call_including_load=seconds/len(rows),
        members=len(outputs),threshold=.5,caveat="Cold-path timing including member hash checks and loading; repeat trials on same CPU")


def infer(checkpoint, label_dir, device="cpu", long_checkpoint=None, margin=.05, limit=0, batch=8):
    started = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = checkpoint_model(payload).to(device).eval()
    tokenizer = tokenizer_from_files(payload["files"])
    paths = sorted(Path(label_dir).rglob("*.json"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No input JSON files")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate input file names")
    # Explicitly discard labels and all metadata at the input boundary.
    rows = [dict(name=p.name, doc={"utterances":[dict(text=base.read_text(p))]}) for p in paths]
    indices = list(range(len(rows)))
    if device == "cuda": torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    started = time.perf_counter()
    # One warm-up, excluded from the reported prediction interval.
    predict(model, tokenizer, rows, indices[:1], payload["max_length"], torch.device(device), batch=1)
    if device == "cuda": torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-started
    started = time.perf_counter()
    probabilities = predict(model, tokenizer, rows, indices, payload["max_length"], torch.device(device), batch=batch)
    reviewed, long_load_seconds = 0, 0.
    if long_checkpoint is not None:
        if payload["max_length"] != 256:
            raise ValueError("Cascade front model must use 256 tokens")
        lengths = [len(tokenizer(base.transcript(r["doc"]), add_special_tokens=False)["input_ids"]) for r in rows]
        chosen = core.route_mask(probabilities, lengths, margin)
        reviewed = int(chosen.sum())
        if reviewed:
            long_start = time.perf_counter()
            second = torch.load(long_checkpoint, map_location="cpu", weights_only=True)
            if second["max_length"] != 512:
                raise ValueError("Cascade long model must use 512 tokens")
            second_tokenizer = tokenizer_from_files(second["files"])
            long_model = checkpoint_model(second).to(device).eval()
            long_load_seconds = time.perf_counter()-long_start
            probabilities[chosen] = predict(long_model, second_tokenizer, rows, np.flatnonzero(chosen).tolist(), 512,
                                             torch.device(device), batch=batch)
    if device == "cuda": torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    return rows, probabilities, dict(rows=len(rows), device=device, torch_threads=torch.get_num_threads(),
        load_seconds=load_seconds, warmup_seconds=warmup_seconds, inference_seconds=seconds,
        total_seconds_including_load=load_seconds+warmup_seconds+seconds,
        long_load_included_seconds=long_load_seconds, seconds_per_call=seconds/len(rows),
        reviewed_calls=reviewed, threshold=.5, checkpoint_bytes=Path(checkpoint).stat().st_size,
        caveat="Wall-clock measurement; long model loading is included when first needed. No F1 is measured here.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ckpt-path", type=Path)
    selection.add_argument("--teacher-dir", type=Path)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--long-checkpoint", type=Path)
    p.add_argument("--margin", type=float, default=.05)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args = p.parse_args()
    if args.threads < 1 or args.batch < 1 or args.limit < 0:
        p.error("Invalid batch/thread/limit")
    torch.set_num_threads(args.threads)
    if args.teacher_dir:
        if args.device!="cpu" or args.long_checkpoint: p.error("Teacher benchmark is CPU-only without cascade")
        rows,probabilities,timing=infer_teacher(args.teacher_dir,args.label_dir,args.limit,args.batch)
    else:
        rows, probabilities, timing = infer(args.ckpt_path, args.label_dir, args.device,
            args.long_checkpoint, args.margin, args.limit, args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label file name","symptom"])
        for row, probs in zip(rows, probabilities):
            writer.writerow([row["name"], str([label for label,p in zip(base.LABELS,probs) if p >= .5])])
    args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__": main()
