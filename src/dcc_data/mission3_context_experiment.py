"""Reproducible M3 context experiment; official validation is a separate command.

No official labels enter preparation, retrieval, head fitting, or encoder fitting.
The exact existing holdout is recovered by file name from saved baseline arrays.
Feature shards and completed runs resume only after source/config hash checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import functional as F

from dcc_inference import mission3_bert as baseline
from dcc_inference import mission3_context as context

RECIPE = dict(head_epochs=10, head_lr=.001, head_batch=128, weight_decay=.01,
              fine_epochs=1, encoder_lr=.000001, fine_head_lr=.0001,
              fine_batch=16, micro_batch=8, bootstrap=1000, seeds=[42, 43, 44])
KEYWORDS = {
    "고열": r"열이|고열|열\s*나|\d{2}\.\d\s*도|체온",
    "구토": r"토했|토해|토하|토를|구토|게워|토가|오바이트",
    "두통": r"머리.{0,8}(아프|아파|아퍼|깨질|지끈|띵|쑤)|두통|편두통",
    "복통": r"배.{0,6}(아프|아파|아퍼)|복통|배가|명치|아랫배|윗배",
    "어지러움": r"어지러|어지럽|어질|빙빙|핑\s*돌|현기증",
    "열상": r"찢어|베였|베이|피가|피\s*나|출혈|꿰매|상처",
    "오심": r"메스껍|메슥|울렁|구역|토할\s*것|토할\s*거|속이\s*(안\s*좋|이상|불편|뒤집)|오심",
    "전신쇠약": r"기운이?\s*없|힘이?\s*없|쳐지|처지|축\s*늘어|기력|무기력|못\s*일어|쇠약",
    "호흡곤란": r"숨.{0,6}(못|안|차|막|쉬기|가빠|헐떡)|호흡|숨이",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(label_dir):
    paths = sorted(Path(label_dir).rglob("*.json"))
    if not paths or {p.casefold() for p in Path(label_dir).parts} & {"validation", "test"}:
        raise ValueError("A nonempty Training label directory is required")
    rows = []
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        text, spans = context.text_and_spans(doc)
        labels = doc.get("symptom", [])
        if isinstance(labels, str):
            labels = [labels]
        labels = {str(s).strip() for s in labels}
        # Strip every non-text input field before caching or encoding.
        allowed = {"utterances": [{"text": text[start:end]} for start, end in spans]}
        rows.append(dict(name=path.name, call=context.call_key(path.name), text_hash=context.text_hash(text),
                         doc=allowed, gold=[int(s in labels) for s in context.LABELS]))
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate file names; cannot align with baseline predictions")
    return rows


def prepare_data(label_dir, reference_dir, smoke_rows=0):
    rows = read_rows(label_dir)
    references = {}
    for name in ("r3", "k12", "p0"):
        with np.load(Path(reference_dir) / f"{name}_holdout.npz", allow_pickle=False) as archive:
            references[name] = {key:archive[key].copy() for key in ("names", "gold", "probs")}
    holdout_names = references["r3"]["names"].astype(str).tolist()
    if len(set(holdout_names)) != len(holdout_names):
        raise ValueError("Duplicate baseline holdout names")
    lookup = {row["name"]:i for i,row in enumerate(rows)}
    if not set(holdout_names) <= set(lookup):
        raise ValueError("Baseline holdout files missing from Training")
    holdout = [lookup[name] for name in holdout_names]
    holdout_set = set(holdout_names)
    train = [i for i in range(len(rows)) if rows[i]["name"] not in holdout_set]
    expected_gold = np.asarray([rows[i]["gold"] for i in holdout])
    aligned = {}
    for name, arrays in references.items():
        indices = {str(file):i for i,file in enumerate(arrays["names"])}
        if len(indices) != len(holdout) or set(indices) != set(holdout_names):
            raise ValueError(f"{name}: holdout file set differs")
        order = [indices[file] for file in holdout_names]
        if not np.array_equal(expected_gold, arrays["gold"][order]):
            raise ValueError(f"{name}: baseline gold differs from provided labels")
        aligned[name] = arrays["probs"][order]
    ensemble = .45*aligned["r3"] + .15*aligned["k12"] + .40*aligned["p0"]
    if not smoke_rows and (len(train), len(holdout)) != (23408, 5792):
        raise ValueError(f"Expected existing 23408/5792 split; got {len(train)}/{len(holdout)}")
    if {rows[i]["call"] for i in train} & {rows[i]["call"] for i in holdout}:
        raise ValueError("Call identity overlaps between Training and holdout")
    if smoke_rows:
        train = train[:smoke_rows]
        selected = list(range(min(max(8, smoke_rows//4), len(holdout))))
        holdout = [holdout[i] for i in selected]
        ensemble = ensemble[selected]
        aligned = {name:values[selected] for name,values in aligned.items()}
    return rows, train, holdout, aligned["r3"], ensemble


class BudgetReached(RuntimeError):
    """A bounded run can resume its completed shards and epochs later."""


class RunBudget:
    def __init__(self, output, hours):
        if not 0 < hours <= 10:
            raise ValueError("The approved run budget must be between zero and ten hours")
        self.path = Path(output)/"budget.json"
        self.limit = hours*3600
        previous = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.used = previous.get("used_seconds", 0.)
        self.started = self.saved = time.monotonic()

    def check(self, force=False):
        now = time.monotonic()
        used = self.used + now-self.started
        if force or now-self.saved >= 60 or used >= self.limit:
            write_json(self.path,dict(used_seconds=used,limit_seconds=self.limit))
            self.saved = now
        if used >= self.limit:
            raise BudgetReached("Run budget reached; completed shards/epochs are resumable")


def check_budget(budget):
    if budget is not None:
        budget.check()


def metrics(gold, probabilities, documents=None):
    gold, probabilities = np.asarray(gold), np.asarray(probabilities)
    if gold.shape != probabilities.shape or gold.ndim != 2 or gold.shape[1] != 9:
        raise ValueError("Metric arrays must align as (calls, 9)")
    if not len(gold) or not np.isfinite(probabilities).all():
        raise ValueError("Empty/nonfinite metric arrays")
    prediction = probabilities >= .5
    result = dict(rows=len(gold), macro_f1=float(f1_score(gold, prediction, average="macro", zero_division=0)),
                  micro_f1=float(f1_score(gold, prediction, average="micro", zero_division=0)),
                  empty_predictions=int((prediction.sum(1)==0).sum()),
                  per_label={})
    p = precision_score(gold, prediction, average=None, zero_division=0)
    r = recall_score(gold, prediction, average=None, zero_division=0)
    f = f1_score(gold, prediction, average=None, zero_division=0)
    for i,label in enumerate(context.LABELS):
        info = dict(f1=float(f[i]), precision=float(p[i]), recall=float(r[i]), support=int(gold[:, i].sum()))
        if documents is not None:
            texts = [context.text_and_spans(doc)[0] for doc in documents]
            match = np.array([bool(re.search(KEYWORDS[label], text)) for text in texts])
            for key, group in (("regex_match", match), ("regex_no_match", ~match)):
                positive = group & (gold[:, i]==1)
                info[key] = dict(rows=int(group.sum()), positives=int(positive.sum()),
                                 recall=float(prediction[positive, i].mean()) if positive.any() else None)
            question, negation = [], []
            for doc in documents:
                utterances = [u["text"] for u in doc["utterances"]]
                pairs = [(u, utterances[j+1]) for j,u in enumerate(utterances[:-1])
                         if re.search(KEYWORDS[label], u) and re.search(r"\?|세요|나요|습니까|시죠|는지",u)]
                question.append(bool(pairs))
                negation.append(any(re.search(r"없|아니|안\s|않", answer) for _,answer in pairs))
            for key, mask in (("question_answer", question), ("negative_answer", negation)):
                mask = np.asarray(mask, dtype=bool)
                info[key] = dict(rows=int(mask.sum()), false_positive=int(((gold[:,i]==0)&prediction[:,i]&mask).sum()),
                                 f1=float(f1_score(gold[mask,i], prediction[mask,i], zero_division=0)) if mask.any() else None)
        result["per_label"][label] = info
    return result


def loss(logits, targets, pos_weight):
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    probability = logits.sigmoid()
    f1 = 2*(probability*targets).sum(0)/(probability.sum(0)+targets.sum(0)+1e-8)
    return bce + 1-f1.mean()


def positive_weights(rows, train, device):
    gold = torch.tensor([rows[i]["gold"] for i in train], dtype=torch.float32, device=device)
    positive = gold.sum(0)
    return (.5*((len(train)-positive)/positive.clamp_min(1)).sqrt()).clamp(1,6)


def initialize(checkpoint, method, device, bank=None, config=None):
    from transformers import AutoTokenizer
    import tempfile
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != baseline.FORMAT or tuple(payload.get("labels",())) != context.LABELS:
        raise ValueError("Expected an exported ROBERTA-003 baseline checkpoint")
    base = baseline._build(payload["files"], payload)
    base.load_state_dict(payload.pop("model"), strict=True)
    with tempfile.TemporaryDirectory() as folder:
        for name,text in payload["files"].items():
            (Path(folder)/Path(name).name).write_text(text,encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
    model = context.ContextModel(base, method, config, bank).to(device)
    return model, tokenizer, payload


def extract_features(model, tokenizer, rows, indices, directory, signature, batch_size=32, budget=None):
    """Resumable bounded shards; retained features contain no transcript strings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "index.json"
    spec = dict(signature=signature, indices=indices, batch_size=batch_size)
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != spec:
        raise ValueError("Feature cache source/split/config changed")
    write_json(marker,spec)
    model.eval()
    device = next(model.parameters()).device
    features = {}
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            check_budget(budget)
            path = directory / f"shard_{start:06d}.pt"
            chunk = indices[start:start+batch_size]
            if path.exists():
                records = torch.load(path,map_location="cpu",weights_only=True)
                if [record["index"] for record in records] != chunk:
                    raise ValueError("Feature shard row order changed")
            else:
                encoded,_ = context.tokenize_documents(tokenizer,[rows[i]["doc"] for i in chunk])
                ids,mask,turns = [encoded[key].to(device) for key in ("input_ids","attention_mask","turn_ids")]
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    cls,logits,pooled,visible = model.extract(ids,mask,turns)
                records = [dict(index=i,cls=cls[j].half().cpu(),base_logits=logits[j].cpu(),
                                turns=pooled[j,visible[j]].half().cpu()) for j,i in enumerate(chunk)]
                save_torch(path,records)
            for record in records:
                features[record["index"]] = record
            if start % (batch_size*20)==0:
                print(f"feature cache {min(start+batch_size,len(indices))}/{len(indices)}",flush=True)
    return features


def make_bank(features,rows,train):
    return context.MemoryBank(torch.stack([features[i]["cls"] for i in train]),
                              [rows[i]["gold"] for i in train], [rows[i]["call"] for i in train],
                              [rows[i]["text_hash"] for i in train])


def feature_batch(features,rows,indices,device):
    cls = torch.stack([features[i]["cls"] for i in indices]).float().to(device)
    logits = torch.stack([features[i]["base_logits"] for i in indices]).float().to(device)
    length = max(1,max(len(features[i]["turns"]) for i in indices))
    turns = torch.zeros(len(indices),length,cls.shape[-1],device=device)
    mask = torch.zeros(len(indices),length,dtype=torch.bool,device=device)
    for j,i in enumerate(indices):
        n = len(features[i]["turns"])
        turns[j,:n] = features[i]["turns"].float().to(device)
        mask[j,:n] = True
    return cls,logits,turns,mask


def evaluate_features(head,bank,features,rows,holdout,device,batch_size=128):
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0,len(holdout),batch_size):
            chunk = holdout[start:start+batch_size]
            logits = head(*feature_batch(features,rows,chunk,device),bank=bank,
                          calls=[rows[i]["call"] for i in chunk],hashes=[rows[i]["text_hash"] for i in chunk])
            outputs.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def fit_head(model,features,rows,train,holdout,seed,directory,signature,recipe,budget=None):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    state_path = directory / "head_last.pt"
    specification = dict(signature=signature,method=model.head.method,seed=seed,recipe=recipe)
    device = next(model.head.parameters()).device
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    model.base.eval()
    optimizer = torch.optim.AdamW(model.head.parameters(),lr=recipe["head_lr"],weight_decay=recipe["weight_decay"])
    weights = positive_weights(rows,train,device)
    start,history = 0,[]
    if state_path.exists():
        state = torch.load(state_path,map_location=device,weights_only=True)
        if state["specification"] != specification:
            raise ValueError("Head resume source/config mismatch")
        model.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start,history = state["epoch"],state["history"]
    for epoch in range(start+1,recipe["head_epochs"]+1):
        seed_all(seed+epoch)  # epoch-boundary resume has the same order and dropout RNG
        model.head.train()
        order = np.random.default_rng(seed+epoch).permutation(train).tolist()
        total,started = 0.,time.monotonic()
        for offset in range(0,len(order),recipe["head_batch"]):
            check_budget(budget)
            chunk = order[offset:offset+recipe["head_batch"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(*feature_batch(features,rows,chunk,device),bank=model.bank,
                                calls=[rows[i]["call"] for i in chunk],hashes=[rows[i]["text_hash"] for i in chunk])
            targets = torch.tensor([rows[i]["gold"] for i in chunk],device=device,dtype=torch.float32)
            value = loss(logits,targets,weights)
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite head loss")
            value.backward()
            optimizer.step()
            total += value.item()*len(chunk)
        record = dict(epoch=epoch,loss=total/len(train),seconds=time.monotonic()-started)
        if holdout:
            probs = evaluate_features(model.head,model.bank,features,rows,holdout,device)
            record.update(metrics([rows[i]["gold"] for i in holdout],probs))
        history.append(record)
        save_torch(state_path,dict(specification=specification,head=model.head.state_dict(),optimizer=optimizer.state_dict(),
                                   epoch=epoch,history=history))
        write_json(directory/"history.json",history)
        print(f"{model.head.method} seed={seed} epoch={epoch} loss={record['loss']:.4f} macro={record.get('macro_f1')}",flush=True)
    probabilities = evaluate_features(model.head,model.bank,features,rows,holdout,device) if holdout else None
    if probabilities is not None:
        np.savez_compressed(directory/"holdout.npz",names=[rows[i]["name"] for i in holdout],
                            gold=[rows[i]["gold"] for i in holdout],probs=probabilities)
    return probabilities


def fit_encoder(model,tokenizer,rows,train,holdout,seed,directory,signature,recipe,budget=None):
    directory = Path(directory)
    state_path = directory/"encoder_last.pt"
    specification = dict(signature=signature,seed=seed,recipe=recipe,method=model.head.method)
    device = next(model.parameters()).device
    if state_path.exists():
        state = torch.load(state_path,map_location=device,weights_only=True)
        if state["specification"] != specification:
            raise ValueError("Encoder resume configuration changed")
        model.load_state_dict(state["model"])
    else:
        seed_all(seed)
        for parameter in model.base.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([dict(params=model.base.parameters(),lr=recipe["encoder_lr"]),
                                      dict(params=model.head.parameters(),lr=recipe["fine_head_lr"])],
                                     weight_decay=recipe["weight_decay"])
        from transformers import get_linear_schedule_with_warmup
        updates = math_ceil(len(train)/recipe["fine_batch"])
        scheduler = get_linear_schedule_with_warmup(optimizer,math_ceil(updates*.1),updates)
        scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        weights = positive_weights(rows,train,device)
        order = np.random.default_rng(seed).permutation(train).tolist()
        model.train()
        total,started = 0.,time.monotonic()
        for offset in range(0,len(order),recipe["fine_batch"]):
            check_budget(budget)
            group = order[offset:offset+recipe["fine_batch"]]
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(group),recipe["micro_batch"]):
                chunk = group[micro:micro+recipe["micro_batch"]]
                encoded,hashes = context.tokenize_documents(tokenizer,[rows[i]["doc"] for i in chunk])
                ids,mask,turns = [encoded[key].to(device) for key in ("input_ids","attention_mask","turn_ids")]
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    logits = model(ids,mask,turns,[rows[i]["call"] for i in chunk],hashes)
                targets = torch.tensor([rows[i]["gold"] for i in chunk],device=device,dtype=torch.float32)
                value = loss(logits,targets,weights)
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite encoder loss")
                scaler.scale(value*len(chunk)/len(group)).backward()
                total += value.item()*len(chunk)
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous:
                scheduler.step()
            if offset % (recipe["fine_batch"]*100)==0:
                print(f"encoder {model.head.method} seed={seed} {offset}/{len(order)}",flush=True)
        save_torch(state_path,dict(specification=specification,model=model.state_dict(),epoch=1,
                                   loss=total/len(train),seconds=time.monotonic()-started))
    model.eval()
    probs = context.probabilities(model,tokenizer,[rows[i]["doc"] for i in holdout],
                                  [rows[i]["name"] for i in holdout],batch_size=recipe["micro_batch"]) if holdout else None
    if probs is not None:
        np.savez_compressed(directory/"fine_holdout.npz",names=[rows[i]["name"] for i in holdout],
                            gold=[rows[i]["gold"] for i in holdout],probs=probs)
    return probs


def math_ceil(value):
    import math
    return math.ceil(value)


def choose_pilot(scores,control,reference_scores=()):
    threshold=max([control,*reference_scores])
    successful = [method for method in ("memory","evidence") if scores[method] > threshold]
    if not successful:
        return None
    if len(successful)==2 and abs(scores["memory"]-scores["evidence"]) < .001:
        return "evidence"
    return max(successful,key=lambda method:scores[method])


def bootstrap_difference(gold,candidates,reference,count=1000,seed=42,calls=None):
    gold,reference = np.asarray(gold),np.asarray(reference)
    candidates = [np.asarray(values) for values in candidates]
    def contributions(probability):
        prediction = probability >= .5
        return np.stack([prediction & (gold==1),prediction & (gold==0),(~prediction)&(gold==1)],axis=1).astype(np.int32)
    ref = contributions(reference)
    cand = [contributions(probability) for probability in candidates]
    def score(counts):
        tp,fp,fn = counts.sum(0)
        return np.divide(2*tp,2*tp+fp+fn,out=np.zeros(9,dtype=float),where=(2*tp+fp+fn)>0).mean()
    randomizer = np.random.default_rng(seed)
    groups=None
    if calls is not None:
        if len(calls)!=len(gold):
            raise ValueError("Bootstrap call identities must align with predictions")
        if len(set(calls))!=len(calls):
            buckets={}
            for i,call in enumerate(calls):
                buckets.setdefault(call,[]).append(i)
            groups=list(buckets.values())
    values = []
    for _ in range(count):
        indices = (randomizer.integers(0,len(gold),len(gold)) if groups is None else
                   np.concatenate([groups[j] for j in randomizer.integers(0,len(groups),len(groups))]))
        values.append(float(np.mean([score(item[indices]) for item in cand])-score(ref[indices])))
    return [float(value) for value in np.quantile(values,[.025,.5,.975])]


def adoption_gate(gold,candidates,ensemble,controls,ablations,*,smoke=False,bootstrap=1000,calls=None):
    base = metrics(gold,ensemble)
    candidate = [metrics(gold,p) for p in candidates]
    if len(candidate)!=3 or len(controls)!=3 or len(ablations)!=3:
        raise ValueError("Adoption requires three candidates, controls and ablations")
    mean = float(np.mean([m["macro_f1"] for m in candidate]))
    interval = bootstrap_difference(gold,candidates,ensemble,bootstrap,calls=calls)
    checks = dict(full_dataset=not smoke, mean_gain_ge_003=mean-base["macro_f1"] >= .003-1e-12,
                  at_least_two_seeds_win=sum(m["macro_f1"]>base["macro_f1"] for m in candidate)>=2,
                  ci_lower_positive=interval[0]>0,
                  micro_drop_le_005=np.mean([m["micro_f1"] for m in candidate])>=base["micro_f1"]-.005)
    contribution=dict(beats_additional_training_control=bool(mean>np.mean([metrics(gold,p)["macro_f1"] for p in controls])),
                      beats_ablation=bool(mean>np.mean([metrics(gold,p)["macro_f1"] for p in ablations])))
    return dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                mean_macro_f1=mean, reference_macro_f1=base["macro_f1"],gain=mean-base["macro_f1"],
                paired_bootstrap_ci95=interval,seed_metrics=candidate,contribution_checks=contribution)


def report(output,summary):
    output=Path(output)
    write_json(output/"metrics.json",summary)
    lines=["# M3 context experiment", "",f"Status: **{summary['status']}**", "",
           "Threshold >= 0.5. Training-only retrieval. Official Validation excluded from fitting.","",
           "| Model | Macro F1 | Micro F1 |","|---|---:|---:|"]
    for name,item in summary.get("pilot",{}).items():
        lines.append(f"| {name} | {item['macro_f1']:.4f} | {item['micro_f1']:.4f} |")
    if "gate" in summary:
        lines.extend(["", "## Adoption gate", "", "```json",json.dumps(summary["gate"],ensure_ascii=False,indent=2),"```"])
    if summary.get("repeat"):
        lines.extend(["", "## Three-seed comparison", "", "| Method | Stage | Mean Macro F1 | Mean Micro F1 |",
                      "|---|---|---:|---:|"])
        for method,stages in summary["repeat"].items():
            for stage,items in stages.items():
                lines.append(f"| {method} | {stage} | {np.mean([m['macro_f1'] for m in items]):.4f} | "
                             f"{np.mean([m['micro_f1'] for m in items]):.4f} |")
    if summary.get("smoke"):
        lines.extend(["", "This bounded smoke run is an engineering check; its scores are not competition performance."])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if summary.get("pilot"):
            names=list(summary["pilot"])
            fig,ax=plt.subplots(figsize=(8,4))
            ax.barh(names,[summary["pilot"][name]["macro_f1"] for name in names])
            ax.set(xlabel="Macro F1 at threshold 0.5",xlim=(0,1),
                   title="M3 internal holdout comparison"+(" (smoke only)" if summary.get("smoke") else ""))
            fig.tight_layout();fig.savefig(output/"comparison.png",dpi=180);plt.close(fig)
    except ImportError:
        pass


def run_experiment(args):
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    budget=RunBudget(output,getattr(args,"max_hours",10))
    try:
        return _run_experiment(args,budget)
    except BudgetReached:
        path=output/"metrics.json"
        summary=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        summary.update(status="budget_reached_resume_pending",selected_method=summary.get("selected_method"),
                       smoke=bool(args.smoke_rows),elapsed_seconds=budget.used+time.monotonic()-budget.started)
        report(output,summary)
        return summary
    finally:
        try:
            budget.check(force=True)
        except BudgetReached:
            pass


def _run_experiment(args,budget):
    started=time.monotonic()
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.cpu_threads)
    rows,train,holdout,base_probs,ensemble=prepare_data(args.train_label_dir,args.reference_probs_dir,args.smoke_rows)
    recipe=dict(RECIPE)
    if args.smoke_rows:
        recipe.update(head_epochs=2,bootstrap=100)
    source=dict(base_checkpoint_sha256=file_hash(args.base_ckpt),
                rows_sha256=digest([{k:row[k] for k in ("name","call","text_hash","gold")} for row in rows]),
                split_sha256=digest(dict(train=train,holdout=holdout)),
                recipe=recipe,context_config=context.DEFAULTS,smoke=bool(args.smoke_rows),
                feature_precision="cuda_fp16" if device.type=="cuda" else "cpu_fp32",
                implementation_sha256=digest({context.__name__:file_hash(context.__file__),
                                              baseline.__name__:file_hash(baseline.__file__),
                                              __name__:file_hash(__file__)}))
    signature=digest(source)
    previous=output/"run.json"
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8"))!=source:
        raise ValueError("Output directory belongs to a different dataset/checkpoint/recipe")
    write_json(previous,source)
    seed_all(42)
    model,tokenizer,payload=initialize(args.base_ckpt,"control",device)
    if not args.smoke_rows and (payload.get("source",{}).get("exp_id")!="M3-ROBERTA-003" or payload["source"].get("seed")!=42):
        raise ValueError("Full runs must start from ROBERTA-003 seed42")
    indices=sorted(set(train+holdout))
    features=extract_features(model,tokenizer,rows,indices,output/"features",signature,args.feature_batch,budget)
    bank=make_bank(features,rows,train).to(device)
    initial_base={key:value.detach().cpu().clone() for key,value in model.base.state_dict().items()}
    gold=np.asarray([rows[i]["gold"] for i in holdout])
    docs=[rows[i]["doc"] for i in holdout]
    summary=dict(status="pilot",smoke=bool(args.smoke_rows),source=source,selected_method=None,
                 device=str(device),pilot={"roberta003":metrics(gold,base_probs,docs),
                                          "existing_ensemble":metrics(gold,ensemble,docs)})
    recached=torch.stack([features[i]["base_logits"] for i in holdout]).sigmoid().numpy()
    summary["pilot"]["recached_roberta003"]=metrics(gold,recached,docs)
    summary["runtime"]=dict(torch=str(torch.__version__),numpy=np.__version__,
                             gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None)
    predictions={}
    for method in ("control","memory","evidence"):
        seed_all(42)
        model.head=context.ResidualHead(model.base.classifier.in_features,method,context.DEFAULTS).to(device)
        model.bank=bank if method.startswith("memory") else None
        predictions[method]=fit_head(model,features,rows,train,holdout,42,output/f"{method}_42",signature,recipe,budget)
        summary["pilot"][method]=metrics(gold,predictions[method],docs)
        report(output,summary)
    winner=choose_pilot({method:summary["pilot"][method]["macro_f1"] for method in ("memory","evidence")},
                        summary["pilot"]["control"]["macro_f1"],
                        [summary["pilot"][name]["macro_f1"] for name in ("roberta003","existing_ensemble")])
    summary["selected_method"]=winner
    if winner is None:
        summary["status"]="both_pilots_failed"
    elif args.phase=="pilot":
        summary["status"]="pilot_complete_repeat_pending"
    else:
        ablation="memory_removed" if winner=="memory" else "answers_removed"
        candidates,controls,ablations=[],[],[]
        frozen_candidates,frozen_controls,frozen_ablations=[],[],[]
        summary["repeat"]={method:dict(frozen=[],fine=[]) for method in ("control",winner,ablation)}
        for seed in recipe["seeds"]:
            for method,container in (("control",controls),(winner,candidates),(ablation,ablations)):
                seed_all(seed)
                model.base.load_state_dict(initial_base)
                model.head=context.ResidualHead(model.base.classifier.in_features,method,context.DEFAULTS).to(device)
                model.bank=bank if method.startswith("memory") else None
                directory=output/f"{method}_{seed}"
                frozen=fit_head(model,features,rows,train,holdout,seed,directory,signature,recipe,budget)
                frozen_container={"control":frozen_controls,winner:frozen_candidates,ablation:frozen_ablations}[method]
                frozen_container.append(frozen)
                frozen_metrics=metrics(gold,frozen,docs)
                summary["repeat"][method]["frozen"].append(frozen_metrics)
                write_json(directory/"frozen_metrics.json",frozen_metrics)
                probability=fit_encoder(model,tokenizer,rows,train,holdout,seed,directory,signature,recipe,budget)
                container.append(probability)
                fine_metrics=metrics(gold,probability,docs)
                summary["repeat"][method]["fine"].append(fine_metrics)
                write_json(directory/"fine_metrics.json",fine_metrics)
                report(output,summary)
        calls=[rows[i]["call"] for i in holdout]
        summary["gate"]=adoption_gate(gold,candidates,ensemble,controls,ablations,smoke=bool(args.smoke_rows),bootstrap=recipe["bootstrap"],calls=calls)
        summary["frozen_gate"]=adoption_gate(gold,frozen_candidates,ensemble,frozen_controls,frozen_ablations,
                                             smoke=bool(args.smoke_rows),bootstrap=recipe["bootstrap"],calls=calls)
        summary["status"]="internal_gate_passed" if summary["gate"]["passed"] else "internal_gate_failed"
        report(output,summary)
        if summary["gate"]["passed"] and args.phase=="all":
            all_train=list(range(len(rows)))
            # Holdout is now part of Training after recipe selection; official Val remains excluded.
            seed_all(42)
            model.base.load_state_dict(initial_base)
            model.head=context.ResidualHead(model.base.classifier.in_features,winner,context.DEFAULTS).to(device)
            model.bank=make_bank(features,rows,all_train).to(device) if winner=="memory" else None
            final_dir=output/"final"
            final_signature=digest(dict(source=source,training="all_29200",method=winner))
            fit_head(model,features,rows,all_train,[],42,final_dir,final_signature,recipe,budget)
            fit_encoder(model,tokenizer,rows,all_train,[],42,final_dir,final_signature,recipe,budget)
            final_source=dict(source,representative_seed=42,training_rows=len(rows),selected_method=winner,
                              internal_gate=summary["gate"],official_validation_used=False)
            save_torch(final_dir/"candidate.pt",context.export_payload(model,payload,final_source))
            write_json(final_dir/"candidate.pt.json",dict(source=final_source,
                       checkpoint_sha256=file_hash(final_dir/"candidate.pt")))
            summary["status"]="final_candidate_ready_official_pending"
    summary["elapsed_seconds"]=time.monotonic()-started
    summary["primary_metric"]=dict(name="macro_f1",value=max(summary["pilot"][name]["macro_f1"] for name in
                                                          ("control","memory","evidence")),split="internal_holdout")
    report(output,summary)
    return summary
