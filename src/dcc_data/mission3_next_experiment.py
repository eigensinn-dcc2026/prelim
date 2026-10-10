"""Controlled M3 loss, coverage, and local evidence-state experiments.

Only Training is loaded. Full frozen and fine pilots are measured at fixed
epochs; official Validation is reserved for a separately gated final candidate.
"""
from __future__ import annotations
import copy
import csv
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import mission3_context_experiment as exp
from . import mission3_evidence_data as data
from dcc_inference import mission3_bert as baseline, mission3_evidence as evidence

RECIPE = dict(head_epochs=10, head_batch=128, head_lr=.001, encoder_epochs=1,
              encoder_lr=1e-6, head_fine_lr=1e-4, effective_batch=16,
              base_micro_batch=8, window_micro_batch=2, state_weight=.1,
              weight_decay=.01, gamma_pos=0., gamma_neg=4., asl_clip=.05,
              threshold=.5, seeds=[42,43,44], max_windows=6)
INPUT_METHODS = ("front", "head_tail", "multi", "states", "states_removed")


def objective(logits, targets, weights, kind):
    logits, targets = logits.float(), targets.float()
    if kind == "original":
        return exp.loss(logits, targets, weights)
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
    if kind != "asl":
        raise ValueError(f"Unknown loss: {kind}")
    positive = logits.sigmoid()
    negative = (1-positive+RECIPE["asl_clip"]).clamp(max=1)
    weight = (1-positive).pow(RECIPE["gamma_pos"])*targets + (1-negative).pow(RECIPE["gamma_neg"])*(1-targets)
    log_likelihood = targets*positive.clamp_min(1e-8).log()+(1-targets)*negative.clamp_min(1e-8).log()
    return -(weight.detach()*log_likelihood).mean()


def auxiliary_loss(state_logits, states):
    valid = states != data.IGNORE
    if not valid.any():
        return state_logits.sum()*0
    return F.cross_entropy(state_logits[valid].float(), states[valid])


def source_signature(rows, train, holdout, checkpoint, smoke, device="cpu"):
    source = dict(base_checkpoint_sha256=exp.file_hash(checkpoint),
                  rows_sha256=exp.digest([{k: row[k] for k in ("name","call","text_hash","gold")} for row in rows]),
                  split_sha256=exp.digest(dict(train=train, holdout=holdout)),
                  recipe=RECIPE, config=evidence.DEFAULTS, smoke=bool(smoke),
                  feature_precision="cuda_fp16" if str(device).startswith("cuda") else "cpu_fp32",
                  implementation_sha256=exp.digest({module.__name__:exp.file_hash(module.__file__)
                      for module in (exp, data, evidence, baseline, __import__(__name__, fromlist=['']))}))
    return source, exp.digest(source)


def inspect_coverage(rows, train, holdout, tokenizer, base_probs, out, budget):
    """Token-location diagnostics are lexical proxies, not annotated symptom evidence."""
    out = Path(out)
    if (out/"coverage.json").exists():
        return json.loads((out/"coverage.json").read_text(encoding="utf-8"))
    holdout_position = {index: j for j,index in enumerate(holdout)}
    counters = {label:dict(false_negative=0, no_lexical_cue=0, cue_only_after_front=0,
                           cue_in_front=0) for label in baseline.LABELS}
    lengths, truncated, weak_counts, audit = [], 0, np.zeros((9,3),dtype=int), []
    sampled=np.zeros((9,3),dtype=int)
    training = set(train)
    for index, row in enumerate(rows):
        if index % 1000 == 0:
            budget.check(); print(f"coverage audit {index}/{len(rows)}", flush=True)
        text = baseline.transcript(row["doc"])
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        lengths.append(len(encoded["input_ids"]))
        capacity = 256-tokenizer.num_special_tokens_to_add(pair=False)
        offsets = encoded["offset_mapping"]
        cutoff = offsets[min(capacity,len(offsets))-1][1] if offsets else 0
        truncated += len(offsets)>capacity
        if index in holdout_position:
            j = holdout_position[index]
            for label_index,label in enumerate(baseline.LABELS):
                if row["gold"][label_index] and base_probs[j,label_index]<.5:
                    item = counters[label]; item["false_negative"] += 1
                    matches = list(re.finditer(exp.KEYWORDS[label], text))
                    if not matches:
                        item["no_lexical_cue"] += 1
                    elif all(match.start()>=cutoff for match in matches):
                        item["cue_only_after_front"] += 1
                    else:
                        item["cue_in_front"] += 1
        # Aux annotation candidates come only from the existing Training split.
        if index in training:
            parts, _ = data.utterance_spans(row["doc"])
            for start in range(len(parts)):
                window = parts[start:start+3]
                states = data.weak_states(window)
                for label_index,state in enumerate(states):
                    if state != data.IGNORE:
                        weak_counts[label_index,state] += 1
                        if sampled[label_index,state]<3:
                            sampled[label_index,state]+=1
                            audit.append(dict(name=row["name"], turn=start, label=baseline.LABELS[label_index],
                                              proposed_state=["positive","negative","unknown"][state],
                                              reviewed_state="", text=" | ".join(window)))
    report = dict(rows=len(rows), truncated_calls=int(truncated),
                  token_quantiles={str(q):float(np.quantile(lengths,q)) for q in (.5,.9,.95,.99)},
                  false_negative_lexical_location=counters,
                  training_weak_state_counts={label:weak_counts[i].tolist() for i,label in enumerate(baseline.LABELS)},
                  state_order=["positive","negative","unknown"],
                  caveat="Rules are weak supervision, not manual evidence labels. No holdout states are used for training.")
    exp.write_json(out/"coverage.json",report)
    with (out/"training_state_audit.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["name","turn","label","proposed_state","reviewed_state","text"])
        writer.writeheader(); writer.writerows(audit)
    return report


def make_base(payload, device):
    model = baseline._build(payload["files"], payload)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def train_base(payload, tokenizer, rows, train, holdout, weights, kind, seed, out, signature, budget, device):
    spec=dict(signature=signature, kind=kind, seed=seed, epochs=1, training_indices_sha256=exp.digest(train),
              holdout_indices_sha256=exp.digest(holdout))
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():
        if json.loads((out/"spec.json").read_text()) != spec:
            raise ValueError("Baseline resume configuration changed")
        return np.load(out/"holdout.npz",allow_pickle=False)["probs"].copy() if holdout else None
    exp.write_json(out/"spec.json",spec); exp.seed_all(seed)
    model=make_base(payload,device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE["encoder_lr"],weight_decay=RECIPE["weight_decay"])
    from transformers import get_linear_schedule_with_warmup
    updates=exp.math_ceil(len(train)/RECIPE["effective_batch"])
    scheduler=get_linear_schedule_with_warmup(optimizer,exp.math_ceil(updates*.1),updates)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    order=np.random.default_rng(seed).permutation(train).tolist()
    for start in range(0,len(order),RECIPE["effective_batch"]):
        budget.check(); group=order[start:start+RECIPE["effective_batch"]]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(group),RECIPE["base_micro_batch"]):
            chunk=group[offset:offset+RECIPE["base_micro_batch"]]
            packed,_=evidence.pack(tokenizer,[rows[i]["doc"] for i in chunk],"front",evidence.DEFAULTS)
            packed=evidence.to_device(packed,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(packed["ids"],packed["attention_mask"])
            gold=torch.tensor([rows[i]["gold"] for i in chunk],device=device,dtype=torch.float32)
            scaler.scale(objective(logits,gold,weights,kind)*len(chunk)/len(group)).backward()
        previous=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()>=previous:
            scheduler.step()
        if start%(RECIPE["effective_batch"]*100)==0:
            print(f"base_{kind} seed={seed} {start}/{len(train)}",flush=True)
    model.eval()
    texts=[baseline.transcript(rows[i]["doc"]) for i in holdout]
    probs=baseline.probabilities(model,tokenizer,texts,batch_size=8) if holdout else None
    export=dict(payload,model={key:value.detach().cpu() for key,value in model.state_dict().items()},
                source=dict(experimental=True,qualified_for_submission=False,signature=signature,kind=kind,seed=seed,
                            training_rows=len(train),official_validation_used=False))
    exp.save_torch(out/"experimental.pt",export)
    if holdout:
        np.savez_compressed(out/"holdout.npz",names=[rows[i]["name"] for i in holdout],
                            gold=[rows[i]["gold"] for i in holdout],probs=probs)
        result=exp.metrics(np.asarray([rows[i]["gold"] for i in holdout]),probs,[rows[i]["doc"] for i in holdout])
    else:result=dict(status="full_training_complete",training_rows=len(train))
    exp.write_json(out/"result.json",result)
    del model, export
    if device.type=="cuda": torch.cuda.empty_cache()
    return probs


def feature_cache(payload, tokenizer, rows, indices, out, signature, budget, device):
    """Cache all views on the same encoder; no holdout state labels enter fitting."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    specification=dict(signature=signature,indices=indices,encoder_sha256=exp.digest(payload.get("source",{})))
    marker=out/"index.json"
    if marker.exists() and json.loads(marker.read_text()) != specification:
        raise ValueError("Feature cache source changed")
    exp.write_json(marker,specification)
    model=make_base(payload,device).eval(); records={}
    for start in range(0,len(indices),64):
        budget.check(); chunk=indices[start:start+64]; path=out/f"shard_{start:06d}.pt"
        if path.exists():
            records.update(torch.load(path,map_location="cpu",weights_only=True)); continue
        shard={}
        for index in chunk:
            item={}
            for mode in ("multi","head_tail"):
                packed,views=evidence.pack(tokenizer,[rows[index]["doc"]],mode,evidence.DEFAULTS)
                packed=evidence.to_device(packed,device)
                with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    hidden=model.bert(input_ids=packed["ids"],attention_mask=packed["attention_mask"]).last_hidden_state[:,0]
                    logits=model.classifier(hidden)
                item[mode]=dict(cls=hidden.half().cpu(),logits=logits.float().cpu(),
                                states=torch.tensor(data.window_states(rows[index]["doc"],views[0],data.STATE_KEYWORDS)))
            shard[index]=item
        exp.save_torch(path,shard); records.update(shard)
        print(f"window features {start+len(chunk)}/{len(indices)}",flush=True)
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    return records


def feature_batch(records, indices, mode, device):
    selected=[records[i]["head_tail" if mode=="head_tail" else "multi"] for i in indices]
    width=1 if mode=="front" else max(len(item["cls"]) for item in selected)
    hidden=selected[0]["cls"].shape[-1]
    cls=torch.zeros(len(indices),width,hidden,device=device)
    logits=torch.zeros(len(indices),width,9,device=device)
    mask=torch.zeros(len(indices),width,dtype=torch.bool,device=device)
    states=torch.full((len(indices),width,9),data.IGNORE,dtype=torch.long,device=device)
    for j,item in enumerate(selected):
        n=min(width,len(item["cls"]))
        cls[j,:n]=item["cls"][:n].to(device);logits[j,:n]=item["logits"][:n].to(device)
        mask[j,:n]=True;states[j,:n]=item["states"][:n].to(device)
    return cls,logits,mask,states


def head_probabilities(head,records,indices,mode,device):
    outputs=[];head.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),128):
            cls,logits,mask,_=feature_batch(records,indices[start:start+128],mode,device)
            predicted,_=head(cls,logits,mask); outputs.append(predicted.sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def train_head(records,rows,train,holdout,mode,seed,out,signature,budget,device):
    specification=dict(signature=signature,mode=mode,seed=seed,epochs=RECIPE["head_epochs"],
                       training_indices_sha256=exp.digest(train),holdout_indices_sha256=exp.digest(holdout))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);exp.seed_all(seed)
    head=evidence.EvidenceHead(records[train[0]]["multi"]["cls"].shape[-1],mode,evidence.DEFAULTS).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=RECIPE["head_lr"],weight_decay=RECIPE["weight_decay"])
    weights=exp.positive_weights(rows,train,device); history=[];begin=0
    if (out/"head.pt").exists():
        saved=torch.load(out/"head.pt",map_location=device,weights_only=True)
        if saved["specification"]!=specification:raise ValueError("Head resume configuration changed")
        head.load_state_dict(saved["head"]);optimizer.load_state_dict(saved["optimizer"])
        begin=saved["epoch"]; history=saved["history"]
    for epoch in range(begin,RECIPE["head_epochs"]):
        exp.seed_all(seed+epoch);head.train(); order=np.random.default_rng(seed+epoch).permutation(train).tolist(); total=0
        for start in range(0,len(order),RECIPE["head_batch"]):
            budget.check(); chunk=order[start:start+RECIPE["head_batch"]]
            cls,logits,mask,states=feature_batch(records,chunk,mode,device)
            predicted,state_logits=head(cls,logits,mask)
            gold=torch.tensor([rows[i]["gold"] for i in chunk],device=device,dtype=torch.float32)
            value=objective(predicted,gold,weights,"bce")
            if mode in ("states","states_removed"):
                value=value+RECIPE["state_weight"]*auxiliary_loss(state_logits,states)
            if not torch.isfinite(value):raise RuntimeError("Nonfinite head loss")
            optimizer.zero_grad(set_to_none=True);value.backward();optimizer.step();total+=value.item()*len(chunk)
        record=dict(epoch=epoch+1,loss=total/len(train));history.append(record)
        exp.save_torch(out/"head.pt",dict(specification=specification,head=head.state_dict(),optimizer=optimizer.state_dict(),
                                         epoch=epoch+1,history=history))
        print(f"{mode} seed={seed} head epoch={epoch+1} loss={record['loss']:.4f}",flush=True)
    probs=head_probabilities(head,records,holdout,mode,device) if holdout else None
    if probs is not None:
        np.savez_compressed(out/"holdout.npz",names=[rows[i]["name"] for i in holdout],
                            gold=[rows[i]["gold"] for i in holdout],probs=probs)
        exp.write_json(out/"result.json",exp.metrics(np.asarray([rows[i]["gold"] for i in holdout]),probs,
                                                    [rows[i]["doc"] for i in holdout]))
    exp.write_json(out/"history.json",history)
    return head,probs


def fine_model(payload,tokenizer,rows,train,holdout,mode,seed,head,out,signature,budget,device):
    """One encoder epoch after a fixed frozen-head fit; final epoch only."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=dict(signature=signature,mode=mode,seed=seed,train=exp.digest(train),holdout=exp.digest(holdout))
    if (out/"experimental.pt").exists():
        saved=torch.load(out/"experimental.pt",map_location="cpu",weights_only=True)
        if saved["source"]["specification"]!=spec:raise ValueError("Fine resume configuration changed")
        return saved,np.load(out/"holdout.npz")["probs"].copy() if holdout else None
    exp.seed_all(seed)
    model=evidence.EvidenceModel(make_base(payload,device),mode).to(device)
    model.head.load_state_dict(head.state_dict());model.train()
    optimizer=torch.optim.AdamW([
        dict(params=model.base.parameters(),lr=RECIPE["encoder_lr"]),
        dict(params=model.head.parameters(),lr=RECIPE["head_fine_lr"])],weight_decay=RECIPE["weight_decay"])
    weights=exp.positive_weights(rows,train,device)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    from transformers import get_linear_schedule_with_warmup
    updates=exp.math_ceil(len(train)/RECIPE["effective_batch"])
    scheduler=get_linear_schedule_with_warmup(optimizer,exp.math_ceil(updates*.1),updates)
    order=np.random.default_rng(seed).permutation(train).tolist()
    for start in range(0,len(order),RECIPE["effective_batch"]):
        budget.check();group=order[start:start+RECIPE["effective_batch"]];optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(group),RECIPE["window_micro_batch"]):
            chunk=group[offset:offset+RECIPE["window_micro_batch"]]
            documents=[rows[i]["doc"] for i in chunk]
            packed,views=evidence.pack(tokenizer,documents,mode,model.config)
            states=torch.full((len(chunk),packed["width"],9),data.IGNORE,device=device,dtype=torch.long)
            for j,(doc,windows) in enumerate(zip(documents,views)):
                states[j,:len(windows)]=torch.tensor(data.window_states(doc,windows,data.STATE_KEYWORDS),device=device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits,state_logits=model(evidence.to_device(packed,device))
            gold=torch.tensor([rows[i]["gold"] for i in chunk],device=device,dtype=torch.float32)
            value=objective(logits,gold,weights,"bce")
            if mode in ("states","states_removed"):value+=RECIPE["state_weight"]*auxiliary_loss(state_logits,states)
            if not torch.isfinite(value):raise RuntimeError("Nonfinite fine loss")
            scaler.scale(value*len(chunk)/len(group)).backward()
        previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()>=previous:scheduler.step()
        if start%(RECIPE["effective_batch"]*100)==0:print(f"fine {mode} seed={seed} {start}/{len(train)}",flush=True)
    model.eval()
    probs=evidence.probabilities(model,tokenizer,[rows[i]["doc"] for i in holdout]) if holdout else None
    if holdout:
        np.savez_compressed(out/"holdout.npz",names=[rows[i]["name"] for i in holdout],gold=[rows[i]["gold"] for i in holdout],probs=probs)
        exp.write_json(out/"result.json",exp.metrics(np.asarray([rows[i]["gold"] for i in holdout]),probs))
    exported=evidence.export_payload(model,payload,dict(experimental=True,qualified_for_submission=False,specification=spec))
    exp.save_torch(out/"experimental.pt",exported)
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    return exported,probs


def checked_gate(gold,candidate,ensemble,control,ablation,rows,holdout):
    gate=exp.adoption_gate(gold,candidate,ensemble,control,ablation,calls=[rows[i]["call"] for i in holdout],smoke=False)
    gate["checks"].update(gate["contribution_checks"])
    gate["passed"]=all(gate["checks"].values())
    return gate


def run(args):
    started=time.monotonic();out=args.out;out.mkdir(parents=True,exist_ok=True)
    RECIPE["head_epochs"]=2 if args.smoke_rows else 10
    budget=exp.RunBudget(out,args.max_hours);device=torch.device(args.device);torch.set_num_threads(4)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    rows,train,holdout,base_probs,ensemble=exp.prepare_data(args.train_label_dir,args.reference_probs_dir,args.smoke_rows)
    source,signature=source_signature(rows,train,holdout,args.base_ckpt,args.smoke_rows,device)
    marker=out/"source.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))!=source:
        raise ValueError("Experiment resume source/configuration changed")
    exp.write_json(marker,source)
    payload=torch.load(args.base_ckpt,map_location="cpu",weights_only=True)
    if payload.get("format")!=baseline.FORMAT or tuple(payload.get("labels",[]))!=baseline.LABELS:
        raise ValueError("A self-contained RoBERTa baseline is required")
    if not args.smoke_rows and (payload.get("source",{}).get("exp_id")!="M3-ROBERTA-003" or payload["source"].get("seed")!=42):
        raise ValueError("ROBERTA-003 seed42 required")
    from transformers import AutoTokenizer
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        for name,value in payload["files"].items():
            if Path(name).name!=name:raise ValueError("Unsafe tokenizer file")
            (Path(directory)/name).write_text(value,encoding="utf-8")
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
    summary=dict(status="running",smoke=bool(args.smoke_rows),source=source,selected_method=None,
                 official_validation_used=False,pilot={},loss_comparison={},gate={"passed":False})
    gold=np.asarray([rows[i]["gold"] for i in holdout]);docs=[rows[i]["doc"] for i in holdout]
    summary["pilot"].update(roberta003=exp.metrics(gold,base_probs,docs),existing_ensemble=exp.metrics(gold,ensemble,docs))
    def save():
        summary["elapsed_seconds"]=time.monotonic()-started;exp.report(out,summary)
        report=out/"report.md"
        text=report.read_text(encoding="utf-8").replace("# M3 context experiment","# M3 loss, coverage and evidence-state experiment",1)
        text=text.replace("Training-only retrieval.","Training-only weak state supervision; at most six windows.")
        if summary["loss_comparison"]:
            text+="\n## Matched one-epoch loss comparison\n\n| Loss | Macro F1 | Micro F1 |\n|---|---:|---:|\n"
            for name,item in summary["loss_comparison"].items():text+=f"| {name} | {item['macro_f1']:.4f} | {item['micro_f1']:.4f} |\n"
        report.write_text(text,encoding="utf-8")
    try:
        inspect_coverage(rows,train,holdout,tokenizer,base_probs,out,budget)
        save()
        if args.phase=="audit":summary["status"]="audit_complete";save();return summary
        weights=exp.positive_weights(rows,train,device)
        for kind in ("original","bce","asl"):
            probs=train_base(payload,tokenizer,rows,train,holdout,weights,kind,42,out/f"loss_{kind}_42",signature,budget,device)
            summary["loss_comparison"][kind]=exp.metrics(gold,probs,docs);save()
        # Loss selection is a screening decision, not a submission qualification.
        loss_winner=max(("bce","asl"),key=lambda name:summary["loss_comparison"][name]["macro_f1"])
        minimum=max(summary["loss_comparison"]["original"]["macro_f1"],summary["pilot"]["roberta003"]["macro_f1"])
        if summary["loss_comparison"][loss_winner]["macro_f1"]>minimum:
            chosen=torch.load(out/f"loss_{loss_winner}_42/experimental.pt",map_location="cpu",weights_only=True)
            del payload;payload=chosen;summary["feature_encoder_loss"]=loss_winner
        else:summary["feature_encoder_loss"]="original_checkpoint"
        if args.phase=="loss":summary["status"]="loss_screen_complete";save();return summary
        records=feature_cache(payload,tokenizer,rows,sorted(set(train+holdout)),out/"features",signature,budget,device)
        methods=("front","head_tail","multi","states","states_removed")
        for mode in methods:
            _,probs=train_head(records,rows,train,holdout,mode,42,out/f"{mode}_42",signature,budget,device)
            summary["pilot"][mode]=exp.metrics(gold,probs,docs);save()
        floor=max(summary["pilot"][name]["macro_f1"] for name in ("front","roberta003","existing_ensemble"))
        if summary["feature_encoder_loss"]!="original_checkpoint":
            floor=max(floor,summary["loss_comparison"][loss_winner]["macro_f1"])
        successful=[name for name in ("head_tail","multi","states") if summary["pilot"][name]["macro_f1"]>floor]
        winner=max(successful,key=lambda name:summary["pilot"][name]["macro_f1"]) if successful else None
        summary.update(status="smoke_complete" if args.smoke_rows else "input_pilots_failed" if winner is None else "repeat_pending",selected_method=winner)
        save()
        if args.smoke_rows or args.phase=="pilot":return summary
        # A loss-only winner is useful even when richer inputs do not help.
        if winner is None:
            if summary["loss_comparison"][loss_winner]["macro_f1"]<=max(minimum,summary["pilot"]["existing_ensemble"]["macro_f1"]):return summary
            candidates=[];controls=[]
            for seed in RECIPE["seeds"]:
                candidates.append(train_base(torch.load(args.base_ckpt,map_location="cpu",weights_only=True),tokenizer,rows,train,holdout,weights,loss_winner,seed,out/f"loss_{loss_winner}_{seed}",signature,budget,device))
                controls.append(train_base(torch.load(args.base_ckpt,map_location="cpu",weights_only=True),tokenizer,rows,train,holdout,weights,"original",seed,out/f"loss_original_{seed}",signature,budget,device))
            summary["gate"]=checked_gate(gold,candidates,ensemble,controls,controls,rows,holdout)
            summary["selected_method"]="loss_"+loss_winner
            summary["status"]="repeat_gate_failed"
            if summary["gate"]["passed"]:
                full=list(range(len(rows)))
                original=torch.load(args.base_ckpt,map_location="cpu",weights_only=True)
                train_base(original,tokenizer,rows,full,[],exp.positive_weights(rows,full,device),loss_winner,42,out/"full_loss",signature,budget,device)
                final=torch.load(out/"full_loss/experimental.pt",map_location="cpu",weights_only=True)
                final["source"].update(smoke=False,training_rows=len(full),representative_seed=42,internal_gate=summary["gate"],official_validation_used=False,qualified_for_submission=False)
                exp.save_torch(out/"final_candidate.pt",final);summary["status"]="final_candidate_ready"
            save();return summary
        candidate,control,ablation=[],[],[]
        ablated="states_removed" if winner=="states" else "front"
        summary["repeat"]={winner:{"frozen":[]}}
        for seed in RECIPE["seeds"]:
            for mode,destination in ((winner,candidate),("front",control),(ablated,ablation)):
                _,probs=train_head(records,rows,train,holdout,mode,seed,out/f"{mode}_{seed}",signature,budget,device)
                destination.append(probs)
                if mode==winner:summary["repeat"][winner]["frozen"].append(exp.metrics(gold,probs,docs))
            save()
        summary["frozen_gate"]=checked_gate(gold,candidate,ensemble,control,ablation,rows,holdout)
        summary["gate"]=summary["frozen_gate"]
        fine_candidate=[];fine_control=[];fine_ablation=[]
        summary["repeat"][winner]["fine"]=[]
        for seed in RECIPE["seeds"]:
            for mode,destination in ((winner,fine_candidate),("front",fine_control),(ablated,fine_ablation)):
                head,_=train_head(records,rows,train,holdout,mode,seed,out/f"{mode}_{seed}",signature,budget,device)
                _,probs=fine_model(payload,tokenizer,rows,train,holdout,mode,seed,head,out/f"fine_{mode}_{seed}",signature,budget,device)
                destination.append(probs)
                if mode==winner:summary["repeat"][winner]["fine"].append(exp.metrics(gold,probs,docs))
            save()
        summary["fine_gate"]=checked_gate(gold,fine_candidate,ensemble,fine_control,fine_ablation,rows,holdout)
        # Fix the recipe using internal results before any official access.
        stage="fine" if summary["fine_gate"]["passed"] and (not summary["frozen_gate"]["passed"] or summary["fine_gate"]["mean_macro_f1"]>summary["frozen_gate"]["mean_macro_f1"]) else "frozen"
        summary["gate"]=summary[stage+"_gate"];summary["selected_stage"]=stage
        summary["status"]="repeat_gate_failed"
        if summary["gate"]["passed"]:
            full=list(range(len(rows)))
            # Recreate the chosen loss adaptation on all Training, not the internal split.
            if summary["feature_encoder_loss"]!="original_checkpoint":
                original=torch.load(args.base_ckpt,map_location="cpu",weights_only=True)
                train_base(original,tokenizer,rows,full,[],exp.positive_weights(rows,full,device),loss_winner,42,out/"full_loss",signature,budget,device)
                payload=torch.load(out/"full_loss/experimental.pt",map_location="cpu",weights_only=True)
                records=feature_cache(payload,tokenizer,rows,full,out/"full_features",signature,budget,device)
            head,_=train_head(records,rows,full,[],winner,42,out/"full_head",signature,budget,device)
            if stage=="fine":final,_=fine_model(payload,tokenizer,rows,full,[],winner,42,head,out/"full_fine",signature,budget,device)
            else:
                model=evidence.EvidenceModel(make_base(payload,device),winner).to(device);model.head.load_state_dict(head.state_dict())
                final=evidence.export_payload(model,payload,{})
            final["source"].update(smoke=False,training_rows=len(full),representative_seed=42,internal_gate=summary["gate"],signature=signature,official_validation_used=False,selected_stage=stage,qualified_for_submission=False,experimental=True)
            exp.save_torch(out/"final_candidate.pt",final);summary["status"]="final_candidate_ready"
        save();return summary
    except exp.BudgetReached:
        summary["status"]="budget_reached_resume_pending";save();return summary
    finally:
        try:budget.check(force=True)
        except exp.BudgetReached:pass
