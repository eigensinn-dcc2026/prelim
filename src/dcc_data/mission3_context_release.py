"""One final official check and a separate, hash-verified offline package."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dcc_inference import mission3_bert as baseline
from dcc_inference import mission3_context as context
from .mission3_context_experiment import digest, file_hash, metrics, write_json


def validate_final(payload):
    source=payload.get("source",{})
    gate=source.get("internal_gate",{})
    old=payload.get("format")==context.FORMAT and payload.get("method") in ("memory","evidence")
    new=payload.get("format")=="dcc-m3-evidence-v2" and payload.get("mode") in ("head_tail","multi","states")
    loss=payload.get("format")==baseline.FORMAT and source.get("kind") in ("bce","asl")
    if (not (old or new or loss)
        or source.get("smoke",True) or source.get("training_rows")!=29200
        or source.get("representative_seed")!=42 or source.get("official_validation_used") is not False
        or not gate.get("passed") or not gate.get("checks") or not all(gate["checks"].values())):
        raise ValueError("Official evaluation requires a full Training seed42 candidate with a passed internal gate")
    if (new or loss) and (not gate.get("contribution_checks") or not all(gate["contribution_checks"].values())):
        raise ValueError("New information/loss must beat its matched controls")


def candidate_probabilities(model,tokenizer,docs,names,batch_size):
    if getattr(model,"checkpoint_format",None)=="dcc-m3-evidence-v2":
        from dcc_inference.mission3_evidence import probabilities
        return probabilities(model,tokenizer,docs,names,batch_size=batch_size)
    if getattr(model,"requires_documents",False):
        return context.probabilities(model,tokenizer,docs,names,batch_size=batch_size)
    return baseline.probabilities(model,tokenizer,[baseline.transcript(doc) for doc in docs],batch_size=batch_size)


def evaluate_official(checkpoint,label_dir,output,registry,device=None,batch_size=32,expected_rows=3640):
    """Reserve a single recipe; resume only this identical candidate and dataset.

    Inference shards can resume after interruption. A completed evaluation returns
    its existing result without running another forward pass. The registry is
    shared across output directories, so a different directory does not reset it.
    """
    output,registry=Path(output),Path(registry)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    validate_final(payload)
    paths=sorted(Path(label_dir).rglob("*.json"))
    if len(paths)!=expected_rows or len({p.name for p in paths})!=len(paths):
        raise ValueError(f"Official Validation must contain {expected_rows} uniquely named calls")
    docs=[json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
    names=[p.name for p in paths]
    gold=np.array([[int(label in doc.get("symptom",[])) for label in context.LABELS] for doc in docs])
    specification=dict(checkpoint_sha256=file_hash(checkpoint),rows=len(paths),
                       dataset_sha256=digest([dict(name=name,transcript=context.text_hash(context.text_and_spans(doc)[0]),
                                                  gold=row.tolist()) for name,doc,row in zip(names,docs,gold)]))
    registry.parent.mkdir(parents=True,exist_ok=True)
    if registry.exists():
        reserved=json.loads(registry.read_text(encoding="utf-8"))
        if reserved["specification"]!=specification:
            raise ValueError("Official Validation already reserved for another final candidate; no tuning against it")
        output=Path(reserved["output"])
        if reserved["status"]=="completed":
            return json.loads((output/"official_metrics.json").read_text(encoding="utf-8"))
    else:
        with registry.open("x",encoding="utf-8") as stream:
            json.dump(dict(specification=specification,status="in_progress",output=str(output.resolve())),stream,indent=2)
    output.mkdir(parents=True,exist_ok=True)
    model,tokenizer,_,_=baseline.load_checkpoint(checkpoint,device)
    started=time.monotonic()
    predictions=[]
    for start in range(0,len(paths),batch_size):
        shard=output/f"probabilities_{start:06d}.npz"
        chunk_names=names[start:start+batch_size]
        if shard.exists():
            with np.load(shard,allow_pickle=False) as cached:
                if cached["names"].astype(str).tolist()!=chunk_names:
                    raise ValueError("Official inference shard order changed")
                probs=cached["probs"].copy()
        else:
            probs=candidate_probabilities(model,tokenizer,docs[start:start+batch_size],chunk_names,batch_size)
            np.savez_compressed(shard,names=chunk_names,probs=probs)
        if probs.shape!=(len(chunk_names),9) or not np.isfinite(probs).all():
            raise ValueError("Invalid official prediction shard")
        predictions.append(probs)
        if start%(batch_size*20)==0:
            print(f"official final inference {min(start+batch_size,len(paths))}/{len(paths)}",flush=True)
    probabilities=np.concatenate(predictions)
    np.savez_compressed(output/"official.npz",names=names,gold=gold,probs=probabilities)
    pd.DataFrame({"label file name":names,"symptom":[str([label for label,positive in zip(context.LABELS,row) if positive])
                                                  for row in probabilities>=.5]})\
        .to_csv(output/"mission3.csv",index=False,encoding="utf-8")
    result=dict(specification=specification,metrics=metrics(gold,probabilities,docs),
                elapsed_seconds=time.monotonic()-started,threshold=.5,adoption_threshold=.6540)
    result["adopt"]=bool(result["metrics"]["macro_f1"]>=.6540)
    result["decision"]="replace_candidate" if result["adopt"] else "retain_roberta003"
    write_json(output/"official_metrics.json",result)
    write_json(registry,dict(specification=specification,status="completed",output=str(output.resolve()),adopt=result["adopt"]))
    return result


def package_candidate(checkpoint,official_result,output,root):
    checkpoint,output,root=Path(checkpoint),Path(output),Path(root)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    validate_final(payload)
    result=json.loads(Path(official_result).read_text(encoding="utf-8"))
    if (not result.get("adopt") or result["metrics"]["macro_f1"]<.6540
        or result["specification"]["checkpoint_sha256"]!=file_hash(checkpoint)):
        raise ValueError("Package promotion requires a matching official score >= 0.6540")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new candidate directory; existing submission files are protected")
    output.mkdir(parents=True,exist_ok=True)
    legacy=payload.get("format")==context.FORMAT
    template=root/"submission"/("mission3_context" if legacy else "mission3_evidence")
    for name in ("inference.py","requirements.txt","README.md"):
        shutil.copy2(template/name,output/name)
    for name in (("mission3_bert.py","mission3_context.py") if legacy else ("mission3_bert.py","mission3_evidence.py")):
        shutil.copy2(root/"src"/"dcc_inference"/name,output/name)
    if not legacy:shutil.copy2(root/"src"/"dcc_data"/"mission3_evidence_data.py",output/"mission3_evidence_data.py")
    shutil.copy2(checkpoint,output/"mission3.pt")
    shutil.copy2(official_result,output/"official_metrics.json")
    manifest={path.name:file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output/"freeze.json",dict(method=payload.get("method",payload.get("mode",payload.get("source",{}).get("kind"))),seed=42,threshold=.5,
                                       official_macro_f1=result["metrics"]["macro_f1"],files=manifest))
    return output
