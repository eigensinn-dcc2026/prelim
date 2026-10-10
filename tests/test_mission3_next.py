"""Coverage, weak-state leakage, loss equivalence, training and offline tests."""
from argparse import Namespace
import copy
import json
import socket
from pathlib import Path
import numpy as np
import pytest
from test_mission3_context import tiny
from dcc_data import mission3_evidence_data as data, mission3_next_experiment as next_exp
from dcc_inference import mission3_evidence as evidence, mission3_bert as baseline

torch=next_exp.torch


def test_small_batch_f1_changes_objective_but_bce_accumulation_does_not():
    logits=torch.tensor([[2.]*9,[-2.]*9,[1.]*9,[-1.]*9],requires_grad=True)
    gold=torch.tensor([[1.]*9,[0.]*9,[0.]*9,[0.]*9]);weights=torch.ones(9)
    for kind in ("bce","asl"):
        full=next_exp.objective(logits,gold,weights,kind)
        split=sum(next_exp.objective(logits[i:i+2],gold[i:i+2],weights,kind) for i in (0,2))/2
        torch.testing.assert_close(full,split)
    full=next_exp.objective(logits,gold,weights,"original")
    split=sum(next_exp.objective(logits[i:i+2],gold[i:i+2],weights,"original") for i in (0,2))/2
    assert abs(full.item()-split.item())>.01
    for kind in ("original","bce","asl"):
        value=next_exp.objective(logits,gold,weights,kind);value.backward(retain_graph=True)
        assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_weak_states_do_not_turn_unknown_into_negative_or_read_gold():
    keywords=next_exp.exp.KEYWORDS; label=baseline.LABELS.index("두통")
    assert data.weak_states(["머리 아프세요?"],keywords)[label]==data.UNKNOWN
    assert data.weak_states(["머리 아프세요?","네"],keywords)[label]==data.POSITIVE
    assert data.weak_states(["머리 아프세요?","아니요"],keywords)[label]==data.NEGATIVE
    assert data.weak_states(["머리 아파요","머리 안 아파요"],keywords)[label]==data.IGNORE
    assert data.weak_states(["주소 알려주세요"],keywords)==[data.IGNORE]*9


def test_views_preserve_front_and_recover_late_tokens_without_extra_budget(tiny):
    _,tokenizer,_=tiny
    doc={"utterances":[{"text":"열 "*300},{"text":"머리"},{"text":"배"}],"symptom":["두통"],"speaker":99}
    front=data.tokenize_views(tokenizer,doc,"front")
    direct=tokenizer(baseline.transcript(doc),truncation=True,max_length=256)["input_ids"]
    assert front[0]["ids"]==direct
    tail_id=tokenizer("배",add_special_tokens=False)["input_ids"][0]
    assert tail_id not in front[0]["ids"]
    for mode in ("head_tail","multi","states"):
        views=data.tokenize_views(tokenizer,doc,mode,max_windows=3)
        assert len(views)<=3 and all(len(view["ids"])<=256 for view in views)
        assert any(tail_id in view["ids"] for view in views)
        assert views==data.tokenize_views(tokenizer,{"utterances":doc["utterances"]},mode,max_windows=3)
    assert data.tokenize_views(tokenizer,{"utterances":[]},"multi")[0]["ids"]


def test_partial_utterance_does_not_generate_state_supervision(tiny):
    _,tokenizer,_=tiny
    doc={"utterances":[{"text":"머리 아프세요?"},{"text":"네 "*400}]}
    front=data.tokenize_views(tokenizer,doc,"front")
    states=data.window_states(doc,front,next_exp.exp.KEYWORDS)
    assert states[0][baseline.LABELS.index("두통")]==data.UNKNOWN


@pytest.mark.parametrize("mode",["front","head_tail","multi","states","states_removed"])
def test_offline_roundtrip_and_encoder_gradients(tiny,tmp_path,mode):
    context,tokenizer,payload=tiny
    model=evidence.EvidenceModel(context.base,mode).eval()
    docs=[{"utterances":[{"text":"열 "*260},{"text":"머리"},{"text":"배"}]},{"utterances":[]}]
    expected=evidence.probabilities(model,tokenizer,docs,batch_size=1)
    original=baseline.probabilities(context.base,tokenizer,[baseline.transcript(doc) for doc in docs],batch_size=1)
    if mode!="head_tail": np.testing.assert_array_equal(expected,original)
    exported=evidence.export_payload(model,payload,{"experimental":True,"internal_gate":{"passed":False}})
    restored,rtok,labels,_=evidence.load_payload(exported,"cpu")
    np.testing.assert_array_equal(expected,evidence.probabilities(restored,rtok,docs,batch_size=1))
    assert labels==list(baseline.LABELS)
    with torch.no_grad():model.head.residual.weight.fill_(.01)
    packed,_=evidence.pack(tokenizer,docs,mode,model.config)
    logits,states=model(packed)
    (logits.square().mean()+states.square().mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.base.parameters())
    assert torch.isfinite(next_exp.auxiliary_loss(states,torch.full(states.shape[:-1],data.IGNORE)))


def test_full_smoke_pipeline_is_not_adoptable_and_resumes(tiny,tmp_path):
    model,tokenizer,payload=tiny
    training=tmp_path/"Training";training.mkdir();references=tmp_path/"references";references.mkdir()
    docs=[]
    for i in range(24):
        doc={"utterances":[{"text":"머리 아프세요?"},{"text":"네" if i%2 else "아니요"}],
             "symptom":[baseline.LABELS[i%9]]}
        (training/f"{i:03d}.json").write_text(json.dumps(doc,ensure_ascii=False),encoding="utf-8");docs.append(doc)
    names=[f"{i:03d}.json" for i in range(16,24)]
    gold=np.array([[int(label in docs[i]["symptom"]) for label in baseline.LABELS] for i in range(16,24)])
    for name in ("r3","k12","p0"):
        np.savez(references/f"{name}_holdout.npz",names=names,gold=gold,probs=np.full((8,9),.6))
    ckpt=tmp_path/"base.pt"
    torch.save(dict(payload,model={key:value.detach().clone() for key,value in model.base.state_dict().items()}),ckpt)
    args=Namespace(train_label_dir=training,base_ckpt=ckpt,reference_probs_dir=references,out=tmp_path/"run",
                   device="cpu",phase="all",smoke_rows=16,max_hours=.1)
    result=next_exp.run(args)
    assert result["status"]=="smoke_complete" and result["smoke"] and not result["gate"]["passed"]
    assert not result["official_validation_used"]
    assert len(result["loss_comparison"])==3 and all(mode in result["pilot"] for mode in next_exp.INPUT_METHODS)
    saved=np.load(args.out/"states_42/holdout.npz")["probs"]
    resumed=next_exp.run(args)
    np.testing.assert_array_equal(saved,np.load(args.out/"states_42/holdout.npz")["probs"])
    assert resumed["status"]=="smoke_complete"
    private=(args.out/"training_state_audit.csv").read_text(encoding="utf-8-sig")
    assert all(name not in private for name in names)


def test_fine_resume_and_final_guard(tiny,tmp_path):
    context,tokenizer,payload=tiny
    payload=dict(payload,model=context.base.state_dict())
    rows=[dict(name=f"{i}.json",call=str(i),gold=[i%2]*9,doc={"utterances":[{"text":"머리 아프세요?"},{"text":"네"}]}) for i in range(6)]
    head=evidence.EvidenceHead(context.base.classifier.in_features,"states",evidence.DEFAULTS)
    budget=next_exp.exp.RunBudget(tmp_path,.1)
    first,probs=next_exp.fine_model(payload,tokenizer,rows,[0,1,2,3],[4,5],"states",42,head,tmp_path/"fine","tiny",budget,torch.device("cpu"))
    _,again=next_exp.fine_model(payload,tokenizer,rows,[0,1,2,3],[4,5],"states",42,head,tmp_path/"fine","tiny",budget,torch.device("cpu"))
    np.testing.assert_array_equal(probs,again)
    from dcc_data import mission3_context_release as release
    with pytest.raises(ValueError):release.validate_final(first)
    first["source"].update(smoke=False,training_rows=29200,representative_seed=42,official_validation_used=False,internal_gate=dict(passed=True,checks={"full_dataset":True},contribution_checks={"beats_control":True}))
    release.validate_final(first)
    first["source"]["internal_gate"]["contribution_checks"]["beats_control"]=False
    with pytest.raises(ValueError,match="matched controls"):release.validate_final(first)


def test_intrinsic_absence_and_unrelated_denial_are_not_negative():
    def state(text,label):return data.weak_states([text])[baseline.LABELS.index(label)]
    assert state("기력이 없으셔서요","전신쇠약")==data.POSITIVE
    assert state("먹을 수도 없고 토할 거 같고 기침하고","오심")==data.POSITIVE
    assert state("머리 너무 아파 가지고, 아니 와서 봐 주세요","두통")==data.POSITIVE
    assert state("두통은 없어요","두통")==data.NEGATIVE
    assert state("머리가 안 아파요","두통")==data.NEGATIVE
    assert state("호흡은 정상이에요","호흡곤란")==data.IGNORE
    assert state("숨이 차요","호흡곤란")==data.POSITIVE
    assert data.weak_states(["머리 아프세요?","주소를 말씀하셨나요?","네"])[baseline.LABELS.index("두통")]==data.UNKNOWN


def test_v2_official_once_and_standalone_csv(tiny,tmp_path):
    import subprocess,sys
    import pandas as pd
    from dcc_data import mission3_context_release as release
    context,tokenizer,payload=tiny
    model=evidence.EvidenceModel(context.base,"states")
    candidate=evidence.export_payload(model,payload,dict(smoke=False,training_rows=29200,representative_seed=42,official_validation_used=False,
        internal_gate=dict(passed=True,checks={"full_dataset":True},contribution_checks={"beats_control":True,"beats_ablation":True})))
    checkpoint=tmp_path/"final.pt";torch.save(candidate,checkpoint)
    labels=tmp_path/"Validation";labels.mkdir()
    (labels/"0.json").write_text(json.dumps(dict(utterances=[dict(text="머리")],symptom=list(baseline.LABELS))),encoding="utf-8")
    result=release.evaluate_official(checkpoint,labels,tmp_path/"official",tmp_path/"registry.json","cpu",batch_size=1,expected_rows=1)
    assert result["adopt"]
    assert release.evaluate_official(checkpoint,labels,tmp_path/"other",tmp_path/"registry.json","cpu",expected_rows=1)==result
    root=Path(__file__).resolve().parents[1]
    package=release.package_candidate(checkpoint,tmp_path/"official/official_metrics.json",tmp_path/"package",root)
    run=subprocess.run([sys.executable,str(package/"inference.py"),"--audio_dir",str(tmp_path),"--label_dir",str(labels),"--ckpt_path",str(package/"mission3.pt"),"--output",str(tmp_path/"output.csv")],capture_output=True,text=True,encoding="utf-8",timeout=240)
    assert run.returncode==0,run.stderr
    assert pd.read_csv(tmp_path/"output.csv").columns.tolist()==["label file name","symptom"]
    assert pd.read_csv(tmp_path/"output.csv")["symptom"].iloc[0]==str(list(baseline.LABELS))
