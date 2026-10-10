"""Offline tiny-model tests for data boundaries, residuals, resume, and promotion."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dcc_inference import mission3, mission3_bert as baseline, mission3_context as context
from dcc_data import mission3_context_experiment as experiment
from dcc_data import mission3_context_release as release
from test_inference_mission3_bert import _model_dir, _lab_ckpt


@pytest.fixture
def tiny(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    def refuse(*args,**kwargs):
        raise AssertionError("Context inference must not access the network")
    monkeypatch.setattr(socket.socket,"connect",refuse)
    directory=_model_dir(tmp_path)
    checkpoint,_=_lab_ckpt(tmp_path,directory,bias=[2.]*9)
    exported=tmp_path/"baseline.pt"
    baseline.export_checkpoint(checkpoint,directory,exported)
    model,tokenizer,payload=experiment.initialize(exported,"evidence","cpu")
    return model.eval(),tokenizer,payload


def test_offsets_preserve_normalization_and_ignore_metadata(tiny):
    _,tokenizer,_=tiny
    doc={"utterances":[{"text":" 열 \n 머리","speaker":9},{"text":""},{"text":"배","speaker":2}],
         "symptom":["고열"],"age":99}
    text,spans=context.text_and_spans(doc)
    assert text==baseline.transcript(doc)=="열 머리 배"
    assert [text[a:b] for a,b in spans]==["열 머리","배"]
    encoded,_=context.tokenize_documents(tokenizer,[doc,{"utterances":[]}])
    assert set(encoded["turn_ids"][0].tolist())=={-1,0,1}
    assert (encoded["turn_ids"][1]==-1).all()
    changed={"utterances":[{"text":u["text"]} for u in doc["utterances"]]}
    clean,_=context.tokenize_documents(tokenizer,[changed,{"utterances":[]}])
    for key in encoded:
        assert torch.equal(encoded[key],clean[key])
    long={"utterances":[{"text":"열 "*254},{"text":"배"}]}
    truncated,_=context.tokenize_documents(tokenizer,[long])
    assert truncated["input_ids"].shape[1]==256
    assert (truncated["turn_ids"]==1).sum()==0


def test_pooling_excludes_specials_and_remains_differentiable():
    hidden=torch.tensor([[[99.],[1.],[3.],[6.],[99.]]],requires_grad=True)
    turns,mask=context.pool_turns(hidden,torch.tensor([[-1,0,0,1,-1]]))
    assert turns.tolist()==[[[2.],[6.]]]
    assert mask.tolist()==[[True,True]]
    turns.sum().backward()
    assert hidden.grad.flatten().tolist()==[0.,.5,.5,1.,0.]


def test_memory_excludes_call_and_identical_full_transcript():
    labels=[[1]*9,[1]*9,[0]*9,[1]*9]
    bank=context.MemoryBank([[1.,0.],[1.,0.],[0.,1.],[.6,.8]],labels,
                            ["self","other","negative","positive"],["a","same","c","d"])
    scores=bank.similarities(torch.tensor([[1.,0.]]),["self"],["same"])
    assert torch.isneginf(scores[0,:2]).all()
    prototypes,stats=bank.retrieve(torch.tensor([[1.,0.]]),["self"],["same"],context.DEFAULTS)
    torch.testing.assert_close(prototypes[0,:,0],torch.tensor([[.6,.8]]*9))
    torch.testing.assert_close(prototypes[0,:,1],torch.tensor([[0.,1.]]*9))
    assert torch.isfinite(stats).all() and (stats[:,:,5]==1).all()
    with pytest.raises(ValueError,match="Every query"):
        bank.similarities(torch.tensor([[1.,0.]]),[],[])
    with pytest.raises(ValueError,match="Only Training"):
        context.MemoryBank([[1.,0.]],[[0]*9],["a"],["b"],split="validation")


@pytest.mark.parametrize("empty",[False,True])
def test_memory_missing_neighbors_is_finite_baseline_fallback(empty):
    vectors=torch.empty(0,2) if empty else torch.tensor([[1.,0.]])
    labels=torch.empty(0,9) if empty else torch.ones(1,9)
    bank=context.MemoryBank(vectors,labels,[] if empty else ["a"],[] if empty else ["b"])
    head=context.ResidualHead(2,"memory",context.DEFAULTS).eval()
    with torch.no_grad():
        head.residual[-1].bias.fill_(3)
    base=torch.randn(1,9)
    got=head(torch.ones(1,2),base,torch.zeros(1,1,2),torch.zeros(1,1,dtype=torch.bool),
             bank=bank,calls=["a"],hashes=["b"])
    torch.testing.assert_close(got,base,rtol=0,atol=0)


@pytest.mark.parametrize("method",["control","evidence","answers_removed","memory","memory_removed"])
def test_zero_residual_exactly_preserves_baseline(method):
    head=context.ResidualHead(3,method,context.DEFAULTS).eval()
    bank=context.MemoryBank(torch.randn(4,3),torch.tensor([[0]*9,[1]*9]*2),list("abcd"),list("1234"))
    base=torch.randn(2,9)
    got=head(torch.randn(2,3),base,torch.randn(2,4,3),torch.ones(2,4,dtype=torch.bool),
             bank=bank,calls=["x","y"],hashes=["X","Y"])
    torch.testing.assert_close(got,base,atol=0,rtol=0)


def test_answers_ablation_removes_following_slots_and_masked_turns():
    config=dict(context.DEFAULTS,dimension=4,dropout=0.)
    evidence=context.ResidualHead(3,"evidence",config).eval()
    removed=context.ResidualHead(3,"answers_removed",config).eval()
    with torch.no_grad():
        evidence.residual[-1].weight.fill_(1.)
    removed.load_state_dict(evidence.state_dict())
    cls,base=torch.ones(1,3),torch.zeros(1,9)
    turns=torch.randn(1,3,3)
    mask=torch.tensor([[True,False,False]])
    changed=turns.clone();changed[:,1:]+=20
    torch.testing.assert_close(removed(cls,base,turns,mask),removed(cls,base,changed,mask),atol=0,rtol=0)
    # A malformed padded turn must not leak into the evidence window either.
    torch.testing.assert_close(evidence(cls,base,turns,mask),evidence(cls,base,changed,mask),atol=0,rtol=0)
    live=torch.ones_like(mask)
    assert not torch.allclose(evidence(cls,base,turns,live),evidence(cls,base,changed,live))


def test_offline_roundtrip_and_existing_predict_schema(tmp_path,tiny):
    model,tokenizer,payload=tiny
    labels=tmp_path/"labels";labels.mkdir()
    docs=[{"utterances":[{"text":"머리"},{"text":"배"}],"symptom":["두통"]},
          {"utterances":[],"symptom":[]}]
    for name,doc in zip(["a.json","b.json"],docs):
        (labels/name).write_text(json.dumps(doc,ensure_ascii=False),encoding="utf-8")
    checkpoint=tmp_path/"context.pt"
    torch.save(context.export_payload(model,payload,{"smoke":True}),checkpoint)
    loaded,loaded_tokenizer,order,length=baseline.load_checkpoint(checkpoint,"cpu")
    assert order==list(context.LABELS) and length==256
    with pytest.raises(ValueError,match="raw documents"):
        baseline.probabilities(loaded,loaded_tokenizer,["머리 배"])
    expected=context.probabilities(model,tokenizer,docs,["a.json","b.json"])
    got=context.probabilities(loaded,loaded_tokenizer,docs,["a.json","b.json"])
    np.testing.assert_allclose(got,expected,atol=1e-6)
    frame=mission3.predict(tmp_path,labels,checkpoint)
    assert list(frame.columns)==["label file name","symptom"]
    assert frame["label file name"].tolist()==["a.json","b.json"]


def test_fixed_split_aligns_shuffled_reference_arrays_and_rejects_gold(tmp_path):
    labels=tmp_path/"Training";labels.mkdir()
    for i in range(20):
        (labels/f"call{i:02d}_20261010.json").write_text(json.dumps({"utterances":[{"text":f"열 {i}"}],
                                                         "symptom":["고열"]}),encoding="utf-8")
    references=tmp_path/"reference";references.mkdir()
    names=np.array([f"call{i:02d}_20261010.json" for i in range(8)])
    gold=np.tile([1]+[0]*8,(8,1));probs=gold*.8
    for model in ("r3","k12","p0"):
        order=np.arange(8) if model=="r3" else np.arange(7,-1,-1)
        np.savez(references/f"{model}_holdout.npz",names=names[order],gold=gold[order],probs=probs[order])
    rows,train,holdout,_,ensemble=experiment.prepare_data(labels,references,smoke_rows=4)
    assert [rows[i]["name"] for i in holdout]==names.tolist()
    assert len(train)==4 and not set(train)&set(holdout)
    np.testing.assert_allclose(ensemble,probs)
    np.savez(references/"p0_holdout.npz",names=names,gold=1-gold,probs=probs)
    with pytest.raises(ValueError,match="gold differs"):
        experiment.prepare_data(labels,references,smoke_rows=4)
    validation=tmp_path/"validation";validation.mkdir();(validation/"x.json").write_text("{}")
    with pytest.raises(ValueError,match="Training"):
        experiment.read_rows(validation)


def test_selection_tie_and_bootstrap_gate():
    assert experiment.choose_pilot({"memory":.651,"evidence":.6505},.65)=="evidence"
    assert experiment.choose_pilot({"memory":.648,"evidence":.649},.65) is None
    assert experiment.choose_pilot({"memory":.651,"evidence":.6505},.64,[.6441,.6539]) is None
    assert experiment.choose_pilot({"memory":.656,"evidence":.6555},.64,[.6441,.6539])=="evidence"
    gold=np.tile([[1]*9,[0]*9],(100,1))
    perfect=gold*.8+.1
    reference=perfect.copy();reference[:20]=1-reference[:20]
    controls=[reference]*3
    good=experiment.adoption_gate(gold,[perfect]*3,reference,controls,controls,bootstrap=100)
    assert good["passed"] and good["paired_bootstrap_ci95"][0]>0
    smoke=experiment.adoption_gate(gold,[perfect]*3,reference,controls,controls,smoke=True,bootstrap=100)
    assert not smoke["passed"] and not smoke["checks"]["full_dataset"]
    tied=experiment.adoption_gate(gold,controls,reference,controls,controls,bootstrap=100)
    assert not tied["passed"]


def test_head_feature_cache_resume_and_source_guard(tmp_path,tiny):
    model,tokenizer,_=tiny
    rows=[dict(name=f"{i}.json",call=str(i),text_hash=str(i),gold=[i%2]*9,
               doc={"utterances":[{"text":"머리"},{"text":"배"}]}) for i in range(8)]
    features=experiment.extract_features(model,tokenizer,rows,list(range(8)),tmp_path/"features","test",4)
    recipe=dict(experiment.RECIPE,head_epochs=2,head_batch=4)
    directory=tmp_path/"head"
    first=experiment.fit_head(model,features,rows,list(range(6)),[6,7],42,directory,"test",recipe)
    second=experiment.fit_head(model,features,rows,list(range(6)),[6,7],42,directory,"test",recipe)
    np.testing.assert_array_equal(first,second)
    assert len(json.loads((directory/"history.json").read_text()))==2
    with pytest.raises(ValueError,match="changed"):
        experiment.extract_features(model,tokenizer,rows,list(range(8)),tmp_path/"features","different",4)
    with pytest.raises(ValueError,match="mismatch"):
        experiment.fit_head(model,features,rows,list(range(6)),[6,7],42,directory,"different",recipe)
    fine=experiment.fit_encoder(model,tokenizer,rows,list(range(6)),[6,7],42,directory,"test",recipe)
    assert fine.shape==(2,9) and np.isfinite(fine).all()
    again=experiment.fit_encoder(model,tokenizer,rows,list(range(6)),[6,7],42,directory,"test",recipe)
    np.testing.assert_array_equal(fine,again)


def test_official_reserved_once_and_offline_package(tmp_path,tiny):
    model,tokenizer,payload=tiny
    candidate=context.export_payload(model,payload,dict(smoke=False,training_rows=29200,representative_seed=42,
                           official_validation_used=False,internal_gate=dict(passed=True,checks={"full_dataset":True})))
    checkpoint=tmp_path/"candidate.pt";torch.save(candidate,checkpoint)
    labels=tmp_path/"Validation";labels.mkdir()
    for i in range(2):
        (labels/f"{i}.json").write_text(json.dumps({"utterances":[{"text":"열"}],
                                                 "symptom":list(context.LABELS)}),encoding="utf-8")
    output,registry=tmp_path/"official",tmp_path/"official_once.json"
    result=release.evaluate_official(checkpoint,labels,output,registry,"cpu",batch_size=1,expected_rows=2)
    assert result["adopt"] and result["metrics"]["macro_f1"]==1.
    assert release.evaluate_official(checkpoint,labels,tmp_path/"another",registry,"cpu",expected_rows=2)==result
    candidate["source"]["representative_seed"]=43;torch.save(candidate,tmp_path/"wrong.pt")
    with pytest.raises(ValueError,match="seed42"):
        release.evaluate_official(tmp_path/"wrong.pt",labels,output,registry,"cpu",expected_rows=2)
    candidate["source"]["representative_seed"]=42;candidate["source"]["different"]=True
    torch.save(candidate,tmp_path/"different.pt")
    with pytest.raises(ValueError,match="already reserved"):
        release.evaluate_official(tmp_path/"different.pt",labels,output,registry,"cpu",expected_rows=2)
    package=release.package_candidate(checkpoint,output/"official_metrics.json",tmp_path/"package",ROOT)
    run=subprocess.run([sys.executable,str(package/"inference.py"),"--audio_dir",str(tmp_path),"--label_dir",str(labels),
                        "--ckpt_path",str(package/"mission3.pt"),"--output",str(tmp_path/"out.csv")],
                       capture_output=True,text=True,encoding="utf-8",timeout=240)
    assert run.returncode==0,run.stderr
    assert pd.read_csv(tmp_path/"out.csv")["label file name"].tolist()==["0.json","1.json"]
    manifest=json.loads((package/"freeze.json").read_text())
    assert all(release.file_hash(package/name)==digest for name,digest in manifest["files"].items())
    with pytest.raises(ValueError,match="protected"):
        release.package_candidate(checkpoint,output/"official_metrics.json",package,ROOT)
