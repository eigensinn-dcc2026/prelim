import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from dcc_data import mission3_approved_core as core
from dcc_inference import mission3_bert as base

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import train_mission3_approved as runner


def test_global_gradient_matches_full_batch_and_differs_from_micro():
    torch.manual_seed(4)
    x, y = torch.randn(13,5), (torch.rand(13,9) > .8).float()
    net = torch.nn.Linear(5,9)
    weights = torch.linspace(1,3,9)
    logits = net(x)
    p = logits.sigmoid()
    direct = F.binary_cross_entropy_with_logits(logits,y,pos_weight=weights) + 1 - (2*(p*y).sum(0)/(p.sum(0)+y.sum(0)+1e-8)).mean()
    expected = torch.autograd.grad(direct, net.weight)[0]
    split = torch.cat([net(x[i:i+4]) for i in range(0,13,4)])
    actual = torch.autograd.grad(core.grouped_objective(split,y,weights,"global",4),net.weight)[0]
    torch.testing.assert_close(actual,expected)
    split = torch.cat([net(x[i:i+4]) for i in range(0,13,4)])
    micro = torch.autograd.grad(core.grouped_objective(split,y,weights,"micro",4),net.weight)[0]
    assert (micro-expected).abs().max() > 1e-4


def test_micro_partial_batch_matches_original_accumulation():
    torch.manual_seed(2)
    x, y = torch.randn(11,9,requires_grad=True), (torch.rand(11,9)>.7).float()
    weights = torch.ones(9)
    from dcc_data.mission3_context_experiment import loss
    expected = sum(loss(x[i:i+8],y[i:i+8],weights)*len(x[i:i+8])/len(x) for i in range(0,len(x),8))
    actual = core.grouped_objective(x,y,weights,"micro",8)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])


def rows(n=6):
    return [dict(name=f"call{i}.json",call=str(i),text_hash=str(i),
                 gold=[i%2]*9,doc={"utterances":[{"text":"머리가 아프다"}]}) for i in range(n)]


@pytest.mark.parametrize("key", ["name","call","text_hash"])
def test_split_rejects_identity_and_transcript_leaks(key):
    data=rows(); data[5][key]=data[0][key]
    with pytest.raises(ValueError,match="leakage"):
        core.assert_disjoint(data,[0,1,2,3],[4,5])


def test_oof_rejects_outer_holdout_and_task_finetuned_initialization():
    data=rows()
    folds=[dict(fit=[0,1],predict=[2,3],initialization="public_pretrained"),
           dict(fit=[2,3],predict=[0,1],initialization="public_pretrained")]
    core.verify_oof(data,[0,1,2,3],[4,5],folds,np.full((4,9),.5))
    folds[0]["initialization"]="finetuned_on_all_training"
    with pytest.raises(ValueError,match="initialize"):
        core.verify_oof(data,[0,1,2,3],[4,5],folds,np.full((4,9),.5))
    folds[0]["initialization"]="public_pretrained"; folds[0]["fit"].append(4)
    with pytest.raises(ValueError):
        core.verify_oof(data,[0,1,2,3],[4,5],folds,np.full((4,9),.5))


def test_human_annotations_cannot_be_rules_holdout_or_invalid_offsets(tmp_path):
    data=rows()
    item=dict(name="call0.json",label=base.LABELS[2],state="positive",span=[0,7],reviewer="reviewer_1",provenance="human")
    path=tmp_path/"review.jsonl"
    path.write_text(json.dumps(item),encoding="utf-8")
    assert core.read_annotations(path,data,[0,1])[(0,2)]["state"]==0
    for changes in (dict(name="call5.json"),dict(provenance="rule"),dict(span=[0,100]),dict(reviewer="")):
        path.write_text(json.dumps(dict(item,**changes)),encoding="utf-8")
        with pytest.raises(ValueError): core.read_annotations(path,data,[0,1])


def test_truncated_evidence_does_not_supervise_state():
    states=torch.randn(1,9,3,requires_grad=True)
    attention=torch.softmax(torch.randn(1,9,4,requires_grad=True),-1)
    annotations={(0,0):dict(state=0,span=[10,15])}
    loss=core.annotation_objective(states,attention,[[(0,0),(0,2),(2,4),(0,0)]],[0],annotations)
    assert loss.item()==0
    loss.backward()
    assert states.grad.abs().sum()==0


def tiny_payload():
    from transformers import BertConfig
    config=BertConfig(vocab_size=12,hidden_size=12,num_hidden_layers=1,
                      num_attention_heads=3,intermediate_size=20,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    files={"config.json":config.to_json_string(),
           "vocab.txt":"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\na\nb\nc\nd\ne\n",
           "tokenizer_config.json":json.dumps(dict(tokenizer_class="BertTokenizer",do_lower_case=True))}
    payload=dict(format=base.FORMAT,labels=list(base.LABELS),max_length=256,dropout=0.,files=files,source={})
    model=base._build(files,payload)
    payload["model"]=model.state_dict()
    return payload


def test_evidence_initially_preserves_base_masks_padding_and_roundtrips(tmp_path):
    payload=tiny_payload()
    original=runner.checkpoint_model(payload).eval()
    model=core.EvidenceClassifier(original).eval()
    ids=torch.tensor([[2,5,3,0]])
    mask=torch.tensor([[1,1,1,0]])
    logits,states,attention=model(ids,mask,True)
    torch.testing.assert_close(logits,original(ids,mask))
    assert attention[:,:,-1].abs().sum()==0
    stored=dict(payload,format="dcc-m3-human-evidence-v1",model=model.state_dict())
    torch.save(stored,tmp_path/"model.pt")
    restored=runner.checkpoint_model(torch.load(tmp_path/"model.pt",weights_only=True)).eval()
    torch.testing.assert_close(restored(ids,mask),logits)


def test_offline_inference_ignores_labels_and_metadata(tmp_path,monkeypatch):
    import inference_mission3_approved as inference
    torch.set_num_threads(1)
    payload=tiny_payload(); checkpoint=tmp_path/"model.pt";torch.save(payload,checkpoint)
    inputs=tmp_path/"input";inputs.mkdir()
    doc=dict(utterances=[dict(text="hello world",speaker=1)],symptom=["두통"],triage="private")
    (inputs/"a.json").write_text(json.dumps(doc),encoding="utf-8")
    _,p1,_=inference.infer(checkpoint,inputs)
    doc.update(symptom=[],triage="changed");doc["utterances"][0]["speaker"]=99
    (inputs/"a.json").write_text(json.dumps(doc),encoding="utf-8")
    _,p2,report=inference.infer(checkpoint,inputs)
    np.testing.assert_array_equal(p1,p2)
    assert report["threshold"]==.5


def test_cascade_review_only_long_calls_with_out_of_fold_selection():
    rng=np.random.default_rng(4)
    y=(rng.random((100,9))>.6).astype(int)
    front=rng.random((100,9));long=np.where(y,.8,.2)
    lengths=np.r_[np.full(50,100),np.full(50,350)]
    report,p=core.crossfit_cascade(y,front,long,lengths)
    np.testing.assert_array_equal(p[:50],front[:50])
    assert report["reviewed_fraction"]<=.5 and len(report["folds"])==5
    for a in report["folds"]: assert a["test_rows"]==20


def test_gpu_execution_is_disabled_by_default(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,"argv",["runner","loss","--bundle",str(tmp_path),"--out",str(tmp_path/"out")])
    with pytest.raises(SystemExit) as error: runner.main()
    assert error.value.code==2


def test_real_tiny_training_export_and_completed_resume(tmp_path):
    torch.set_num_threads(1)
    payload=tiny_payload()
    tokenizer=runner.tokenizer_from_files(payload["files"])
    data=rows(8)
    for row in data: row["doc"]={"utterances":[{"text":"hello world"}]}
    class Budget:
        def check(self): pass
    kwargs=dict(device=torch.device("cpu"),epochs=1,effective=3,micro=2,length=16,aggregation="global")
    output=tmp_path/"fit"
    p=runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                 output,"signature",Budget(),**kwargs)
    restored=runner.checkpoint_model(torch.load(output/"model.pt",weights_only=True)).eval()
    q=runner.predict(restored,tokenizer,data,[5,6,7],16,torch.device("cpu"))
    np.testing.assert_allclose(p,q,rtol=1e-6)
    again=runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                     output,"signature",Budget(),**kwargs)
    np.testing.assert_array_equal(p,again)
    with pytest.raises(ValueError,match="Resume configuration changed"):
        runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                   output,"changed",Budget(),**kwargs)


def test_summary_distinguishes_prepared_from_measured(tmp_path):
    from summarize_mission3_approved import summarize
    report=summarize(tmp_path)
    assert report["status"]=="prepared_no_gpu_results"
    assert not report["qualified_for_submission"]


def test_epoch_resume_matches_uninterrupted_optimizer_state(tmp_path):
    torch.set_num_threads(1)
    payload=tiny_payload();tokenizer=runner.tokenizer_from_files(payload["files"])
    data=rows(8)
    for row in data: row["doc"]={"utterances":[{"text":"hello world"}]}
    class Budget:
        def check(self): pass
    class Interrupted:
        def __init__(self): self.count=0
        def check(self):
            self.count+=1
            if self.count==3: raise RuntimeError("simulated interruption after epoch one")
    kwargs=dict(device=torch.device("cpu"),epochs=2,effective=3,micro=2,length=16,aggregation="global")
    with pytest.raises(RuntimeError,match="simulated interruption"):
        runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                   tmp_path/"resume","signature",Interrupted(),**kwargs)
    resumed=runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                       tmp_path/"resume","signature",Budget(),**kwargs)
    direct=runner.fit(runner.checkpoint_model(payload),tokenizer,payload,data,[0,1,2,3,4],[5,6,7],
                      tmp_path/"direct","signature",Budget(),**kwargs)
    np.testing.assert_allclose(resumed,direct,rtol=1e-6,atol=1e-7)


def test_teacher_offline_members_and_checksum_guard(tmp_path):
    import inference_mission3_approved as inference
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(1)
    directory=tmp_path/"teachers";fold=directory/"fold_0";fold.mkdir(parents=True)
    payload=tiny_payload();torch.save(payload,fold/"model.pt")
    vectorizer=TfidfVectorizer();x=vectorizer.fit_transform(["hello world","hello hello","world world","world hello"])
    classifier=OneVsRestClassifier(LogisticRegression()).fit(x,np.asarray([[0]*9,[1]*9,[0]*9,[1]*9]))
    joblib.dump(dict(vectorizer=vectorizer,classifier=classifier),fold/"linear.joblib")
    from dcc_data.mission3_context_experiment import file_hash
    provenance=dict(folds=[dict(neural_weight=.6,checkpoint_sha256=file_hash(fold/"model.pt"),linear_sha256=file_hash(fold/"linear.joblib"))])
    (directory/"provenance.json").write_text(json.dumps(provenance))
    inputs=tmp_path/"input";inputs.mkdir()
    (inputs/"a.json").write_text(json.dumps(dict(utterances=[dict(text="hello world")],symptom=["ignored"])))
    _,p,timing=inference.infer_teacher(directory,inputs)
    assert p.shape==(1,9) and np.isfinite(p).all() and timing["members"]==1
    provenance["folds"][0]["linear_sha256"]="wrong"
    (directory/"provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match="checksum"):
        inference.infer_teacher(directory,inputs)


def test_cascade_does_not_select_margin_using_heldout_fold_labels():
    from sklearn.model_selection import KFold
    rng=np.random.default_rng(4)
    y=(rng.random((80,9))>.5).astype(int)
    front=rng.random((80,9));long=rng.random((80,9));lengths=np.full(80,350)
    report,p=core.crossfit_cascade(y,front,long,lengths)
    _,test=next(KFold(5,shuffle=True,random_state=42).split(y))
    changed=y.copy();changed[test]=1-changed[test]
    other,q=core.crossfit_cascade(changed,front,long,lengths)
    assert report["folds"][0]["margin"]==other["folds"][0]["margin"]
    np.testing.assert_array_equal(p[test],q[test])
