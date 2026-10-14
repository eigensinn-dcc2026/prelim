"""Validate acquired M3 final deliverables and stage/activate a standalone folder.

Requires a completed export directory (not a live Drive folder), matching seed
sidecars, executed notebook, and full Validation probabilities/score. Does not
train, change ensemble weights, invent notebook outputs, or update the team ZIP.
Use --activate only after the reported candidate is accepted for submission.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "wt-m3-kit")
    parser.add_argument("--expected-bert-weight", type=float, required=True,
                        help="Approved fixed weight, not selected from Validation results")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    source = args.artifacts.resolve()
    required = ["m3_ensemble_full.pt", "m3_ensemble_full.pt.json", "m3_final_train.executed.ipynb",
                "metrics.json", "config.json", "ensemble/val_score.json", "ensemble/val_probs.npy"]
    required += [f"m3_roberta003_full_seed{s}.pt.json" for s in (42, 43, 44)]
    missing = [name for name in required if not (source / name).is_file()]
    require(not missing, f"Incomplete export; missing {missing}")
    config, metrics = read(source / "config.json"), read(source / "metrics.json")
    require(config.get("rows") == 29200 and config.get("epochs") == 3 and not config.get("quick"),
            "Not the full 29,200-row, 3-epoch Training run")
    require(metrics.get("exp_id") in ("M3-FINAL-001", "M3-FINAL-002"), "Unexpected final run identity")
    notebook = read(source / "m3_final_train.executed.ipynb")
    codes = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    require(codes and all(c.get("execution_count") is not None for c in codes), "Notebook not fully executed")
    require(any(c.get("outputs") for c in codes), "Notebook lacks execution outputs")
    require(not any(o.get("output_type") == "error" for c in codes for o in c.get("outputs", [])),
            "Notebook contains execution errors")
    log_text = json.dumps([c.get("outputs", []) for c in codes], ensure_ascii=False)
    require("29200" in log_text and all(f"seed {s}" in log_text or f"seed={s}" in log_text for s in (42,43,44)),
            "Notebook outputs do not identify full data and all three seeds")
    ckpt = source / "m3_ensemble_full.pt"
    record = read(source / "m3_ensemble_full.pt.json")
    require(record.get("output_bytes") == ckpt.stat().st_size and record.get("output_sha256") == sha(ckpt),
            "Ensemble export SHA256/size mismatch")
    bert = [m for m in record["members"] if m["kind"] == "bert"]
    tfidf = [m for m in record["members"] if m["kind"] == "tfidf"]
    require(len(bert) == 3 and len(tfidf) == 1, "Expected three seeds and one TF-IDF member")
    require(abs(sum(m["weight"] for m in bert)-args.expected_bert_weight) < 1e-6 and
            abs(tfidf[0]["weight"]-(1-args.expected_bert_weight)) < 1e-6, "Approved ensemble weights differ")
    for seed, member in zip((42,43,44), bert):
        name = f"m3_roberta003_full_seed{seed}.pt"
        sidecar = read(source / (name + ".json"))
        require(member["file"] == name and member["sha256"] == sidecar["output_sha256"], "Seed provenance differs")
        require(sidecar.get("rows") == 29200 and sidecar.get("epochs") == 3 and len(sidecar.get("history", [])) == 3,
                "Seed lacks full-Training epoch history")
    import torch
    payload = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    require(payload.get("format") == "dcc-m3-ens-v1" and payload.get("labels") == LABELS
            and payload.get("threshold") == 0.5, "Checkpoint format/labels/threshold mismatch")
    require(len(payload.get("members", [])) == len(record["members"]), "Embedded member count differs")
    for embedded, provenance in zip(payload["members"], record["members"]):
        require(embedded["kind"] == provenance["kind"] and
                abs(embedded["weight"]-provenance["weight"]) < 1e-9 and
                embedded["source"]["file"] == provenance["file"] and
                embedded["source"]["sha256"] == provenance["sha256"], "Embedded member differs from sidecar")
        if embedded["kind"] == "bert":
            require(embedded["checkpoint"]["labels"] == LABELS, "Embedded seed label order differs")
    del payload
    import numpy as np
    from sklearn.metrics import f1_score
    probs = np.load(source / "ensemble/val_probs.npy", allow_pickle=False)
    require(probs.shape == (3640,9) and np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all(),
            "Invalid full Validation probability array")
    data = ROOT.parent / "dcc2026-prelim/data/raw_merged/Validation/2.라벨링데이터/VL_서울_구급"
    paths = sorted(data.rglob("*.json"))
    require(len(paths) == 3640, "Full Validation labels unavailable")
    import unicodedata
    gold = []
    for p in paths:
        symptoms = json.loads(p.read_text(encoding="utf-8-sig")).get("symptom", [])
        symptoms = [symptoms] if isinstance(symptoms, str) else symptoms
        symptoms = {unicodedata.normalize("NFC", s.strip()) for s in symptoms if isinstance(s,str)}
        gold.append([int(label in symptoms) for label in LABELS])
    score = read(source / "ensemble/val_score.json")
    macro = float(f1_score(gold, probs>=0.5, average="macro", zero_division=0))
    require(score.get("rows") == 3640 and abs(macro-score["macro_f1"]) < 1e-8, "Validation ordering/score mismatch")
    target = ROOT / "outputs/m3_final_candidate/mission3"
    require(not (target / "m3_ensemble_full.pt").exists(), "Existing staged weights: choose a fresh stage before rerunning")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("inference.py", "mission3_model.py", "requirements.txt"):
        shutil.copy2(args.kit / "submission/mission3" / name, target / name)
    shutil.copy2(ckpt, target / ckpt.name)
    shutil.copy2(source / "m3_final_train.executed.ipynb", target / "training.ipynb")
    shutil.copy2(source / "m3_ensemble_full.pt.json", target / "m3_ensemble_full.pt.json")
    shutil.copy2(source / "config.json", target / "training_config.json")
    # Preserve the scripts called by the notebook and their original repository layout.
    training_source = target / "training_source"
    for directory in ("scripts/m3_final", "src"):
        shutil.copytree(args.kit / directory, training_source / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.executed.ipynb"), dirs_exist_ok=True)
    launcher = '''import os, socket, runpy, sys
def deny(*a,**k):
    open(os.environ['NETWORK_AUDIT'],'a').write('attempt\\n')
    raise RuntimeError('Network forbidden')
socket.socket.connect=deny
socket.socket.connect_ex=deny
socket.create_connection=deny
socket.getaddrinfo=deny
sys.argv=['inference.py']+sys.argv[1:]
runpy.run_path('inference.py',run_name='__main__')
'''
    with tempfile.TemporaryDirectory(prefix="dcc-final-check-") as temp:
        work = Path(temp)
        label_dir = work / "labels"
        label_dir.mkdir()
        for p in paths[:2]:
            doc = json.loads(p.read_text(encoding="utf-8-sig"))
            sanitized = {"utterances":[{"text":u.get("text","")} for u in doc["utterances"]]}
            (label_dir / p.name).write_text(json.dumps(sanitized,ensure_ascii=False),encoding="utf-8")
        launcher_file = work / "launcher.py"
        launcher_file.write_text(launcher,encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH",None)
        env.update(HF_HOME=str(work/"empty_cache"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                   USE_TF="0",USE_FLAX="0",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="",NETWORK_AUDIT=str(work/"network.txt"))
        began = time.perf_counter()
        proc = subprocess.run([sys.executable,str(launcher_file),"--audio_dir",str(work/"absent"),
            "--label_dir",str(label_dir),"--ckpt_path",str(target/ckpt.name),"--output",str(work/"out.csv")],
            cwd=target,env=env,capture_output=True,text=True,timeout=900)
        require(proc.returncode == 0, "Isolated inference failed: "+proc.stderr[-1200:])
        require(not (work/"network.txt").exists(),"Network access attempted")
        with (work/"out.csv").open(encoding="utf-8",newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ["label file name","symptom"],"CSV header mismatch")
            rows = list(reader)
        expected = [str([l for l,f in zip(LABELS,row) if f]) for row in (probs[:2]>=0.5)]
        require([r["label file name"] for r in rows] == [p.name for p in paths[:2]],"CSV rows differ")
        require([r["symptom"] for r in rows] == expected,"CPU smoke predictions differ from saved GPU probabilities")
        for r in rows:
            require(isinstance(ast.literal_eval(r["symptom"]),list),"Invalid symptom cell")
        seconds = round(time.perf_counter()-began,2)
    report = dict(status="verified_candidate",full_training_rows=29200,seeds=[42,43,44],threshold=0.5,
                  bert_weight=args.expected_bert_weight,tfidf_weight=1-args.expected_bert_weight,
                  checkpoint_sha256=record["output_sha256"],validation_macro_f1=macro,validation_rows=3640,
                  notebook_sha256=sha(target/"training.ipynb"),offline_smoke_rows=2,network_attempts=0,
                  smoke_seconds=seconds,activated=False,team_zip_updated=False)
    (target/"final_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (target/"README.md").write_text(
        f"# Mission 3 final candidate\n\nFull Training 29,200 calls; seeds 42/43/44 + TF-IDF. "
        f"Fixed weights {args.expected_bert_weight}:{1-args.expected_bert_weight}; threshold >= 0.5, no post-processing. "
        f"Official Validation Macro F1 {macro:.8f}.\n\n"
        "python inference.py --audio_dir WAV_FOLDER --label_dir JSON_FOLDER --ckpt_path m3_ensemble_full.pt --output outputs/mission3.csv\n\n"
        "Executed training notebook: training.ipynb. Original scripts/layout: training_source/. "
        "See final_validation.json for hashes and offline smoke evidence.\n",encoding="utf-8")
    if args.activate:
        current = ROOT/"submission/mission3"
        backup = ROOT/"outputs/m3_submission_backups"/time.strftime("%Y%m%d-%H%M%S")
        backup.parent.mkdir(parents=True,exist_ok=True)
        current.rename(backup)
        try:
            target.rename(current)
        except Exception:
            backup.rename(current)
            raise
        report.update(activated=True,backup=str(backup))
        (current/"final_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
