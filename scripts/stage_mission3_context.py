"""Stage a private GPU input bundle containing Training only and local source.

No audio, patient metadata, speaker numbers, official Validation, or credentials
are included. Transcript text and its nine Training labels remain private data.
"""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dcc_data.mission3_context_experiment import prepare_data,file_hash,digest
from dcc_inference.mission3_context import LABELS


def stage(training,checkpoint,references,output):
    rows,train,holdout,_,_=prepare_data(training,references)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+".tmp")
    def add_text(archive,name,text):
        data=text.encode("utf-8")
        member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o600
        archive.addfile(member,io.BytesIO(data))
    with tarfile.open(temporary,"w:gz",compresslevel=1) as archive:
        for path in sorted((ROOT/"src").rglob("*.py")):
            archive.add(path,arcname=path.relative_to(ROOT).as_posix())
        for name in ("train_mission3_context.py","evaluate_mission3_context.py","package_mission3_context.py"):
            archive.add(ROOT/"scripts"/name,arcname="scripts/"+name)
        for path in sorted((ROOT/"submission/mission3_context").iterdir()):
            if path.is_file():
                archive.add(path,arcname=path.relative_to(ROOT).as_posix())
        for i,row in enumerate(rows):
            doc=dict(row["doc"],symptom=[label for label,positive in zip(LABELS,row["gold"]) if positive])
            add_text(archive,"training/"+row["name"],json.dumps(doc,ensure_ascii=False))
            if i%5000==0:
                print(f"staging text-only Training {i}/{len(rows)}",flush=True)
        for name in ("r3","k12","p0"):
            archive.add(Path(references)/f"{name}_holdout.npz",arcname=f"reference/{name}_holdout.npz")
        archive.add(checkpoint,arcname="baseline.pt")
        add_text(archive,"bundle.json",json.dumps(dict(training_rows=len(rows),train_rows=len(train),
                  internal_holdout_rows=len(holdout),official_validation_included=False,
                  checkpoint_sha256=file_hash(checkpoint),
                  rows_sha256=digest([{k:row[k] for k in ("name","call","text_hash","gold")} for row in rows])),indent=2))
    temporary.replace(output)
    print(json.dumps(dict(bundle=str(output.resolve()),bytes=output.stat().st_size,training_rows=len(rows))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-label-dir",required=True,type=Path)
    parser.add_argument("--base-ckpt",required=True,type=Path)
    parser.add_argument("--reference-probs-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    stage(args.train_label_dir,args.base_ckpt,args.reference_probs_dir,args.out)
