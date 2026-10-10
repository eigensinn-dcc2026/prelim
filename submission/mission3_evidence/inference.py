"""Offline M3 inference, preserving the existing CLI and CSV schema."""
import argparse
import json
from pathlib import Path
import pandas as pd
import mission3_bert as baseline
import mission3_evidence as evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("audio_dir","label_dir","ckpt_path"):
        parser.add_argument("--"+option,type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("outputs/mission3.csv"))
    args=parser.parse_args()
    paths=sorted(args.label_dir.rglob("*.json"))
    if not paths or len({p.name for p in paths})!=len(paths):raise ValueError("Expected uniquely named JSON files")
    docs=[json.loads(path.read_text(encoding="utf-8-sig")) for path in paths]
    model,tokenizer,labels,length=baseline.load_checkpoint(args.ckpt_path)
    if getattr(model,"requires_documents",False):
        probs=evidence.probabilities(model,tokenizer,docs,max_length=length,batch_size=2)
    else:probs=baseline.probabilities(model,tokenizer,[baseline.transcript(doc) for doc in docs],length,8)
    frame=pd.DataFrame({"label file name":[path.name for path in paths],"symptom":[str([label for label,positive in zip(labels,row) if positive]) for row in probs>=.5]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False,encoding="utf-8")
    print(f"Mission 3 CSV saved: {args.output} (rows={len(frame)})")


if __name__=="__main__":main()
