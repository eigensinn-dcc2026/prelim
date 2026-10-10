"""Offline M3 context inference; same command and CSV schema as the accepted model."""
import argparse
import json
from pathlib import Path

import pandas as pd
import mission3_bert as baseline
import mission3_context as context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir",type=Path,required=True)
    parser.add_argument("--label_dir",type=Path,required=True)
    parser.add_argument("--ckpt_path",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("outputs/mission3.csv"))
    args=parser.parse_args()
    paths=sorted(args.label_dir.rglob("*.json"))
    if not paths or len({p.name for p in paths})!=len(paths):
        raise ValueError("Expected nonempty, uniquely named label JSON files")
    docs=[json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
    model,tokenizer,_,length=baseline.load_checkpoint(args.ckpt_path)
    if not getattr(model,"requires_documents",False):
        raise ValueError("This package requires a context checkpoint")
    probs=context.probabilities(model,tokenizer,docs,[p.name for p in paths],length,32)
    frame=pd.DataFrame({"label file name":[p.name for p in paths],
                        "symptom":[str([label for label,positive in zip(context.LABELS,row) if positive]) for row in probs>=.5]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False,encoding="utf-8")
    print(f"Mission 3 CSV saved: {args.output} (rows={len(frame)})")


if __name__=="__main__":
    main()
