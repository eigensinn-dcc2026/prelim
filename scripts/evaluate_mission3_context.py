"""Use official Validation once for the internally accepted final candidate."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dcc_data.mission3_context_release import evaluate_official


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt",required=True,type=Path)
    parser.add_argument("--official-label-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--registry",type=Path,default=ROOT/"reports/mission3/context/official_once.json")
    parser.add_argument("--device",choices=("cpu","cuda"))
    parser.add_argument("--batch",type=int,default=32)
    args=parser.parse_args()
    if args.batch<1:
        parser.error("batch must be positive")
    result=evaluate_official(args.ckpt,args.official_label_dir,args.out,args.registry,args.device,args.batch)
    print(json.dumps(dict(macro_f1=result["metrics"]["macro_f1"],decision=result["decision"])))


if __name__=="__main__":
    main()
