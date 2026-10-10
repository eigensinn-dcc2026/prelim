"""Run M3 controlled loss and complete-window evidence pilots on Training only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcc_data.mission3_next_experiment import run

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("train-label-dir","base-ckpt","reference-probs-dir","out"):
        parser.add_argument("--"+option,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--phase",choices=("audit","loss","pilot","all"),default="all")
    parser.add_argument("--smoke-rows",type=int,default=0)
    parser.add_argument("--max-hours",type=float,default=6.)
    args=parser.parse_args()
    if args.smoke_rows<0 or not 0<args.max_hours<=10:parser.error("Invalid smoke rows/budget")
    result=run(args)
    print(json.dumps({key:result[key] for key in ("status","selected_method","elapsed_seconds")},ensure_ascii=False))
