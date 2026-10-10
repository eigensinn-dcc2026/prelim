"""Run the approved M3 two-method experiment without official Validation access."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcc_data.mission3_context_experiment import run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-label-dir",required=True,type=Path)
    parser.add_argument("--base-ckpt",required=True,type=Path)
    parser.add_argument("--reference-probs-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--phase",choices=("pilot","repeat","all"),default="all")
    parser.add_argument("--device",choices=("cpu","cuda"))
    parser.add_argument("--feature-batch",type=int,default=32)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--smoke-rows",type=int,default=0,help="Bounded engineering check; never eligible for adoption")
    parser.add_argument("--max-hours",type=float,default=10,help="Cumulative resumable wall-clock budget, at most 10 hours")
    args=parser.parse_args()
    if args.feature_batch<1 or args.cpu_threads<1 or args.smoke_rows<0:
        parser.error("batch/threads must be positive; smoke-rows must be nonnegative")
    if not 0 < args.max_hours <= 10:
        parser.error("max-hours must be between zero and ten")
    summary=run_experiment(args)
    print(json.dumps({key:summary[key] for key in ("status","selected_method","elapsed_seconds")},ensure_ascii=False))


if __name__=="__main__":
    main()
