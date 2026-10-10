"""Freeze a separately named offline package only after official acceptance."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dcc_data.mission3_context_release import package_candidate


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt",required=True,type=Path)
    parser.add_argument("--official-result",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    print(package_candidate(args.ckpt,args.official_result,args.out,ROOT))
