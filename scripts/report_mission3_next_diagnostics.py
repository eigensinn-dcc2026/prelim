"""Publish aggregate coverage diagnostics, excluding private transcripts."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.audit/"coverage.json").read_text(encoding="utf-8"))
    report.pop("training_weak_state_counts",None)
    report["weak_state_counts_omitted_reason"]="Initial rules were revised after Training-only audit; regenerate counts in the full run."
    report["provenance"]=json.loads((args.audit/"source.json").read_text(encoding="utf-8"))
    (args.out/"coverage_metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    n=report["rows"];truncated=report["truncated_calls"]
    axes[0].bar(["Fits front256","Truncated"],[n-truncated,truncated],color=["#547c9e","#d79b45"])
    axes[0].set(title="Training token coverage",ylabel="Calls")
    for i,value in enumerate([n-truncated,truncated]):axes[0].text(i,value+150,f"{value:,} ({value/n:.1%})",ha="center")
    labels=["Nausea","Headache"]
    counts=[report["false_negative_lexical_location"][label] for label in ["오심","두통"]]
    bottom=[0,0]
    for key,title,color in [("no_lexical_cue","No legacy regex cue","#547c9e"),("cue_in_front","Cue in front","#90b6a9"),("cue_only_after_front","Cue only later","#d79b45")]:
        values=[item[key] for item in counts]
        axes[1].bar(labels,values,bottom=bottom,label=title,color=color)
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].set(title="RoBERTa internal false negatives",ylabel="Label-call pairs")
    axes[1].legend(fontsize=8)
    figure.suptitle("Lexical diagnostics do not establish causal evidence")
    figure.tight_layout();figure.savefig(args.out/"coverage_diagnostics.png",dpi=180)
    plt.close(figure)
    print(args.out/"coverage_metrics.json")

if __name__=="__main__":main()
