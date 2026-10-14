"""Summarize actual pilot artifacts without mistaking missing runs for results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcc_data import mission3_context_experiment as exp


def paired(run, candidate, reference):
    with np.load(run/candidate/"predictions.npz",allow_pickle=False) as a, np.load(run/reference/"predictions.npz",allow_pickle=False) as b:
        if not np.array_equal(a["names"],b["names"]) or not np.array_equal(a["gold"],b["gold"]):
            raise ValueError("Cannot compare unaligned predictions")
        gold,p,q=a["gold"].copy(),a["probs"].copy(),b["probs"].copy()
    delta=exp.metrics(gold,p)["macro_f1"]-exp.metrics(gold,q)["macro_f1"]
    interval=exp.bootstrap_difference(gold,[p],q,count=1000,seed=42)
    return dict(delta_macro_f1=delta,paired_bootstrap=interval,
                caveat="Fixed pilot and evaluation sample only; excludes seed/selection uncertainty")


def summarize(run):
    run=Path(run)
    result=dict(status="prepared_no_gpu_results",qualified_for_submission=False,
                official_validation_used=False,models={},comparisons={})
    for path in sorted(run.glob("*/result.json")):
        if path.parent.name=="teachers": continue
        result["models"][path.parent.name]=json.loads(path.read_text())
    if (run/"reference/predictions.npz").exists():
        for name in result["models"]:
            result["comparisons"][name+"_vs_existing_ensemble"]=paired(run,name,"reference")
    for seed in (42,43,44):
        for candidate,reference in [(f"loss_b{batch}_global_s{seed}",f"loss_b{batch}_micro_s{seed}") for batch in (16,64)]+[
            (f"student_kd_s{seed}",f"student_hard_s{seed}"),
            (f"evidence_human_s{seed}",f"evidence_control_s{seed}"),
            (f"cascade_512_s{seed}",f"cascade_256_s{seed}")]:
            if candidate in result["models"] and reference in result["models"]:
                result["comparisons"][candidate+"_vs_"+reference]=paired(run,candidate,reference)
    if result["models"]: result["status"]="pilot_results_require_seed_and_latency_validation"
    result["cascade"]={path.stem:json.loads(path.read_text()) for path in run.glob("cascade_crossfit_s*.json")}
    if (run/"teachers/result.json").exists(): result["teacher"]=json.loads((run/"teachers/result.json").read_text())
    if "teacher" in result:
        for name,metrics in result["models"].items():
            if name.startswith("student_kd"):
                delta=metrics["macro_f1"]-result["teacher"]["macro_f1"]
                result["comparisons"][name+"_vs_teacher"]=dict(delta_macro_f1=delta,
                    accuracy_target_met=delta>=-.003,latency_target_status="not_measured")
    run.mkdir(parents=True,exist_ok=True)
    exp.write_json(run/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    summarize(parser.parse_args().run)
