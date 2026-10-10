"""Finish the already-running local evaluations without a network connection."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
HEAD_HASH = "4fffb8aa97c2f3083b05820ca45b84ecddbc60c5162e02961546f37d133dd9d5"
M3_HASH = "bbc0bd7a5f320df78bc20e59fc98827c6d9bb9367d23ac45367d7c2ff9a56e7e"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--m3-started-at", required=True, help="Process start time; wall duration includes startup")
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    out = ROOT / "outputs/model_selection"
    status_path = out / "completion_status.json"
    state = {"m1": "running", "m2": "comparison_complete_artifacts_unavailable", "m3": "running"}
    deadline = time.monotonic() + args.timeout_hours * 3600
    previous_csv = None
    last_progress = None
    while time.monotonic() < deadline:
        m1 = out / "m1_xvec001/metrics.json"
        if state["m1"] == "running" and m1.exists():
            try:
                result = json.loads(m1.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(15)
                continue
            if result["eval_rows"] != 3640 or result["identity"]["head_sha256"] != HEAD_HASH or result["fitting_performed"]:
                raise ValueError("M1 evaluation identity/row count mismatch")
            baseline = 3462 / 3640
            accuracy = result["metrics"]["accuracy"]
            result["identity"].pop("data_root", None)
            result["comparison"] = {"baseline_accuracy": baseline, "baseline_correct": 3462,
                "delta_percentage_points": (accuracy - baseline) * 100, "minimum_gain_percentage_points": .5,
                "decision": "adopt_xvector_candidate" if accuracy - baseline >= .005 else "retain_existing_candidate"}
            write_json(ROOT / "reports/mission1/m1_xvec001_official_validation.json", result)
            state["m1"] = "complete"
            state["m1_accuracy"] = accuracy
            state["m1_decision"] = result["comparison"]["decision"]
            print("M1 official Validation finalized", flush=True)
        csv = out / "mission3.csv"
        if state["m3"] == "running" and csv.exists():
            signature = (csv.stat().st_size, csv.stat().st_mtime_ns)
            if signature == previous_csv:
                spec = importlib.util.spec_from_file_location("score_m3", ROOT / "scripts/mission3_score_csv.py")
                scoring = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scoring)
                result = scoring.score(csv, args.label_dir)
                if result["rows"] != 3640:
                    raise ValueError("M3 official Validation row count mismatch")
                packaged = ROOT / "submission/mission3/m3_roberta003_seed42.pt"
                evaluated = ROOT / "checkpoints/mission3/m3_roberta003_seed42.pt"
                hashes = {"evaluated": digest(evaluated), "packaged": digest(packaged)}
                if set(hashes.values()) != {M3_HASH}:
                    raise ValueError("M3 checkpoint copy mismatch")
                metadata = json.loads(evaluated.with_suffix(".pt.json").read_text(encoding="utf-8"))
                result.update(source=metadata["source"], checkpoint_sha256=hashes,
                    checkpoint_bytes=packaged.stat().st_size, parameter_count=336665609,
                    threshold=.5, max_length=256, batch_size=32, inference_device="cpu",
                    offline_environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                    process_wall_seconds=csv.stat().st_mtime - datetime.fromisoformat(args.m3_started_at).timestamp(),
                    process_wall_duration_includes_startup=True,
                    historical_official_macro_f1_rounded=.6433,
                    matches_historical_rounding=round(result["macro_f1"], 4) == .6433)
                write_json(ROOT / "reports/mission3/official_val/roberta003_submission_check.json", result)
                write_json(ROOT / "reports/mission3/official_val/m3_roberta003_seed42.pt.json", metadata)
                state["m3"] = "complete"
                state["m3_macro_f1"] = result["macro_f1"]
                state["m3_reproduction_matches"] = result["matches_historical_rounding"]
                print("M3 offline submission check finalized", flush=True)
            previous_csv = signature
        chunks = len(list((out / "m1_xvec001").glob("features_*.npy")))
        state.update(checked_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                     m1_feature_chunks_complete=chunks, m1_feature_chunks_total=91)
        write_json(status_path, state)
        if (chunks, state["m1"], state["m3"]) != last_progress:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            last_progress = (chunks, state["m1"], state["m3"])
        if state["m1"] == state["m3"] == "complete":
            return
        time.sleep(15)
    raise TimeoutError("Evaluation finalization exceeded the local wait budget; existing computations were not stopped")


if __name__ == "__main__":
    main()
