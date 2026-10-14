"""Audit the acquired submission; never train, repack, or mark missing logs complete.

Run from any directory: python scripts/preflight_submission.py
Reports contain hashes and aggregate checks, never call text or per-call predictions.
Exit 0: ready, 2: required artifacts/validation still pending, 1: check failed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    output = ROOT / "reports/submission/preflight.json"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "acquired package; M3 real offline smoke and CLI boundary checks",
              "checks": {}, "pending": [], "failures": []}

    def check(name, function):
        print(f"Checking {name}", flush=True)
        try:
            report["checks"][name] = {"status": "passed", **function()}
        except Exception as error:
            report["checks"][name] = {"status": "failed", "error": str(error)}
            report["failures"].append(name)

    manifest = json.loads((ROOT / "reports/submission/submission_manifest.json").read_text(encoding="utf-8"))

    def inventory():
        for entry in manifest["files"]:
            path = ROOT / "submission" / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Manifest mismatch: {entry['path']}")
        return {"files": len(manifest["files"]), "sha256_verified": True}

    check("package_file_hashes", inventory)

    def archive_check():
        path = ROOT / "outputs/deliverables" / manifest["archive_name"]
        if digest(path) != manifest["archive_sha256"]:
            raise ValueError("Archive SHA256 mismatch")
        expected = {entry["path"]: entry for entry in manifest["files"]}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError("Archive file inventory differs")
            for name in names:
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError("Unsafe archive path")
                with archive.open(name) as stream:
                    if digest_stream(stream) != expected[name]["sha256"]:
                        raise ValueError(f"Archive member SHA256 mismatch: {name}")
        return {"files": len(expected), "sha256": manifest["archive_sha256"],
                "member_hashes_and_crc_verified": True}

    check("archive_integrity", archive_check)

    def required():
        found = {}
        for mission in (1, 2, 3):
            folder = ROOT / "submission" / f"mission{mission}"
            needed = ["inference.py", f"mission{mission}_model.py", "requirements.txt"]
            absent = [name for name in needed if not (folder / name).is_file()]
            if absent:
                raise ValueError(f"Mission {mission} missing {absent}")
            checkpoints = [p.name for p in folder.iterdir() if p.suffix in (".pt", ".pth", ".ckpt")]
            notebooks = list(folder.glob("*.ipynb"))
            executed = []
            for path in notebooks:
                nb = json.loads(path.read_text(encoding="utf-8"))
                codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
                if any(c.get("execution_count") is not None and c.get("outputs") for c in codes):
                    executed.append(path.name)
            found[f"mission{mission}"] = {"checkpoints": checkpoints,
                "notebooks": [p.name for p in notebooks], "notebooks_with_execution_outputs": executed}
            if not checkpoints:
                report["pending"].append(f"M{mission}: selected final checkpoint missing")
            # Cell outputs alone cannot establish correspondence to the selected checkpoint.
            if not executed:
                report["pending"].append(f"M{mission}: selected training notebook with execution outputs unavailable")
        if not list((ROOT / "submission").glob("*.pptx")):
            report["pending"].append("Results PPTX missing")
        report["pending"].extend([
            "M3: replace with full-Training final .pt and matching executed notebook, then rerun preflight",
            "Fresh environment requirements installation not verified",
            "M1/M2 runtime checks retained from prior manifest; not repeated in this M3 preflight",
        ])
        return found

    check("required_artifact_inventory", required)

    def versions():
        result = {}
        for line in (ROOT / "submission/mission3/requirements.txt").read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.startswith("#"):
                name, expected = line.split("==")
                actual = importlib.metadata.version(name)
                result[name] = {"expected": expected, "installed": actual}
                # PEP 440 pins without a local suffix accept builds such as +cpu.
                if actual.split("+", 1)[0] != expected:
                    raise ValueError(f"{name}: installed {actual}, expected {expected}")
        return {"versions": result}

    check("m3_installed_versions", versions)

    # Fresh work directory and HF cache, no repository modules or audio inputs.
    # This launcher denies Python TCP/DNS access even if an offline flag is ignored.
    launcher = '''import os, runpy, socket, sys
def denied(*args, **kwargs):
    with open(os.environ["NETWORK_AUDIT"], "a", encoding="utf-8") as f:
        f.write("attempt\\n")
    raise RuntimeError("Network access forbidden by submission preflight")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
sys.argv = ["inference.py"] + sys.argv[1:]
runpy.run_path("inference.py", run_name="__main__")
'''

    def m3_runtime():
        folder = ROOT / "submission/mission3"
        ckpt = folder / "m3_roberta003_seed42.pt"
        sidecar = ROOT / "checkpoints/mission3/m3_roberta003_seed42.pt.json"
        expected_hash = json.loads(sidecar.read_text(encoding="utf-8"))["output_sha256"]
        if digest(ckpt) != expected_hash:
            raise ValueError("Checkpoint differs from export provenance")
        with tempfile.TemporaryDirectory(prefix="dcc-m3-preflight-") as temp:
            stage = Path(temp)
            for name in ("inference.py", "mission3_model.py"):
                shutil.copy2(folder / name, stage / name)
            (stage / "launcher.py").write_text(launcher, encoding="utf-8")
            labels = stage / "labels"
            labels.mkdir()
            samples = sorted((ROOT / "outputs/submission_check/labels3").glob("*.json"))
            if len(samples) != 2:
                raise ValueError("Expected two existing allowed-text-only Validation smoke samples")
            for sample in samples:
                doc = json.loads(sample.read_text(encoding="utf-8"))
                safe = {"utterances": [{"text": u.get("text", "")} for u in doc["utterances"]]}
                (labels / sample.name).write_text(json.dumps(safe, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update(HF_HOME=str(stage / "empty_hf_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", NETWORK_AUDIT=str(stage / "network.txt"))
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, "launcher.py", "--audio_dir", str(stage / "absent_audio"),
                "--label_dir", str(labels), "--ckpt_path", str(ckpt), "--output", "result.csv"],
                cwd=stage, env=env, capture_output=True, text=True, timeout=args.timeout)
            if proc.returncode:
                raise ValueError(f"Isolated CLI failed: {proc.stderr[-1500:]}")
            if (stage / "network.txt").exists():
                raise ValueError("CLI attempted network access")
            with (stage / "result.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["label file name", "symptom"]:
                    raise ValueError("CSV columns differ")
                rows = list(reader)
            with (ROOT / "outputs/model_selection/mission3.csv").open(encoding="utf-8", newline="") as stream:
                reference = {r["label file name"]: r for r in csv.DictReader(stream)}
            if len(rows) != 2 or [r["label file name"] for r in rows] != [p.name for p in samples]:
                raise ValueError("Row count or sorted filename order differs")
            for row in rows:
                predicted = ast.literal_eval(row["symptom"])
                if not isinstance(predicted, list) or predicted != [label for label in LABELS if label in predicted]:
                    raise ValueError("Symptom label set/order differs")
                if row != reference[row["label file name"]]:
                    raise ValueError("Smoke prediction differs from previously verified full Validation CSV")
            return {"rows": len(rows), "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "checkpoint_sha256": expected_hash, "network_attempts": 0, "empty_hf_cache": True,
                    "no_audio_or_disallowed_annotation": True, "matches_full_validation_csv": True,
                    "csv_columns_and_label_order": True, "sample_check_only": True}

    check("m3_offline_real_checkpoint", m3_runtime)

    def boundary():
        # Exercise the actual CLI decision/serialization with controlled probabilities.
        # These weights are not model predictions and are never placed in the package.
        code = '''import importlib.util, pathlib, numpy as np, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
spec = importlib.util.spec_from_file_location("m3_cli", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert list(module.model_code.LABELS) == %r
assert module.model_code.THRESHOLD == 0.5
module.model_code.load_checkpoint = lambda p: (None, None, None, 256)
module.model_code.probabilities = lambda *a: np.array([[0.499999]*9, [0.5]*9, [0.0,0.0,0.5,0.500001,0,0,0,0,0]])
module.main(sys.argv[2:])
''' % LABELS
        with tempfile.TemporaryDirectory(prefix="dcc-m3-boundary-") as temp:
            stage = Path(temp)
            for name in ("c.json", "a.json", "b.json"):
                (stage / name).write_text('{"utterances":[]}', encoding="utf-8")
            ckpt = stage / "dummy.pt"
            ckpt.touch()
            proc = subprocess.run([sys.executable, "-c", code, str(ROOT / "submission/mission3/inference.py"),
                "--audio_dir", str(stage / "absent"), "--label_dir", str(stage), "--ckpt_path", str(ckpt),
                "--output", str(stage / "result.csv")], capture_output=True, text=True, timeout=120)
            if proc.returncode:
                raise ValueError(proc.stderr[-1500:])
            with (stage / "result.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [r["label file name"] for r in rows] != ["a.json", "b.json", "c.json"]:
                raise ValueError("Filename sorting failed")
            if [r["symptom"] for r in rows] != ["[]", str(LABELS), str(["두통", "복통"])]:
                raise ValueError("Threshold boundary, empty list, or label serialization failed")
        return {"below_half_empty": True, "exact_half_positive": True,
                "all_nine_in_order": True, "python_list_cell_roundtrip": True,
                "no_minimum_one_rule": True, "controlled_probabilities_only": True}

    check("m3_cli_threshold_boundary", boundary)
    report["ready_for_final_submission"] = not report["failures"] and not report["pending"]
    report["status"] = "failed" if report["failures"] else "passed_with_pending_artifacts" if report["pending"] else "ready"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output}", flush=True)
    return 1 if report["failures"] else 2 if report["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
