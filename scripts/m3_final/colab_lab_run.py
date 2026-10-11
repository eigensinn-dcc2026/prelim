"""Run the M3 final-training notebook on a dcc-lab Colab worker (job M3-FINAL-001).

    python scripts/m3_final/colab_lab_run.py --out "$LAB_OUT" --work "$LAB_WORK"

1. Finds the official data folder on the mounted Drive and extracts only the label zips
   (Training/TL_서울_구급.zip, Validation/VL_서울_구급.zip) to the local work disk.
2. Executes scripts/m3_final/m3_final_train.ipynb with nbconvert (seeds 42 43 44, ensemble on,
   best.ckpt deleted after export). Large files are written to the local disk, not to Drive.
3. Copies the deliverables to --out (Drive): the executed notebook (training-log submission file),
   m3_ensemble_full.pt, the seed .pt files, TF-IDF joblib, *.json records, and writes metrics.json
   (the runner marks the job done only if it exists).
The official Validation is used only for scoring inside the notebook.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
NOTEBOOK = ROOT / "scripts" / "m3_final" / "m3_final_train.ipynb"
SEEDS = os.environ.get("M3_FINAL_SEEDS", "42 43 44")  # override only for local tests


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def extract(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for item in z.infolist():
            if not (dest / item.filename).resolve().is_relative_to(dest.resolve()):
                raise ValueError(f"unsafe zip member {item.filename}")
        z.extractall(dest)
    n = sum(1 for _ in dest.rglob("*.json"))
    log(f"extracted {zip_path.name}: {n} JSON")
    if n == 0:
        raise RuntimeError(f"no JSON in {zip_path}")
    return dest


def copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if (dst_dir / src.name).exists() and (dst_dir / src.name).stat().st_size == src.stat().st_size and src.suffix == ".pt"             and src.name.startswith("m3_roberta003_full_seed"):
        log(f"{src.name} already on Drive (resume copy), skipped")
        return
    tmp = dst_dir / (src.name + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst_dir / src.name)
    log(f"copied {src.name} ({src.stat().st_size:,} bytes) -> {dst_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Drive output folder ($LAB_OUT)")
    ap.add_argument("--work", type=Path, required=True, help="local work folder ($LAB_WORK)")
    args = ap.parse_args()
    from dcc_data.colab_env import find_data_root

    data = find_data_root()
    log(f"data root {data}")
    work = args.work
    train_dir = extract(data / "Training" / "TL_서울_구급.zip", work / "data" / "train")
    val_dir = extract(data / "Validation" / "VL_서울_구급.zip", work / "data" / "val")
    runs = work / "runs"
    # Inside the repo checkout: nbconvert runs the kernel in the notebook's folder, and the notebook
    # locates the repository by walking up from there.
    executed = NOTEBOOK.with_name("m3_final_train.executed.ipynb")
    env = dict(os.environ, M3_TRAIN_DIR=str(train_dir), M3_VAL_DIR=str(val_dir), M3_OUT=str(runs),
               M3_MODEL_CACHE=str(Path(os.environ["DCC_RUN_ROOT"]) / "models"), M3_SEEDS=SEEDS,
               M3_RUN_QUICK="0", M3_ENSEMBLE="1", M3_DROP_LAB_CKPT="1",
               # survive a lost Colab session: finished seeds are kept on Drive and reused by the next attempt
               M3_RESUME_DIR=str(args.out), M3_PROGRESS_LOG=str(args.out / "progress.log"))
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(NOTEBOOK, executed)  # run a copy so the tracked notebook stays clean
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", str(executed)]
    log("running notebook: " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(NOTEBOOK.parent), env=env)
    # Always keep the notebook (it holds the log, even on failure).
    copy(executed, args.out)
    if rc != 0:
        raise SystemExit(f"notebook failed with exit code {rc}")

    for f in sorted(runs.glob("*.pt")) + sorted(runs.glob("*.pt.json")) + sorted(runs.glob("*.joblib")) + [runs / "config.json"]:
        copy(f, args.out)
    for d in sorted(runs.glob("seed_*")) + [runs / "ensemble"]:
        for f in sorted(d.glob("*.json")) + sorted(d.glob("*.npy")):
            copy(f, args.out / d.name)
    scores = {d.name: json.loads((d / "val_score.json").read_text(encoding="utf-8"))
              for d in sorted(runs.glob("seed_*")) + [runs / "ensemble"] if (d / "val_score.json").exists()}
    ens = scores.get("ensemble", {})
    metrics = dict(exp_id="M3-FINAL-001",
                   primary_metric=dict(name="macro_f1", value=ens.get("macro_f1"), split="official_validation"),
                   metrics=dict(official_val=scores),
                   notes="Final training on 100% Training (ROBERTA-003 settings, seeds 42/43/44) + TF-IDF; "
                         "ensemble 0.55 x seed mean + 0.45 x TF-IDF fixed in advance on the 80:20 holdout; "
                         "official Validation used for scoring only; threshold >= 0.5, no post-processing.",
                   official_val=True, deliverables=sorted(p.name for p in args.out.iterdir()))
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log("metrics.json written: ensemble official Val Macro F1 " + str(ens.get("macro_f1")))


if __name__ == "__main__":
    main()
