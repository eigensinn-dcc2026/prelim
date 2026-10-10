"""Assemble local mission folders without publishing models or source data."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission"


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_notebook(mission, exp_id, script_path, job_path, official_path):
    """Preserve real lab records as cited markdown, never fabricate executed cells."""
    job = json.loads(job_path.read_text(encoding="utf-8"))
    record = {key: job[key] for key in ("commit", "worker", "gpu", "started_at", "finished_at", "elapsed_min", "metrics") if key in job}
    cells = [dict(cell_type="markdown", metadata={}, source=[
        f"# Mission {mission}: {exp_id} 학습 코드와 실제 실행 기록\n\n",
        "아래 코드는 실제 GPU 실험에서 사용한 원본입니다. 이 노트북에서 학습을 다시 실행한 것으로 표시하지 않습니다.\n",
        "학습 기록은 완료된 lab 작업 JSON에서 가져왔으며, 코드 셀의 execution_count와 outputs는 비워 둡니다.\n",
        "학습은 Training만 사용합니다. 공식 Validation 결과는 학습 기록과 구분해 마지막 셀에 보존합니다.\n",
        "재학습 시 Training TL/TS zip, DCC_RUN_ROOT, LAB_WORK, LAB_OUT을 지정하고 로컬 dcc_data를 PYTHONPATH에 추가하세요.\n\n",
        f"원본 코드: scripts/lab/{script_path.name}\n원본 실행 기록: {job_path.name}\n\n",
        "```json\n" + json.dumps(record, ensure_ascii=False, indent=2) + "\n```\n"
    ])]
    # The original argparse entry point is intentionally excluded in a notebook.
    code = script_path.read_text(encoding="utf-8").split('if __name__ == "__main__":')[0]
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True)))
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=[
        "# Run explicitly only when reproducing training on the configured GPU environment.\n",
        "# import sys\n# sys.argv = ['training', '--out', str(Path(os.environ['LAB_OUT']))]\n# main()\n"
    ]))
    official = json.loads(official_path.read_text(encoding="utf-8"))
    cells.append(dict(cell_type="markdown", metadata={}, source=[
        "## 제출 체크포인트의 공식 Validation 검증 기록\n\n",
        "```json\n" + json.dumps(official, ensure_ascii=False, indent=2) + "\n```\n"
    ]))
    for index, cell in enumerate(cells):
        cell["id"] = f"m{mission}-record-{index}"
    write_json(DEST / f"mission{mission}" / "training.ipynb", dict(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    }, nbformat=4, nbformat_minor=5))


def main():
    # Vendor only modules required by inference/training records; no parent repo access.
    common = ["__init__", "audio", "guards", "manifest", "mission2", "mission2_features"]
    extras = {
        1: ["mission1", "mission1_features", "mission1_train", "mission2_train", "colab_env"],
        2: ["mission2_train", "mission2_checkpoint", "mission2_spectrogram", "mission2_cnn"],
        3: ["colab_env"],
    }
    for mission in (1, 2, 3):
        directory = DEST / f"mission{mission}"
        directory.mkdir(parents=True, exist_ok=True)
        names = common + extras[mission] if mission != 3 else ["__init__", "manifest", "guards", "colab_env"]
        for name in names:
            copy(ROOT / "src/dcc_data" / f"{name}.py", directory / "dcc_data" / f"{name}.py")
        (directory / ".gitignore").write_text("*.pt\n*.pth\n*.ckpt\n*.safetensors\n__pycache__/\noutputs/\n", encoding="utf-8")
    m1 = DEST / "mission1"
    copy(ROOT / "checkpoints/mission1/m1_xvec001_combined.ckpt", m1 / "m1_xvec001_combined.ckpt")
    for path in (ROOT / "checkpoints/mission1/xvector-model").iterdir():
        if path.is_file():
            copy(path, m1 / "xvector-model" / path.name)
    m2 = DEST / "mission2"
    model = (ROOT / "src/dcc_inference/mission2.py").read_text(encoding="utf-8")
    model = model.replace('Path(__file__).resolve().parents[2]', 'Path(__file__).resolve().parent')
    model = model.replace('REPOSITORY_ROOT / "scripts" / "mission2_inference.py"', 'REPOSITORY_ROOT / "mission2_pipeline.py"')
    (m2 / "mission2_model.py").write_text(model, encoding="utf-8")
    pipeline = (ROOT / "scripts/mission2_inference.py").read_text(encoding="utf-8")
    pipeline = pipeline.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
    pipeline = pipeline.replace('REPOSITORY_ROOT / "src"', 'REPOSITORY_ROOT')
    (m2 / "mission2_pipeline.py").write_text(pipeline, encoding="utf-8")
    packer = (ROOT / "scripts/pack_mission2_l2_ensemble.py").read_text(encoding="utf-8")
    packer = packer.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent').replace('REPOSITORY_ROOT / "src"', 'REPOSITORY_ROOT')
    (m2 / "pack_ensemble.py").write_text(packer, encoding="utf-8")
    copy(ROOT / "notebooks/mission2_l2_retrain.ipynb", m2 / "training_reference.ipynb")
    # Freeze versions actually installed and used by the audited environment.
    packages = ["torch", "transformers", "numpy", "pandas", "librosa", "scipy", "joblib", "soundfile", "safetensors"]
    versions = {name: importlib.metadata.version(name).split("+")[0] for name in packages}
    versions["scikit-learn"] = "1.6.1"  # version of the fixed LR checkpoint
    (m1 / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n", encoding="utf-8")
    m2_versions = {k: v for k, v in versions.items() if k not in ("transformers", "safetensors")}
    m2_versions["lightgbm"] = importlib.metadata.version("lightgbm")
    (m2 / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in m2_versions.items()) + "\n", encoding="utf-8")
    source_notebook(1, "M1-XVEC-001", ROOT / "scripts/lab/m1-xvec-001.py", ROOT.parent / "wt-lab-model-selection/done/02_M1-XVEC-001.json", ROOT / "reports/mission1/m1_xvec001_official_validation.json")
    source_notebook(3, "M3-ROBERTA-003", ROOT / "scripts/lab/m3-roberta-003.py", ROOT.parent / "wt-lab-model-selection/done/02_M3-ROBERTA-003.json", ROOT / "reports/mission3/official_val/roberta003_submission_check.json")
    print("Mission folders assembled locally. M2 selected weights remain unavailable.")


if __name__ == "__main__":
    main()
