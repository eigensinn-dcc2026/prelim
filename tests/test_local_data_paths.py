"""Path/setup checks use temporary files only; no competition pipeline runs."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["eda_dcc_dataset", "analyze_mission1_validation_errors", "verify_timestamps"])
def test_data_root_precedence(name, monkeypatch, tmp_path):
    module = load_script(name)
    monkeypatch.delenv("DCC_DATA_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", [name])
    assert Path(module.parse_args().data_root) == ROOT / "data/raw_original"
    env_root = tmp_path / "한글 data"
    monkeypatch.setenv("DCC_DATA_ROOT", str(env_root))
    assert Path(module.parse_args().data_root) == env_root
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(sys, "argv", [name, "--data-root", str(explicit)])
    assert Path(module.parse_args().data_root) == explicit


@pytest.mark.parametrize("name", ["eda_dcc_dataset", "analyze_mission1_validation_errors", "verify_timestamps"])
def test_missing_inputs_stop_before_outputs(name, monkeypatch, tmp_path):
    module = load_script(name)
    monkeypatch.setattr(sys, "argv", [name, "--data-root", str(tmp_path / "missing"), "--manifest", str(tmp_path / "missing.csv")])
    with pytest.raises(FileNotFoundError, match="missing"):
        module.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["eda_dcc_dataset", "analyze_mission1_validation_errors", "verify_timestamps"])
def test_missing_manifest_stops_before_outputs(name, monkeypatch, tmp_path):
    module = load_script(name)
    monkeypatch.setattr(sys, "argv", [name, "--data-root", str(tmp_path), "--manifest", str(tmp_path / "missing.csv")])
    with pytest.raises(FileNotFoundError, match="Manifest.*missing.csv"):
        module.main()
    assert list(tmp_path.iterdir()) == []


def notebook(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path,cells", [("notebooks/mission2_local.ipynb", [1, 2, 4, 5]), ("reports/M3_baseline.ipynb", [0, 1, 2])])
@pytest.mark.parametrize("external", [False, True])
def test_notebook_setup_is_local_and_read_only(path, cells, external, monkeypatch, tmp_path):
    repo = tmp_path / "한글 repo"
    (repo / "src/dcc_data").mkdir(parents=True)
    (repo / "scripts").mkdir()
    data = tmp_path / "외부 data" if external else repo / "data/raw_original"
    for folder in ["Training/2.라벨링데이터/TL_서울_구급", "Validation/2.라벨링데이터/VL_서울_구급", "Validation/1.원천데이터/VS_서울_구급"]:
        (data / folder).mkdir(parents=True)
    workdir = repo / Path(path).parent
    workdir.mkdir(exist_ok=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("DCC_DATA_ROOT", raising=False)
    if external:
        monkeypatch.setenv("DCC_DATA_ROOT", str(data))
    monkeypatch.delenv("DCC_PYTHON", raising=False)
    before = set(repo.rglob("*"))
    namespace = {}
    for index in cells:
        exec(compile("".join(notebook(path)["cells"][index]["source"]), path, "exec"), namespace)
    assert namespace["REPO_ROOT"] == repo
    assert namespace.get("DATA_ROOT", namespace.get("DATA_DIR")) == data
    assert set(repo.rglob("*")) == before


def test_existing_local_mode_skips_zip_preparation():
    namespace = {"DATA_MODE": "existing"}
    for index in [5, 7, 10]:
        exec("".join(notebook("notebooks/mission2_local.ipynb")["cells"][index]["source"]), namespace)
    # No ZIP variables, subprocess helper or filesystem module is needed.
    assert "shutil" not in namespace


def test_m3_commands_use_selected_python_and_absolute_scripts(monkeypatch, tmp_path):
    calls = []
    namespace = {
        "PYTHON_EXECUTABLE": sys.executable, "REPO_ROOT": tmp_path,
        "TRAIN_LABEL_DIR": tmp_path / "한글 train", "VALID_LABEL_DIR": tmp_path / "labels",
        "VALID_AUDIO_DIR": tmp_path / "audio", "ARTIFACT_DIR": tmp_path / "artifacts",
        "PREDICTION_PATH": tmp_path / "outputs/mission3.csv",
        "subprocess": types.SimpleNamespace(run=lambda args, **kwargs: calls.append((args, kwargs))),
    }
    for index in [7, 8]:
        exec("".join(notebook("reports/M3_baseline.ipynb")["cells"][index]["source"]), namespace)
    assert calls[0][0][:2] == [sys.executable, str(tmp_path / "scripts/mission3_baseline.py")]
    assert calls[1][0][:2] == [sys.executable, str(tmp_path / "scripts/inference.py")]
    assert calls[0][0][-6:] == ["--threshold", "0.5", "--test_size", "0.2", "--seed", "42"]
    assert all(kwargs == {"cwd": tmp_path, "check": True} for _, kwargs in calls)


def test_colab_mount_is_conditional(monkeypatch, tmp_path):
    (tmp_path / "src/dcc_data").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("DCC_PYTHON", raising=False)
    mounts = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mounts.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    cell = "".join(notebook("reports/M3_baseline.ipynb")["cells"][1]["source"])
    monkeypatch.delenv("DCC_DATA_ROOT", raising=False)
    exec(cell, {})
    assert mounts == ["/content/drive"]
    mounts.clear()
    monkeypatch.setenv("DCC_DATA_ROOT", str(tmp_path))
    exec(cell, {})
    assert mounts == []


def test_timestamp_cli_uses_only_synthetic_data(monkeypatch, tmp_path):
    module = load_script("verify_timestamps")
    raw = tmp_path / "한글 data"
    raw.mkdir()
    (raw / "sample.json").write_text(json.dumps({"endAt": 1000, "utterances": [{"endAt": 1000}]}), encoding="utf-8")
    import wave
    with wave.open(str(raw / "sample.wav"), "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * 8000)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,stem,wav_path,json_path\nTraining,sample,sample.wav,sample.json\n", encoding="utf-8")
    output = tmp_path / "report/result.json"
    monkeypatch.setattr(sys, "argv", ["verify_timestamps", "--data-root", str(raw), "--manifest", str(manifest), "--output", str(output)])
    module.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["training"]["n_checked"] == 1
    assert report["training"]["violation_count"] == 0
    assert report["validation"]["n_checked"] == 0


def test_m3_loader_and_columns(monkeypatch, tmp_path):
    (tmp_path / "sample.json").write_text(json.dumps({"recordId": "sample", "utterances": [{"text": "도와주세요"}], "symptom": ["고열"]}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = {"TRAIN_LABEL_DIR": tmp_path, "display": lambda value: None}
    exec("".join(notebook("reports/M3_baseline.ipynb")["cells"][5]["source"]), namespace)
    assert len(namespace["dataframe"]) == 1
    exec("".join(notebook("reports/M3_baseline.ipynb")["cells"][6]["source"]), namespace)
