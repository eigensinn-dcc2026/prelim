"""The L2 retrain notebook is generated from notebooks/gen/make_l2_nb.py and must stay in sync.

These checks cover the Colab-run gates (code zip sha256, Training packed sha256,
AMP epoch-1 stop, fp32 weights, env.json, fresh output folder) without launching training.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "mission2_l2_retrain.ipynb"
GEN_PATH = REPO / "notebooks" / "gen" / "make_l2_nb.py"
CODE_ZIP_SHA = "f1526b1df25fe9c9a5c985e01521df244955ceea5af256604000b4d717c92407"
TRAIN_PACKED_LINE = (
    'TRAIN_PACKED_SHA256 = {"m2_spectrograms_packed.npy": "", '
    '"m2_spectrogram_index_packed.parquet": ""}'
)


def _notebook():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return ["".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _cell_containing(snippet: str) -> str:
    matches = [cell for cell in _code_cells() if snippet in cell]
    assert len(matches) == 1, snippet
    return matches[0]


def _exec_functions(source: str, namespace: dict) -> None:
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, "<notebook-cell>", "exec"), namespace)


def test_generator_rewrites_the_committed_notebook() -> None:
    dst = NB_PATH.with_suffix(".regen.ipynb")
    try:
        shutil.copyfile(NB_PATH, dst)
        subprocess.run([sys.executable, str(GEN_PATH), str(dst)], check=True, cwd=REPO)
        assert dst.read_bytes() == NB_PATH.read_bytes()
    finally:
        dst.unlink(missing_ok=True)


def test_notebook_pins_code_zip_training_sha_amp_env_and_output_folder() -> None:
    text = "\n".join(_code_cells())
    assert 'Path("/content/drive/MyDrive/dcc2026-prelim-main-6cc1181.zip")' in text
    assert f'CODE_ZIP_SHA256_EXPECTED = "{CODE_ZIP_SHA}"' in text
    assert "if CODE_ZIP_SHA256 != CODE_ZIP_SHA256_EXPECTED:" in text
    assert "raise RuntimeError" in _cell_containing("CODE_ZIP_SHA256_EXPECTED")
    assert "zip sha256" in _cell_containing("CODE_ZIP_SHA256_EXPECTED")
    assert "commit" in _cell_containing("ZIP_COMMENT")
    assert "zip comment (git archive)" in text
    assert TRAIN_PACKED_LINE in text
    assert "require_train_packed_sha256_before_training" in text
    assert "FP32_EPOCH1_REF = None" in text
    assert "0.9133/0.9134" in text
    assert "USE_AMP    = True" in text
    assert 'cmd.append("--amp")' in text
    assert "AMP_SUSPECT_HELD_OUT_BELOW = 0.903" in text
    assert "assert_amp_seed1_epoch1" in text
    assert 'torch.load(path, map_location="cpu", weights_only=False)' in text
    assert "torch.float32" in text
    assert '"torch.version.cuda": torch.version.cuda' in text
    assert '"cudnn_version": torch.backends.cudnn.version()' in text
    assert "nvidia-smi" in text
    assert "pip_freeze.txt" in text
    assert 'OUT / "env.json"' in text
    assert 'OUT  = RUN_ROOT / "m2_l2_rerun_6cc1181"' in text
    assert "MyDrive/dcc_runs/m2_l2_rerun_6cc1181" in text
    assert "guard_fresh_output_folder" in text
    # hyperparameters stay on the main recipe
    assert '"--model", "large"' in text
    assert '"--seed", "42"' in text
    assert '"--torch-seed", seed' in text
    assert '"--test-size", "0.2"' in text
    assert '"--epochs", n_epochs' in text
    assert '"--batch-size", "64"' in text
    assert '"--lr", "1e-3"' in text
    assert "SEEDS      = [1, 2, 3]" in text
    assert "EPOCHS     = 30" in text
    assert "zip top-level:" in text


def test_train_packed_sha256_refuses_empty_or_mismatch() -> None:
    namespace = {
        "RUN_TRAIN": True,
        "RUN_BUILD": False,
        "TRAIN_PACKED_NAME": "m2_spectrograms_packed.npy",
        "TRAIN_INDEX_NAME": "m2_spectrogram_index_packed.parquet",
        "TRAIN_PACKED_SHA256": {
            "m2_spectrograms_packed.npy": "",
            "m2_spectrogram_index_packed.parquet": "",
        },
        "DATA_SHAS": {"train_packed": "abc", "train_index": "def"},
    }
    _exec_functions(_cell_containing("def require_train_packed_sha256_before_training"), namespace)
    with pytest.raises(RuntimeError, match="비어 있어 학습을 시작하지 않음"):
        namespace["require_train_packed_sha256_before_training"]()

    namespace["TRAIN_PACKED_SHA256"] = {
        "m2_spectrograms_packed.npy": "abc",
        "m2_spectrogram_index_packed.parquet": "not-def",
    }
    with pytest.raises(RuntimeError, match="sha256이 Drive 파일과 달라"):
        namespace["require_train_packed_sha256_before_training"]()

    namespace["TRAIN_PACKED_SHA256"]["m2_spectrogram_index_packed.parquet"] = "def"
    namespace["require_train_packed_sha256_before_training"]()


def test_amp_epoch1_gate_stops_on_nan_or_nondecreasing_loss() -> None:
    import math

    recorded = {}

    def epoch_log_status(_seed):
        return {"by_epoch": {1: recorded["e1"]}}

    namespace = {
        "USE_AMP": True,
        "FP32_EPOCH1_REF": None,
        "np": __import__("numpy"),
    }
    _exec_functions(_cell_containing("def assert_amp_seed1_epoch1"), namespace)
    namespace["epoch_log_status"] = epoch_log_status
    gate = namespace["assert_amp_seed1_epoch1"]

    recorded["e1"] = {"train_loss": math.nan, "val_acc_internal": 0.5, "lr": 1e-3}
    with pytest.raises(RuntimeError, match="NaN/inf"):
        gate()

    recorded["e1"] = {"train_loss": 1.2, "val_acc_internal": 0.4, "lr": 1e-3}
    with pytest.raises(RuntimeError, match="감소하지 않음"):
        gate()

    recorded["e1"] = {"train_loss": 0.4, "val_acc_internal": 0.8, "lr": 1e-3}
    gate()


def test_saved_checkpoint_floating_tensors_must_be_fp32(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    good = tmp_path / "good.pt"
    torch.save(
        {
            "best_model_state_dict": {
                "classifier.weight": torch.zeros(2, 2),
                "features.1.num_batches_tracked": torch.tensor(3, dtype=torch.long),
            },
            "last_model_state_dict": {"classifier.weight": torch.ones(2, 2)},
        },
        good,
    )
    namespace = {"torch": torch, "Path": Path}
    _exec_functions(_cell_containing("def assert_ckpt_weights_fp32"), namespace)
    namespace["assert_ckpt_weights_fp32"](good)

    bad = tmp_path / "fp16.pt"
    torch.save({"best_model_state_dict": {"classifier.weight": torch.zeros(2, 2, dtype=torch.float16)}}, bad)
    with pytest.raises(AssertionError, match="torch.float32"):
        namespace["assert_ckpt_weights_fp32"](bad)

    odd = tmp_path / "int.pt"
    torch.save({"best_model_state_dict": {"classifier.weight": torch.zeros(2, dtype=torch.long)}}, odd)
    with pytest.raises(AssertionError, match="fp32 가중치가 아님"):
        namespace["assert_ckpt_weights_fp32"](odd)


def test_env_json_records_cuda_cudnn_and_nvidia_smi_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = {
        "subprocess": __import__("subprocess"),
        "json": json,
        "COMMIT_FULL": "6cc11813650a85e172229f6d0618a9ca7b3fec7a",
        "COMMIT_SOURCE": "zip comment (git archive)",
        "CODE_ZIP": Path("dcc2026-prelim-main-6cc1181.zip"),
        "CODE_ZIP_SHA256": CODE_ZIP_SHA,
        "OUT": tmp_path,
        "torch": type(
            "Torch",
            (),
            {
                "__version__": "2.1.0+cu121",
                "version": type("V", (), {"cuda": "12.1"})(),
                "backends": type(
                    "B",
                    (),
                    {"cudnn": type("C", (), {"version": staticmethod(lambda: 8902)})()},
                )(),
                "cuda": type("Cuda", (), {"is_available": staticmethod(lambda: False)})(),
            },
        )(),
    }

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi missing")

    monkeypatch.setattr(namespace["subprocess"], "run", boom)
    _exec_functions(_cell_containing("def write_env_json"), namespace)
    namespace["write_env_json"]()
    env = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
    assert env["torch.version.cuda"] == "12.1"
    assert env["cudnn_version"] == 8902
    assert env["nvidia_smi_gpu_name"] is None
    assert "FileNotFoundError" in env["nvidia_smi_error"]
    assert "nvidia-smi missing" in env["nvidia_smi_error"]


def test_output_folder_stops_for_unmarked_checkpoints_and_resumes_this_run(tmp_path: Path) -> None:
    out = tmp_path / "MyDrive" / "dcc_runs" / "m2_l2_rerun_6cc1181"
    out.mkdir(parents=True)
    (out / "m2_cnn_large_raw_ts1.pt").write_bytes(b"ckpt")
    namespace = {
        "OUT": out,
        "json": json,
        "os": os,
        "RUN_IDENTITY_PATH": out / "run_identity.json",
        "CODE_ZIP_SHA256": CODE_ZIP_SHA,
        "THIS_RUN": {
            "folder": "m2_l2_rerun_6cc1181",
            "commit": "6cc11813650a85e172229f6d0618a9ca7b3fec7a",
            "code_zip": "dcc2026-prelim-main-6cc1181.zip",
            "code_zip_sha256": CODE_ZIP_SHA,
            "amp": True,
        },
    }
    _exec_functions(_cell_containing("def guard_fresh_output_folder"), namespace)
    with pytest.raises(RuntimeError, match="이전 실행 체크포인트"):
        namespace["guard_fresh_output_folder"]()
    assert not (out / "run_identity.json").exists()

    (out / "train_ts1_epochs.jsonl").write_text(
        json.dumps({"run_folder": "m2_l2_rerun_6cc1181", "code_zip_sha256": CODE_ZIP_SHA}) + "\n",
        encoding="utf-8",
    )
    namespace["guard_fresh_output_folder"]()
    ident = json.loads((out / "run_identity.json").read_text(encoding="utf-8"))
    assert ident["code_zip_sha256"] == CODE_ZIP_SHA
    assert ident["amp"] is True
