"""Synthetic-only tests; never read competition caches or evaluate their rows."""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def screen():
    return importlib.import_module("dcc_data.mission2_model_family_screen")


@pytest.fixture
def context(monkeypatch):
    module = screen()
    from dcc_data.mission2_clean_stage4 import clean_59d_feature_names

    monkeypatch.setattr(module.models, "EXPECTED_INTERNAL_SPLIT", {
        "total_rows": 10, "train_rows": 8, "validation_rows": 2,
        "train_calls": 4, "validation_calls": 1, "call_overlap": 0,
    })
    calls = np.repeat(["a", "b", "c", "d", "e"], 2)
    clean = SimpleNamespace(
        X=np.zeros((10, 59), dtype=np.float32), y=np.tile([0, 1], 5),
        record_keys=np.array([f"Training|{call}|{i % 2}" for i, call in enumerate(calls)]),
        call_ids=calls, feature_names=clean_59d_feature_names(),
        provenance_fingerprint=module.CLEAN_FINGERPRINT,
    )
    clean.X[2:4] = 10000
    return module.prepare_screen(clean)


def test_registry_configs_and_capacity_control():
    module = screen()
    assert set(module.CANDIDATES) == {
        "mlp_reference", "lightgbm", "xgboost_hist", "catboost",
        "numerical_embedding_mlp", "stronger_mlp",
    }
    reference = module.get_candidate_config("mlp_reference")["parameters"]
    stronger = module.get_candidate_config("stronger_mlp")["parameters"]
    assert {k for k in reference if reference[k] != stronger[k]} == {"hidden_layer_sizes"}
    assert stronger["hidden_layer_sizes"] == (128,)
    assert reference["max_iter"] == 300
    for candidate in module.CANDIDATES:
        config = module.get_candidate_config(candidate)
        assert config["seed"] == 42
        assert set(config["parameter_evidence"]) == set(config["parameters"])
        assert json.loads(json.dumps(config))["config_id"] == config["config_id"]


def test_scaler_and_contract(context):
    module = screen()
    assert context.prepared.scaler.n_samples_seen_ == 8
    np.testing.assert_array_equal(context.prepared.scaler.mean_, np.zeros(59))
    module.validate_screen_contract(context)
    assert context.contract["threshold"] == 0.5
    assert context.contract["source_split"] == "Training"


@pytest.mark.parametrize("damage", ["dimension", "feature_order", "labels", "keys", "calls", "split", "scaler", "nan", "threshold", "official"])
def test_contract_damage_rejected_before_fit(context, damage):
    module = screen()
    if damage == "dimension":
        object.__setattr__(context.prepared, "X_train_scaled", context.prepared.X_train_scaled[:, :58])
    elif damage == "feature_order":
        object.__setattr__(context.prepared, "feature_names", context.prepared.feature_names[::-1])
    elif damage == "labels":
        context.prepared.y_train[0] ^= 1
    elif damage == "keys":
        context.prepared.training_record_keys[0] = "bad"
    elif damage == "calls":
        context.validation_calls[0] = context.train_calls[0]
    elif damage == "split":
        context.prepared.train_idx[0] = context.prepared.validation_idx[0]
    elif damage == "scaler":
        context.prepared.scaler.fit(np.ones((10, 59)))
    elif damage == "nan":
        context.prepared.X_train_scaled[0, 0] = np.nan
    elif damage == "threshold":
        context.contract["threshold"] = 0.4
    else:
        context.contract["source_split"] = "Validation"
    with pytest.raises(ValueError):
        module.validate_screen_contract(context)


def test_metrics_threshold_and_hce():
    module = screen()
    result, frame = module.evaluate_candidate(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.95, 0.5, 0.05]),
        np.array(["a", "b", "c", "d"]), np.array(["u", "v", "w", "x"]),
    )
    assert result["validation_accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["recall_0"] == result["recall_1"] == result["balanced_accuracy"] == 0.5
    assert result["errors"] == result["hce"] == 2
    assert frame.prediction.tolist() == [0, 1, 1, 0]
    assert frame.error_type.tolist() == ["TN", "FP", "TP", "FN"]
    with pytest.raises(ValueError):
        module.evaluate_candidate(np.array([0]), np.array([1.1]), np.array(["k"]), np.array(["c"]))


def test_environment_no_silent_fallback(monkeypatch):
    module = screen()
    monkeypatch.setattr(module, "environment_info", lambda: {"cuda_available": False})
    monkeypatch.setattr(module, "_optional_import", lambda name: SimpleNamespace(__version__="3.0.0"))
    assert module.check_candidate_environment("xgboost_hist", "gpu")["status"] == "GPU_UNAVAILABLE"
    assert module.check_candidate_environment("stronger_mlp", "gpu")["status"] == "BACKEND_UNAVAILABLE"
    assert module.check_candidate_environment("numerical_embedding_mlp", "cpu")["status"] == "BACKEND_UNAVAILABLE"
    assert module.check_candidate_environment("mlp_reference", "cpu")["status"] == "READY_CPU"
    with pytest.raises(ValueError, match="fallback prohibited"):
        module.build_candidate("mlp_reference", module.get_candidate_config("mlp_reference"), "gpu")
    with pytest.raises(ValueError, match="fallback prohibited"):
        module.build_candidate("numerical_embedding_mlp", module.get_candidate_config("numerical_embedding_mlp"), "cpu")
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(module, "_optional_import", missing)
    assert module.check_candidate_environment("lightgbm", "gpu")["status"] == "MISSING_DEPENDENCY"


def test_gpu_backend_probe_failure_is_explicit(monkeypatch):
    module = screen()
    monkeypatch.setattr(module, "environment_info", lambda: {"cuda_available": True})
    monkeypatch.setattr(module, "_optional_import", lambda name: SimpleNamespace(__version__="4.6.0"))
    def unavailable(*args):
        raise RuntimeError("GPU Tree Learner was not enabled in this build")
    monkeypatch.setattr(module, "_probe_gpu_backend", unavailable)
    result = module.check_candidate_environment("lightgbm", "gpu")
    assert result["status"] == "BACKEND_UNAVAILABLE"
    assert result["actual_device"] is None


def test_dependency_abi_import_failure_is_explicit(monkeypatch):
    module = screen()
    monkeypatch.setattr(module, "environment_info", lambda: {"cuda_available": True})
    def broken(name):
        raise ValueError("binary ABI mismatch")
    monkeypatch.setattr(module, "_optional_import", broken)
    result = module.check_candidate_environment("catboost", "gpu")
    assert result["status"] == "BACKEND_UNAVAILABLE"
    assert result["actual_device"] is None
    assert "ABI mismatch" in result["reason"]


def test_artifacts_comparison_failure_isolation_and_no_overwrite(context, tmp_path, monkeypatch):
    module = screen()
    run = module.create_run(tmp_path, context, run_id="synthetic")
    class FakeModel:
        classes_ = np.array([0, 1])
        def fit(self, X, y):
            assert X is context.prepared.X_train_scaled
            assert y is context.prepared.y_train
            return self
        def predict_proba(self, X):
            p = np.tile([0.1, 0.9], len(X) // 2)
            return np.column_stack([1 - p, p])
    monkeypatch.setattr(module, "check_candidate_environment", lambda *a: {
        "status": "READY_CPU", "requested_device": "cpu", "actual_device": "cpu",
        "backend": "synthetic", "library_version": "synthetic", "environment": {},
    })
    monkeypatch.setattr(module, "build_candidate", lambda *a: FakeModel())
    reference = module.run_candidate("mlp_reference", context, run, device="cpu")
    candidate = module.run_candidate("stronger_mlp", context, run, device="cpu")
    table = module.build_comparison(run, context)
    assert table.delta_vs_mlp300.tolist() == [0, 0]
    assert candidate["paired"]["both_correct"] == 2
    assert candidate["paired"]["net_correction"] == 0
    directory = run / "candidates/mlp_reference"
    assert {p.name for p in directory.iterdir()} >= {
        "config.json", "contract.json", "environment.json", "metrics.json",
        "warnings.json", "timing.json", "validation_predictions.parquet",
    }
    assert pd.read_parquet(directory / "validation_predictions.parquet").columns.tolist() == module.PREDICTION_COLUMNS
    with pytest.raises(FileExistsError):
        module.run_candidate("mlp_reference", context, run, device="cpu")
    with pytest.raises(FileExistsError):
        module.create_run(tmp_path, context, run_id="synthetic")
    def fail(*args):
        raise RuntimeError("isolated failure")
    monkeypatch.setattr(module, "build_candidate", fail)
    failed = module.run_candidate("lightgbm", context, run, device="cpu")
    assert failed["status"] == "FAILED"
    assert reference["status"] == "SUCCESS"
    assert "FAILED" in module.build_comparison(run, context).status.tolist()


def test_paired_comparison_and_delta(context):
    module = screen()
    _, ref = module.evaluate_candidate(np.array([0, 1]), np.array([0.1, 0.1]), np.array(["a", "b"]), np.array(["u", "v"]))
    _, cand = module.evaluate_candidate(np.array([0, 1]), np.array([0.9, 0.9]), np.array(["a", "b"]), np.array(["u", "v"]))
    assert module.paired_comparison(ref, cand) == {
        "both_correct": 0, "both_wrong": 0, "MLP_correct_candidate_wrong": 1,
        "MLP_wrong_candidate_correct": 1, "net_correction": 0,
    }
    with pytest.raises(ValueError):
        module.paired_comparison(ref, cand.iloc[::-1])


def test_optional_imports_are_lazy():
    import subprocess
    import sys
    command = "import sys; from dcc_data import mission2_model_family_screen; assert not any(n in sys.modules for n in ('torch', 'lightgbm', 'catboost', 'xgboost', 'rtdl_num_embeddings'))"
    result = subprocess.run([sys.executable, "-c", command], env={**__import__('os').environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_backend_parameters_seed_and_no_validation_arguments(monkeypatch):
    module = screen()
    captured = []
    def constructor(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()
    libraries = {
        "lightgbm": SimpleNamespace(LGBMClassifier=constructor),
        "xgboost": SimpleNamespace(XGBClassifier=constructor),
        "catboost": SimpleNamespace(CatBoostClassifier=constructor),
    }
    monkeypatch.setattr(module, "_optional_import", libraries.__getitem__)
    for candidate in libraries:
        candidate_id = "xgboost_hist" if candidate == "xgboost" else candidate
        module.build_candidate(candidate_id, module.get_candidate_config(candidate_id), "gpu")
    assert captured[0]["device_type"] == "gpu"
    assert captured[0]["random_state"] == 42
    assert captured[1]["device"] == "cuda"
    assert captured[1]["tree_method"] == "hist"
    assert captured[2]["task_type"] == "GPU"
    assert captured[2]["devices"] == "0"
    assert captured[2]["random_seed"] == 42
    assert captured[2]["allow_writing_files"] is False
    assert all("eval_set" not in config and "early_stopping_rounds" not in config for config in captured)
    config = module.get_candidate_config("mlp_reference")
    config["parameters"]["max_iter"] = 1
    with pytest.raises(ValueError):
        module.build_candidate("mlp_reference", config, "cpu")


def test_xgboost_fitted_cpu_fallback_rejected():
    module = screen()
    booster = SimpleNamespace(save_config=lambda: json.dumps({"learner": {"generic_param": {"device": "cpu"}}}))
    model = SimpleNamespace(get_booster=lambda: booster)
    with pytest.raises(RuntimeError, match="fallback rejected"):
        module._assert_actual_backend("xgboost_hist", model, "gpu")


def test_nonzero_deltas_reference_reuse_and_artifact_tamper(context, tmp_path, monkeypatch):
    module = screen()
    run = module.create_run(tmp_path, context, run_id="delta")
    class FakeModel:
        classes_ = np.array([0, 1])
        def __init__(self, correct):
            self.correct = correct
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            p = np.tile([0.1, 0.9] if self.correct else [0.1, 0.1], len(X) // 2)
            return np.column_stack([1 - p, p])
    monkeypatch.setattr(module, "check_candidate_environment", lambda *a: {
        "status": "READY_CPU", "actual_device": "cpu", "backend": "synthetic"})
    monkeypatch.setattr(module, "build_candidate", lambda candidate, *a: FakeModel(candidate != "mlp_reference"))
    module.run_candidate("mlp_reference", context, run, device="cpu")
    module.run_candidate("stronger_mlp", context, run, device="cpu")
    table = module.build_comparison(run, context)
    row = table.loc[table.candidate == "stronger_mlp"].iloc[0]
    assert row.delta_vs_mlp300 == 0.5
    assert row.error_delta_vs_mlp300 == -1
    assert row.net_correction == 1
    destination = module.create_run(tmp_path, context, run_id="reused")
    module.load_reference_artifact(run / "candidates/mlp_reference", context, destination)
    assert module.build_comparison(destination, context).validation_accuracy.iloc[0] == 0.5
    module.save_comparison(run, context)
    with pytest.raises(FileExistsError):
        module.save_comparison(run, context)
    contract_path = run / "candidates/stronger_mlp/contract.json"
    contract = json.loads(contract_path.read_text())
    contract["identity"]["train_matrix"] = "different"
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="contract differs"):
        module.build_comparison(run, context)


def test_readiness_exception_isolated(context, tmp_path, monkeypatch):
    module = screen()
    run = module.create_run(tmp_path, context, run_id="broken-import")
    def broken(*args):
        raise RuntimeError("binary ABI failure")
    monkeypatch.setattr(module, "check_candidate_environment", broken)
    result = module.run_candidate("catboost", context, run, device="gpu")
    assert result["status"] == "BACKEND_UNAVAILABLE"
    assert "ABI" in result["reason"]


@pytest.mark.parametrize("candidate,package", [("lightgbm", "lightgbm"), ("xgboost_hist", "xgboost")])
def test_optional_real_library_synthetic_smoke(candidate, package):
    pytest.importorskip(package)
    module = screen()
    X = np.random.default_rng(42).normal(size=(16, 59)).astype(np.float32)
    config = module.get_candidate_config(candidate)["parameters"]
    config["n_estimators"] = 2  # Adapter smoke only, never a screen configuration.
    if candidate == "lightgbm":
        config["min_child_samples"] = 1
    model = module._estimator(candidate, config, "cpu")
    model.fit(X, np.tile([0, 1], 8))
    values = module.models.validate_binary_probabilities(model.predict_proba(X), expected_rows=16, classes=model.classes_)
    assert values.shape == (16, 2)


def test_optional_plr_adapter_synthetic_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rtdl_num_embeddings")
    from dcc_data.mission2_numerical_embedding_mlp import NumericalEmbeddingMLP
    parameters = screen().get_candidate_config("numerical_embedding_mlp")["parameters"]
    parameters["max_epochs"] = 2  # Private adapter CPU smoke, not public full-run fallback.
    parameters["batch_size"] = 8
    X = np.random.default_rng(42).normal(size=(16, 59)).astype(np.float32)
    state = torch.random.get_rng_state().clone()
    model = NumericalEmbeddingMLP(parameters, device="cpu")
    model.fit(X, np.tile([0, 1], 8))
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.n_iter_ == 2
    assert model.predict_proba(X).shape == (16, 2)
    model.save(tmp_path / "model.pt")
    model.release()
    assert model.model is None


def test_notebook_structure_sources_only_and_shared_runner():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "notebooks/mission2_model_family_screen_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    checkpoints = [source.index("Checkpoint " + checkpoint + " —") for checkpoint in ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]]
    assert checkpoints == sorted(checkpoints)
    assert "RUN_MODEL_FIT" not in source
    assert "partial(screen.run_candidate" in source
    assert "screen.load_shared_screen(" in source
    assert "screen.shared_training_archive_paths(SHARED_ARCHIVE_ROOT)" in source
    assert "mission2_shared_training_raw" in source
    assert "mission2_egemaps_transfer_identity.json" in source
    assert "'opensmile'" in source
    for candidate in screen().CANDIDATES:
        assert "run_candidate('" + candidate in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            code = "".join(cell["source"])
            compile(code, str(path), "exec")
            assert "C:\\Users" not in code


def test_shared_screen_connects_training_staging_and_strict_loader(context, tmp_path, monkeypatch):
    module = screen()
    from dcc_data import mission2_egemaps_colab as colab
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame({"split": ["Training", "Validation"], "call_id": ["a", "unused"],
                  "audio_path": ["Training/a.wav", "Validation/unused.wav"],
                  "json_path": ["Training/a.json", "Validation/unused.json"]}).to_parquet(manifest_path)
    proof_path = tmp_path / "transfer.json"
    proof_path.write_text("{}")
    monkeypatch.setattr(module, "resolve_repo_root", lambda root: repo)
    monkeypatch.setattr(module.models, "standalone_training_cache_paths", lambda root: SimpleNamespace(
        manifest_parquet=manifest_path, manifest_metadata=tmp_path / "manifest.json"))
    seen = []
    def stage(manifest, archives, raw_root):
        assert manifest["split"].tolist() == ["Training"]
        assert archives == [tmp_path / "shared.zip"]
        seen.append("stage")
        return {"status": "PASS", "wav_files": 1, "json_files": 1}
    def load(chain, manifest, metadata, raw_root, *, transfer_identity_path):
        assert raw_root == tmp_path / "staged"
        assert transfer_identity_path == proof_path
        assert seen == ["stage"]
        seen.append("load")
        return SimpleNamespace()
    monkeypatch.setattr(colab, "prepare_shared_training_raw", stage)
    monkeypatch.setattr(module, "load_clean_stage4_59d", load)
    monkeypatch.setattr(module, "prepare_screen", lambda clean: context)
    assert module.load_shared_screen(repo, [tmp_path / "shared.zip"], tmp_path / "staged", proof_path) is context
    assert seen == ["stage", "load"]
    seen.clear()
    with pytest.raises(FileNotFoundError):
        module.load_shared_screen(repo, [tmp_path / "shared.zip"], tmp_path / "staged", tmp_path / "missing.json")
    assert seen == []


@pytest.mark.parametrize("shared", [False, True])
def test_cli_reference_load_does_not_refit(tmp_path, monkeypatch, shared):
    import importlib.util
    from pathlib import Path
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/run_mission2_model_family_screen.py"
    spec = importlib.util.spec_from_file_location("screen_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    module = cli.screen
    argv = [str(script), "--candidate", "mlp_reference", "--device", "cpu", "--reference-artifact", str(tmp_path / "reference")]
    if shared:
        argv.extend(["--shared-archive-root", str(tmp_path / "archives"), "--raw-root", str(tmp_path / "staged"),
                     "--transfer-identity", str(tmp_path / "proof.json")])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module, "check_candidate_environment", lambda *a: {"status": "READY_CPU"})
    monkeypatch.setattr(module, "resolve_repo_root", lambda *a: tmp_path)
    monkeypatch.setattr(module, "load_screen", lambda *a: None)
    def load_shared(root, archives, raw, proof):
        assert len(archives) == 12
        assert archives[0].name.endswith("001.zip") and archives[-1].name.endswith("012.zip")
        assert raw == tmp_path / "staged" and proof == tmp_path / "proof.json"
    monkeypatch.setattr(module, "load_shared_screen", load_shared)
    monkeypatch.setattr(module, "create_run", lambda *a, **k: tmp_path / "run")
    monkeypatch.setattr(module, "load_reference_artifact", lambda *a: {"status": "SUCCESS"})
    monkeypatch.setattr(module, "build_comparison", lambda *a: pd.DataFrame())
    def forbidden(*args, **kwargs):
        pytest.fail("Reference reuse must not fit again")
    monkeypatch.setattr(module, "run_candidate", forbidden)
    cli.main()
