"""Frozen, Training-only clean-59D screen shared by Colab and CLI.

Importing this module never loads caches, imports GPU packages, or fits a model.
Environment GPU probes use 16 synthetic rows, never competition data.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
import uuid
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier

from . import mission2_model_experiments as models
from . import mission2_acoustic_feature_expansion as acoustic
from .mission2_clean_stage4 import clean_59d_feature_names, load_clean_stage4_59d
from .mission2_mlp_bottleneck import build_mlp_configs, prepare_clean_59d


CLEAN_FINGERPRINT = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
CLEAN_CHAIN_RELATIVE = Path("data/processed/mission2_stage4_clean/stage4_clean_v1_090422f783f0d911")
SCHEMA = "mission2_model_family_screen_v1"
PREDICTION_COLUMNS = ["record_key", "call_id", "y_true", "probability_speaker1",
                      "prediction", "confidence", "correct", "error_type"]
CANDIDATES = {
    "mlp_reference": ("MLP", "sklearn", "scikit-learn"),
    "lightgbm": ("GBDT", "lightgbm", "lightgbm"),
    "xgboost_hist": ("GBDT", "xgboost", "xgboost"),
    "catboost": ("GBDT", "catboost", "catboost"),
    "numerical_embedding_mlp": ("MLP", "torch", "torch"),
    "stronger_mlp": ("MLP", "sklearn", "scikit-learn"),
}
SOURCES = {
    "sklearn": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    "lightgbm": "https://lightgbm.readthedocs.io/en/stable/Parameters.html",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/parameter.html",
    "catboost": "https://catboost.ai/docs/en/references/training-parameters/common",
    "torch": "https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package",
}


def get_candidate_config(candidate_id: str) -> dict:
    """Return a fresh copy of the sole predeclared config; no score selection."""
    family, implementation, _ = CANDIDATES[candidate_id]
    reference = build_mlp_configs()[1]
    configs = {
        "mlp_reference": reference,
        "stronger_mlp": {**reference, "hidden_layer_sizes": (128,)},
        "lightgbm": dict(objective="binary", boosting_type="gbdt", n_estimators=300,
                         learning_rate=0.05, num_leaves=31, max_depth=-1,
                         min_child_samples=20, max_bin=255, reg_alpha=0.0,
                         reg_lambda=1.0, subsample=1.0, subsample_freq=0,
                         colsample_bytree=1.0, class_weight=None, scale_pos_weight=1.0,
                         is_unbalance=False, boost_from_average=True, early_stopping_round=0,
                         random_state=42, n_jobs=2, verbosity=-1),
        "xgboost_hist": dict(objective="binary:logistic", booster="gbtree",
                             tree_method="hist", eval_metric="logloss", n_estimators=300,
                             learning_rate=0.05, max_depth=6, grow_policy="depthwise",
                             min_child_weight=1.0, max_bin=256, gamma=0.0,
                             reg_alpha=0.0, reg_lambda=1.0, subsample=1.0,
                             colsample_bytree=1.0, scale_pos_weight=1.0,
                             base_score=0.5, random_state=42, n_jobs=2, verbosity=1),
        "catboost": dict(loss_function="Logloss", eval_metric="Logloss", iterations=300,
                         learning_rate=0.05, depth=6, grow_policy="SymmetricTree",
                         boosting_type="Plain", l2_leaf_reg=3.0, border_count=254,
                         random_strength=1.0, bootstrap_type="No",
                         class_weights=None, auto_class_weights=None, use_best_model=False,
                         random_seed=42, thread_count=2,
                         allow_writing_files=False, verbose=False),
        "numerical_embedding_mlp": dict(
            embedding_type="PLR", d_embedding=24, n_frequencies=48,
            frequency_init_scale=0.01, lite=False,
            hidden_sizes=(256, 256), activation="relu", dropout=0.1,
            optimizer="AdamW", learning_rate=0.001, weight_decay=0.0001,
            adam_betas=(0.9, 0.999), adam_eps=1e-8, batch_size=1024,
            max_epochs=100, stopping_rule="fixed_epochs_last_checkpoint",
            random_state=42, shuffle=True, dtype="float32", deterministic=True,
        ),
    }
    reasons = {
        "mlp_reference": "Reuse the observed MLP-300 config, not its score. No validation stopping.",
        "stronger_mlp": "Double only the single hidden layer 64 -> 128; all other reference settings identical.",
        "lightgbm": "Moderate leaf capacity, fixed 300 rounds and shrinkage; full rows/features, unweighted binary task.",
        "xgboost_hist": "Histogram depthwise trees, moderate depth, fixed rounds, L2 regularization; no validation selection.",
        "catboost": "Symmetric depth-6 trees, Plain boosting and no bootstrap; fixed rounds and no class weighting.",
        "numerical_embedding_mlp": "Official non-lite PLR embedding defaults; two 256-unit ReLU layers, fixed 100 epochs. This is a first-screen recipe, not paper-score reproduction.",
    }
    parameters = deepcopy(configs[candidate_id])
    evidence = {name: {"parameter": name, "value": value,
                       "selection_rationale": reasons[candidate_id],
                       "source": SOURCES[implementation]} for name, value in parameters.items()}
    config = {"candidate": candidate_id, "model_family": family,
              "implementation": implementation, "seed": 42, "threshold": 0.5,
              "parameters": parameters, "parameter_evidence": evidence}
    if candidate_id == "numerical_embedding_mlp":
        config["embedding_contract"] = {
            "learned_components": "frequencies and feature-specific linear projections",
            "fixed_components": "2*pi cosine/sine basis followed by ReLU",
            "bin_basis_construction": "No data-derived bins or statistics; random initialization then train-only optimization",
            "paper": "https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf",
            "backbone_structure": "Canonical MLP blocks: Linear -> ReLU -> Dropout, then a linear output",
            "backbone_reference": "https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py",
        }
    config["config_id"] = models.config_fingerprint(config)
    return config


def resolve_repo_root(explicit: str | Path | None = None) -> Path:
    """Use an explicit path or cwd ancestry; never guess a Drive directory."""
    options = [Path(explicit).expanduser().resolve()] if explicit else [Path.cwd(), *Path.cwd().parents]
    for root in options:
        if (root / "src/dcc_data/mission2_clean_stage4.py").is_file() and (root / "requirements.txt").is_file():
            return root.resolve()
    raise FileNotFoundError("Set REPO_ROOT explicitly to the mounted repository; no Drive path was selected.")


def _optional_import(name):
    return importlib.import_module(name)


def environment_info() -> dict:
    info = {"python": sys.version, "platform": platform.platform(),
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__, "librosa_version": importlib.metadata.version("librosa"),
            "cuda_available": False, "torch_cuda_available": None,
            "gpu_names": [], "gpu_count": 0}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10, check=True)
        info["gpu_names"] = [line.strip() for line in output.stdout.splitlines() if line.strip()]
        info["gpu_count"] = len(info["gpu_names"])
        info["cuda_available"] = bool(info["gpu_count"])
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        torch = _optional_import("torch")
        info["torch_version"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if info["torch_cuda_available"]:
            info["cuda_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
    except Exception as error:
        info["torch_import_note"] = str(error)
    return info


def _backend_parameters(candidate_id, device):
    if candidate_id == "lightgbm":
        return {"device_type": "cpu" if device == "cpu" else device,
                **({"deterministic": True, "force_col_wise": True} if device == "cpu" else {})}
    if candidate_id == "xgboost_hist":
        return {"device": "cpu" if device == "cpu" else "cuda"}
    if candidate_id == "catboost":
        return {"task_type": "CPU" if device == "cpu" else "GPU",
                **({"devices": "0"} if device != "cpu" else {})}
    return {}


def _estimator(candidate_id, parameters, device):
    parameters = {**parameters, **_backend_parameters(candidate_id, device)}
    if candidate_id in ("mlp_reference", "stronger_mlp"):
        return MLPClassifier(**parameters)
    if candidate_id == "numerical_embedding_mlp":
        from .mission2_numerical_embedding_mlp import NumericalEmbeddingMLP
        return NumericalEmbeddingMLP(parameters, device=device)
    constructors = {"lightgbm": "LGBMClassifier", "xgboost_hist": "XGBClassifier", "catboost": "CatBoostClassifier"}
    return getattr(_optional_import(CANDIDATES[candidate_id][1]), constructors[candidate_id])(**parameters)


def _assert_actual_backend(candidate_id, model, device):
    """XGBoost may emit a warning and silently fall back; inspect fitted state."""
    if candidate_id == "xgboost_hist":
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        expected = "cpu" if device == "cpu" else "cuda"
        if not actual.startswith(expected):
            raise RuntimeError(f"Requested {expected}, fitted XGBoost backend is {actual}; fallback rejected")


def _probe_gpu_backend(candidate_id, device):
    if candidate_id == "numerical_embedding_mlp":
        torch = _optional_import("torch")
        (torch.ones(1, device="cuda") + 1).cpu()
        return
    config = get_candidate_config(candidate_id)["parameters"]
    config["iterations" if candidate_id == "catboost" else "n_estimators"] = 1
    if candidate_id == "lightgbm":
        config.update(min_child_samples=1, min_data_in_bin=1)
    model = _estimator(candidate_id, config, device)
    X = np.random.default_rng(42).normal(size=(16, 59)).astype(np.float32)
    model.fit(X, np.tile([0, 1], 8))
    _assert_actual_backend(candidate_id, model, device)


def check_candidate_environment(candidate_id: str, device: str) -> dict:
    family, implementation, package = CANDIDATES[candidate_id]
    if device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be explicitly cpu, gpu or cuda")
    result = {"candidate": candidate_id, "requested_device": device,
              "actual_device": None, "backend": None, "status": "BACKEND_UNAVAILABLE",
              "library_version": None, "environment": environment_info()}
    if implementation == "sklearn" and device != "cpu":
        result["reason"] = "sklearn MLP supports CPU only"
        return result
    if candidate_id == "numerical_embedding_mlp" and device == "cpu":
        result["reason"] = "This candidate requires explicitly requested CUDA; no CPU full-run fallback"
        return result
    dependencies = [implementation] if implementation != "sklearn" else []
    if candidate_id == "numerical_embedding_mlp":
        dependencies.append("rtdl_num_embeddings")
    for name in dependencies:
        try:
            library = _optional_import(name)
            result.setdefault("dependency_versions", {})[name] = getattr(library, "__version__", "unknown")
        except (ImportError, OSError) as error:
            result.update(status="MISSING_DEPENDENCY", reason=f"MISSING_DEPENDENCY: {name}: {error}")
            return result
        except Exception as error:
            result.update(status="BACKEND_UNAVAILABLE",
                          reason=f"{name} import failed: {type(error).__name__}: {error}")
            return result
    result["library_version"] = (result.get("dependency_versions", {}).get(implementation)
                                  or importlib.metadata.version(package))
    if candidate_id == "xgboost_hist" and int(result["library_version"].split(".")[0]) not in (2, 3):
        result["reason"] = "Supported API: XGBoost >=2,<4; tree_method=hist with device=cpu/cuda"
        return result
    if candidate_id == "numerical_embedding_mlp":
        embedding = _optional_import("rtdl_num_embeddings")
        required = {"n_features", "d_embedding", "n_frequencies", "frequency_init_scale", "activation", "lite"}
        if not hasattr(embedding, "PeriodicEmbeddings") or not required.issubset(inspect.signature(embedding.PeriodicEmbeddings).parameters):
            result["reason"] = "Installed rtdl PeriodicEmbeddings does not support the verified non-lite PLR API"
            return result
    if device != "cpu":
        env = result["environment"]
        available = env.get("torch_cuda_available") if implementation == "torch" else env["cuda_available"]
        if not available:
            result.update(status="GPU_UNAVAILABLE", reason="Requested GPU is unavailable; choose CPU explicitly where supported")
            return result
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _probe_gpu_backend(candidate_id, device)
            result["probe_warnings"] = [str(item.message) for item in caught]
        except Exception as error:
            result.update(status="BACKEND_UNAVAILABLE", reason=str(error))
            return result
    result.update(status="READY_CPU" if device == "cpu" else "READY_GPU",
                  actual_device="cpu" if device == "cpu" else "cuda" if implementation in {"torch", "xgboost", "catboost"} else device,
                  backend=f"{implementation}:{device}",
                  backend_parameters=_backend_parameters(candidate_id, device))
    result["reproducibility_note"] = (
        "CatBoost GPU training is not bitwise deterministic despite fixed seed."
        if candidate_id == "catboost" and device != "cpu"
        else "Fixed seed; reproducibility is scoped to the recorded versions, device and backend."
    )
    result["backend_parameter_evidence"] = {
        name: {"parameter": name, "value": value,
               "selection_rationale": "Explicit requested backend; CPU determinism where supported. No fallback.",
               "source": SOURCES[implementation]}
        for name, value in result["backend_parameters"].items()
    }
    return result


def build_candidate(candidate_id, config, device):
    if device not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be explicitly cpu, gpu or cuda")
    if CANDIDATES[candidate_id][1] == "sklearn" and device != "cpu":
        raise ValueError("sklearn MLP supports CPU only; fallback prohibited")
    if candidate_id == "numerical_embedding_mlp" and device == "cpu":
        raise ValueError("Numerical embedding screen requires CUDA; fallback prohibited")
    if models.config_fingerprint(config) != models.config_fingerprint(get_candidate_config(candidate_id)):
        raise ValueError("Only the frozen first-screen config is accepted")
    return _estimator(candidate_id, config["parameters"], device)


@dataclass
class ScreenContext:
    prepared: object
    source_y: np.ndarray
    source_keys: np.ndarray
    source_calls: np.ndarray
    train_calls: np.ndarray
    validation_calls: np.ndarray
    contract: dict
    frozen_contract: dict


def _data_identity(context):
    p = context.prepared
    result = acoustic._prepared_array_fingerprints(p)
    for name, value in {"train_matrix": p.X_train_scaled, "validation_matrix": p.X_evaluation_scaled,
                        "train_y": p.y_train, "validation_y": p.y_evaluation,
                        "train_keys": p.training_record_keys.astype(str),
                        "validation_keys": p.evaluation_record_keys.astype(str),
                        "source_y": context.source_y, "source_keys": context.source_keys.astype(str),
                        "source_calls": context.source_calls.astype(str)}.items():
        result[name] = models._array_fingerprint(value)
    return result


def prepare_screen(clean) -> ScreenContext:
    if clean.provenance_fingerprint != CLEAN_FINGERPRINT:
        raise ValueError("Current clean chain provenance differs")
    if tuple(clean.feature_names) != clean_59d_feature_names() or clean.X.shape != (len(clean.y), 59):
        raise ValueError("Expected exact clean 59D feature order and row shape")
    if clean.X.dtype != np.float32:
        raise ValueError("Preserve the clean float32 dtype")
    if not (len(clean.y) == len(clean.record_keys) == len(clean.call_ids)):
        raise ValueError("Clean row alignment differs")
    # The existing strict loader guarantees source alignment; this also protects
    # callers passing an in-memory object. Tracking columns never enter X.
    keys = np.asarray(clean.record_keys).astype(str)
    calls = np.asarray(clean.call_ids).astype(str)
    if len(np.unique(keys)) != len(keys) or any(
        not key.startswith(f"Training|{call}|") for key, call in zip(keys, calls)
    ):
        raise ValueError("Training record_key/call_id alignment differs")
    prepared = prepare_clean_59d(clean)
    context = ScreenContext(prepared, clean.y.copy(), keys.copy(), calls.copy(),
                            calls[prepared.train_idx], calls[prepared.validation_idx], {}, {})
    context.contract = {
        "schema_version": SCHEMA, "source_split": "Training", "evaluation_set": "fixed_internal_validation",
        "provenance_fingerprint": CLEAN_FINGERPRINT, "feature_names": list(clean.feature_names),
        "feature_dim": 59, "dtype": "float32", "seed": 42, "threshold": 0.5,
        "scaler_fit_scope": "internal_train_only", "validation_preprocessing": "transform_only",
        "official_validation_used": False, "fit_inputs": ["X_train_scaled", "y_train"],
        "split": dict(prepared.split_audit), "identity": _data_identity(context),
    }
    if getattr(clean, "raw_source_verification", None) is not None:
        context.contract["raw_source_verification"] = deepcopy(clean.raw_source_verification)
    context.frozen_contract = deepcopy(context.contract)
    validate_screen_contract(context)
    return context


def validate_screen_contract(context):
    """Recheck the prepared payload immediately before any full fit/load."""
    p = context.prepared
    if context.contract != context.frozen_contract:
        raise ValueError("Frozen comparison contract changed")
    if tuple(p.feature_names) != clean_59d_feature_names():
        raise ValueError("Feature order differs or forbidden feature present")
    for X, y in [(p.X_train_scaled, p.y_train), (p.X_evaluation_scaled, p.y_evaluation)]:
        if X.shape != (len(y), 59) or X.dtype != np.float32 or not np.isfinite(X).all():
            raise ValueError("59D row/dtype/finite guard failed")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Binary labels must contain both classes")
    if set(context.train_calls) & set(context.validation_calls):
        raise ValueError("call overlap guard failed")
    for idx, y, keys, calls in [(p.train_idx, p.y_train, p.training_record_keys, context.train_calls),
                                (p.validation_idx, p.y_evaluation, p.evaluation_record_keys, context.validation_calls)]:
        if not (np.array_equal(y, context.source_y[idx]) and np.array_equal(keys, context.source_keys[idx])
                and np.array_equal(calls, context.source_calls[idx])):
            raise ValueError("Row/key/label/call alignment guard failed")
    if int(p.scaler.n_samples_seen_) != len(p.train_idx):
        raise ValueError("Scaler must fit internal train only")
    if _data_identity(context) != context.contract["identity"]:
        raise ValueError("Fixed split, scaling or prepared array identity changed")
    return deepcopy(context.contract)


def load_screen(repo_root: str | Path) -> ScreenContext:
    root = resolve_repo_root(repo_root)
    paths = models.standalone_training_cache_paths(root)
    clean = load_clean_stage4_59d(root / CLEAN_CHAIN_RELATIVE, paths.manifest_parquet,
                                 paths.manifest_metadata, root / "data/raw_original")
    return prepare_screen(clean)


def shared_training_archive_paths(archive_root: str | Path) -> list[Path]:
    """The 12 shared ZIP names recorded by the existing Colab Notebook."""
    return [Path(archive_root) / f"대학부 데이터-20260901T090511Z-1-{number:03d}.zip"
            for number in range(1, 13)]


def load_shared_screen(repo_root, archive_paths, raw_root, transfer_identity_path) -> ScreenContext:
    """Stage shared Training only, verify transfer bytes, reuse frozen 59D."""
    from .mission2_egemaps_colab import prepare_shared_training_raw

    root = resolve_repo_root(repo_root)
    proof = Path(transfer_identity_path)
    if not proof.is_file():
        raise FileNotFoundError(f"Existing transfer identity required before staging: {proof}")
    paths = models.standalone_training_cache_paths(root)
    manifest = pd.read_parquet(paths.manifest_parquet, columns=["split", "call_id", "audio_path", "json_path"])
    training = manifest.loc[manifest["split"].astype(str) == "Training"].reset_index(drop=True)
    staged = prepare_shared_training_raw(training, archive_paths, raw_root)
    print("Shared Training prepared:", staged)
    del manifest, training
    clean = load_clean_stage4_59d(root / CLEAN_CHAIN_RELATIVE, paths.manifest_parquet,
                                 paths.manifest_metadata, raw_root, transfer_identity_path=proof)
    return prepare_screen(clean)


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def create_run(repo_root, context, run_id=None):
    validate_screen_contract(context)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a single safe directory name")
    root = Path(repo_root) / "data/processed/mission2_model_family_screen" / run_id
    root.mkdir(parents=True, exist_ok=False)
    _write_json_new(root / "run_config.json", {"schema_version": SCHEMA, "run_id": run_id,
                    "candidates": {c: get_candidate_config(c) for c in CANDIDATES},
                    "historical_reference_note": "MLP-300 previously stopped at 130; historical metrics are not assertions or reusable predictions.",
                    "timing_note": "CPU/GPU times describe recorded backend runs, not pure algorithm speed."})
    _write_json_new(root / "contract.json", context.contract)
    _write_json_new(root / "environment.json", environment_info())
    scaler = context.prepared.scaler
    _write_json_new(root / "scaler.json", {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_), "fit_scope": "internal_train_only",
        "feature_names": context.contract["feature_names"],
    })
    (root / "candidates").mkdir()
    return root


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_run_contract(root, context):
    if _read_json(Path(root) / "contract.json") != context.contract:
        raise ValueError("Run result contract differs")
    configs = _read_json(Path(root) / "run_config.json")["candidates"]
    for candidate in CANDIDATES:
        if models.config_fingerprint(configs[candidate]) != models.config_fingerprint(get_candidate_config(candidate)):
            raise ValueError("Saved run frozen configuration differs")


def evaluate_candidate(y_true, probability_speaker1, record_keys, call_ids):
    p1 = np.asarray(probability_speaker1, dtype=np.float64)
    values = models.validate_binary_probabilities(np.column_stack([1 - p1, p1]), expected_rows=len(y_true))
    prediction = (values[:, 1] >= 0.5).astype(np.int64)
    metrics = models.calculate_prediction_metrics(y_true, prediction)
    confidence = np.maximum(values[:, 0], values[:, 1])
    truth = np.asarray(y_true)
    correct = prediction == truth
    error_type = np.where(truth == 0, np.where(prediction == 0, "TN", "FP"), np.where(prediction == 0, "FN", "TP"))
    frame = pd.DataFrame(dict(record_key=record_keys, call_id=call_ids, y_true=truth,
                              probability_speaker1=p1, prediction=prediction, confidence=confidence,
                              correct=correct, error_type=error_type), columns=PREDICTION_COLUMNS)
    metrics.update(validation_accuracy=metrics["accuracy"], balanced_accuracy=(metrics["recall_0"] + metrics["recall_1"]) / 2,
                   fp=int(metrics["confusion_matrix"][0][1]), fn=int(metrics["confusion_matrix"][1][0]),
                   errors=int((~correct).sum()), hce=int((~correct & (confidence >= 0.90)).sum()),
                   mean_confidence=float(confidence.mean()))
    return metrics, frame


def paired_comparison(reference, candidate):
    for name in ("record_key", "call_id", "y_true"):
        if not np.array_equal(reference[name].to_numpy(), candidate[name].to_numpy()):
            raise ValueError("Paired prediction row/label/call alignment differs")
    a = reference.correct.to_numpy(dtype=bool)
    b = candidate.correct.to_numpy(dtype=bool)
    lost, gained = int((a & ~b).sum()), int((~a & b).sum())
    return {"both_correct": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "MLP_correct_candidate_wrong": lost, "MLP_wrong_candidate_correct": gained,
            "net_correction": gained - lost}


def _validated_saved_result(directory, context):
    if _read_json(directory / "contract.json") != context.contract:
        raise ValueError("Candidate result contract differs; deltas prohibited")
    metrics = _read_json(directory / "metrics.json")
    candidate = metrics["candidate"]
    config = _read_json(directory / "config.json")
    if models.config_fingerprint(config) != models.config_fingerprint(get_candidate_config(candidate)):
        raise ValueError("Candidate result frozen config differs")
    if metrics["status"] != "SUCCESS":
        return metrics, None
    if metrics["config_id"] != config["config_id"]:
        raise ValueError("Candidate result config_id differs")
    frame = pd.read_parquet(directory / "validation_predictions.parquet")
    p = context.prepared
    if frame.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Prediction schema differs")
    for name, expected in [("record_key", p.evaluation_record_keys), ("call_id", context.validation_calls), ("y_true", p.y_evaluation)]:
        if not np.array_equal(frame[name].to_numpy(), expected):
            raise ValueError("Saved prediction identity differs")
    observed, expected_frame = evaluate_candidate(p.y_evaluation, frame.probability_speaker1.to_numpy(), p.evaluation_record_keys, context.validation_calls)
    if not frame.equals(expected_frame) or any(metrics.get(name) != value for name, value in observed.items()):
        raise ValueError("Saved probabilities, derived fields or metrics differ")
    return metrics, frame


def run_candidate(candidate_id, context, run_root, *, device):
    """Fit using train only; persist one result, release model, isolate failures."""
    config = get_candidate_config(candidate_id)
    validate_screen_contract(context)
    root = Path(run_root)
    _assert_run_contract(root, context)
    directory = root / "candidates" / candidate_id
    directory.mkdir(exist_ok=False)  # Reserve before work; reruns cannot overwrite.
    _write_json_new(directory / "config.json", config)
    _write_json_new(directory / "contract.json", context.contract)
    try:
        environment = check_candidate_environment(candidate_id, device)
    except Exception as error:
        environment = {"status": "BACKEND_UNAVAILABLE", "requested_device": device,
                       "actual_device": None, "backend": None,
                       "reason": f"Environment check failed: {type(error).__name__}: {error}"}
    _write_json_new(directory / "environment.json", environment)
    metrics = {"candidate": candidate_id, "model_family": config["model_family"],
               "implementation": config["implementation"], "config_id": config["config_id"],
               "device": environment["actual_device"], "backend": environment["backend"],
               "status": environment["status"]}
    timings = {"fit_seconds": None, "train_predict_seconds": None,
               "validation_predict_seconds": None, "predict_seconds": None}
    caught_messages = []
    model = None
    if environment["status"] in {"READY_CPU", "READY_GPU"}:
        try:
            p = context.prepared
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = build_candidate(candidate_id, config, device)
                started = perf_counter()
                model.fit(p.X_train_scaled, p.y_train)
                timings["fit_seconds"] = perf_counter() - started
                _assert_actual_backend(candidate_id, model, device)
                started = perf_counter()
                train_prob = models.validate_binary_probabilities(model.predict_proba(p.X_train_scaled), expected_rows=len(p.y_train), classes=model.classes_)
                timings["train_predict_seconds"] = perf_counter() - started
                train_metrics = models.calculate_prediction_metrics(p.y_train, (train_prob[:, 1] >= 0.5).astype(np.int64))
                del train_prob
                started = perf_counter()
                val_prob = models.validate_binary_probabilities(model.predict_proba(p.X_evaluation_scaled), expected_rows=len(p.y_evaluation), classes=model.classes_)
                timings["validation_predict_seconds"] = perf_counter() - started
                timings["predict_seconds"] = timings["train_predict_seconds"] + timings["validation_predict_seconds"]
                observed, frame = evaluate_candidate(p.y_evaluation, val_prob[:, 1], p.evaluation_record_keys, context.validation_calls)
                metrics.update(observed, train_accuracy=train_metrics["accuracy"],
                               train_val_gap=train_metrics["accuracy"] - observed["validation_accuracy"], status="SUCCESS")
                if hasattr(model, "n_iter_"):
                    metrics["n_iter_"] = int(model.n_iter_)
                if hasattr(model, "loss_"):
                    metrics["final_loss"] = float(model.loss_)
                metrics["input_dtype"] = str(p.X_train_scaled.dtype)
                metrics["dtype_conversion"] = "none; float32 preserved (Torch batches use float32)"
                frame.to_parquet(directory / "validation_predictions.parquet", index=False)
                if candidate_id == "numerical_embedding_mlp":
                    model.save(directory / "model.pt")
                reference_dir = root / "candidates/mlp_reference"
                if candidate_id != "mlp_reference" and (reference_dir / "metrics.json").is_file():
                    ref_metrics, reference = _validated_saved_result(reference_dir, context)
                    if reference is not None:
                        metrics["paired"] = paired_comparison(reference, frame)
                caught_messages = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        except Exception as error:
            metrics.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
            if "caught" in locals():
                caught_messages = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        finally:
            if model is not None and hasattr(model, "release"):
                model.release()
            del model
            gc.collect()
    else:
        metrics["reason"] = environment.get("reason")
    metrics["convergence_warning_raised"] = any(w["category"] == "ConvergenceWarning" for w in caught_messages)
    metrics.update(timings)
    _write_json_new(directory / "metrics.json", metrics)
    _write_json_new(directory / "timing.json", timings)
    _write_json_new(directory / "warnings.json", caught_messages)
    return metrics


def load_reference_artifact(source_directory, context, run_root):
    """Reuse only a complete matching screen-schema MLP reference artifact."""
    import shutil
    validate_screen_contract(context)
    _assert_run_contract(run_root, context)
    source = Path(source_directory)
    metrics, frame = _validated_saved_result(source, context)
    if metrics["candidate"] != "mlp_reference" or metrics["status"] != "SUCCESS":
        raise ValueError("A successful MLP reference prediction artifact is required")
    required = ["metrics.json", "config.json", "contract.json", "environment.json", "warnings.json", "timing.json", "validation_predictions.parquet"]
    if not all((source / name).is_file() for name in required):
        raise ValueError("Reference artifact incomplete")
    destination = Path(run_root) / "candidates/mlp_reference"
    destination.mkdir(exist_ok=False)
    for name in required:
        shutil.copyfile(source / name, destination / name)
    _write_json_new(destination / "reuse.json", {"source": str(source.resolve()), "action": "REUSED_STRICT_REFERENCE"})
    return metrics


def build_comparison(run_root, context):
    validate_screen_contract(context)
    root = Path(run_root)
    _assert_run_contract(root, context)
    results = {}
    for candidate in CANDIDATES:
        directory = root / "candidates" / candidate
        if (directory / "metrics.json").is_file():
            results[candidate] = _validated_saved_result(directory, context)
            if results[candidate][0]["candidate"] != candidate:
                raise ValueError("Candidate artifact ID/directory differs")
    reference, ref_frame = results.get("mlp_reference", ({}, None))
    rows = []
    for candidate, (metrics, frame) in results.items():
        row = dict(metrics)
        row["delta_vs_mlp300"] = None
        row["error_delta_vs_mlp300"] = None
        if ref_frame is not None and frame is not None:
            row["delta_vs_mlp300"] = metrics["validation_accuracy"] - reference["validation_accuracy"]
            row["error_delta_vs_mlp300"] = metrics["errors"] - reference["errors"]
            row.update(paired_comparison(ref_frame, frame))
        rows.append(row)
    columns = ["candidate", "model_family", "implementation", "device", "backend", "config_id",
               "train_accuracy", "validation_accuracy", "delta_vs_mlp300", "balanced_accuracy",
               "recall_0", "recall_1", "fp", "fn", "errors", "error_delta_vs_mlp300",
               "train_val_gap", "hce", "fit_seconds", "predict_seconds", "status",
               "both_correct", "both_wrong", "MLP_correct_candidate_wrong", "MLP_wrong_candidate_correct", "net_correction"]
    return pd.DataFrame(rows).reindex(columns=columns)


def save_comparison(run_root, context):
    table = build_comparison(run_root, context)
    with (Path(run_root) / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        table.to_csv(stream, index=False)
    return table
