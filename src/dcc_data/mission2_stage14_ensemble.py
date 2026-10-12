"""Stage 14: train joint F01+F02 members and compare fixed soft-voting ensembles.

All three members receive the same 149D Clean59 + F01(80D) + F02(10D)
matrix. The module reuses the fixed-split, feature-cache, model-recipe,
train-only scaler, fit, and metric contracts from the Mission 2 feature grid.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import mission2_acoustic_feature_grid as grid


MODEL_IDS = ("tabm", "tabular_resnet", "gandalf")
MEMBER_IDS = tuple(f"{model}_F01_F02" for model in MODEL_IDS)
FEATURE_IDS = ("F01", "F02")
INPUT_DIMENSION = 149
RUN_SUFFIX = "_stage14_joint_f01_f02"
RUN_SCHEMA = "mission2_stage14_joint_f01_f02_v1"
LEDGER_SCHEMA = "mission2_stage14_ensemble_ledger_v1"
ENSEMBLE_SPECS = {
    "E_TR": (("tabm_F01_F02", 0.5), ("tabular_resnet_F01_F02", 0.5)),
    "E_TG": (("tabm_F01_F02", 0.5), ("gandalf_F01_F02", 0.5)),
    "E_RG": (("tabular_resnet_F01_F02", 0.5), ("gandalf_F01_F02", 0.5)),
    "E_TRG": (("tabm_F01_F02", 1 / 3),
              ("tabular_resnet_F01_F02", 1 / 3), ("gandalf_F01_F02", 1 / 3)),
}


def build_joint_input(context):
    """Join verified Clean59, F01 and F02 blocks in their frozen column order."""
    clean = np.asarray(context.clean_X)
    if (clean.ndim != 2 or clean.shape != (len(context.y), 59)
            or clean.dtype != np.float32 or not np.isfinite(clean).all()):
        raise ValueError("Clean 59D matrix must be aligned float32 and finite")
    names = list(context.feature_names_59d)
    if len(names) != 59:
        raise ValueError("Clean 59D feature_names must contain exactly 59 columns")
    blocks = []
    for feature_id in FEATURE_IDS:
        block = grid.BLOCK_BY_ID[feature_id]
        values = context.feature_matrices.get(feature_id)
        if values is None:
            raise RuntimeError(f"Verified feature matrix is not loaded: {feature_id}")
        values = np.asarray(values)
        if (values.dtype != np.float32
                or values.shape != (len(context.y), block.dimension)
                or not np.isfinite(values).all()):
            raise ValueError(f"{feature_id} matrix shape/dtype/finiteness differs")
        blocks.append(values)
        names.extend(block.feature_names)
    if len(names) != INPUT_DIMENSION or len(set(names)) != len(names):
        raise ValueError("Joint 149D feature schema has a wrong dimension or duplicate names")
    matrix = np.concatenate((clean, *blocks), axis=1).astype(np.float32, copy=False)
    if matrix.shape != (len(context.y), INPUT_DIMENSION) or not np.isfinite(matrix).all():
        raise ValueError("Joint Clean59+F01+F02 input is invalid")
    return matrix, tuple(names)


def ensemble_candidate_specs():
    """Return every fixed equal-weight pair and the fixed three-member average."""
    return deepcopy(ENSEMBLE_SPECS)


def build_ensemble_predictions(member_predictions, ensemble_id):
    """Average aligned member probabilities and apply the fixed >=0.5 rule."""
    started = time.perf_counter()
    if ensemble_id not in ENSEMBLE_SPECS:
        raise KeyError(f"Unknown fixed Stage 14 ensemble: {ensemble_id}")
    spec = ENSEMBLE_SPECS[ensemble_id]
    first_id = spec[0][0]
    if first_id not in member_predictions:
        raise ValueError(f"Missing ensemble member predictions: {first_id}")
    reference = member_predictions[first_id]
    required = ("record_key", "call_id", "y_true", "probability_speaker1")
    if any(column not in reference for column in required):
        raise ValueError("Ensemble member predictions lack required identity/probability columns")
    keys = reference.record_key.astype(str).to_numpy()
    calls = reference.call_id.astype(str).to_numpy()
    labels = reference.y_true.to_numpy(dtype=np.int64)
    if len(set(keys)) != len(keys):
        raise ValueError("Ensemble prediction identity contains duplicate record keys")
    probability = np.zeros(len(reference), dtype=np.float64)
    for member_id, weight in spec:
        frame = member_predictions.get(member_id)
        if frame is None or any(column not in frame for column in required):
            raise ValueError(f"Missing/incomplete ensemble member predictions: {member_id}")
        for column, expected in (("record_key", keys), ("call_id", calls), ("y_true", labels)):
            if not np.array_equal(frame[column].astype(str).to_numpy(), expected.astype(str)):
                raise ValueError(f"Ensemble prediction identity differs: {member_id}/{column}")
        values = frame.probability_speaker1.to_numpy(dtype=np.float64)
        if (values.shape != probability.shape or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))):
            raise ValueError(f"Invalid member probabilities: {member_id}")
        probability += weight * values
    if not np.isclose(sum(weight for _, weight in spec), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Frozen ensemble weights must sum to one")
    metrics, predictions = grid._metric_bundle(labels, probability, keys, calls)
    metrics.update({
        "ensemble_id": ensemble_id,
        "members": [{"member_id": member, "weight": weight} for member, weight in spec],
        "weight_policy": "fixed_equal_weight_no_search",
        "probability_combination_seconds": time.perf_counter() - started,
    })
    return metrics, predictions


def load_joint_feature_blocks(source_root, context):
    """Load only hash-verified Training F01/F02 caches from an existing run."""
    from . import mission2_acoustic_feature_grid_compact as compact

    source = Path(source_root).resolve(strict=True)
    cache_dir = source / "feature_cache"
    manifest_path = cache_dir / "manifest.json"
    manifest = grid._read_json(manifest_path)
    if (manifest.get("schema_version") != "mission2_acoustic_feature_grid_cache_v1"
            or manifest.get("source_split") != "Training"
            or manifest.get("official_validation_used") is not False
            or manifest.get("record_key_signature") != grid._record_key_signature(context.record_keys)
            or manifest.get("call_id_hash") != grid._array_fingerprint(context.call_ids)
            or manifest.get("labels_hash") != grid._array_fingerprint(context.y)):
        raise ValueError("Source F01/F02 feature-cache manifest differs from fixed Training rows")
    matrices, hashes = {}, {}
    for feature_id in FEATURE_IDS:
        matrices[feature_id], hashes[feature_id] = compact._load_cached_block(
            source, context, feature_id, manifest,
        )
    context.feature_matrices = matrices
    return matrices, hashes, grid._sha256_file(manifest_path)


def _schema(context):
    return {
        "clean_59d": list(context.feature_names_59d),
        "F01": list(grid.BLOCK_BY_ID["F01"].feature_names),
        "F02": list(grid.BLOCK_BY_ID["F02"].feature_names),
    }


def _run_identity(context):
    return {
        "context_identity": grid._context_identity(context),
        "split": deepcopy(context.split_audit),
        "source_split": "Training",
        "evaluation_set": "fixed_internal_validation",
        "split_seed": grid.SEED,
        "group_key": "call_id",
        "threshold": grid.THRESHOLD,
        "official_validation_used": False,
    }


def prepare_stage14_run(context, source_run_root, output_root):
    """Create a separate Stage 14 run or validate its existing immutable state."""
    source = Path(source_run_root).resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    if output.name != grid.COLAB_OUTPUT_DIR_NAME or output.parent.name != "MyDrive":
        raise ValueError("Stage 14 output must be a direct child of MyDrive/mission2_grid_runs")
    if output == source or output.is_relative_to(source):
        raise ValueError("Stage 14 output must be separate from the read-only source run")
    matrices, feature_hashes, feature_manifest_sha = load_joint_feature_blocks(source, context)
    _, names = build_joint_input(context)
    run_id = f"{source.name}{RUN_SUFFIX}"
    root = output / run_id
    if root.exists():
        audit = validate_stage14_run(context, root, source_run_root=source)
        return root, audit

    configs = {member_id: grid._model_config(model_id, INPUT_DIMENSION)
               for member_id, model_id in zip(MEMBER_IDS, MODEL_IDS)}
    output.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    (root / "members").mkdir(exist_ok=False)
    (root / "ensembles").mkdir(exist_ok=False)
    (root / "fit_environments").mkdir(exist_ok=False)
    schema = _schema(context)
    cache_manifest = grid._read_json(source / "feature_cache" / "manifest.json")
    contract = {
        "schema_version": RUN_SCHEMA, "run_id": run_id,
        **_run_identity(context),
        "model_ids": list(MODEL_IDS), "member_ids": list(MEMBER_IDS),
        "input_dimension": INPUT_DIMENSION, "feature_schema": schema,
        "feature_names": list(names),
        "feature_schema_fingerprint": grid._json_fingerprint(schema),
        "source_run_root": str(source),
        "source_feature_manifest_sha256": feature_manifest_sha,
        "source_feature_cache": {
            feature_id: {
                "path": str(source / "feature_cache" / f"{feature_id}.npz"),
                "sha256": feature_hashes[feature_id],
                "feature_names": list(grid.BLOCK_BY_ID[feature_id].feature_names),
                "dimension": grid.BLOCK_BY_ID[feature_id].dimension,
            } for feature_id in FEATURE_IDS
        },
        "source_feature_manifest_schema": cache_manifest.get("schema_version"),
        "model_configs": configs,
        "ensemble_candidates": {
            name: [{"member_id": member, "weight": weight} for member, weight in spec]
            for name, spec in ENSEMBLE_SPECS.items()
        },
        "recipe_policy": "Stage13 recipe; input_dim only changed to 149",
        "scaler_fit_scope": "fixed_internal_train_only",
        "selection_bias_disclosure_required": True,
        "internal_validation_reuse_disclosure_required": True,
        "attempt_policy": {"sequential": True, "automatic_retry_or_overwrite": False},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    grid._write_json_new(root / "contract.json", contract)
    grid._write_json_new(root / "run_ledger.json", {
        "schema_version": LEDGER_SCHEMA, "run_id": run_id, "status": "READY",
        "member_order": list(MEMBER_IDS), "next_member": MEMBER_IDS[0],
        "members": {member: "PENDING" for member in MEMBER_IDS},
        "ensemble_candidates": {name: "PENDING" for name in ENSEMBLE_SPECS},
        "completed_member_count": 0,
    })
    grid._write_json_new(root / "environment.json", {
        "status": "NOT_CHECKED", "device": None, "official_validation_used": False,
    })
    audit = validate_stage14_run(context, root, source_run_root=source)
    audit.update({"feature_cache": "PASS", "joint_dimensions": INPUT_DIMENSION})
    return root, audit


def validate_stage14_run(context, run_root, *, source_run_root=None):
    """Validate frozen provenance, cached blocks, and all completed member artifacts."""
    root = Path(run_root).resolve(strict=True)
    contract = grid._read_json(root / "contract.json")
    ledger = grid._read_json(root / "run_ledger.json")
    expected_root_identity = _run_identity(context)
    if (contract.get("schema_version") != RUN_SCHEMA
            or contract.get("run_id") != root.name
            or contract.get("member_ids") != list(MEMBER_IDS)
            or contract.get("input_dimension") != INPUT_DIMENSION
            or contract.get("source_split") != "Training"
            or contract.get("evaluation_set") != "fixed_internal_validation"
            or contract.get("split_seed") != grid.SEED
            or contract.get("group_key") != "call_id"
            or contract.get("context_identity") != expected_root_identity["context_identity"]
            or contract.get("split") != context.split_audit
            or contract.get("official_validation_used") is not False
            or contract.get("threshold") != 0.5
            or ledger.get("schema_version") != LEDGER_SCHEMA
            or ledger.get("member_order") != list(MEMBER_IDS)):
        raise ValueError("Stage 14 fixed-split/input/threshold contract differs")
    source = Path(source_run_root or contract["source_run_root"]).resolve(strict=True)
    if str(source) != contract.get("source_run_root"):
        raise ValueError("Stage 14 source run path differs from frozen contract")
    matrices, hashes, manifest_sha = load_joint_feature_blocks(source, context)
    _, names = build_joint_input(context)
    if (manifest_sha != contract.get("source_feature_manifest_sha256")
            or tuple(contract.get("feature_names", ())) != names
            or contract.get("feature_schema_fingerprint") != grid._json_fingerprint(_schema(context))):
        raise ValueError("Stage 14 source cache/schema identity differs")
    for feature_id in FEATURE_IDS:
        if hashes[feature_id] != contract["source_feature_cache"][feature_id]["sha256"]:
            raise ValueError(f"Stage 14 source cache hash differs: {feature_id}")
    states = ledger.get("members", {})
    if set(states) != set(MEMBER_IDS):
        raise ValueError("Stage 14 member ledger does not match the frozen model set")
    completed = []
    for member_id in MEMBER_IDS:
        status = states[member_id]
        member_dir = root / "members" / member_id
        if status == "COMPLETE":
            verify_completed_member(context, root, member_id)
            completed.append(member_id)
        elif status in ("RUNNING", "FAILED") or member_dir.exists():
            raise RuntimeError(
                f"Stage 14 member {member_id} is {status} or has partial artifacts; "
                "preserve it and start a separately identified attempt"
            )
        elif status != "PENDING":
            raise ValueError(f"Unexpected member state: {member_id}={status}")
    pending = [member for member in MEMBER_IDS if states[member] == "PENDING"]
    if pending and pending[0] != ledger.get("next_member"):
        raise ValueError("Stage 14 next_member pointer differs from sequential ledger order")
    return {
        "status": ledger.get("status"), "run_root": str(root),
        "completed_members": completed, "completed_member_count": len(completed),
        "next_member": pending[0] if pending else None,
        "feature_cache_hashes": hashes,
    }


def record_environment(run_root, *, device="cuda"):
    """Record one explicitly selected Colab device; never switch devices implicitly."""
    import torch

    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported explicit training device: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; no CPU fallback is permitted")
    environment = {
        "status": "READY", "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cuda_build": torch.version.cuda if device == "cuda" else None,
        "python": __import__("sys").version.split()[0],
        "platform": __import__("platform").platform(),
        "packages": {
            package: __import__("importlib.metadata", fromlist=["version"]).version(package)
            for package in ("numpy", "pandas", "scikit-learn", "torch", "tabm",
                            "rtdl-revisiting-models", "rtdl-num-embeddings", "pytorch-tabular")
        },
        "official_validation_used": False,
    }
    grid._write_json_replace(Path(run_root) / "environment.json", environment)
    return environment


def validate_preflight(context, run_root):
    """Check 149D schema, feature identity, all recipes and Train-only scaler fit."""
    root = Path(run_root).resolve(strict=True)
    validate_stage14_run(context, root)
    matrix, names = build_joint_input(context)
    contract = grid._read_json(root / "contract.json")
    checks = []
    for member_id, model_id in zip(MEMBER_IDS, MODEL_IDS):
        config = contract["model_configs"].get(member_id)
        if (config != grid._model_config(model_id, INPUT_DIMENSION)
                or config.get("input_dim") != INPUT_DIMENSION
                or config.get("seed") != grid.SEED):
            raise ValueError(f"Stage 13 model recipe changed beyond input dimension: {member_id}")
        scaler, train_scaled, validation_scaled = grid._scaler_for_experiment(matrix, context)
        if (len(scaler.mean_) != INPUT_DIMENSION
                or int(scaler.n_samples_seen_) != len(context.train_idx)
                or not np.isfinite(train_scaled).all()
                or not np.isfinite(validation_scaled).all()):
            raise ValueError(f"Train-only scaler preflight failed: {member_id}")
        checks.append({"member_id": member_id, "input_dimension": INPUT_DIMENSION,
                       "schema_unique": len(names) == len(set(names)),
                       "finite": True, "scaler_fit_rows": int(scaler.n_samples_seen_),
                       "device_recipe_seed": config["seed"]})
        del scaler, train_scaled, validation_scaled
    return {"status": "PASS", "member_count": len(checks), "members": checks}


def _member_manifest(root, member_id):
    return grid._read_json(root / "members" / member_id / "artifact_manifest.json")


def verify_completed_member(context, run_root, member_id):
    root = Path(run_root).resolve(strict=True)
    if member_id not in MEMBER_IDS:
        raise KeyError(f"Unknown Stage 14 member: {member_id}")
    directory = root / "members" / member_id
    status = grid._read_json(directory / "status.json")
    contract = grid._read_json(root / "contract.json")
    config = contract["model_configs"][member_id]
    manifest = grid._verify_artifact_manifest(directory)
    if (status.get("status") != "COMPLETE"
            or manifest.get("experiment_id") != member_id
            or manifest.get("config_id") != config.get("config_id")
            or manifest.get("split_identity") != contract.get("context_identity")
            or manifest.get("feature_schema_fingerprint")
            != contract.get("feature_schema_fingerprint")
            or grid._read_json(directory / "config.json") != config):
        raise ValueError(f"Stage 14 member artifact config/status differs: {member_id}")
    metrics = grid._read_json(directory / "metrics.json")
    expected_env_hash = grid._sha256_file(directory / "environment.json")
    if (metrics.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS")
            or metrics.get("split_identity") != contract.get("context_identity")
            or metrics.get("environment_sha256") != expected_env_hash
            or metrics.get("input_dimension") != INPUT_DIMENSION
            or metrics.get("threshold") != 0.5):
        raise ValueError(f"Stage 14 member fit/reload/split evidence differs: {member_id}")
    if not grid._checkpoint_files_from_manifest(manifest.get("files", {})):
        raise ValueError(f"Stage 14 member checkpoint is missing: {member_id}")
    frame = pd.read_parquet(directory / "predictions.parquet")
    grid._prediction_identity(frame, context.validation_record_keys,
                              context.validation_call_ids, context.validation_labels)
    observed, reconstructed = grid._metric_bundle(
        context.validation_labels,
        frame.probability_speaker1.to_numpy(dtype=np.float64),
        context.validation_record_keys, context.validation_call_ids,
    )
    if not frame.equals(reconstructed):
        raise ValueError(f"Stage 14 member predictions do not reconstruct: {member_id}")
    if any(metrics.get(name) != value for name, value in observed.items()):
        raise ValueError(f"Stage 14 member metrics do not reconstruct: {member_id}")
    return {"status": "VERIFIED_COMPLETE", "member_id": member_id,
            "validation_accuracy": observed["validation_accuracy"]}


def _update_ledger(root, member_id, status):
    path = root / "run_ledger.json"
    ledger = grid._read_json(path)
    states = ledger["members"]
    states[member_id] = status
    completed = [name for name in MEMBER_IDS if states[name] == "COMPLETE"]
    pending = [name for name in MEMBER_IDS if states[name] == "PENDING"]
    ledger["completed_member_count"] = len(completed)
    ledger["next_member"] = pending[0] if pending else None
    ledger["status"] = ("FAILED_STOP" if status == "FAILED" else
                        "ALL_MEMBERS_COMPLETE" if len(completed) == len(MEMBER_IDS) else
                        "RUNNING" if status == "RUNNING" else "READY")
    grid._write_json_replace(path, ledger)


def fit_next_member(context, run_root, *, device="cuda"):
    """Fit exactly one next member; completed fits are verified, partial fits stop."""
    root = Path(run_root).resolve(strict=True)
    audit = validate_stage14_run(context, root)
    if audit["next_member"] is None:
        return {"status": "ALL_MEMBERS_COMPLETE", "member_id": None}
    member_id = audit["next_member"]
    contract = grid._read_json(root / "contract.json")
    environment = grid._read_json(root / "environment.json")
    if environment.get("status") != "READY" or environment.get("device") != device:
        raise RuntimeError("The explicitly selected device must be checked and recorded before fitting")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable; refusing an implicit CPU retry")
    matrix, names = build_joint_input(context)
    model_id = MODEL_IDS[MEMBER_IDS.index(member_id)]
    config = contract["model_configs"][member_id]
    scaler, x_train, x_validation = grid._scaler_for_experiment(matrix, context)
    y_train = context.y[context.train_idx]
    y_validation = context.y[context.validation_idx]
    directory = root / "members" / member_id
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    grid._write_json_new(directory / "config.json", config)
    grid._write_json_new(directory / "status.json", {
        "status": "RUNNING", "member_id": member_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "run_id": root.name,
    })
    grid._save_scaler(scaler, directory, names)
    grid._write_json_new(directory / "environment.json", environment)
    fit_env_path = root / "fit_environments" / f"{member_id}.json"
    grid._write_json_new(fit_env_path, environment)
    _update_ledger(root, member_id, "RUNNING")
    try:
        model, train_probability, validation_probability, history, fit = grid._fit_one_model(
            model_id, config, names, x_train, y_train, x_validation, y_validation,
            directory, device,
        )
        if fit.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS"):
            raise RuntimeError(f"Checkpoint reload did not pass for {member_id}")
        metrics, predictions = grid._metric_bundle(
            y_validation, validation_probability,
            context.validation_record_keys, context.validation_call_ids,
        )
        train_probability = np.asarray(train_probability, dtype=np.float64)
        if (train_probability.shape != y_train.shape or not np.isfinite(train_probability).all()
                or np.any((train_probability < 0) | (train_probability > 1))):
            raise ValueError(f"Invalid training probabilities: {member_id}")
        metrics.update({
            "member_id": member_id, "model_id": model_id,
            "input_dimension": INPUT_DIMENSION, "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
            "config_id": config["config_id"], "seed": grid.SEED,
            "train_accuracy": float(((train_probability >= 0.5) == y_train).mean()),
            "parameter_count": grid._parameter_count(model),
            "best_epoch": fit.get("best_epoch"), "epochs_run": fit.get("epochs_run"),
            "history_kind": fit.get("history_kind", "epoch_history"),
            "fit_seconds": float(fit["fit_seconds"]),
            "wall_seconds": time.perf_counter() - started,
            "checkpoint_reload": fit["checkpoint_reload"],
            "feature_names": list(names),
            "feature_cache_sha256": {
                name: contract["source_feature_cache"][name]["sha256"] for name in FEATURE_IDS
            },
            "scaler_fit_scope": "fixed_internal_train_only",
            "split_identity": deepcopy(contract["context_identity"]),
            "environment_sha256": grid._sha256_file(directory / "environment.json"),
            "official_validation_used": False,
        })
        history.to_csv(directory / "history.csv", index=False, mode="x")
        predictions.to_parquet(directory / "predictions.parquet", index=False)
        grid._write_json_new(directory / "metrics.json", metrics)
        artifact_hashes = {
            path.relative_to(directory).as_posix(): grid._sha256_file(path)
            for path in grid._run_artifact_files(directory)
        }
        grid._write_json_new(directory / "artifact_manifest.json", {
            "schema_version": "mission2_stage14_member_artifacts_v1",
            "experiment_id": member_id, "config_id": config["config_id"],
            "split_identity": contract["context_identity"],
            "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
            "files": artifact_hashes,
        })
        grid._write_experiment_status(directory / "status.json", "COMPLETE",
                                      checkpoint_reload=fit["checkpoint_reload"])
        _update_ledger(root, member_id, "COMPLETE")
        result = verify_completed_member(context, root, member_id)
        return {**result, "fit_seconds": metrics["fit_seconds"],
                "wall_seconds": metrics["wall_seconds"]}
    except Exception as error:
        grid._write_experiment_status(directory / "status.json", "FAILED",
                                      error_type=type(error).__name__, error=str(error))
        _update_ledger(root, member_id, "FAILED")
        raise
    finally:
        del scaler, x_train, x_validation, matrix


def run_remaining_member_fits(context, run_root, *, device="cuda"):
    """Run the remaining members in frozen order, skipping only verified COMPLETE fits."""
    results = []
    while True:
        audit = validate_stage14_run(context, run_root)
        if audit["next_member"] is None:
            break
        result = fit_next_member(context, run_root, device=device)
        results.append(result)
        print(f"COMPLETE {result['member_id']} accuracy={result['validation_accuracy']:.8f} "
              f"fit_seconds={result['fit_seconds']:.1f}", flush=True)
    return {"status": "ALL_MEMBERS_COMPLETE", "completed_member_count": len(MEMBER_IDS),
            "results": results}


def _load_member_predictions(context, root):
    frames = {}
    for member_id in MEMBER_IDS:
        verify_completed_member(context, root, member_id)
        frame = pd.read_parquet(root / "members" / member_id / "predictions.parquet")
        frames[member_id] = frame
    return frames


def _duration_overlap_slices(context, predictions):
    """Use manifest-only duration/overlap for error analysis, never model inputs."""
    from . import mission2_model_family_screen as screen
    from . import mission2_standard_representation_comparison as diagnostics

    validation_manifest = context.manifest.iloc[context.validation_idx].reset_index(drop=True).copy()
    duration = ((pd.to_numeric(validation_manifest.endAt_ms, errors="raise").to_numpy(dtype=np.float64)
                 - pd.to_numeric(validation_manifest.startAt_ms, errors="raise").to_numpy(dtype=np.float64))
                / 1000.0)
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("Validation duration slice metadata is invalid")
    overlap = screen.models._overlap_mask(validation_manifest)
    for name, expected in (("record_key", context.validation_record_keys),
                           ("call_id", context.validation_call_ids),
                           ("speaker", context.validation_labels)):
        if not np.array_equal(validation_manifest[name].astype(str).to_numpy(),
                              np.asarray(expected).astype(str)):
            raise ValueError(f"Validation slice metadata alignment differs: {name}")
    masks = diagnostics.slice_masks(pd.DataFrame({
        "duration_sec": duration, "overlap": overlap,
        "y_true": context.validation_labels,
    }))
    rows = []
    for candidate_id, frame in predictions.items():
        correct = frame.correct.to_numpy(dtype=bool)
        for slice_name, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            rows.append({"candidate_id": candidate_id, "slice": slice_name,
                         "rows": int(mask.sum()),
                         "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return pd.DataFrame(rows)


def _call_bootstrap_accuracy(context, prediction_frames, *, replicates=2000):
    """Estimate row-accuracy uncertainty by resampling whole call_id groups."""
    validation_calls = np.asarray(context.validation_call_ids).astype(str)
    unique_calls, inverse = np.unique(validation_calls, return_inverse=True)
    if len(unique_calls) < 2:
        raise ValueError("Call-level bootstrap requires at least two validation calls")
    rng = np.random.default_rng(grid.SEED)
    rows = []
    correct_vectors = {name: frame.correct.to_numpy(dtype=np.float64)
                       for name, frame in prediction_frames.items()}
    call_sizes = np.bincount(inverse, minlength=len(unique_calls)).astype(np.float64)
    call_correct = {
        name: np.bincount(inverse, weights=values, minlength=len(unique_calls))
        for name, values in correct_vectors.items()
    }
    sampled_calls = rng.integers(0, len(unique_calls), size=(replicates, len(unique_calls)))
    denom = call_sizes[sampled_calls].sum(axis=1)
    for name, values in call_correct.items():
        estimates = values[sampled_calls].sum(axis=1) / denom
        rows.append({"candidate_id": name, "point_accuracy": float(correct_vectors[name].mean()),
                     "call_bootstrap_replicates": replicates,
                     "accuracy_ci95_low": float(np.quantile(estimates, 0.025)),
                     "accuracy_ci95_high": float(np.quantile(estimates, 0.975)),
                     "call_groups": int(len(unique_calls))})
    return pd.DataFrame(rows)


def evaluate_ensembles(context, run_root, *, bootstrap_replicates=2000):
    """Evaluate all frozen ensembles and write predictions, paired errors and slices."""
    root = Path(run_root).resolve(strict=True)
    audit = validate_stage14_run(context, root)
    if audit["next_member"] is not None:
        raise RuntimeError("All three verified member fits are required before ensemble evaluation")
    ledger = grid._read_json(root / "run_ledger.json")
    if ledger.get("status") == "ENSEMBLES_COMPLETE":
        return _load_finished_analysis(root)
    if ledger.get("status") not in ("ALL_MEMBERS_COMPLETE", "ENSEMBLES_COMPLETE"):
        raise RuntimeError(f"Member fit ledger is not complete: {ledger.get('status')}")
    members = _load_member_predictions(context, root)
    all_predictions = dict(members)
    ensemble_metrics = {}
    for ensemble_id in ENSEMBLE_SPECS:
        directory = root / "ensembles" / ensemble_id
        state = ledger["ensemble_candidates"].get(ensemble_id)
        if state == "COMPLETE":
            verify_completed_ensemble(context, root, ensemble_id, members=members)
            frame = pd.read_parquet(directory / "predictions.parquet")
            ensemble_metrics[ensemble_id] = grid._read_json(directory / "metrics.json")
            all_predictions[ensemble_id] = frame
            continue
        if state != "PENDING" or directory.exists():
            raise RuntimeError(f"Ensemble {ensemble_id} is {state} or partial; no overwrite/retry")
        directory.mkdir(exist_ok=False)
        grid._write_json_new(directory / "status.json", {
            "status": "RUNNING", "ensemble_id": ensemble_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        try:
            metrics, frame = build_ensemble_predictions(members, ensemble_id)
            metrics.update({"official_validation_used": False,
                            "split_identity": grid._context_identity(context),
                            "feature_schema_fingerprint": grid._read_json(root / "contract.json")
                            ["feature_schema_fingerprint"]})
            frame.to_parquet(directory / "predictions.parquet", index=False)
            grid._write_json_new(directory / "metrics.json", metrics)
            grid._write_json_new(directory / "weights.json", {
                "ensemble_id": ensemble_id,
                "members": [{"member_id": member, "weight": weight}
                            for member, weight in ENSEMBLE_SPECS[ensemble_id]],
                "policy": "fixed equal probability average; no weight/threshold search",
            })
            artifact_hashes = {
                path.relative_to(directory).as_posix(): grid._sha256_file(path)
                for path in grid._run_artifact_files(directory)
            }
            grid._write_json_new(directory / "artifact_manifest.json", {
                "schema_version": "mission2_stage14_ensemble_artifacts_v1",
                "ensemble_id": ensemble_id,
                "files": artifact_hashes,
            })
            grid._write_experiment_status(directory / "status.json", "COMPLETE")
            _set_ensemble_status(root, ensemble_id, "COMPLETE")
            verify_completed_ensemble(context, root, ensemble_id, members=members)
            ensemble_metrics[ensemble_id] = metrics
            all_predictions[ensemble_id] = frame
        except Exception as error:
            grid._write_experiment_status(directory / "status.json", "FAILED",
                                          error_type=type(error).__name__, error=str(error))
            _set_ensemble_status(root, ensemble_id, "FAILED")
            raise

    correlations = _probability_correlations(all_predictions)
    paired_rows = []
    for ensemble_id in ENSEMBLE_SPECS:
        candidate = all_predictions[ensemble_id]
        for member_id, _ in ENSEMBLE_SPECS[ensemble_id]:
            paired, _, _ = grid._paired_counts(all_predictions[member_id], candidate)
            paired_rows.append({"ensemble_id": ensemble_id, "reference_member": member_id, **paired})
    paired_frame = pd.DataFrame(paired_rows)
    metrics_frame = pd.DataFrame([
        {"candidate_id": member_id, **grid._read_json(root / "members" / member_id / "metrics.json")}
        for member_id in MEMBER_IDS
    ] + [{"candidate_id": name, **ensemble_metrics[name]} for name in ENSEMBLE_SPECS])
    slices = _duration_overlap_slices(context, all_predictions)
    bootstrap = _call_bootstrap_accuracy(context, all_predictions,
                                         replicates=bootstrap_replicates)
    (root / "analysis").mkdir(exist_ok=True)
    metrics_frame.to_csv(root / "analysis" / "metrics_summary.csv", index=False)
    paired_frame.to_csv(root / "analysis" / "paired_error_transitions.csv", index=False)
    correlations.to_csv(root / "analysis" / "probability_correlations.csv")
    slices.to_csv(root / "analysis" / "duration_overlap_slices.csv", index=False)
    bootstrap.to_csv(root / "analysis" / "call_bootstrap_accuracy.csv", index=False)
    _write_report(root, metrics_frame, paired_frame, correlations, slices, bootstrap)
    _set_ensemble_batch_complete(root)
    _write_stage14_manifest(root)
    return {"status": "ENSEMBLES_COMPLETE", "metrics": metrics_frame,
            "paired_errors": paired_frame, "probability_correlations": correlations,
            "duration_overlap_slices": slices, "call_bootstrap": bootstrap,
            "report_path": root / "STAGE14_REPORT_KO.md"}


def _analysis_paths(root):
    return {
        "metrics": root / "analysis" / "metrics_summary.csv",
        "paired": root / "analysis" / "paired_error_transitions.csv",
        "correlations": root / "analysis" / "probability_correlations.csv",
        "slices": root / "analysis" / "duration_overlap_slices.csv",
        "bootstrap": root / "analysis" / "call_bootstrap_accuracy.csv",
        "report": root / "STAGE14_REPORT_KO.md",
    }


def _write_stage14_manifest(root):
    excluded = {"stage14_manifest.json"}
    files = {
        path.relative_to(root).as_posix(): grid._sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
        and "__pycache__" not in path.parts
    }
    if not files:
        raise ValueError("Stage 14 final artifact manifest cannot be empty")
    grid._write_json_new(root / "stage14_manifest.json", {
        "schema_version": "mission2_stage14_final_manifest_v1",
        "run_id": root.name, "files": files,
        "source_feature_cache": grid._read_json(root / "contract.json")["source_feature_cache"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })


def _verify_stage14_manifest(root):
    manifest_path = root / "stage14_manifest.json"
    manifest = grid._read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Stage 14 final manifest is missing file hashes")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path and "__pycache__" not in path.parts
    }
    if set(actual) != set(files):
        raise ValueError("Stage 14 final artifact inventory differs from manifest")
    for relative, path in actual.items():
        if grid._sha256_file(path) != files[relative]:
            raise ValueError(f"Stage 14 final artifact hash differs: {relative}")
    return manifest


def _load_finished_analysis(root):
    _verify_stage14_manifest(root)
    paths = _analysis_paths(root)
    return {
        "status": "ENSEMBLES_COMPLETE",
        "metrics": pd.read_csv(paths["metrics"]),
        "paired_errors": pd.read_csv(paths["paired"]),
        "probability_correlations": pd.read_csv(paths["correlations"], index_col=0),
        "duration_overlap_slices": pd.read_csv(paths["slices"]),
        "call_bootstrap": pd.read_csv(paths["bootstrap"]),
        "report_path": paths["report"],
    }


def _probability_correlations(frames):
    names = list(frames)
    matrix = np.column_stack([frames[name].probability_speaker1.to_numpy(dtype=np.float64)
                              for name in names])
    corr = np.corrcoef(matrix, rowvar=False)
    corr[~np.isfinite(corr)] = 1.0
    return pd.DataFrame(corr, index=names, columns=names)


def _set_ensemble_status(root, ensemble_id, status):
    path = root / "run_ledger.json"
    ledger = grid._read_json(path)
    ledger["ensemble_candidates"][ensemble_id] = status
    ledger["status"] = "FAILED_STOP" if status == "FAILED" else "RUNNING"
    grid._write_json_replace(path, ledger)


def _set_ensemble_batch_complete(root):
    path = root / "run_ledger.json"
    ledger = grid._read_json(path)
    if any(value != "COMPLETE" for value in ledger["ensemble_candidates"].values()):
        raise RuntimeError("Ensemble ledger cannot complete with pending/failed candidates")
    ledger["status"] = "ENSEMBLES_COMPLETE"
    ledger["next_member"] = None
    grid._write_json_replace(path, ledger)


def verify_completed_ensemble(context, run_root, ensemble_id, *, members=None):
    root = Path(run_root).resolve(strict=True)
    if ensemble_id not in ENSEMBLE_SPECS:
        raise KeyError(f"Unknown Stage 14 ensemble: {ensemble_id}")
    directory = root / "ensembles" / ensemble_id
    if grid._read_json(directory / "status.json").get("status") != "COMPLETE":
        raise RuntimeError(f"Stage 14 ensemble is not COMPLETE: {ensemble_id}")
    manifest = grid._verify_artifact_manifest(directory)
    if manifest.get("ensemble_id") != ensemble_id:
        raise ValueError(f"Stage 14 ensemble artifact manifest differs: {ensemble_id}")
    metrics = grid._read_json(directory / "metrics.json")
    weights = grid._read_json(directory / "weights.json")
    expected = [{"member_id": member, "weight": weight}
                for member, weight in ENSEMBLE_SPECS[ensemble_id]]
    if weights.get("members") != expected or metrics.get("members") != expected:
        raise ValueError(f"Stage 14 fixed ensemble weights differ: {ensemble_id}")
    frame = pd.read_parquet(directory / "predictions.parquet")
    grid._prediction_identity(frame, context.validation_record_keys,
                              context.validation_call_ids, context.validation_labels)
    observed, reconstructed = grid._metric_bundle(
        context.validation_labels, frame.probability_speaker1.to_numpy(dtype=np.float64),
        context.validation_record_keys, context.validation_call_ids,
    )
    if not frame.equals(reconstructed) or any(metrics.get(key) != value
                                               for key, value in observed.items()):
        raise ValueError(f"Stage 14 ensemble predictions/metrics do not reconstruct: {ensemble_id}")
    if members is not None:
        expected_metrics, expected_frame = build_ensemble_predictions(members, ensemble_id)
        if not frame.equals(expected_frame) or metrics.get("members") != expected_metrics.get("members"):
            raise ValueError(f"Stage 14 ensemble does not equal fixed member average: {ensemble_id}")
    return {"status": "VERIFIED_COMPLETE", "ensemble_id": ensemble_id,
            "validation_accuracy": observed["validation_accuracy"]}


def _write_report(root, metrics, paired, correlations, slices, bootstrap):
    report = [
        "# Mission 2 Stage 14 — F01+F02 고정 앙상블 결과",
        "",
        "## 실험 계약",
        "",
        "- 세 멤버 모두 Clean 59D + F01 80D + F02 10D, 총 149D를 입력받았다.",
        "- 기존 고정 Internal Split / seed 42 / call_id 그룹 분리와 Train-only StandardScaler를 사용했다.",
        "- 임계값은 0.5로 고정했고 가중치 탐색은 하지 않았다. Official Validation은 사용하지 않았다.",
        "- 여러 과거 실험에서 같은 Internal Validation을 반복 사용했고, 후보 조합 비교도 같은 결과를 기준으로 하므로 선택 편향 가능성이 있다.",
        "- 아래 결과는 내부 후보 비교 자료이며 자동으로 최종 모델을 확정하지 않는다.",
        "",
        "## 후보 지표",
        "",
        _markdown_table(metrics[["candidate_id", "validation_accuracy", "recall_0", "recall_1", "hce"]]),
        "",
        "## paired 교정·회귀",
        "",
        _markdown_table(paired),
        "",
        "## 확률 상관",
        "",
        _markdown_table(correlations.reset_index(names="candidate_id")),
        "",
        "## Duration / overlap slice",
        "",
        _markdown_table(slices),
        "",
        "## call_id 단위 bootstrap 95% 구간",
        "",
        _markdown_table(bootstrap),
        "",
        "체크포인트·Scaler·예측·지표·설정·해시는 각 `members/` 및 `ensembles/` 아티팩트에 기록했다. 최종 후보 선택과 이후 Official Validation 사용은 기존 로드맵 및 사용자 검토에 따른다.",
    ]
    (root / "STAGE14_REPORT_KO.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _markdown_table(frame):
    """Render a small report table without depending on optional tabulate."""
    values = frame.copy().astype(object)
    values = values.where(pd.notna(values), "")
    headers = [str(name).replace("|", "\\|") for name in values.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row]
            for row in values.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
