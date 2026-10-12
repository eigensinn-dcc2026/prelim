"""Small, resumable 3-model x 3-feature Mission 2 grid for the Colab handoff.

The seven-block/40-run implementation remains the historical source of truth.
This module reuses its fixed-split, cache, recipe, scaler, fit, and metric helpers
while writing a separate compact-run contract and artifact tree.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import mission2_acoustic_feature_grid as grid


MODEL_IDS = ("tabm", "tabular_resnet", "gandalf")
FEATURE_IDS = ("F01", "F02", "F08")
F08_NAMES = (
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
)
RUN_SUFFIX = "_compact3x3"
RUN_SCHEMA = "mission2_acoustic_feature_grid_compact_v1"
LEDGER_SCHEMA = "mission2_acoustic_feature_grid_compact_ledger_v1"
COMPLETED_STATUS = "ALL_9_COMPLETE_WITH_3_BASELINES_REUSED"


def candidate_experiment_ids() -> tuple[str, ...]:
    """Keep a deterministic model-major order for one-at-a-time fitting."""
    return tuple(f"{model}_{feature}" for model in MODEL_IDS for feature in FEATURE_IDS)


def _validate_device(device, *, cuda_available):
    """Require an explicitly selected device; never fall back implicitly."""
    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported training device: {device!r}")
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly selected but is unavailable")
    return device


def _candidate_environment_path(run_root, experiment_id):
    return Path(run_root) / "fit_environments" / f"{experiment_id}.json"


def select_voice_quality_matrix(matrix, names, record_keys, expected_keys):
    """Select the exact 6D jitter/shimmer/HNR schema without changing row order."""
    values = np.asarray(matrix)
    feature_names = tuple(str(name) for name in names)
    keys = np.asarray(record_keys).astype(str)
    expected = np.asarray(expected_keys).astype(str)
    if (values.ndim != 2 or values.shape[1] != len(feature_names)
            or len(keys) != len(values) or not np.array_equal(keys, expected)):
        raise ValueError("eGeMAPS record_key order or matrix schema differs")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("eGeMAPS feature_names contain duplicates")
    missing = [name for name in F08_NAMES if name not in feature_names]
    if missing:
        raise ValueError(f"F08 eGeMAPS feature_names are missing: {missing}")
    positions = [feature_names.index(name) for name in F08_NAMES]
    selected = np.asarray(values[:, positions], dtype=np.float32)
    if selected.shape != (len(expected), 6) or not np.isfinite(selected).all():
        raise ValueError("F08 voice-quality matrix has an invalid shape or NaN/Inf")
    return selected


def _feature_schema():
    return {
        "F01": list(grid.BLOCK_BY_ID["F01"].feature_names),
        "F02": list(grid.BLOCK_BY_ID["F02"].feature_names),
        "F08": list(F08_NAMES),
    }


def _load_cached_block(source_root, context, block_id, source_manifest):
    cache_dir = Path(source_root) / "feature_cache"
    block = grid.BLOCK_BY_ID[block_id]
    path = cache_dir / f"{block_id}.npz"
    audit = source_manifest.get("feature_blocks", {}).get(block_id, {})
    if not path.is_file() or grid._sha256_file(path) != audit.get("artifact_sha256"):
        raise ValueError(f"Existing cached feature artifact is missing/corrupt: {block_id}")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"X", "record_keys", "call_ids", "feature_names"}:
            raise ValueError(f"Existing feature cache schema differs: {block_id}")
        values = saved["X"]
        keys = saved["record_keys"].astype(str)
        calls = saved["call_ids"].astype(str)
        names = tuple(saved["feature_names"].tolist())
    if (names != block.feature_names or values.dtype != np.float32
            or values.shape != (len(context.y), block.dimension)
            or not np.array_equal(keys, context.record_keys)
            or not np.array_equal(calls, context.call_ids)
            or not np.isfinite(values).all()):
        raise ValueError(f"Cached feature rows/schema/values differ: {block_id}")
    return np.asarray(values), grid._sha256_file(path)


def _load_compact_matrices(source_root, context, egemaps_cache_path):
    source_cache = Path(source_root) / "feature_cache"
    manifest_path = source_cache / "manifest.json"
    source_manifest = grid._read_json(manifest_path)
    if (source_manifest.get("schema_version") != "mission2_acoustic_feature_grid_cache_v1"
            or source_manifest.get("source_split") != "Training"
            or source_manifest.get("official_validation_used") is not False
            or source_manifest.get("record_key_signature") != grid._record_key_signature(context.record_keys)
            or source_manifest.get("call_id_hash") != grid._array_fingerprint(context.call_ids)
            or source_manifest.get("labels_hash") != grid._array_fingerprint(context.y)):
        raise ValueError("Source feature-cache manifest differs from fixed Training rows")

    matrices, source_hashes = {}, {}
    for block_id in ("F01", "F02"):
        matrices[block_id], source_hashes[block_id] = _load_cached_block(
            source_root, context, block_id, source_manifest,
        )

    from . import mission2_standard_representation as standard

    cache_path = Path(egemaps_cache_path).resolve(strict=True)
    if grid._sha256_file(cache_path) != context.egemaps_audit.get("cache_sha256"):
        raise ValueError("88D eGeMAPS cache hash differs from the loaded Training context")
    egemaps = standard.load_egemaps_cache(
        cache_path, context.record_keys, context.call_ids,
        source_provenance=context.clean_fingerprint,
    )
    matrices["F08"] = select_voice_quality_matrix(
        egemaps.X, egemaps.feature_names, egemaps.record_keys, context.record_keys,
    )
    source_hashes["F08"] = grid._sha256_file(cache_path)
    del egemaps
    return matrices, source_hashes, grid._sha256_file(manifest_path)


def _input_for(context, experiment_id):
    model_id, feature_id = experiment_id.rsplit("_F", 1)
    feature_id = "F" + feature_id
    if model_id not in MODEL_IDS or feature_id not in FEATURE_IDS:
        raise ValueError(f"Experiment is outside the compact grid: {experiment_id}")
    block_names = tuple(_feature_schema()[feature_id])
    values = context.feature_matrices.get(feature_id)
    if values is None:
        raise RuntimeError(f"Verified compact feature matrix is not loaded: {feature_id}")
    expected_dim = {"F01": 80, "F02": 10, "F08": 6}[feature_id]
    if (values.shape != (len(context.y), expected_dim) or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError(f"Compact feature matrix shape/dtype/finiteness differs: {feature_id}")
    names = (*context.feature_names_59d, *block_names)
    X = np.concatenate((context.clean_X, values), axis=1).astype(np.float32, copy=False)
    if X.shape != (len(context.y), 59 + expected_dim) or len(set(names)) != len(names):
        raise ValueError(f"Compact model input schema differs: {experiment_id}")
    return model_id, feature_id, X, tuple(names)


def _baseline_inventory(context, source_root):
    source = Path(source_root).resolve(strict=True)
    contract_path, ledger_path = source / "contract.json", source / "run_ledger.json"
    if not contract_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("Source Run contract or ledger is missing")
    contract, ledger = grid._read_json(contract_path), grid._read_json(ledger_path)
    if (contract.get("source_split") != "Training"
            or contract.get("evaluation_set") != "fixed_internal_validation"
            or contract.get("official_validation_used") is not False
            or contract.get("context_identity") != grid._context_identity(context)
            or contract.get("split") != context.split_audit):
        raise ValueError("Source baseline run differs from the loaded fixed Training split")
    environment_path = source / "environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError("Source baseline environment record is missing")
    baselines = {}
    for model_id in MODEL_IDS:
        experiment_id = f"baseline_{model_id}"
        if ledger.get("experiments", {}).get(experiment_id) != "COMPLETE":
            raise RuntimeError(f"Source baseline is not COMPLETE: {experiment_id}")
        grid.verify_completed_experiment(context, source, experiment_id)
        directory = source / experiment_id
        manifest = grid._verify_artifact_manifest(directory)
        checkpoints = grid._checkpoint_files_from_manifest(manifest["files"])
        if not checkpoints:
            raise ValueError(f"Baseline weights are missing: {experiment_id}")
        baselines[experiment_id] = {
            "source_directory": str(directory),
            "artifact_manifest_sha256": grid._sha256_file(directory / "artifact_manifest.json"),
            "artifact_files": manifest["files"],
            "checkpoint_files": checkpoints,
        }
    return {
        "source_run_root": str(source),
        "source_contract_sha256": grid._sha256_file(contract_path),
        "source_ledger_sha256": grid._sha256_file(ledger_path),
        "source_environment_sha256": grid._sha256_file(environment_path),
        "baselines": baselines,
    }


def _save_feature_cache(root, context, matrices, source_hashes, source_manifest_sha):
    cache_dir = root / "feature_cache"
    cache_dir.mkdir(exist_ok=False)
    blocks = {}
    for feature_id in FEATURE_IDS:
        values = np.asarray(matrices[feature_id], dtype=np.float32)
        names = tuple(_feature_schema()[feature_id])
        if values.shape != (len(context.y), len(names)) or not np.isfinite(values).all():
            raise ValueError(f"Feature cache preflight failed: {feature_id}")
        path = cache_dir / f"{feature_id}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, X=values, record_keys=context.record_keys,
                call_ids=context.call_ids, feature_names=np.asarray(names, dtype=str),
            )
        blocks[feature_id] = {
            "block_id": feature_id, "rows": int(len(values)), "dimension": int(values.shape[1]),
            "dtype": str(values.dtype), "non_finite_values": 0,
            "record_key_signature": grid._record_key_signature(context.record_keys),
            "feature_names": list(names), "source_cache_sha256": source_hashes[feature_id],
            "artifact_sha256": grid._sha256_file(path),
        }
    manifest = {
        "schema_version": "mission2_acoustic_feature_grid_compact_cache_v1",
        "source_split": "Training", "official_validation_used": False,
        "clean_fingerprint": context.clean_fingerprint,
        "record_key_signature": grid._record_key_signature(context.record_keys),
        "call_id_hash": grid._array_fingerprint(context.call_ids),
        "labels_hash": grid._array_fingerprint(context.y),
        "source_feature_manifest_sha256": source_manifest_sha,
        "feature_blocks": blocks,
    }
    grid._write_json_new(cache_dir / "manifest.json", manifest)
    grid._write_json_new(cache_dir / "status.json", {
        "status": "COMPLETE", "rows": len(context.y),
        "manifest_sha256": grid._sha256_file(cache_dir / "manifest.json"),
    })
    return manifest


def _load_feature_cache(root, context):
    cache_dir = Path(root) / "feature_cache"
    manifest = grid._read_json(cache_dir / "manifest.json")
    status = grid._read_json(cache_dir / "status.json")
    if (status.get("status") != "COMPLETE"
            or status.get("manifest_sha256") != grid._sha256_file(cache_dir / "manifest.json")
            or manifest.get("schema_version") != "mission2_acoustic_feature_grid_compact_cache_v1"
            or manifest.get("source_split") != "Training"
            or manifest.get("official_validation_used") is not False
            or manifest.get("record_key_signature") != grid._record_key_signature(context.record_keys)
            or manifest.get("call_id_hash") != grid._array_fingerprint(context.call_ids)
            or manifest.get("labels_hash") != grid._array_fingerprint(context.y)):
        raise ValueError("Compact feature cache completion/source identity differs")
    matrices = {}
    for feature_id in FEATURE_IDS:
        entry = manifest.get("feature_blocks", {}).get(feature_id, {})
        path = cache_dir / f"{feature_id}.npz"
        if not path.is_file() or grid._sha256_file(path) != entry.get("artifact_sha256"):
            raise ValueError(f"Compact feature cache artifact missing/corrupt: {feature_id}")
        with np.load(path, allow_pickle=False) as saved:
            values = saved["X"]
            keys = saved["record_keys"].astype(str)
            calls = saved["call_ids"].astype(str)
            names = tuple(saved["feature_names"].tolist())
        if (names != tuple(_feature_schema()[feature_id])
                or values.dtype != np.float32
                or not np.array_equal(keys, context.record_keys)
                or not np.array_equal(calls, context.call_ids)
                or not np.isfinite(values).all()):
            raise ValueError(f"Compact feature-cache identity/schema differs: {feature_id}")
        matrices[feature_id] = values
    context.feature_matrices = matrices
    return matrices, manifest


def _model_configs():
    configs = {}
    for model_id in MODEL_IDS:
        configs[f"baseline_{model_id}"] = grid._model_config(model_id, 59)
        for feature_id, width in (("F01", 80), ("F02", 10), ("F08", 6)):
            configs[f"{model_id}_{feature_id}"] = grid._model_config(model_id, 59 + width)
    return configs


def _verify_reused_baselines(context, root, contract, reuse):
    baseline_root = Path(root) / "baseline_reuse"
    copied_environment = baseline_root / "source_environment.json"
    if grid._sha256_file(copied_environment) != reuse.get("source_environment_sha256"):
        raise ValueError("Copied baseline environment hash differs")
    for model_id in MODEL_IDS:
        experiment_id = f"baseline_{model_id}"
        reference = reuse.get("baselines", {}).get(experiment_id)
        directory = baseline_root / experiment_id
        if (reference is None
                or grid._sha256_file(directory / "artifact_manifest.json")
                != reference.get("artifact_manifest_sha256")):
            raise ValueError(f"Reused baseline manifest differs: {experiment_id}")
        artifact_manifest = grid._verify_artifact_manifest(directory)
        if artifact_manifest.get("files") != reference.get("artifact_files"):
            raise ValueError(f"Reused baseline artifacts differ: {experiment_id}")
        if not grid._checkpoint_files_from_manifest(artifact_manifest["files"]):
            raise ValueError(f"Reused baseline checkpoint is missing: {experiment_id}")
        metrics = grid._read_json(directory / "metrics.json")
        config = grid._read_json(directory / "config.json")
        scaler = grid._read_json(directory / "scaler.json")
        predictions = pd.read_parquet(directory / "predictions.parquet")
        if config != contract["model_configs"].get(experiment_id):
            raise ValueError(f"Reused baseline recipe differs: {experiment_id}")
        if (scaler.get("fit_scope") != "fixed_internal_train_only"
                or scaler.get("training_rows") != len(context.train_idx)
                or len(scaler.get("mean", [])) != 59
                or len(scaler.get("scale", [])) != 59
                or len(scaler.get("var", [])) != 59):
            raise ValueError(f"Reused baseline scaler differs: {experiment_id}")
        grid._prediction_identity(
            predictions, context.validation_record_keys,
            context.validation_call_ids, context.validation_labels,
        )
        observed, reconstructed = grid._metric_bundle(
            context.validation_labels,
            predictions.probability_speaker1.to_numpy(dtype=np.float64),
            context.validation_record_keys, context.validation_call_ids,
        )
        if not predictions.equals(reconstructed):
            raise ValueError(f"Reused baseline predictions do not reconstruct: {experiment_id}")
        if any(metrics.get(name) != value for name, value in observed.items()):
            raise ValueError(f"Reused baseline metrics do not reconstruct: {experiment_id}")
        if (metrics.get("split_identity") != contract.get("context_identity")
                or metrics.get("environment_sha256") != reuse.get("source_environment_sha256")
                or metrics.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS")):
            raise ValueError(f"Reused baseline split/checkpoint evidence differs: {experiment_id}")


def validate_compact_run(context, run_root):
    """Verify the compact run and refuse partial or changed artifacts."""
    root = Path(run_root).resolve(strict=True)
    contract_path, ledger_path = root / "contract.json", root / "run_ledger.json"
    if not contract_path.is_file() or not ledger_path.is_file():
        raise RuntimeError(f"Compact run is partial (contract/ledger missing): {root}")
    contract, ledger = grid._read_json(contract_path), grid._read_json(ledger_path)
    candidates = candidate_experiment_ids()
    if (contract.get("schema_version") != RUN_SCHEMA
            or contract.get("candidate_experiments") != list(candidates)
            or contract.get("models") != list(MODEL_IDS)
            or contract.get("features") != list(FEATURE_IDS)
            or contract.get("context_identity") != grid._context_identity(context)
            or contract.get("split") != context.split_audit
            or contract.get("official_validation_used") is not False
            or contract.get("run_id") != root.name
            or contract.get("output_root") != str(root.parent)):
        raise ValueError("Compact run contract differs from model/feature/split scope")
    expected_ids = {
        *(f"baseline_{model}" for model in MODEL_IDS), *candidates,
    }
    if (ledger.get("schema_version") != LEDGER_SCHEMA
            or set(ledger.get("experiments", {})) != expected_ids
            or any(ledger["experiments"].get(f"baseline_{model}") != "REUSED_VERIFIED"
                   for model in MODEL_IDS)):
        raise ValueError("Compact run ledger differs from its 3 baseline + 9 candidate contract")
    completed = []
    pending = []
    for experiment_id in candidates:
        status = ledger["experiments"].get(experiment_id)
        directory = root / experiment_id
        if status == "COMPLETE":
            verify_completed_candidate(context, root, experiment_id)
            completed.append(experiment_id)
        elif status == "PENDING" and not directory.exists():
            pending.append(experiment_id)
        else:
            raise RuntimeError(
                f"Compact candidate is failed or partial; do not retry/overwrite: {experiment_id}={status}"
            )
    if ledger.get("completed_count") != len(completed):
        raise ValueError("Compact ledger completed_count differs from verified artifacts")
    reuse = grid._read_json(root / "baseline_reuse_manifest.json")
    if (grid._sha256_file(root / "baseline_reuse_manifest.json")
            != contract.get("baseline_reuse_manifest_sha256")
            or reuse.get("source_run_root") != contract.get("baseline_source_run_root")
            or reuse.get("source_contract_sha256") != contract.get("baseline_source_contract_sha256")):
        raise ValueError("Compact reused-baseline provenance differs")
    _verify_reused_baselines(context, root, contract, reuse)
    matrices, cache_manifest = _load_feature_cache(root, context)
    context.feature_matrices = matrices
    if (grid._sha256_file(root / "feature_cache" / "manifest.json")
            != contract.get("feature_cache_manifest_sha256")
            or cache_manifest.get("feature_blocks") is None):
        raise ValueError("Compact feature-cache hash differs from contract")
    if ledger.get("status") in ("FAILED_STOP", "RUNNING"):
        raise RuntimeError(f"Compact run cannot resume from ledger status {ledger['status']}")
    return {
        "status": ledger.get("status"), "run_root": str(root),
        "completed_count": len(completed), "candidate_count": len(candidates),
        "next_experiment": pending[0] if pending else None,
    }


def prepare_compact_run(context, source_run_root, output_root, egemaps_cache_path):
    """Reuse three verified baselines and the existing F01/F02 + 88D caches."""
    source = Path(source_run_root).resolve(strict=True)
    base = Path(output_root).expanduser().resolve()
    if base.name != grid.COLAB_OUTPUT_DIR_NAME or base.parent.name != "MyDrive":
        raise ValueError("Compact output must be a direct child of MyDrive/mission2_grid_runs")
    if base == source or base.is_relative_to(source):
        raise ValueError("Compact output must remain separate from the read-only source Run")

    baseline_inventory = _baseline_inventory(context, source)
    matrices, source_feature_hashes, source_feature_manifest_sha = _load_compact_matrices(
        source, context, egemaps_cache_path,
    )
    schema = _feature_schema()
    overlap = set(schema["F08"]) & set(grid.F04_NAMES + grid.F05_NAMES)
    if overlap:
        raise ValueError(f"F08 duplicates the existing eGeMAPS F04/F05 schema: {sorted(overlap)}")

    run_id = f"{source.name}{RUN_SUFFIX}"
    root = base / run_id
    if root.exists():
        result = validate_compact_run(context, root)
        contract = grid._read_json(root / "contract.json")
        if (contract.get("baseline_source_run_root") != str(source)
                or contract.get("source_feature_manifest_sha256") != source_feature_manifest_sha):
            raise ValueError("Existing compact run belongs to a different source/cache")
        return root, {**result, "baseline_reuse": "PASS", "feature_cache": "PASS"}

    configs = _model_configs()
    base.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    reuse_root = root / "baseline_reuse"
    reuse_root.mkdir(exist_ok=False)
    try:
        copied = {}
        for experiment_id, reference in baseline_inventory["baselines"].items():
            target = reuse_root / experiment_id
            shutil.copytree(reference["source_directory"], target)
            copied_manifest = grid._verify_artifact_manifest(target)
            if (copied_manifest.get("files") != reference["artifact_files"]
                    or grid._sha256_file(target / "artifact_manifest.json")
                    != reference["artifact_manifest_sha256"]):
                raise ValueError(f"Copied baseline changed during transfer: {experiment_id}")
            copied[experiment_id] = reference
        shutil.copy2(source / "environment.json", reuse_root / "source_environment.json")

        feature_manifest = _save_feature_cache(
            root, context, matrices, source_feature_hashes, source_feature_manifest_sha,
        )
        reuse_manifest = {
            "source_run_root": str(source),
            "source_contract_sha256": baseline_inventory["source_contract_sha256"],
            "source_ledger_sha256": baseline_inventory["source_ledger_sha256"],
            "source_environment_sha256": baseline_inventory["source_environment_sha256"],
            "copied_environment_sha256": grid._sha256_file(reuse_root / "source_environment.json"),
            "baselines": copied,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        grid._write_json_new(root / "baseline_reuse_manifest.json", reuse_manifest)
        context_identity = grid._context_identity(context)
        contract = {
            "schema_version": RUN_SCHEMA, "run_id": run_id,
            "source_run_id": source.name, "source_split": "Training",
            "evaluation_set": "fixed_internal_validation", "split_seed": grid.SEED,
            "group_key": "call_id", "threshold": grid.THRESHOLD,
            "official_validation_used": False,
            "split": deepcopy(context.split_audit), "context_identity": context_identity,
            "models": list(MODEL_IDS), "features": list(FEATURE_IDS),
            "feature_schema": schema,
            "feature_schema_fingerprint": grid._json_fingerprint(schema),
            "candidate_experiments": list(candidate_experiment_ids()),
            "baseline_experiments": [f"baseline_{model}" for model in MODEL_IDS],
            "model_configs": configs,
            "baseline_source_run_root": str(source),
            "baseline_source_contract_sha256": baseline_inventory["source_contract_sha256"],
            "baseline_reuse_manifest_sha256": grid._sha256_file(root / "baseline_reuse_manifest.json"),
            "feature_cache_source_run_root": str(source),
            "source_feature_manifest_sha256": source_feature_manifest_sha,
            "feature_cache_manifest_sha256": grid._sha256_file(root / "feature_cache" / "manifest.json"),
            "feature_cache_manifest": feature_manifest,
            "output_root": str(base),
            "attempt_policy": {"sequential": True, "automatic_retry_or_overwrite": False,
                               "device_required": "cuda", "cpu_fallback": False},
            "comparison_policy": {
                "primary": "reused_same_model_59d_baseline_vs_single_feature_candidate_on_exact_rows",
                "secondary": "Stage10_to_13_predictions_after_exact_key_call_label_alignment",
                "internal_validation_reuse_disclosed": True,
                "three_feature_selection_bias_disclosed": True,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        grid._write_json_new(root / "contract.json", contract)
        grid._write_json_new(root / "run_ledger.json", {
            "schema_version": LEDGER_SCHEMA, "run_id": run_id, "status": "READY",
            "next_experiment": candidate_experiment_ids()[0],
            "experiments": {
                **{f"baseline_{model}": "REUSED_VERIFIED" for model in MODEL_IDS},
                **{name: "PENDING" for name in candidate_experiment_ids()},
            },
            "completed_count": 0, "candidate_count": len(candidate_experiment_ids()),
            "reused_baseline_count": len(MODEL_IDS),
        })
        grid._write_json_new(root / "environment.json", {
            "status": "NOT_CHECKED", "device_required": "cuda", "cpu_fallback": False,
        })
    except Exception:
        # A partial directory is evidence of an interrupted prepare and is never overwritten.
        raise
    context.feature_matrices = matrices
    result = validate_compact_run(context, root)
    return root, {**result, "baseline_reuse": "PASS", "feature_cache": "PASS"}


def prepare_cpu_resume_run(context, prior_run_root, output_root):
    """Start a separate CPU attempt, reusing only hash-verified finished fits.

    The prior run is read-only. Any RUNNING/partial candidate remains there as
    evidence; the CPU attempt gets a fresh directory for that candidate.
    """
    prior = Path(prior_run_root).resolve(strict=True)
    base = Path(output_root).resolve(strict=True)
    prior_contract = grid._read_json(prior / "contract.json")
    prior_ledger = grid._read_json(prior / "run_ledger.json")
    candidates = candidate_experiment_ids()
    if (prior_contract.get("schema_version") != RUN_SCHEMA
            or prior_contract.get("candidate_experiments") != list(candidates)
            or prior_contract.get("context_identity") != grid._context_identity(context)
            or prior_contract.get("split") != context.split_audit
            or prior_contract.get("official_validation_used") is not False
            or prior_ledger.get("schema_version") != LEDGER_SCHEMA
            or set(prior_ledger.get("experiments", {})) != {
                *(f"baseline_{model}" for model in MODEL_IDS), *candidates,
            }):
        raise ValueError("Prior compact run differs from this fixed-split 3x3 experiment")
    if not (prior / "environment.json").is_file():
        raise FileNotFoundError("Prior compact run environment record is missing")

    completed = []
    prior_states = {}
    for experiment_id in candidates:
        status = prior_ledger["experiments"].get(experiment_id)
        prior_states[experiment_id] = status
        candidate_dir = prior / experiment_id
        if status == "COMPLETE":
            verify_completed_candidate(context, prior, experiment_id)
            completed.append(experiment_id)
        elif status in ("PENDING", "RUNNING", "FAILED"):
            # Do not repair, delete, or reuse any incomplete directory.
            continue
        else:
            raise RuntimeError(f"Unexpected prior candidate state: {experiment_id}={status}")
    if not completed:
        raise RuntimeError("No completed candidate artifacts are available to reuse")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / f"{prior.name}_cpu_resume_{stamp}"
    root.mkdir(exist_ok=False)
    shutil.copytree(prior / "baseline_reuse", root / "baseline_reuse")
    shutil.copy2(prior / "baseline_reuse_manifest.json", root / "baseline_reuse_manifest.json")
    shutil.copytree(prior / "feature_cache", root / "feature_cache")
    fit_environment_root = root / "fit_environments"
    fit_environment_root.mkdir(exist_ok=False)

    for experiment_id in completed:
        shutil.copytree(prior / experiment_id, root / experiment_id)
        prior_fit_environment = _candidate_environment_path(prior, experiment_id)
        if not prior_fit_environment.is_file():
            prior_fit_environment = prior / "environment.json"
        shutil.copy2(prior_fit_environment, _candidate_environment_path(root, experiment_id))

    contract = deepcopy(prior_contract)
    contract["run_id"] = root.name
    contract["output_root"] = str(base)
    contract["resume_of"] = str(prior)
    contract["attempt_policy"] = {
        "sequential": True, "automatic_retry_or_overwrite": False,
        "device_required": "cpu", "cpu_fallback": False,
        "partial_parent_attempt_preserved": True,
    }
    grid._write_json_new(root / "contract.json", contract)
    experiments = {
        **{f"baseline_{model}": "REUSED_VERIFIED" for model in MODEL_IDS},
        **{name: ("COMPLETE" if name in completed else "PENDING") for name in candidates},
    }
    next_experiment = next((name for name in candidates if name not in completed), None)
    grid._write_json_new(root / "run_ledger.json", {
        "schema_version": LEDGER_SCHEMA, "run_id": root.name, "status": "READY",
        "next_experiment": next_experiment, "experiments": experiments,
        "completed_count": len(completed), "candidate_count": len(candidates),
        "reused_baseline_count": len(MODEL_IDS),
        "reused_completed_candidate_count": len(completed),
    })
    grid._write_json_new(root / "environment.json", {
        "status": "NOT_CHECKED", "device_required": "cpu", "cpu_fallback": False,
    })
    grid._write_json_new(root / "cpu_resume_amendment.json", {
        "schema_version": "mission2_grid_cpu_resume_amendment_v1",
        "parent_run_root": str(prior), "parent_run_root_preserved": True,
        "selected_device": "cpu", "reused_verified_candidates": completed,
        "prior_candidate_states": prior_states,
        "official_validation_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })

    context.feature_matrices, _ = _load_feature_cache(root, context)
    result = validate_compact_run(context, root)
    return root, {**result, "resume_parent": str(prior),
                  "reused_candidates": completed, "baseline_reuse": "PASS",
                  "feature_cache": "PASS", "partial_parent_preserved": True}


def record_environment(run_root, *, device="cuda"):
    """Record the explicitly selected Colab device without implicit fallback."""
    import torch

    _validate_device(device, cuda_available=torch.cuda.is_available())
    if device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Explicit CPU mode requires a CPU-only Colab runtime")
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


def validate_candidate_preflight(context, run_root):
    """Validate all nine dimensions and Train-only scaler fits before training."""
    root = Path(run_root).resolve()
    contract = grid._read_json(root / "contract.json")
    if (contract.get("schema_version") != RUN_SCHEMA
            or contract.get("candidate_experiments") != list(candidate_experiment_ids())
            or contract.get("official_validation_used") is not False):
        raise ValueError("Preflight requires the frozen compact 9-fit Training-only contract")
    matrices, _ = _load_feature_cache(root, context)
    checks = []
    for experiment_id in candidate_experiment_ids():
        model_id, feature_id, X, names = _input_for(context, experiment_id)
        config = contract["model_configs"][experiment_id]
        if (config.get("input_dim") != X.shape[1] or config.get("seed") != grid.SEED
                or len(names) != len(set(names))):
            raise ValueError(f"Frozen model input/recipe differs: {experiment_id}")
        scaler, X_train, X_validation = grid._scaler_for_experiment(X, context)
        if (int(scaler.n_samples_seen_) != len(context.train_idx)
                or not np.isfinite(X_train).all() or not np.isfinite(X_validation).all()):
            raise ValueError(f"Train-only scaler preflight failed: {experiment_id}")
        checks.append({"experiment_id": experiment_id, "model": model_id,
                       "feature": feature_id, "input_dimension": int(X.shape[1]),
                       "finite": True, "train_only_scaler": True})
    return {"status": "PASS", "candidate_count": 9, "features": checks}


def verify_completed_candidate(context, run_root, experiment_id):
    root = Path(run_root).resolve()
    if experiment_id not in candidate_experiment_ids():
        raise ValueError(f"Not a compact Feature Grid candidate: {experiment_id}")
    directory = root / experiment_id
    if grid._read_json(directory / "status.json").get("status") != "COMPLETE":
        raise RuntimeError(f"Candidate is not COMPLETE: {experiment_id}")
    manifest = grid._verify_artifact_manifest(directory)
    contract = grid._read_json(root / "contract.json")
    config = contract["model_configs"][experiment_id]
    if (manifest.get("experiment_id") != experiment_id
            or manifest.get("config_id") != config["config_id"]
            or manifest.get("split_identity") != contract.get("context_identity")
            or manifest.get("feature_schema_fingerprint") != contract.get("feature_schema_fingerprint")
            or grid._read_json(directory / "config.json") != config):
        raise ValueError(f"Candidate artifact config/split/schema differs: {experiment_id}")
    metrics = grid._read_json(directory / "metrics.json")
    environment_path = _candidate_environment_path(root, experiment_id)
    if not environment_path.is_file():
        environment_path = root / "environment.json"
    if (metrics.get("status") != "COMPLETE"
            or metrics.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS")
            or metrics.get("split_identity") != contract.get("context_identity")
            or metrics.get("environment_sha256") != grid._sha256_file(environment_path)):
        raise ValueError(f"Candidate metrics/checkpoint evidence differs: {experiment_id}")
    feature_id = experiment_id.rsplit("_F", 1)[1]
    feature_id = "F" + feature_id
    expected_feature_hash = grid._read_json(root / "feature_cache" / "manifest.json")
    expected_feature_hash = expected_feature_hash["feature_blocks"][feature_id]["artifact_sha256"]
    if metrics.get("feature_cache_sha256") != expected_feature_hash:
        raise ValueError(f"Candidate feature-cache hash differs: {experiment_id}")
    scaler = grid._read_json(directory / "scaler.json")
    if (scaler.get("fit_scope") != "fixed_internal_train_only"
            or scaler.get("training_rows") != len(context.train_idx)
            or scaler.get("feature_names") != metrics.get("feature_names")
            or len(scaler.get("mean", [])) != metrics.get("input_dimension")
            or len(scaler.get("scale", [])) != metrics.get("input_dimension")
            or len(scaler.get("var", [])) != metrics.get("input_dimension")):
        raise ValueError(f"Candidate scaler scope/schema differs: {experiment_id}")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    grid._prediction_identity(predictions, context.validation_record_keys,
                              context.validation_call_ids, context.validation_labels)
    observed, reconstructed = grid._metric_bundle(
        context.validation_labels,
        predictions.probability_speaker1.to_numpy(dtype=np.float64),
        context.validation_record_keys, context.validation_call_ids,
    )
    if not predictions.equals(reconstructed) or any(
            metrics.get(name) != value for name, value in observed.items()):
        raise ValueError(f"Candidate prediction/metrics reconstruction differs: {experiment_id}")
    return {"status": "VERIFIED_COMPLETE", "experiment_id": experiment_id,
            "validation_accuracy": observed["validation_accuracy"]}


def _set_candidate_status(root, experiment_id, status):
    ledger_path = Path(root) / "run_ledger.json"
    ledger = grid._read_json(ledger_path)
    current = ledger["experiments"].get(experiment_id)
    allowed_transition = (
        status == "RUNNING" and current == "PENDING"
    ) or (
        status in ("COMPLETE", "FAILED") and current == "RUNNING"
    )
    if not allowed_transition:
        raise RuntimeError(
            f"Invalid compact candidate state transition {experiment_id}: {current} -> {status}"
        )
    ledger["experiments"][experiment_id] = status
    completed = [name for name in candidate_experiment_ids()
                 if ledger["experiments"].get(name) == "COMPLETE"]
    pending = [name for name in candidate_experiment_ids()
               if ledger["experiments"].get(name) == "PENDING"]
    ledger["completed_count"] = len(completed)
    ledger["next_experiment"] = pending[0] if pending else None
    if status == "FAILED":
        ledger["status"] = "FAILED_STOP"
    elif len(completed) == len(candidate_experiment_ids()):
        ledger["status"] = COMPLETED_STATUS
    else:
        ledger["status"] = "RUNNING" if status == "RUNNING" else "READY"
    grid._write_json_replace(ledger_path, ledger)


def _fit_candidate(context, root, experiment_id):
    contract = grid._read_json(root / "contract.json")
    run_environment_path = root / "environment.json"
    environment = grid._read_json(run_environment_path)
    import torch

    device = _validate_device(environment.get("device"),
                             cuda_available=torch.cuda.is_available())
    if environment.get("status") != "READY":
        raise RuntimeError("A recorded live environment is required before fitting")
    if device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Explicit CPU fit requires a CPU-only Colab runtime")
    environment_path = _candidate_environment_path(root, experiment_id)
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    if not environment_path.is_file():
        grid._write_json_new(environment_path, environment)
    model_id, feature_id, X, names = _input_for(context, experiment_id)
    config = contract["model_configs"][experiment_id]
    if config.get("input_dim") != X.shape[1]:
        raise ValueError("Frozen recipe input width differs from feature matrix")
    scaler, X_train, X_validation = grid._scaler_for_experiment(X, context)
    y_train = context.y[context.train_idx]
    y_validation = context.y[context.validation_idx]
    directory = root / experiment_id
    directory.mkdir(exist_ok=False)
    grid._write_json_new(directory / "config.json", config)
    grid._write_json_new(directory / "status.json", {
        "status": "RUNNING", "experiment_id": experiment_id,
        "run_id": root.name, "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    _set_candidate_status(root, experiment_id, "RUNNING")
    grid._save_scaler(scaler, directory, names)
    model = None
    try:
        model, train_p, validation_p, history, fit = grid._fit_one_model(
            model_id, config, names, X_train, y_train, X_validation,
            y_validation, directory, device,
        )
        metrics, predictions = grid._metric_bundle(
            y_validation, validation_p, context.validation_record_keys,
            context.validation_call_ids,
        )
        if not np.array_equal(predictions.y_true.to_numpy(), y_validation):
            raise ValueError("Predicted row labels differ from fixed Internal Validation")
        train_accuracy = float(((train_p >= grid.THRESHOLD) == y_train).mean())
        metrics.update({
            "experiment_id": experiment_id, "model_id": model_id, "block_id": feature_id,
            "input_dimension": int(X.shape[1]), "config_id": config["config_id"],
            "seed": grid.SEED, "train_accuracy": train_accuracy,
            "train_validation_gap": train_accuracy - metrics["validation_accuracy"],
            "parameter_count": grid._parameter_count(model),
            "best_epoch": fit.get("best_epoch"), "epochs_run": fit.get("epochs_run"),
            "history_kind": fit.get("history_kind", "epoch_history"),
            "fit_seconds": float(fit["fit_seconds"]),
            "checkpoint_reload": fit["checkpoint_reload"],
            "feature_names": list(names), "scaler_fit_scope": "fixed_internal_train_only",
            "feature_cache_sha256": grid._read_json(root / "feature_cache" / "manifest.json")
            ["feature_blocks"][feature_id]["artifact_sha256"],
            "split_identity": deepcopy(contract["context_identity"]),
            "environment_sha256": grid._sha256_file(environment_path),
        })
        history.to_csv(directory / "history.csv", index=False, mode="x")
        with (directory / "predictions.parquet").open("xb") as stream:
            predictions.to_parquet(stream, index=False)
        grid._write_json_new(directory / "metrics.json", metrics)
        hashes = {
            path.relative_to(directory).as_posix(): grid._sha256_file(path)
            for path in grid._run_artifact_files(directory)
        }
        grid._write_json_new(directory / "artifact_manifest.json", {
            "schema_version": "mission2_grid_run_artifacts_v1",
            "experiment_id": experiment_id, "config_id": config["config_id"],
            "split_identity": contract["context_identity"],
            "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
            "files": hashes,
        })
        grid._write_experiment_status(directory / "status.json", "COMPLETE",
                                      experiment_id=experiment_id,
                                      completed_utc=datetime.now(timezone.utc).isoformat(),
                                      checkpoint_reload=fit["checkpoint_reload"])
        _set_candidate_status(root, experiment_id, "COMPLETE")
        return metrics
    except Exception as error:
        grid._write_experiment_status(directory / "status.json", "FAILED",
                                      experiment_id=experiment_id,
                                      reason=f"{type(error).__name__}: {error}")
        _set_candidate_status(root, experiment_id, "FAILED")
        raise
    finally:
        del model
        import gc
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_compact_fits(context, run_root, *, device="cuda"):
    """Fit candidates serially, verifying each Drive artifact before advancing."""
    import torch

    _validate_device(device, cuda_available=torch.cuda.is_available())
    if device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Explicit CPU mode requires a CPU-only Colab runtime")
    root = Path(run_root).resolve(strict=True)
    current = validate_compact_run(context, root)
    environment = grid._read_json(root / "environment.json")
    if environment.get("status") != "READY" or environment.get("device") != device:
        raise RuntimeError("Record and validate the explicitly selected device before fitting")
    preflight = validate_candidate_preflight(context, root)
    results = []
    for index, experiment_id in enumerate(candidate_experiment_ids(), start=1):
        ledger = grid._read_json(root / "run_ledger.json")
        status = ledger["experiments"].get(experiment_id)
        if status == "COMPLETE":
            result = verify_completed_candidate(context, root, experiment_id)
            results.append(result)
            print(f"[{index}/9] VERIFIED_EXISTING {experiment_id}", flush=True)
            continue
        if status != "PENDING" or (root / experiment_id).exists():
            raise RuntimeError(
                f"Stopped without retry/overwrite: {experiment_id} is {status} or has partial artifacts"
            )
        result = _fit_candidate(context, root, experiment_id)
        verified = verify_completed_candidate(context, root, experiment_id)
        results.append(verified)
        print(f"[{index}/9] COMPLETE {experiment_id} "
              f"accuracy={result['validation_accuracy']:.8f} "
              f"fit_seconds={result['fit_seconds']:.1f} artifact=VERIFIED_COMPLETE", flush=True)
    final_ledger = grid._read_json(root / "run_ledger.json")
    if final_ledger.get("status") != COMPLETED_STATUS:
        raise RuntimeError("Compact candidate batch ended before all nine artifacts were complete")
    return {"status": COMPLETED_STATUS, "completed_count": 9,
            "candidate_count": 9, "preflight": preflight, "results": results}
