from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dcc_data.mission2_model_experiments"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "mission2_model_experiments.ipynb"

EXPECTED_MFCC40 = tuple(
    [f"mfcc_{index:02d}_mean" for index in range(1, 21)]
    + [f"mfcc_{index:02d}_std" for index in range(1, 21)]
)
EXPECTED_E05 = EXPECTED_MFCC40 + ("spectral_rolloff", "zcr")
EXPECTED_M01_CONFIG = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}
EXPECTED_M02_CONFIG = {
    "n_estimators": 128,
    "criterion": "gini",
    "max_depth": 18,
    "min_samples_split": 2,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "bootstrap": False,
    "class_weight": None,
    "n_jobs": -1,
    "random_state": 42,
}
EXPECTED_M03_CONFIG = {
    "priors": None,
    "reg_param": 0.0,
    "store_covariance": False,
    "tol": 0.0001,
}
EXPECTED_M04_CONFIG = {
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 1024,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_iter": 30,
    "shuffle": True,
    "random_state": 42,
    "tol": 0.0001,
    "early_stopping": False,
    "n_iter_no_change": 10,
}
EXPECTED_M04_ITER60_CONFIG = {**EXPECTED_M04_CONFIG, "max_iter": 60}


def _module_exists() -> bool:
    return importlib.util.find_spec(MODULE_NAME) is not None


@pytest.fixture
def model_module():
    if not _module_exists():
        pytest.skip("production module has not been implemented yet")
    return importlib.import_module(MODULE_NAME)


def _synthetic_e05(n_rows: int = 80):
    row = np.arange(n_rows, dtype=np.float32)[:, None]
    column = np.arange(42, dtype=np.float32)[None, :]
    X = np.sin(row / 5.0 + column / 7.0).astype(np.float32)
    y = ((np.arange(n_rows) // 2) % 2).astype(np.int64)
    record_keys = np.asarray([f"Training/call_{i // 4:03d}/{i:04d}" for i in range(n_rows)])
    call_ids = np.asarray([f"call_{i // 4:03d}" for i in range(n_rows)])
    return X, y, record_keys, call_ids


def _group_split_adapter(call_ids, *, seed: int, test_size: float):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    positions = np.arange(len(call_ids), dtype=np.int64)
    train_idx, validation_idx = next(splitter.split(positions, groups=call_ids))
    train_calls = set(np.asarray(call_ids)[train_idx].tolist())
    validation_calls = set(np.asarray(call_ids)[validation_idx].tolist())
    audit = {
        "seed": seed,
        "test_size": test_size,
        "total_rows": len(call_ids),
        "train_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "train_calls": len(train_calls),
        "validation_calls": len(validation_calls),
        "call_overlap": len(train_calls & validation_calls),
    }
    return train_idx, validation_idx, audit


def _prepared(model_module):
    X, y, record_keys, call_ids = _synthetic_e05()
    return model_module.prepare_internal_comparison(
        X=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        feature_names=EXPECTED_E05,
        split_function=_group_split_adapter,
    )


def _prepared_screening(model_module):
    n_rows = 240
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_rows, 42)).astype(np.float32)
    y = ((np.arange(n_rows) // 2) % 2).astype(np.int64)
    X[:, 0] += y.astype(np.float32) * np.float32(1.5)
    X[:, 1] += (1 - y).astype(np.float32) * np.float32(0.75)
    record_keys = np.asarray(
        [f"Training/screen_call_{i // 4:03d}/{i:04d}" for i in range(n_rows)]
    )
    call_ids = np.asarray([f"screen_call_{i // 4:03d}" for i in range(n_rows)])
    return model_module.prepare_internal_comparison(
        X=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        feature_names=EXPECTED_E05,
        split_function=_group_split_adapter,
    )


def _prepared_official_screening(model_module):
    rng = np.random.default_rng(84)
    training_X = rng.normal(size=(200, 42)).astype(np.float32)
    validation_X = rng.normal(size=(40, 42)).astype(np.float32)
    training_y = ((np.arange(200) // 2) % 2).astype(np.int64)
    validation_y = ((np.arange(40) // 2) % 2).astype(np.int64)
    training_X[:, 0] += training_y.astype(np.float32) * np.float32(1.5)
    validation_X[:, 0] += validation_y.astype(np.float32) * np.float32(1.5)
    return model_module.prepare_official_comparison(
        training_X=training_X,
        training_y=training_y,
        training_record_keys=np.asarray(
            [f"Training/screen/{index:04d}" for index in range(200)]
        ),
        validation_X=validation_X,
        validation_y=validation_y,
        validation_record_keys=np.asarray(
            [f"Validation/screen/{index:04d}" for index in range(40)]
        ),
        feature_names=EXPECTED_E05,
        user_approved=True,
        expected_training_rows=200,
        expected_validation_rows=40,
    )


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _literal_record_key_signature(record_keys) -> str:
    digest = hashlib.sha256()
    for key in sorted(map(str, record_keys)):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_standalone_cache_fixture(tmp_path: Path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    mfcc_npz = cache_root / "training_mfcc.npz"
    mfcc_metadata = cache_root / "training_mfcc.json"
    candidate_parquet = cache_root / "training_candidate.parquet"
    candidate_metadata = cache_root / "training_candidate.json"
    manifest_parquet = cache_root / "utterance_manifest.parquet"
    manifest_metadata = cache_root / "utterance_manifest_meta.json"

    X_mfcc = np.arange(6 * 40, dtype=np.float32).reshape(6, 40) / np.float32(10)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    call_ids = np.array(["call_a"] * 3 + ["call_b"] * 3)
    annotation_rows = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    record_keys = np.array(
        [
            f"Training|{call_id}|{annotation_row}"
            for call_id, annotation_row in zip(call_ids, annotation_rows, strict=True)
        ]
    )
    start_at = np.arange(6, dtype=np.float64) * 100.0
    end_at = start_at + 50.0
    duration = end_at - start_at
    rolloff = np.array([100, 200, 300, 400, 500, 600], dtype=np.float64)
    zcr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)

    np.savez(
        mfcc_npz,
        X=X_mfcc,
        y=y,
        stem=call_ids,
        start_at=start_at,
        end_at=end_at,
        duration=duration,
    )
    mfcc_metadata.write_text(
        json.dumps(
            {
                "cache_schema_version": "mission2_mfcc_mean_std_v1",
                "feature_dimension": 40,
                "mfcc_config": {
                    "hop_length": 160,
                    "n_fft": 512,
                    "n_mfcc": 20,
                    "sample_rate": 8000,
                },
                "source_call_count": 2,
                "source_split": "Training",
                "utterance_count": 6,
            }
        ),
        encoding="utf-8",
    )

    candidate = pd.DataFrame(
        {
            "record_key": record_keys,
            "call_id": call_ids,
            "zcr": zcr,
            "spectral_centroid": rolloff / 2,
            "spectral_rolloff": rolloff,
            "spectral_flatness": zcr / 2,
            "silence_ratio": zcr / 3,
        }
    ).iloc[[4, 1, 5, 0, 3, 2]]
    candidate.to_parquet(candidate_parquet, index=False)
    source_signature = "synthetic-source-signature"
    candidate_metadata.write_text(
        json.dumps(
            {
                "cache_schema_version": "mission2_candidate_audio_features_v1",
                "extraction_config": {
                    "clipping_amplitude": 0.999,
                    "hop_length": 160,
                    "n_fft": 512,
                    "sample_rate": 8000,
                    "silence_threshold_dbfs": -40.0,
                },
                "feature_count": 5,
                "feature_names": [
                    "zcr",
                    "spectral_centroid",
                    "spectral_rolloff",
                    "spectral_flatness",
                    "silence_ratio",
                ],
                "record_key_signature": _literal_record_key_signature(record_keys),
                "row_count": 6,
                "source_signature": source_signature,
                "source_split": "Training",
                "waveform_only": True,
            }
        ),
        encoding="utf-8",
    )

    manifest = pd.DataFrame(
        {
            "split": "Training",
            "call_id": call_ids,
            "record_key": record_keys,
            "annotation_row": annotation_rows,
            "speaker": y,
            "startAt_ms": start_at,
            "endAt_ms": end_at,
            "duration_sec": duration / 1000.0,
        }
    ).iloc[[2, 5, 0, 4, 1, 3]]
    manifest.to_parquet(manifest_parquet, index=False)
    manifest_metadata.write_text(
        json.dumps(
            {
                "schema_version": "mission2_raw_eda_v1",
                "source_signature": source_signature,
                "call_id_definition": "file stem; use (split, call_id) as group key",
                "allowed_annotation_fields": ["id", "startAt", "endAt", "speaker"],
            }
        ),
        encoding="utf-8",
    )
    return {
        "path_values": {
            "mfcc_npz": mfcc_npz,
            "mfcc_metadata": mfcc_metadata,
            "candidate_parquet": candidate_parquet,
            "candidate_metadata": candidate_metadata,
            "manifest_parquet": manifest_parquet,
            "manifest_metadata": manifest_metadata,
        },
        "X_mfcc": X_mfcc,
        "y": y,
        "call_ids": call_ids,
        "annotation_rows": annotation_rows,
        "record_keys": record_keys,
        "start_at": start_at,
        "end_at": end_at,
        "duration": duration,
        "rolloff": rolloff,
        "zcr": zcr,
        "candidate": candidate,
        "manifest": manifest,
    }


def _standalone_paths(model_module, fixture):
    return model_module.StandaloneCachePaths(**fixture["path_values"])


def _load_synthetic_standalone(model_module, fixture):
    return model_module.load_standalone_e05_cache(
        _standalone_paths(model_module, fixture),
        split="Training",
        expected_rows=6,
        expected_calls=2,
    )


def _rewrite_mfcc_fixture(fixture, X_mfcc):
    np.savez(
        fixture["path_values"]["mfcc_npz"],
        X=X_mfcc,
        y=fixture["y"],
        stem=fixture["call_ids"],
        start_at=fixture["start_at"],
        end_at=fixture["end_at"],
        duration=fixture["duration"],
    )


def _write_standalone_validation_cache_fixture(tmp_path: Path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    record_keys = np.asarray(
        [
            f"Validation|{call_id}|{annotation_row}"
            for call_id, annotation_row in zip(
                fixture["call_ids"], fixture["annotation_rows"], strict=True
            )
        ]
    )
    mfcc_metadata = json.loads(
        fixture["path_values"]["mfcc_metadata"].read_text(encoding="utf-8")
    )
    mfcc_metadata["source_split"] = "Validation"
    fixture["path_values"]["mfcc_metadata"].write_text(
        json.dumps(mfcc_metadata), encoding="utf-8"
    )

    source_signature = "synthetic-source-signature"
    candidate_paths = []
    candidate_metadata_paths = []
    for feature_name, values in (
        ("spectral_rolloff", fixture["rolloff"]),
        ("zcr", fixture["zcr"]),
    ):
        parquet_path = (
            fixture["path_values"]["candidate_parquet"].parent
            / f"validation_{feature_name}.parquet"
        )
        metadata_path = parquet_path.with_suffix(".json")
        pd.DataFrame(
            {
                "record_key": record_keys,
                "call_id": fixture["call_ids"],
                feature_name: values,
            }
        ).iloc[[4, 1, 5, 0, 3, 2]].to_parquet(parquet_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_schema_version": "mission2_candidate_audio_features_v1",
                    "extraction_config": {
                        "clipping_amplitude": 0.999,
                        "hop_length": 160,
                        "n_fft": 512,
                        "sample_rate": 8000,
                        "silence_threshold_dbfs": -40.0,
                    },
                    "feature_count": 1,
                    "feature_names": [feature_name],
                    "record_key_signature": _literal_record_key_signature(record_keys),
                    "row_count": 6,
                    "source_signature": source_signature,
                    "source_split": "Validation",
                    "waveform_only": True,
                }
            ),
            encoding="utf-8",
        )
        candidate_paths.append(parquet_path)
        candidate_metadata_paths.append(metadata_path)

    manifest = pd.DataFrame(
        {
            "split": "Validation",
            "call_id": fixture["call_ids"],
            "record_key": record_keys,
            "annotation_row": fixture["annotation_rows"],
            "speaker": fixture["y"],
            "startAt_ms": fixture["start_at"],
            "endAt_ms": fixture["end_at"],
            "duration_sec": fixture["duration"] / 1000.0,
        }
    ).iloc[[2, 5, 0, 4, 1, 3]]
    manifest.to_parquet(fixture["path_values"]["manifest_parquet"], index=False)
    fixture["path_values"]["candidate_parquet"] = tuple(candidate_paths)
    fixture["path_values"]["candidate_metadata"] = tuple(candidate_metadata_paths)
    fixture["record_keys"] = record_keys
    fixture["manifest"] = manifest
    return fixture


def _prepared_official(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    validation_keys = np.asarray(
        [key.replace("Training/", "Validation/", 1) for key in record_keys[60:]]
    )
    return model_module.prepare_official_comparison(
        training_X=X[:60],
        training_y=y[:60],
        training_record_keys=record_keys[:60],
        validation_X=X[60:],
        validation_y=y[60:],
        validation_record_keys=validation_keys,
        feature_names=EXPECTED_E05,
        user_approved=True,
        expected_training_rows=60,
        expected_validation_rows=20,
    )


def test_model_experiment_module_exists():
    assert _module_exists(), "write the model experiment module after observing RED"


def test_e05_contract_is_exactly_42d(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    validated = model_module.validate_e05_contract(X, y, record_keys, EXPECTED_E05)
    assert validated.X.shape == (80, 42)
    assert validated.X.dtype == np.float32

    with pytest.raises(ValueError, match="42"):
        model_module.validate_e05_contract(X[:, :-1], y, record_keys, EXPECTED_E05[:-1])


def test_e05_contract_freezes_mfcc40_then_rolloff_and_zcr(model_module):
    assert model_module.MFCC40_FEATURE_NAMES == EXPECTED_MFCC40
    assert model_module.E05_FEATURE_NAMES[:40] == EXPECTED_MFCC40
    assert model_module.E05_FEATURE_NAMES[-2:] == ("spectral_rolloff", "zcr")

    X, y, record_keys, _ = _synthetic_e05()
    validated = model_module.validate_e05_contract(
        X,
        y,
        record_keys,
        EXPECTED_E05,
        frozen_mfcc40=X[:, :40].copy(),
        expected_extra_features=X[:, 40:].copy(),
    )
    assert np.array_equal(validated.X[:, :40], X[:, :40])
    assert np.array_equal(validated.X[:, 40], X[:, 40])
    assert np.array_equal(validated.X[:, 41], X[:, 41])

    wrong_mfcc = X[:, :40].copy()
    wrong_mfcc[0, 0] += 1.0
    with pytest.raises(ValueError, match="frozen MFCC40"):
        model_module.validate_e05_contract(
            X,
            y,
            record_keys,
            EXPECTED_E05,
            frozen_mfcc40=wrong_mfcc,
            expected_extra_features=X[:, 40:],
        )

    wrong_extra = X[:, 40:].copy()
    wrong_extra[0, 1] += 1.0
    with pytest.raises(ValueError, match="rolloff/zcr"):
        model_module.validate_e05_contract(
            X,
            y,
            record_keys,
            EXPECTED_E05,
            frozen_mfcc40=X[:, :40],
            expected_extra_features=wrong_extra,
        )


def test_e05_contract_rejects_non_float32_input(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    with pytest.raises(ValueError, match="float32"):
        model_module.validate_e05_contract(
            X.astype(np.float64), y, record_keys, EXPECTED_E05
        )


def test_e05_contract_blocks_forbidden_metadata(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    forbidden_names = EXPECTED_E05[:-1] + ("call_id",)
    with pytest.raises(ValueError, match="feature contract"):
        model_module.validate_e05_contract(X, y, record_keys, forbidden_names)


def test_record_key_alignment_is_one_to_one(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    duplicate_keys = record_keys.copy()
    duplicate_keys[-1] = duplicate_keys[0]
    with pytest.raises(ValueError, match="record_key"):
        model_module.validate_e05_contract(X, y, duplicate_keys, EXPECTED_E05)


def test_label_alignment_and_binary_classes_are_guarded(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    with pytest.raises(ValueError, match="row alignment"):
        model_module.validate_e05_contract(X, y[:-1], record_keys, EXPECTED_E05)

    invalid_y = y.copy()
    invalid_y[-1] = 2
    with pytest.raises(ValueError, match="labels"):
        model_module.validate_e05_contract(X, invalid_y, record_keys, EXPECTED_E05)


def test_fixed_call_group_split_has_no_overlap_and_seed_42(model_module):
    _, _, _, call_ids = _synthetic_e05()
    train_idx, validation_idx, audit = model_module.create_fixed_internal_split(
        call_ids, split_function=_group_split_adapter
    )
    assert audit["seed"] == 42
    assert audit["test_size"] == 0.2
    assert audit["call_overlap"] == 0
    assert set(call_ids[train_idx]).isdisjoint(set(call_ids[validation_idx]))


def test_internal_split_count_mismatch_stops(model_module):
    _, _, _, call_ids = _synthetic_e05()
    with pytest.raises(ValueError, match="split contract"):
        model_module.create_fixed_internal_split(
            call_ids,
            split_function=_group_split_adapter,
            expected_contract={
                "total_rows": 889_358,
                "train_rows": 711_467,
                "validation_rows": 177_891,
                "train_calls": 23_360,
                "validation_calls": 5_840,
                "call_overlap": 0,
            },
        )


def test_m00_and_m01_share_identical_split_arrays_and_scaler(model_module):
    prepared = _prepared(model_module)
    m00 = model_module.run_internal_m00(prepared, allow_fit=True)
    m01 = model_module.run_internal_m01(prepared, allow_fit=True)
    assert np.array_equal(m00.fit_audit["train_idx"], m01.fit_audit["train_idx"])
    assert np.array_equal(
        m00.fit_audit["validation_idx"], m01.fit_audit["validation_idx"]
    )
    assert m00.scaler is prepared.scaler
    assert m01.scaler is prepared.scaler


def test_standard_scaler_is_fit_on_internal_train_only(model_module):
    X, y, record_keys, call_ids = _synthetic_e05()
    prepared = model_module.prepare_internal_comparison(
        X=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        feature_names=EXPECTED_E05,
        split_function=_group_split_adapter,
    )
    expected_mean = X[prepared.train_idx].astype(np.float64).mean(axis=0)
    assert np.allclose(prepared.scaler.mean_, expected_mean)
    assert prepared.scaler_fit_scope == "internal_train_only"
    assert np.array_equal(prepared.scaler_fit_record_keys, record_keys[prepared.train_idx])


def test_m01_exact_precommit_config_and_early_stopping_false(model_module):
    assert model_module.M01_HGB_CONFIG == EXPECTED_M01_CONFIG
    model = model_module.build_m01_model()
    params = model.get_params()
    assert {name: params[name] for name in EXPECTED_M01_CONFIG} == EXPECTED_M01_CONFIG
    assert params["early_stopping"] is False


def test_m01_precommit_mapping_cannot_be_mutated(model_module):
    original = dict(model_module.M01_HGB_CONFIG)
    try:
        with pytest.raises(TypeError):
            model_module.M01_HGB_CONFIG["max_iter"] = 101
    finally:
        if isinstance(model_module.M01_HGB_CONFIG, dict):
            model_module.M01_HGB_CONFIG.clear()
            model_module.M01_HGB_CONFIG.update(original)


def test_m01_fingerprint_is_stable_and_config_sensitive(model_module):
    first = model_module.config_fingerprint(model_module.M01_HGB_CONFIG)
    second = model_module.config_fingerprint(dict(reversed(list(EXPECTED_M01_CONFIG.items()))))
    changed = dict(EXPECTED_M01_CONFIG, max_iter=101)
    assert first == second
    assert first != model_module.config_fingerprint(changed)


@pytest.mark.parametrize("runner_name", ["run_internal_m00", "run_internal_m01"])
def test_internal_model_fit_requires_explicit_gate(model_module, runner_name):
    prepared = _prepared(model_module)
    with pytest.raises(RuntimeError, match="allow_fit"):
        getattr(model_module, runner_name)(prepared)


def test_m01_probability_contract_and_binary_classes(model_module):
    result = model_module.run_internal_m01(_prepared(model_module), allow_fit=True)
    assert result.probabilities.shape == (result.y_true.shape[0], 2)
    assert np.isfinite(result.probabilities).all()
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.array_equal(result.model.classes_, np.array([0, 1]))


def test_metric_and_correct_count_calculation(model_module):
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
    metrics = model_module.calculate_prediction_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["recall_speaker_0"] == pytest.approx(0.5)
    assert metrics["recall_speaker_1"] == pytest.approx(1.0)
    assert metrics["correct_count"] == 3


def test_frozen_e05_reference_correct_count_is_confusion_matrix_diagonal(model_module):
    reference = model_module.FROZEN_E05_OFFICIAL_REFERENCE
    diagonal = reference["confusion_matrix"][0][0] + reference["confusion_matrix"][1][1]
    assert reference["correct_count"] == diagonal == 91_026
    assert reference["accuracy"] == pytest.approx(
        reference["correct_count"] / reference["validation_rows"]
    )


def test_paired_error_change_and_disagreement_calculation(model_module):
    keys = np.array(["a", "b", "c", "d", "e"])
    y_true = np.array([0, 0, 1, 1, 0])
    m00 = np.array([0, 1, 0, 1, 0])
    m01 = np.array([0, 0, 1, 0, 1])
    comparison = model_module.calculate_paired_prediction_changes(
        y_true=y_true,
        m00_pred=m00,
        m01_pred=m01,
        m00_record_keys=keys,
        m01_record_keys=keys.copy(),
    )
    assert comparison["m00_wrong_to_m01_correct"] == 2
    assert comparison["m00_correct_to_m01_wrong"] == 2
    assert comparison["prediction_disagreement_count"] == 4
    assert comparison["prediction_disagreement_rate"] == pytest.approx(0.8)


def test_paired_comparison_rejects_record_key_mismatch(model_module):
    keys = np.array(["a", "b"])
    with pytest.raises(ValueError, match="record_key"):
        model_module.calculate_paired_prediction_changes(
            y_true=np.array([0, 1]),
            m00_pred=np.array([0, 1]),
            m01_pred=np.array([0, 1]),
            m00_record_keys=keys,
            m01_record_keys=np.array(["b", "a"]),
        )


def test_internal_comparison_revalidates_shared_split_keys_and_scaler(model_module):
    prepared = _prepared(model_module)
    m00 = model_module.run_internal_m00(prepared, allow_fit=True)
    m01 = model_module.run_internal_m01(prepared, allow_fit=True)
    baseline = model_module.compare_model_results(m00, m01)
    assert "accuracy_delta" in baseline

    different_scaler = StandardScaler().fit(np.zeros((4, 42), dtype=np.float32))
    with pytest.raises(ValueError, match="shared scaler"):
        model_module.compare_model_results(m00, replace(m01, scaler=different_scaler))

    changed_audit = dict(m01.fit_audit)
    changed_audit["train_idx"] = changed_audit["train_idx"][::-1]
    with pytest.raises(ValueError, match="train_idx"):
        model_module.compare_model_results(
            m00, replace(m01, fit_audit=changed_audit)
        )


def test_frozen_row_level_reference_must_match_aggregate_ledger(model_module):
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
    probabilities = np.array(
        [[0.95, 0.05], [0.05, 0.95], [0.10, 0.90], [0.20, 0.80]], dtype=float
    )
    result = SimpleNamespace(
        feature_names=EXPECTED_E05,
        train_shape=(4, 42),
        evaluation_shape=(4, 42),
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        model=model_module.build_m00_model(),
        fit_audit={
            "validation_record_keys": np.array(["a", "b", "c", "d"]),
            "scaler_fit_scope": "Training",
            "model_fit_scope": "Training",
            "evaluation_scope": "Validation",
            "scaler_fit_rows": 4,
            "model_fit_rows": 4,
            "evaluation_rows": 4,
        },
    )
    reference = {
        "training_rows": 4,
        "validation_rows": 4,
        "accuracy": 0.75,
        "confusion_matrix": [[1, 1], [0, 2]],
        "recall_speaker_0": 0.5,
        "recall_speaker_1": 1.0,
        "correct_count": 3,
        "high_confidence_error_count": 1,
    }
    observed = model_module.validate_frozen_e05_row_level_result(
        result, reference=reference
    )
    assert observed["correct_count"] == 3

    invalid_reference = dict(reference, accuracy=0.80)
    with pytest.raises(ValueError, match="accuracy"):
        model_module.validate_frozen_e05_row_level_result(
            result, reference=invalid_reference
        )

    wrong_model = model_module.build_m00_model()
    wrong_model.set_params(max_iter=999)
    with pytest.raises(ValueError, match="LogisticRegression config"):
        model_module.validate_frozen_e05_row_level_result(
            SimpleNamespace(**{**vars(result), "model": wrong_model}),
            reference=reference,
        )


def test_model_report_adapter_preserves_diagnostic_slices(model_module):
    result = model_module.run_internal_m01(_prepared(model_module), allow_fit=True)
    duration_summary = pd.DataFrame([{"slice": "short", "accuracy": 0.5}])
    overlap_summary = pd.DataFrame(
        [
            {"slice": "overlap", "accuracy": 0.4},
            {"slice": "no_overlap", "accuracy": 0.8},
        ]
    )
    confidence_summary = pd.DataFrame(
        [{"slice": "0.90-0.95", "accuracy": 0.7}]
    )
    base_report = {
        "duration_summary": duration_summary,
        "overlap_summary": overlap_summary,
        "confidence_summary": confidence_summary,
        "high_confidence_error_count": 2,
    }
    report = model_module.augment_model_report(result, base_report)
    pd.testing.assert_frame_equal(report["duration_summary"], duration_summary)
    pd.testing.assert_frame_equal(report["confidence_summary"], confidence_summary)
    assert report["overlap_accuracy"] == pytest.approx(0.4)
    assert report["no_overlap_accuracy"] == pytest.approx(0.8)
    assert report["high_confidence_error_count"] == 2
    assert report["model_config"] == EXPECTED_M01_CONFIG


def test_unexecuted_experiment_does_not_fabricate_decision_row(model_module):
    table = model_module.build_decision_table([None, None])
    assert list(table.columns) == list(model_module.DECISION_TABLE_COLUMNS)
    assert table.empty


def test_executed_report_can_be_added_without_winner_claim(model_module):
    result = model_module.run_internal_m01(_prepared(model_module), allow_fit=True)
    report = model_module.augment_model_report(
        result,
        {
            "duration_summary": [],
            "overlap_summary": {},
            "confidence_summary": [],
            "high_confidence_errors": pd.DataFrame(),
        },
    )
    table = model_module.build_decision_table([None, report])
    assert table["experiment_id"].tolist() == ["M01"]
    assert "winner" not in table.columns


def test_cache_miss_never_triggers_extraction(model_module, tmp_path):
    missing = tmp_path / "missing_features.parquet"
    with pytest.raises(FileNotFoundError, match="automatic extraction is disabled"):
        model_module.require_existing_cache_files([missing])
    assert not missing.exists()


def test_official_preparation_and_fit_require_separate_approval(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    with pytest.raises(RuntimeError, match="user_approved"):
        model_module.prepare_official_comparison(
            training_X=X[:60],
            training_y=y[:60],
            training_record_keys=record_keys[:60],
            validation_X=X[60:],
            validation_y=y[60:],
            validation_record_keys=record_keys[60:],
            feature_names=EXPECTED_E05,
        )


def test_notebook_exists_and_has_all_named_checkpoints():
    assert NOTEBOOK_PATH.exists(), "write the notebook only after the first RED run"
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for letter, title in (
        ("A", "Model Experiment Contract"),
        ("B", "Cache / Row Alignment"),
        ("C", "Fixed Training-only GroupSplit"),
        ("D", "M00 Internal LR Reference"),
        ("E", "M01 HGB Precommit"),
        ("F", "M01 Internal Run"),
        ("G", "M00 vs M01 Internal Comparison"),
        ("H", "User Decision Gate"),
        ("I", "Optional Official Validation"),
        ("J", "Decision Table / Interpretation Template"),
    ):
        assert f"Checkpoint {letter} — {title}" in markdown


def test_all_expensive_notebook_switches_have_false_source_defaults():
    assert NOTEBOOK_PATH.exists(), "notebook must exist before source-gate validation"
    tree = ast.parse(_notebook_code())
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value.value

    required_false = {
        "RUN_LOAD_TRAINING_CACHE",
        "RUN_INTERNAL_M00",
        "RUN_INTERNAL_M01",
        "USER_APPROVED_OFFICIAL_VALIDATION",
        "RUN_OFFICIAL_VALIDATION",
    }
    assert required_false <= assignments.keys()
    assert {name: assignments[name] for name in required_false} == {
        name: False for name in required_false
    }
    expensive_names = {
        name
        for name in assignments
        if name.startswith(("RUN_", "BUILD_", "EXTRACT_", "SAVE_"))
    }
    assert all(assignments[name] is False for name in expensive_names)


def test_notebook_contains_no_parameter_search_or_53d_pipeline_imports():
    assert NOTEBOOK_PATH.exists()
    source = _notebook_code()
    for forbidden in (
        "GridSearchCV",
        "RandomizedSearchCV",
        "Optuna",
        "mission2_train",
        "mission2_features",
        "FEATURE_COLUMNS",
    ):
        assert forbidden not in source


def test_notebook_code_cells_compile_independently_of_saved_outputs():
    assert NOTEBOOK_PATH.exists()
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        compile("".join(cell.get("source", [])), str(NOTEBOOK_PATH), "exec")


def test_standalone_module_does_not_import_pr16_feature_helpers(model_module):
    assert model_module.CACHE_BACKEND == "standalone_adapter"
    source = Path(model_module.__file__).read_text(encoding="utf-8")
    assert "from dcc_data.mission2_experiments import" not in source
    assert "from dcc_data.mission2_candidate_features import" not in source


def test_standalone_valid_caches_produce_exact_42d_values(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    loaded = _load_synthetic_standalone(model_module, fixture)
    assert loaded.validated.X.shape == (6, 42)
    assert loaded.validated.X.dtype == np.float32
    assert loaded.validated.feature_names == EXPECTED_E05
    np.testing.assert_array_equal(loaded.validated.X[:, :40], fixture["X_mfcc"])
    np.testing.assert_array_equal(
        loaded.validated.X[:, 40], fixture["rolloff"].astype(np.float32)
    )
    np.testing.assert_array_equal(
        loaded.validated.X[:, 41], fixture["zcr"].astype(np.float32)
    )
    assert loaded.cache_audit["candidate_source_dtype"] == "float64"
    assert loaded.cache_audit["final_dtype"] == "float32"


def test_standalone_alignment_uses_keys_not_input_row_order(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    loaded = _load_synthetic_standalone(model_module, fixture)
    assert np.array_equal(loaded.validated.record_keys, fixture["record_keys"])
    assert np.array_equal(loaded.validated.y, fixture["y"])
    assert loaded.tracking["record_key"].tolist() == fixture["record_keys"].tolist()
    assert loaded.tracking["annotation_row"].tolist() == fixture["annotation_rows"].tolist()


@pytest.mark.parametrize("key_case", ["missing", "extra"])
def test_standalone_rejects_candidate_key_set_mismatch(
    model_module, tmp_path, key_case
):
    fixture = _write_standalone_cache_fixture(tmp_path)
    candidate = fixture["candidate"].copy().reset_index(drop=True)
    if key_case == "missing":
        candidate = candidate.iloc[:-1]
    else:
        extra = candidate.iloc[[0]].copy()
        extra["record_key"] = "Training|call_extra|0"
        extra["call_id"] = "call_extra"
        candidate = pd.concat([candidate, extra], ignore_index=True)
    candidate.to_parquet(fixture["path_values"]["candidate_parquet"], index=False)
    with pytest.raises(ValueError, match="candidate.*(row|key)"):
        _load_synthetic_standalone(model_module, fixture)


def test_standalone_rejects_duplicate_candidate_key(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    candidate = fixture["candidate"].copy().reset_index(drop=True)
    candidate.loc[1, "record_key"] = candidate.loc[0, "record_key"]
    candidate.to_parquet(fixture["path_values"]["candidate_parquet"], index=False)
    with pytest.raises(ValueError, match="candidate record_key"):
        _load_synthetic_standalone(model_module, fixture)


def test_standalone_rejects_duplicate_manifest_key(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    manifest = fixture["manifest"].copy().reset_index(drop=True)
    manifest.loc[1, "record_key"] = manifest.loc[0, "record_key"]
    manifest.to_parquet(fixture["path_values"]["manifest_parquet"], index=False)
    with pytest.raises(ValueError, match="manifest record_key"):
        _load_synthetic_standalone(model_module, fixture)


def test_standalone_rejects_manifest_mfcc_y_mismatch(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    manifest = fixture["manifest"].copy().reset_index(drop=True)
    manifest.loc[0, "speaker"] = 1 - int(manifest.loc[0, "speaker"])
    manifest.to_parquet(fixture["path_values"]["manifest_parquet"], index=False)
    with pytest.raises(ValueError, match="y alignment"):
        _load_synthetic_standalone(model_module, fixture)


def test_standalone_rejects_non_40d_mfcc_cache(model_module, tmp_path):
    fixture = _write_standalone_cache_fixture(tmp_path)
    _rewrite_mfcc_fixture(fixture, fixture["X_mfcc"][:, :39])
    with pytest.raises(ValueError, match="MFCC.*40"):
        _load_synthetic_standalone(model_module, fixture)


@pytest.mark.parametrize("missing_feature", ["spectral_rolloff", "zcr"])
def test_standalone_rejects_missing_selected_candidate_feature(
    model_module, tmp_path, missing_feature
):
    fixture = _write_standalone_cache_fixture(tmp_path)
    fixture["candidate"].drop(columns=[missing_feature]).to_parquet(
        fixture["path_values"]["candidate_parquet"], index=False
    )
    with pytest.raises(ValueError, match=missing_feature):
        _load_synthetic_standalone(model_module, fixture)


@pytest.mark.parametrize(
    ("source", "bad_value"),
    [
        ("mfcc", np.nan),
        ("rolloff", np.nan),
        ("zcr", np.nan),
        ("mfcc", np.inf),
        ("rolloff", np.inf),
        ("zcr", np.inf),
    ],
)
def test_standalone_rejects_nonfinite_model_values(
    model_module, tmp_path, source, bad_value
):
    fixture = _write_standalone_cache_fixture(tmp_path)
    if source == "mfcc":
        X_mfcc = fixture["X_mfcc"].copy()
        X_mfcc[0, 0] = bad_value
        _rewrite_mfcc_fixture(fixture, X_mfcc)
    else:
        candidate = fixture["candidate"].copy()
        column = "spectral_rolloff" if source == "rolloff" else "zcr"
        candidate.iloc[0, candidate.columns.get_loc(column)] = bad_value
        candidate.to_parquet(fixture["path_values"]["candidate_parquet"], index=False)
    with pytest.raises(ValueError, match="NaN or inf"):
        _load_synthetic_standalone(model_module, fixture)


@pytest.mark.parametrize(
    "missing_path_field",
    [
        "mfcc_npz",
        "mfcc_metadata",
        "candidate_parquet",
        "candidate_metadata",
        "manifest_parquet",
        "manifest_metadata",
    ],
)
def test_standalone_cache_miss_hard_fails_without_extraction(
    model_module, tmp_path, missing_path_field
):
    fixture = _write_standalone_cache_fixture(tmp_path)
    paths = _standalone_paths(model_module, fixture)
    missing = tmp_path / f"missing-{missing_path_field}"
    missing_paths = replace(paths, **{missing_path_field: missing})
    with pytest.raises(FileNotFoundError, match="automatic extraction is disabled"):
        model_module.load_standalone_e05_cache(
            missing_paths, split="Training", expected_rows=6, expected_calls=2
        )
    assert not missing.exists()


def test_standalone_output_excludes_forbidden_and_unused_metadata(
    model_module, tmp_path
):
    fixture = _write_standalone_cache_fixture(tmp_path)
    loaded = _load_synthetic_standalone(model_module, fixture)
    assert loaded.validated.feature_names == EXPECTED_E05
    assert loaded.validated.X.shape[1] == 42
    for forbidden in (
        "call_id",
        "record_key",
        "annotation_row",
        "speaker",
        "startAt_ms",
        "endAt_ms",
        "duration_sec",
        "spectral_centroid",
        "spectral_flatness",
        "silence_ratio",
    ):
        assert forbidden not in loaded.validated.feature_names


def test_notebook_uses_standalone_backend_through_checkpoint_h():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell.get("source", [])) for cell in notebook["cells"]}
    assert 'CACHE_BACKEND = "standalone_adapter"' in cells["setup-imports"]
    assert "load_standalone_e05_cache" in cells["checkpoint-b-code"]
    through_h = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["id"] not in {"checkpoint-i-md", "checkpoint-i-load-code", "checkpoint-i-run-code", "checkpoint-j-md", "checkpoint-j-code"}
    )
    assert "FEATURE_HELPERS_AVAILABLE" not in through_h
    assert "mission2_experiments" not in through_h
    assert "mission2_candidate_features" not in through_h


def test_standalone_split_and_diagnostic_report_need_no_pr16_helper(model_module):
    X, y, record_keys, call_ids = _synthetic_e05()
    prepared = model_module.prepare_internal_comparison(
        X=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        feature_names=EXPECTED_E05,
        split_function=model_module.standalone_group_split_indices,
    )
    result = model_module.run_internal_m00(prepared, allow_fit=True)
    tracking = pd.DataFrame(
        {
            "record_key": record_keys,
            "call_id": call_ids,
            "startAt_ms": np.tile(np.array([0.0, 30.0, 60.0, 90.0]), 20),
            "endAt_ms": np.tile(np.array([50.0, 80.0, 110.0, 140.0]), 20),
            "duration_ms": 50.0,
        }
    )
    report = model_module.build_standalone_diagnostic_report(
        result,
        training_tracking=tracking.iloc[prepared.train_idx].reset_index(drop=True),
        evaluation_tracking=tracking.iloc[prepared.validation_idx].reset_index(drop=True),
    )
    assert report["global_metrics"]["sample_count"] == len(prepared.validation_idx)
    assert set(report["overlap_summary"]["slice"]) <= {"overlap", "no_overlap"}
    assert not report["duration_summary"].empty
    assert not report["confidence_summary"].empty


def test_paired_comparison_rechecks_shared_transformed_matrix_fingerprints(
    model_module,
):
    prepared = _prepared(model_module)
    m00 = model_module.run_internal_m00(prepared, allow_fit=True)
    m01 = model_module.run_internal_m01(prepared, allow_fit=True)
    assert model_module.compare_model_results(m00, m01)["both_correct"] >= 0
    changed_audit = dict(m01.fit_audit)
    changed_audit["X_evaluation_fingerprint"] = "different"
    with pytest.raises(ValueError, match="X_evaluation_fingerprint"):
        model_module.compare_model_results(
            m00, replace(m01, fit_audit=changed_audit)
        )


def test_paired_comparison_rechecks_training_label_fingerprint(model_module):
    prepared = _prepared(model_module)
    m00 = model_module.run_internal_m00(prepared, allow_fit=True)
    changed_prepared = replace(prepared, y_train=1 - prepared.y_train)
    m01 = model_module.run_internal_m01(changed_prepared, allow_fit=True)
    with pytest.raises(ValueError, match="y_train_fingerprint"):
        model_module.compare_model_results(m00, m01)


def test_confirmed_validation_cache_paths_and_counts_are_frozen(model_module):
    paths = model_module.standalone_validation_cache_paths(REPO_ROOT)
    assert paths.mfcc_npz.name == "validation_mfcc20_nfft512_hop160_v1.npz"
    assert [path.name for path in paths.candidate_parquet] == [
        "validation_candidate_d6e692b7dcda.parquet",
        "validation_candidate_eaff31c61239.parquet",
    ]
    assert model_module.EXPECTED_OFFICIAL_ROWS["Validation"] == 111_947
    assert model_module.EXPECTED_OFFICIAL_CALLS["Validation"] == 3_640


def test_standalone_validation_loads_two_candidate_caches_as_exact_42d(
    model_module, tmp_path
):
    fixture = _write_standalone_validation_cache_fixture(tmp_path)
    loaded = model_module.load_standalone_e05_cache(
        model_module.StandaloneCachePaths(**fixture["path_values"]),
        split="Validation",
        expected_rows=6,
        expected_calls=2,
    )
    assert loaded.cache_audit["split"] == "Validation"
    assert loaded.validated.X.shape == (6, 42)
    assert loaded.validated.X.dtype == np.float32
    assert np.isfinite(loaded.validated.X).all()
    assert len(np.unique(loaded.validated.record_keys)) == 6
    np.testing.assert_array_equal(loaded.validated.X[:, :40], fixture["X_mfcc"])
    np.testing.assert_array_equal(
        loaded.validated.X[:, 40], fixture["rolloff"].astype(np.float32)
    )
    np.testing.assert_array_equal(
        loaded.validated.X[:, 41], fixture["zcr"].astype(np.float32)
    )


def test_standalone_validation_missing_feature_cache_hard_stops(
    model_module, tmp_path
):
    fixture = _write_standalone_validation_cache_fixture(tmp_path)
    paths = model_module.StandaloneCachePaths(**fixture["path_values"])
    missing = tmp_path / "missing-validation-zcr.parquet"
    broken = replace(
        paths,
        candidate_parquet=(paths.candidate_parquet[0], missing),
    )
    with pytest.raises(FileNotFoundError, match="automatic extraction is disabled"):
        model_module.load_standalone_e05_cache(
            broken, split="Validation", expected_rows=6, expected_calls=2
        )
    assert not missing.exists()


def test_official_preparation_rejects_training_validation_key_overlap(model_module):
    X, y, record_keys, _ = _synthetic_e05()
    with pytest.raises(ValueError, match="Training/Validation record_key overlap"):
        model_module.prepare_official_comparison(
            training_X=X[:60],
            training_y=y[:60],
            training_record_keys=record_keys[:60],
            validation_X=X[:20],
            validation_y=y[:20],
            validation_record_keys=record_keys[:20],
            feature_names=EXPECTED_E05,
            user_approved=True,
            expected_training_rows=60,
            expected_validation_rows=20,
        )


def test_official_scaler_fits_training_and_only_transforms_validation(model_module):
    prepared = _prepared_official(model_module)
    X, _, _, _ = _synthetic_e05()
    np.testing.assert_allclose(prepared.scaler.mean_, X[:60].mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(
        prepared.X_evaluation_scaled,
        prepared.scaler.transform(X[60:]).astype(np.float32),
        rtol=0,
        atol=0,
    )
    assert prepared.scaler.n_samples_seen_ == 60
    assert prepared.scaler_fit_scope == "full_training_only"
    assert prepared.evaluation_scope == "Official Validation"


def test_official_m00_m01_run_without_row_level_artifact_and_share_payload(
    model_module,
):
    prepared = _prepared_official(model_module)
    frozen_e05_row_level_result = None
    assert frozen_e05_row_level_result is None
    m00 = model_module.run_official_m00(
        prepared, allow_fit=True, user_approved=True
    )
    m01 = model_module.run_official_m01(
        prepared, allow_fit=True, user_approved=True
    )
    comparison = model_module.compare_official_model_results(m00, m01)
    assert m00.experiment_id == "M00_OFFICIAL"
    assert m01.experiment_id == "M01_OFFICIAL"
    assert m00.scaler is m01.scaler is prepared.scaler
    assert np.array_equal(m00.y_true, m01.y_true)
    assert np.array_equal(
        m00.fit_audit["validation_record_keys"],
        m01.fit_audit["validation_record_keys"],
    )
    assert comparison["evaluation_rows"] == 20


def test_official_comparison_rechecks_identical_validation_matrix(model_module):
    prepared = _prepared_official(model_module)
    m00 = model_module.run_official_m00(
        prepared, allow_fit=True, user_approved=True
    )
    m01 = model_module.run_official_m01(
        prepared, allow_fit=True, user_approved=True
    )
    changed_audit = dict(m01.fit_audit)
    changed_audit["X_evaluation_fingerprint"] = "different"
    with pytest.raises(ValueError, match="X_evaluation_fingerprint"):
        model_module.compare_official_model_results(
            m00, replace(m01, fit_audit=changed_audit)
        )


def test_official_paired_transition_arithmetic_is_consistent(model_module):
    prepared = _prepared_official(model_module)
    m00 = model_module.run_official_m00(
        prepared, allow_fit=True, user_approved=True
    )
    m01 = model_module.run_official_m01(
        prepared, allow_fit=True, user_approved=True
    )
    comparison = model_module.compare_official_model_results(m00, m01)
    transitions = (
        comparison["m00_wrong_to_m01_correct"]
        + comparison["m00_correct_to_m01_wrong"]
    )
    assert comparison["prediction_disagreement_count"] == transitions
    assert comparison["correct_count_delta"] == (
        comparison["m00_wrong_to_m01_correct"]
        - comparison["m00_correct_to_m01_wrong"]
    )
    assert (
        comparison["both_correct"]
        + comparison["both_wrong"]
        + transitions
        == comparison["evaluation_rows"]
    )


def test_m00_official_reproducibility_gate_uses_aggregate_only(model_module):
    prepared = _prepared_official(model_module)
    result = model_module.run_official_m00(
        prepared, allow_fit=True, user_approved=True
    )
    confidence = result.probabilities.max(axis=1)
    reference = {
        "training_rows": 60,
        "validation_rows": 20,
        **dict(result.metrics),
        "high_confidence_error_count": int(
            np.sum((result.y_pred != result.y_true) & (confidence >= 0.90))
        ),
    }
    observed = model_module.validate_m00_official_reproducibility(
        result, reference=reference
    )
    assert observed["accuracy"] == pytest.approx(result.metrics["accuracy"])
    with pytest.raises(ValueError, match="accuracy"):
        model_module.validate_m00_official_reproducibility(
            result, reference={**reference, "accuracy": reference["accuracy"] + 0.1}
        )


def test_j_table_includes_only_executed_internal_and_official_reports(model_module):
    internal = model_module.run_internal_m00(_prepared(model_module), allow_fit=True)
    official = model_module.run_official_m00(
        _prepared_official(model_module), allow_fit=True, user_approved=True
    )
    base_report = {
        "duration_summary": [],
        "overlap_summary": {},
        "confidence_summary": [],
        "high_confidence_errors": pd.DataFrame(),
    }
    table = model_module.build_decision_table(
        [
            model_module.augment_model_report(internal, base_report),
            None,
            model_module.augment_model_report(official, base_report),
            None,
        ]
    )
    assert table["experiment_id"].tolist() == ["M00", "M00_OFFICIAL"]
    assert "winner" not in table.columns


def test_notebook_i_j_use_single_saved_false_official_switch():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell.get("source", [])) for cell in notebook["cells"]}
    i_source = cells["checkpoint-i-load-code"] + cells["checkpoint-i-run-code"]
    assert "RUN_OFFICIAL_VALIDATION = False" in i_source
    assert "RUN_LOAD_OFFICIAL_CACHE" not in i_source
    assert "RUN_OFFICIAL_M01" not in i_source
    assert "FROZEN_E05_ROW_LEVEL_RESULT is None" not in i_source
    assert "load_standalone_e05_cache" in i_source
    assert "run_official_m00" in i_source
    assert "run_official_m01" in i_source
    assert "m00_official_report" in cells["checkpoint-j-code"]
    assert "m01_official_report" in cells["checkpoint-j-code"]


def test_screening_registry_builds_exactly_three_frozen_new_families(model_module):
    assert model_module.SCREENING_MODEL_IDS == ("M02", "M03", "M04")
    assert dict(model_module.M02_EXTRATREES_CONFIG) == EXPECTED_M02_CONFIG
    assert dict(model_module.M03_QDA_CONFIG) == EXPECTED_M03_CONFIG
    assert dict(model_module.M04_MLP_CONFIG) == EXPECTED_M04_CONFIG

    models = {
        model_id: model_module.build_screening_model(model_id)
        for model_id in model_module.SCREENING_MODEL_IDS
    }
    assert type(models["M02"]) is ExtraTreesClassifier
    assert type(models["M03"]) is QuadraticDiscriminantAnalysis
    assert type(models["M04"]) is MLPClassifier
    expected_configs = {
        "M02": EXPECTED_M02_CONFIG,
        "M03": EXPECTED_M03_CONFIG,
        "M04": EXPECTED_M04_CONFIG,
    }
    for model_id, model in models.items():
        params = model.get_params()
        assert {
            name: params[name] for name in expected_configs[model_id]
        } == expected_configs[model_id]


def test_screening_configs_are_immutable_and_disable_automatic_validation(
    model_module,
):
    with pytest.raises(TypeError):
        model_module.M02_EXTRATREES_CONFIG["n_estimators"] = 256
    with pytest.raises(TypeError):
        model_module.M03_QDA_CONFIG["reg_param"] = 0.1
    with pytest.raises(TypeError):
        model_module.M04_MLP_CONFIG["max_iter"] = 31
    assert model_module.build_screening_model("M04").early_stopping is False
    with pytest.raises(ValueError, match="unknown screening model"):
        model_module.build_screening_model("M05")


def test_internal_screening_reuses_exact_m01_payload_without_feature_change(
    model_module,
):
    prepared = _prepared_screening(model_module)
    m01 = model_module.run_internal_m01(prepared, allow_fit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = model_module.run_internal_screening(prepared, allow_fit=True)
    assert tuple(result.experiment_id for result in candidates) == (
        "M02",
        "M03",
        "M04",
    )
    assert len(candidates) == 3
    for result in candidates:
        assert result.feature_names == EXPECTED_E05
        assert result.scaler is m01.scaler is prepared.scaler
        assert result.train_shape == m01.train_shape
        assert result.evaluation_shape == m01.evaluation_shape
        assert np.array_equal(result.y_true, m01.y_true)
        for audit_name in (
            "train_idx",
            "validation_idx",
            "training_record_keys",
            "validation_record_keys",
            "scaler_fit_record_keys",
        ):
            assert np.array_equal(
                result.fit_audit[audit_name], m01.fit_audit[audit_name]
            )
        for audit_name in (
            "X_train_fingerprint",
            "X_evaluation_fingerprint",
            "y_train_fingerprint",
            "y_evaluation_fingerprint",
        ):
            assert result.fit_audit[audit_name] == m01.fit_audit[audit_name]
        assert result.fit_audit["runtime_seconds"] >= 0.0
        assert result.fit_audit["peak_memory_bytes"] is None


def test_screening_comparison_to_m01_has_consistent_paired_arithmetic(model_module):
    prepared = _prepared_screening(model_module)
    m01 = model_module.run_internal_m01(prepared, allow_fit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = model_module.run_internal_screening(prepared, allow_fit=True)
    comparisons = model_module.build_screening_comparisons(m01, candidates)
    assert [row["candidate_id"] for row in comparisons] == ["M02", "M03", "M04"]
    for row in comparisons:
        transitions = (
            row["reference_wrong_to_candidate_correct"]
            + row["reference_correct_to_candidate_wrong"]
        )
        assert row["prediction_disagreement_count"] == transitions
        assert row["correct_count_delta"] == (
            row["reference_wrong_to_candidate_correct"]
            - row["reference_correct_to_candidate_wrong"]
        )
        assert (
            row["both_correct"]
            + row["both_wrong"]
            + transitions
            == row["evaluation_rows"]
        )
        assert row["accuracy_delta"] == pytest.approx(
            row["candidate_accuracy"] - row["reference_accuracy"]
        )

    changed_audit = dict(candidates[0].fit_audit)
    changed_audit["X_evaluation_fingerprint"] = "different"
    with pytest.raises(ValueError, match="X_evaluation_fingerprint"):
        model_module.compare_candidate_to_reference(
            m01, replace(candidates[0], fit_audit=changed_audit)
        )


def test_decision_table_exposes_screening_config_diagnostics_and_runtime(model_module):
    prepared = _prepared_screening(model_module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model_module.run_internal_screening(
            prepared, allow_fit=True
        )[0]
    report = model_module.augment_model_report(
        result,
        {
            "duration_summary": pd.DataFrame(
                [{"slice": "q25_q75", "accuracy": 0.75}]
            ),
            "overlap_summary": pd.DataFrame(
                [
                    {"slice": "overlap", "accuracy": 0.70},
                    {"slice": "no_overlap", "accuracy": 0.80},
                ]
            ),
            "confidence_summary": pd.DataFrame(
                [{"slice": ">=0.90", "accuracy": 0.90}]
            ),
            "high_confidence_error_count": 3,
        },
    )
    table = model_module.build_decision_table([report])
    assert table.loc[0, "model_config"] == EXPECTED_M02_CONFIG
    assert table.loc[0, "confusion_matrix"] == result.metrics["confusion_matrix"]
    assert table.loc[0, "overlap_accuracy"] == pytest.approx(0.70)
    assert table.loc[0, "no_overlap_accuracy"] == pytest.approx(0.80)
    assert table.loc[0, "runtime_seconds"] >= 0.0
    assert pd.isna(table.loc[0, "peak_memory_bytes"])
    assert "winner" not in table.columns


def test_promotion_freezes_one_or_two_executed_candidates_before_official(
    model_module,
):
    prepared = _prepared_screening(model_module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = model_module.run_internal_screening(prepared, allow_fit=True)
    promotion = model_module.freeze_promotion_selection(
        ("M02", "M04"),
        internal_results=candidates,
        user_approved=True,
    )
    assert promotion.model_ids == ("M02", "M04")
    assert len(promotion.selection_fingerprint) == 64
    assert set(promotion.internal_result_fingerprints) == {"M02", "M04"}

    with pytest.raises(ValueError, match="one or two"):
        model_module.freeze_promotion_selection(
            ("M02", "M03", "M04"),
            internal_results=candidates,
            user_approved=True,
        )
    with pytest.raises(RuntimeError, match="user_approved"):
        model_module.freeze_promotion_selection(
            ("M02",), internal_results=candidates
        )
    with pytest.raises(ValueError, match="executed internal result"):
        model_module.freeze_promotion_selection(
            ("M02",), internal_results=candidates[1:], user_approved=True
        )


def test_promoted_official_run_requires_frozen_selection_user_gate_and_fit_gate(
    model_module,
):
    internal_prepared = _prepared_screening(model_module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = model_module.run_internal_screening(
            internal_prepared, allow_fit=True
        )
    promotion = model_module.freeze_promotion_selection(
        ("M03",), internal_results=candidates, user_approved=True
    )
    official_prepared = _prepared_official_screening(model_module)

    with pytest.raises(RuntimeError, match="user_approved"):
        model_module.run_promoted_official_models(
            official_prepared, promotion=promotion, allow_fit=True
        )
    with pytest.raises(RuntimeError, match="allow_fit"):
        model_module.run_promoted_official_models(
            official_prepared,
            promotion=promotion,
            allow_fit=False,
            user_approved=True,
        )

    forged_fingerprints = {"M05": "forged-internal-result"}
    forged = model_module.PromotionSelection(
        model_ids=("M05",),
        internal_result_fingerprints=forged_fingerprints,
        selection_fingerprint=model_module.config_fingerprint(
            {
                "model_ids": ["M05"],
                "internal_result_fingerprints": forged_fingerprints,
            }
        ),
    )
    with pytest.raises(ValueError, match="frozen M02--M04"):
        model_module.run_promoted_official_models(
            official_prepared,
            promotion=forged,
            allow_fit=True,
            user_approved=True,
        )

    results = model_module.run_promoted_official_models(
        official_prepared,
        promotion=promotion,
        allow_fit=True,
        user_approved=True,
    )
    assert [result.experiment_id for result in results] == ["M03_OFFICIAL"]
    assert results[0].scaler is official_prepared.scaler
    assert results[0].fit_audit["promotion_fingerprint"] == (
        promotion.selection_fingerprint
    )


def test_screening_module_does_not_construct_automatic_search_objects():
    module_path = REPO_ROOT / "src" / "dcc_data" / "mission2_model_experiments.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV"}
    used_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    used_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert forbidden.isdisjoint(used_names | used_attributes)


def test_notebook_appends_screening_and_promotion_gates_with_false_defaults():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell.get("source", [])) for cell in notebook["cells"]}
    ordered_ids = [cell["id"] for cell in notebook["cells"]]
    required_ids = [
        "checkpoint-k-md",
        "checkpoint-k-code",
        "checkpoint-l-md",
        "checkpoint-l-code",
        "checkpoint-m-md",
        "checkpoint-m-code",
        "checkpoint-n-md",
        "checkpoint-n-code",
    ]
    assert all(cell_id in cells for cell_id in required_ids)
    assert ordered_ids.index("checkpoint-k-code") < ordered_ids.index(
        "checkpoint-l-code"
    )
    assert "RUN_INTERNAL_FAMILY_SCREENING = False" in cells["checkpoint-l-code"]
    assert "PROMOTED_MODEL_IDS = ()" in cells["checkpoint-m-code"]
    assert "USER_APPROVED_PROMOTION = False" in cells["checkpoint-m-code"]
    assert (
        "RUN_PROMOTED_OFFICIAL_VALIDATION = False"
        in cells["checkpoint-n-code"]
    )
    assert "build_screening_comparisons" in cells["checkpoint-m-code"]
    assert "run_promoted_official_models" in cells["checkpoint-n-code"]
    assert "GridSearchCV" not in "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")


def test_m04_iter60_config_changes_only_max_iter(model_module):
    assert dict(model_module.M04_CONVERGENCE_CHECK_CONFIG) == (
        EXPECTED_M04_ITER60_CONFIG
    )
    changed = {
        name
        for name in EXPECTED_M04_CONFIG
        if EXPECTED_M04_CONFIG[name]
        != model_module.M04_CONVERGENCE_CHECK_CONFIG[name]
    }
    assert changed == {"max_iter"}
    assert model_module.M04_MLP_CONFIG["max_iter"] == 30
    assert model_module.M04_CONVERGENCE_CHECK_CONFIG["max_iter"] == 60
    with pytest.raises(TypeError):
        model_module.M04_CONVERGENCE_CHECK_CONFIG["max_iter"] = 90
    model = model_module.build_m04_convergence_check_model()
    assert type(model) is MLPClassifier
    assert {
        name: model.get_params()[name] for name in EXPECTED_M04_ITER60_CONFIG
    } == EXPECTED_M04_ITER60_CONFIG


def test_mlp_convergence_summary_exposes_warning_iterations_and_loss_trend(
    model_module,
):
    model = SimpleNamespace(
        n_iter_=60,
        loss_curve_=np.linspace(0.9, 0.4, 60, dtype=np.float64).tolist(),
        get_params=lambda: {"max_iter": 60},
    )
    summary = model_module.summarize_mlp_convergence(
        model,
        convergence_warning_messages=(
            "Stochastic Optimizer: Maximum iterations (60) reached",
        ),
    )
    assert summary["convergence_warning_raised"] is True
    assert summary["n_iter"] == 60
    assert summary["max_iter"] == 60
    assert summary["reached_max_iter"] is True
    assert summary["loss_curve_length"] == 60
    assert summary["initial_loss"] == pytest.approx(0.9)
    assert summary["final_loss"] == pytest.approx(0.4)
    assert summary["tail_loss_delta"] < 0.0
    assert summary["tail_loss_slope"] < 0.0
    assert len(summary["loss_curve"]) == 60

    with pytest.raises(ValueError, match="loss_curve"):
        model_module.summarize_mlp_convergence(
            SimpleNamespace(
                n_iter_=2,
                loss_curve_=[1.0, np.nan],
                get_params=lambda: {"max_iter": 60},
            ),
            convergence_warning_messages=(),
        )


def test_m04_iter60_runner_is_internal_only_and_captures_warning(
    model_module, monkeypatch
):
    prepared = _prepared_screening(model_module)

    class WarningOnlyMLP:
        def fit(self, X, y):
            self.classes_ = np.array([0, 1], dtype=np.int64)
            self.n_iter_ = 60
            self.loss_curve_ = np.linspace(0.8, 0.5, 60).tolist()
            warnings.warn(
                "Maximum iterations (60) reached",
                ConvergenceWarning,
                stacklevel=2,
            )
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X):
            return np.tile(np.array([[0.75, 0.25]]), (len(X), 1))

        def get_params(self, deep=True):
            return dict(EXPECTED_M04_ITER60_CONFIG)

    monkeypatch.setattr(
        model_module,
        "build_m04_convergence_check_model",
        WarningOnlyMLP,
    )
    with pytest.raises(RuntimeError, match="allow_fit"):
        model_module.run_internal_m04_convergence_check(prepared)
    with pytest.raises(ValueError, match="Training/internal_validation"):
        model_module.run_internal_m04_convergence_check(
            _prepared_official_screening(model_module), allow_fit=True
        )

    result = model_module.run_internal_m04_convergence_check(
        prepared, allow_fit=True
    )
    diagnostics = result.fit_audit["mlp_convergence"]
    assert result.experiment_id == "M04_ITER60"
    assert diagnostics["convergence_warning_raised"] is True
    assert diagnostics["n_iter"] == 60
    assert diagnostics["loss_curve_length"] == 60
    assert result.fit_audit["evaluation_scope"] == "Training/internal_validation"


def test_internal_m04_reference_replays_frozen_config_on_supplied_scaler(
    model_module,
):
    X, y, record_keys, call_ids = _synthetic_e05(n_rows=1_600)
    prepared = model_module.prepare_internal_comparison(
        X=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        feature_names=EXPECTED_E05,
        split_function=_group_split_adapter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reference = model_module.run_internal_m04_reference(
            prepared, allow_fit=True
        )

    assert reference.experiment_id == "M04"
    assert dict(reference.model_config) == EXPECTED_M04_CONFIG
    assert reference.model.get_params()["max_iter"] == 30
    assert reference.scaler is prepared.scaler
    assert np.array_equal(reference.fit_audit["train_idx"], prepared.train_idx)
    assert np.array_equal(
        reference.fit_audit["validation_idx"], prepared.validation_idx
    )
    assert np.array_equal(
        reference.fit_audit["training_record_keys"],
        prepared.training_record_keys,
    )
    assert np.array_equal(
        reference.fit_audit["validation_record_keys"],
        prepared.evaluation_record_keys,
    )


def test_m04_iter60_comparison_is_paired_to_original_m04(model_module):
    prepared = _prepared_screening(model_module)
    keys = prepared.evaluation_record_keys
    truth = prepared.y_evaluation
    original_pred = np.zeros_like(truth)
    candidate_pred = original_pred.copy()
    candidate_pred[:4] = truth[:4]

    def fake_result(experiment_id, config, prediction, convergence=None):
        metrics = model_module.calculate_prediction_metrics(truth, prediction)
        audit = {
            "training_scope": prepared.training_scope,
            "evaluation_scope": prepared.evaluation_scope,
            "scaler_fit_scope": prepared.scaler_fit_scope,
            "training_record_keys": prepared.training_record_keys.copy(),
            "validation_record_keys": keys.copy(),
            "scaler_fit_record_keys": prepared.scaler_fit_record_keys.copy(),
            "train_idx": prepared.train_idx.copy(),
            "validation_idx": prepared.validation_idx.copy(),
            "X_train_fingerprint": "same-train-X",
            "X_evaluation_fingerprint": "same-validation-X",
            "y_train_fingerprint": "same-train-y",
            "y_evaluation_fingerprint": "same-validation-y",
            "model_config_fingerprint": model_module.config_fingerprint(config),
            "runtime_seconds": 0.0,
            "peak_memory_bytes": None,
        }
        if convergence is not None:
            audit["mlp_convergence"] = convergence
        return model_module.ModelExperimentResult(
            experiment_id=experiment_id,
            model_name="MLPClassifier",
            model_config=dict(config),
            feature_names=EXPECTED_E05,
            train_shape=prepared.X_train_scaled.shape,
            evaluation_shape=prepared.X_evaluation_scaled.shape,
            y_true=truth.copy(),
            y_pred=prediction.copy(),
            probabilities=np.column_stack(
                [1.0 - prediction * 0.6 - 0.2, prediction * 0.6 + 0.2]
            ),
            metrics=metrics,
            scaler=prepared.scaler,
            model=SimpleNamespace(),
            fit_audit=audit,
        )

    m04 = fake_result("M04", EXPECTED_M04_CONFIG, original_pred)
    convergence = {
        "convergence_warning_raised": True,
        "n_iter": 60,
        "max_iter": 60,
        "reached_max_iter": True,
        "loss_curve_length": 60,
        "initial_loss": 0.8,
        "final_loss": 0.5,
        "tail_loss_delta": -0.05,
        "tail_loss_slope": -0.005,
        "loss_curve": tuple(np.linspace(0.8, 0.5, 60)),
    }
    candidate = fake_result(
        "M04_ITER60", EXPECTED_M04_ITER60_CONFIG, candidate_pred, convergence
    )
    comparison = model_module.compare_m04_convergence_check(m04, candidate)
    transitions = (
        comparison["m04_wrong_to_iter60_correct"]
        + comparison["m04_correct_to_iter60_wrong"]
    )
    assert comparison["prediction_disagreement_count"] == transitions
    assert comparison["correct_count_delta"] == (
        comparison["m04_wrong_to_iter60_correct"]
        - comparison["m04_correct_to_iter60_wrong"]
    )
    assert comparison["accuracy_delta"] == pytest.approx(
        comparison["iter60_accuracy"] - comparison["m04_accuracy"]
    )
    assert comparison["convergence_warning_raised"] is True

    changed_audit = dict(candidate.fit_audit)
    changed_audit["X_evaluation_fingerprint"] = "different"
    with pytest.raises(ValueError, match="X_evaluation_fingerprint"):
        model_module.compare_m04_convergence_check(
            m04, replace(candidate, fit_audit=changed_audit)
        )


def test_notebook_appends_internal_only_checkpoint_o_with_false_switch():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell.get("source", [])) for cell in notebook["cells"]}
    ordered_ids = [cell["id"] for cell in notebook["cells"]]
    assert ordered_ids[-2:] == ["checkpoint-o-md", "checkpoint-o-code"]
    source = cells["checkpoint-o-code"]
    assert "RUN_INTERNAL_M04_CONVERGENCE_CHECK = False" in source
    assert "run_internal_m04_convergence_check" in source
    assert "compare_m04_convergence_check" in source
    assert "ConvergenceWarning" in source
    assert "n_iter" in source
    assert "loss_curve" in source
    assert "prepared_official" not in source
    assert "run_promoted_official_models" not in source
    assert "RUN_OFFICIAL" not in source
    assert "Maximum iterations (30) reached" in NOTEBOOK_PATH.read_text(
        encoding="utf-8"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")


def test_checkpoint_o_passes_split_aligned_tracking_to_internal_diagnostics(
    model_module, monkeypatch
):
    prepared = _prepared(model_module)
    X, y, record_keys, call_ids = _synthetic_e05()
    tracking = pd.DataFrame(
        {
            "record_key": record_keys,
            "call_id": call_ids,
            "startAt_ms": np.tile(np.array([0.0, 30.0, 60.0, 90.0]), 20),
            "endAt_ms": np.tile(np.array([50.0, 80.0, 110.0, 140.0]), 20),
            "duration_ms": 50.0,
        }
    )
    prediction = np.zeros_like(prepared.y_evaluation)
    synthetic_result = SimpleNamespace(
        fit_audit={
            "training_record_keys": prepared.training_record_keys.copy(),
            "validation_record_keys": prepared.evaluation_record_keys.copy(),
        },
        y_true=prepared.y_evaluation.copy(),
        y_pred=prediction,
        probabilities=np.tile(
            np.array([[0.75, 0.25]], dtype=np.float64),
            (len(prediction), 1),
        ),
        metrics=model_module.calculate_prediction_metrics(
            prepared.y_evaluation, prediction
        ),
        scaler=prepared.scaler,
    )
    monkeypatch.setattr(
        model_module,
        "run_internal_m04_convergence_check",
        lambda prepared, allow_fit: synthetic_result,
    )
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(
        next(
            cell for cell in notebook["cells"] if cell["id"] == "checkpoint-o-code"
        )["source"]
    )
    diagnostic_prefix = source.split("m04_iter60_report =", maxsplit=1)[0]
    diagnostic_prefix = diagnostic_prefix.replace(
        "RUN_INTERNAL_M04_CONVERGENCE_CHECK = False",
        "RUN_INTERNAL_M04_CONVERGENCE_CHECK = True",
        1,
    )
    namespace = {
        "M04_MLP_CONFIG": model_module.M04_MLP_CONFIG,
        "prepared_internal": prepared,
        "screening_results": [
            SimpleNamespace(
                experiment_id="M04",
                scaler=prepared.scaler,
                model_config=dict(EXPECTED_M04_CONFIG),
            )
        ],
        "tracking_training": tracking,
        "internal_training_tracking": tracking.iloc[
            prepared.train_idx
        ].reset_index(drop=True),
        "internal_validation_tracking": tracking.iloc[
            prepared.validation_idx
        ].reset_index(drop=True),
        "build_standalone_diagnostic_report": (
            model_module.build_standalone_diagnostic_report
        ),
    }

    exec(compile(diagnostic_prefix, "checkpoint-o-code", "exec"), namespace)

    report = namespace["m04_iter60_base_report"]
    assert report["global_metrics"]["sample_count"] == len(
        prepared.validation_idx
    )


def test_checkpoint_o_preserves_iter60_and_rebuilds_only_stale_m04_reference(
    model_module, monkeypatch, capsys
):
    current_scaler = object()
    stale_scaler = object()
    prepared = SimpleNamespace(scaler=current_scaler)
    stale_reference = SimpleNamespace(experiment_id="M04", scaler=stale_scaler)
    current_reference = SimpleNamespace(
        experiment_id="M04",
        scaler=current_scaler,
        model_config=dict(EXPECTED_M04_CONFIG),
    )
    preserved_iter60 = SimpleNamespace(
        experiment_id="M04_ITER60", scaler=current_scaler
    )

    def forbidden_iter60_fit(*args, **kwargs):
        raise AssertionError("M04-60 must not be fitted when its result exists")

    monkeypatch.setattr(
        model_module,
        "run_internal_m04_convergence_check",
        forbidden_iter60_fit,
    )
    monkeypatch.setattr(
        model_module,
        "run_internal_m04_reference",
        lambda prepared, allow_fit: current_reference,
    )
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(
        next(
            cell for cell in notebook["cells"] if cell["id"] == "checkpoint-o-code"
        )["source"]
    )
    setup_prefix = source.split("m04_iter60_base_report =", maxsplit=1)[0]
    setup_prefix = setup_prefix.replace(
        "RUN_REBUILD_M04_REFERENCE = False",
        "RUN_REBUILD_M04_REFERENCE = True",
        1,
    )
    namespace = {
        "M04_MLP_CONFIG": model_module.M04_MLP_CONFIG,
        "prepared_internal": prepared,
        "screening_results": [stale_reference],
        "m04_iter60_result": preserved_iter60,
    }

    exec(compile(setup_prefix, "checkpoint-o-code", "exec"), namespace)

    assert namespace["m04_iter60_result"] is preserved_iter60
    assert namespace["m04_internal_result"] is current_reference
    assert namespace["m04_internal_result"].scaler is current_scaler
    assert "STALE M04-30 reference" in capsys.readouterr().out


def test_checkpoint_o_stops_on_stale_m04_reference_without_rebuild_gate(
    model_module, monkeypatch
):
    current_scaler = object()
    monkeypatch.setattr(
        model_module,
        "run_internal_m04_convergence_check",
        lambda *args, **kwargs: pytest.fail("M04-60 must not be fitted"),
    )
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(
        next(
            cell for cell in notebook["cells"] if cell["id"] == "checkpoint-o-code"
        )["source"]
    )
    setup_prefix = source.split("m04_iter60_base_report =", maxsplit=1)[0]
    namespace = {
        "M04_MLP_CONFIG": model_module.M04_MLP_CONFIG,
        "prepared_internal": SimpleNamespace(scaler=current_scaler),
        "screening_results": [
            SimpleNamespace(experiment_id="M04", scaler=object())
        ],
        "m04_iter60_result": SimpleNamespace(
            experiment_id="M04_ITER60", scaler=current_scaler
        ),
    }

    with pytest.raises(RuntimeError, match="RUN_REBUILD_M04_REFERENCE"):
        exec(compile(setup_prefix, "checkpoint-o-code", "exec"), namespace)


def test_checkpoint_o_refreshes_new_reference_helper_without_kernel_restart(
    model_module, monkeypatch
):
    monkeypatch.delattr(model_module, "run_internal_m04_reference")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(
        next(
            cell for cell in notebook["cells"] if cell["id"] == "checkpoint-o-code"
        )["source"]
    )
    import_prefix = source.split("m04_config_diff =", maxsplit=1)[0]
    namespace = {}

    exec(compile(import_prefix, "checkpoint-o-code", "exec"), namespace)

    assert callable(namespace["run_internal_m04_reference"])
