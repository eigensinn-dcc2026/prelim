from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcc_data import mission2_candidate_features as candidate_features
from dcc_data.mission2_candidate_features import (
    CANDIDATE_REGISTRY,
    CANDIDATE_EXTRACTION_CONFIG,
    build_candidate_feature_cache,
    freeze_stage2_precommit,
    load_candidate_feature_cache,
    lock_candidate_selection,
    require_stage2_precommit,
    stage2_config_fingerprint,
    stage1_feasibility_diagnostics,
    stage2_training_group_screen,
    validate_candidate_names,
)
from dcc_data.mission2_experiments import (
    BASELINE_FEATURE_NAMES,
    EXPECTED_CACHE_METADATA,
    MFCC_CONFIG,
    REFERENCE_MODEL_CONFIG,
    AlignedSplit,
    append_comparison_row,
    assemble_experiment_matrix,
    assert_single_factor_comparison,
    build_duration_feature_frame,
    build_experiment_report,
    duration_ms_from_bounds,
    empty_experiment_table,
    fit_reference_logistic_regression,
    group_split_indices,
    make_record_key,
    prediction_metrics,
    validate_cache_metadata,
    validate_model_feature_names,
)


def _aligned_split(split: str, rows: int = 6) -> AlignedSplit:
    record_keys = np.array(
        [make_record_key(split, f"call-{index // 2}", index % 2) for index in range(rows)]
    )
    tracking = pd.DataFrame(
        {
            "record_key": record_keys,
            "split": [split] * rows,
            "call_id": [f"call-{index // 2}" for index in range(rows)],
            "annotation_row": [index % 2 for index in range(rows)],
            "startAt_ms": np.arange(rows, dtype=float) * 100.0,
            "endAt_ms": np.arange(rows, dtype=float) * 100.0 + 50.0,
            "duration_ms": np.full(rows, 50.0),
        }
    )
    X_mfcc = np.arange(rows * 40, dtype=float).reshape(rows, 40)
    y = np.array([0, 1] * (rows // 2), dtype=np.int64)
    return AlignedSplit(split, X_mfcc, y, record_keys, tracking)


def _synthetic_stage2_result(
    candidate_name: str, feature_names: tuple[str, ...]
) -> dict[str, object]:
    return {
        "provenance": {
            "source_signature": "synthetic-v2-training",
            "config_fingerprint": stage2_config_fingerprint(_stage2_config()),
        },
        "scope": {
            "source_split": "Training",
            "source_rows": 10,
            "official_validation_used": False,
        },
        "split_audit": pd.DataFrame([{
            "seed": 42, "test_size": 0.2, "train_rows": 8,
            "internal_validation_rows": 2, "train_calls": 4,
            "internal_validation_calls": 1, "call_id_overlap": 0,
        }]),
        "baseline_metrics": {"accuracy": 0.8},
        "candidate_performance": pd.DataFrame([{
            "candidate_name": candidate_name,
            "added_features": list(feature_names),
            "accuracy": 0.81,
            "accuracy_delta_vs_mfcc40": 0.01,
            "official_validation_used": False,
        }]),
    }


def _stage2_config() -> dict[str, object]:
    return {
        "mfcc": MFCC_CONFIG,
        "reference_model": REFERENCE_MODEL_CONFIG,
        "candidate_extraction": CANDIDATE_EXTRACTION_CONFIG,
    }


def test_duration_uses_millisecond_annotation_contract() -> None:
    start_at_ms = np.array([0.0, 125.5, 1_000.0])
    end_at_ms = np.array([250.0, 625.5, 1_001.0])

    observed = duration_ms_from_bounds(start_at_ms, end_at_ms)

    np.testing.assert_array_equal(observed, np.array([250.0, 500.0, 1.0]))


def test_duration_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="endAt_ms must be greater"):
        duration_ms_from_bounds(np.array([10.0]), np.array([10.0]))


def test_forbidden_tracking_fields_cannot_be_model_features() -> None:
    for forbidden in [
        "startAt",
        "endAt",
        "startAt_ms",
        "endAt_ms",
        "utt_idx",
        "crop_start_frame",
        "file_path",
        "overlap_status",
    ]:
        with pytest.raises(ValueError, match="forbidden model feature"):
            validate_model_feature_names([*BASELINE_FEATURE_NAMES, forbidden])


def test_duration_adds_exactly_one_aligned_feature() -> None:
    split = _aligned_split("Training")
    duration_frame = build_duration_feature_frame(split.tracking)

    X_candidate, candidate_names = assemble_experiment_matrix(
        split,
        duration_frame,
        ["duration_ms"],
    )

    assert X_candidate.shape == (6, 41)
    assert candidate_names == (*BASELINE_FEATURE_NAMES, "duration_ms")
    np.testing.assert_array_equal(X_candidate[:, :40], split.X_mfcc)
    np.testing.assert_array_equal(X_candidate[:, 40], np.full(6, 50.0))


def test_duration_experiment_matrix_uses_reference_float32_path() -> None:
    original = _aligned_split("Training")
    split = AlignedSplit(
        split=original.split,
        X_mfcc=original.X_mfcc.astype(np.float32),
        y=original.y,
        record_keys=original.record_keys,
        tracking=original.tracking,
    )
    duration_frame = build_duration_feature_frame(split.tracking)

    X_candidate, _ = assemble_experiment_matrix(
        split,
        duration_frame,
        ["duration_ms"],
    )

    assert duration_frame["duration_ms"].dtype == np.dtype(np.float64)
    assert X_candidate.dtype == np.dtype(np.float32)


def test_single_factor_guard_rejects_changed_mfcc_values() -> None:
    baseline = _aligned_split("Training")
    duration_frame = build_duration_feature_frame(baseline.tracking)
    candidate_X, candidate_names = assemble_experiment_matrix(
        baseline, duration_frame, ["duration_ms"]
    )
    changed = candidate_X.copy()
    changed[0, 0] += 1.0

    with pytest.raises(AssertionError, match="MFCC40 values changed"):
        assert_single_factor_comparison(
            baseline.X_mfcc,
            changed,
            baseline.y,
            baseline.y.copy(),
            BASELINE_FEATURE_NAMES,
            candidate_names,
            ["duration_ms"],
        )


def test_extra_feature_alignment_rejects_missing_or_duplicate_record_key() -> None:
    baseline = _aligned_split("Training")
    duration_frame = build_duration_feature_frame(baseline.tracking)

    with pytest.raises(AssertionError, match="row alignment failed"):
        assemble_experiment_matrix(
            baseline,
            duration_frame.iloc[:-1].copy(),
            ["duration_ms"],
        )

    duplicate = pd.concat([duration_frame, duration_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="record_key is not unique"):
        assemble_experiment_matrix(baseline, duplicate, ["duration_ms"])


def test_scaler_is_fit_from_training_rows_only() -> None:
    X_train = np.array([[-2.0], [0.0], [2.0], [4.0]])
    y_train = np.array([0, 0, 1, 1])
    X_validation = np.array([[100.0], [200.0]])
    y_validation = np.array([0, 1])

    result = fit_reference_logistic_regression(
        experiment_id="synthetic",
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_validation,
        y_evaluation=y_validation,
        feature_names=("waveform_feature",),
        training_scope="Training",
        evaluation_scope="Validation",
    )

    np.testing.assert_allclose(result.scaler.mean_, np.array([1.0]))
    assert result.fit_audit == {
        "scaler_fit_scope": "Training",
        "model_fit_scope": "Training",
        "evaluation_scope": "Validation",
        "scaler_fit_rows": 4,
        "model_fit_rows": 4,
        "evaluation_rows": 2,
    }


def test_reference_logistic_regression_preserves_float32_numerical_path() -> None:
    X_train = np.array(
        [
            [-2.0, 0.5],
            [-1.0, 1.5],
            [1.0, -1.5],
            [2.0, -0.5],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    X_validation = np.array(
        [[-0.5, 0.25], [0.5, -0.25]],
        dtype=np.float32,
    )
    y_validation = np.array([0, 1], dtype=np.int64)

    result = fit_reference_logistic_regression(
        experiment_id="float32-regression",
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_validation,
        y_evaluation=y_validation,
        feature_names=("waveform_feature_1", "waveform_feature_2"),
        training_scope="Training",
        evaluation_scope="Validation",
    )

    assert result.model.coef_.dtype == np.dtype(np.float32)
    assert result.y_pred.shape == y_validation.shape
    assert result.probabilities.shape == (len(y_validation), 2)
    assert result.metrics["sample_count"] == len(y_validation)


def test_group_split_has_zero_call_overlap() -> None:
    groups = np.repeat([f"call-{index}" for index in range(10)], 2)
    train_idx, validation_idx, audit = group_split_indices(
        groups, seed=42, test_size=0.2
    )

    assert set(groups[train_idx]).isdisjoint(set(groups[validation_idx]))
    assert audit["call_id_overlap"] == 0
    assert len(train_idx) + len(validation_idx) == len(groups)


def test_prediction_metrics_uses_true_class_support_for_recalls() -> None:
    metrics = prediction_metrics(
        np.array([0, 0, 0, 1, 1]),
        np.array([0, 1, 1, 1, 0]),
    )

    assert metrics["confusion_matrix"] == [[1, 2], [1, 1]]
    assert metrics["accuracy"] == 0.4
    assert metrics["recall_0"] == pytest.approx(1 / 3)
    assert metrics["recall_1"] == pytest.approx(1 / 2)


def test_experiment_report_and_comparison_table_use_observed_predictions() -> None:
    training = _aligned_split("Training")
    validation = _aligned_split("Validation")
    distinct_durations_ms = np.arange(1, 7, dtype=float) * 10.0
    for split in (training, validation):
        split.tracking["duration_ms"] = distinct_durations_ms
        split.tracking["endAt_ms"] = (
            split.tracking["startAt_ms"] + distinct_durations_ms
        )
    result = fit_reference_logistic_regression(
        experiment_id="synthetic",
        X_train=training.X_mfcc[:, :1],
        y_train=training.y,
        X_evaluation=validation.X_mfcc[:, :1],
        y_evaluation=validation.y,
        feature_names=("waveform_feature",),
        training_scope="Training",
        evaluation_scope="Validation",
    )
    report = build_experiment_report(
        result,
        training_tracking=training.tracking,
        evaluation_tracking=validation.tracking,
        baseline_report=None,
    )

    table = append_comparison_row(empty_experiment_table(), report)

    assert len(table) == 1
    assert table.loc[0, "accuracy"] == report["global_metrics"]["accuracy"]
    assert table.loc[0, "train_rows"] == 6
    assert table.loc[0, "validation_rows"] == 6


def test_cache_metadata_requires_exact_reference_contract() -> None:
    metadata = dict(EXPECTED_CACHE_METADATA["Training"])
    validate_cache_metadata(metadata, "Training")
    changed = json.loads(json.dumps(metadata))
    changed["mfcc_config"]["hop_length"] = 80

    with pytest.raises(AssertionError, match="cache metadata mismatch"):
        validate_cache_metadata(changed, "Training")


def test_stage1_screening_is_training_only_feasibility_not_a_ranking() -> None:
    rows = 8
    sample = pd.DataFrame(
        {
            "split": ["Training"] * rows,
            "speaker": [0, 1] * 4,
            "call_id": [f"call-{index // 2}" for index in range(rows)],
            "record_key": [f"Training|call-{index // 2}|{index % 2}" for index in range(rows)],
            "rms": np.linspace(0.01, 0.08, rows),
            "mfcc_01_mean": np.linspace(-2.0, 2.0, rows),
            "mfcc_01_std": np.linspace(1.0, 2.0, rows),
        }
    )

    diagnostics = stage1_feasibility_diagnostics(
        sample,
        ["rms"],
        expected_training_rows=rows,
    )

    assert diagnostics["scope"]["purpose"] == "feasibility_diagnostic_only"
    assert "rank" not in diagnostics["feature_summary"].columns
    assert "recommended_experiment_id" not in diagnostics["feature_summary"].columns


def test_candidate_registry_rejects_overlap_and_unknown_metadata() -> None:
    assert CANDIDATE_REGISTRY["rms"].waveform_only
    for invalid in ["overlap_status", "call_id", "startAt_ms", "unknown_feature"]:
        with pytest.raises(ValueError):
            validate_candidate_names([invalid])


def test_full_candidate_cache_requires_explicit_large_run_acknowledgement(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "split": ["Training"],
            "record_key": ["Training|call-a|0"],
            "call_id": ["call-a"],
            "audio_path": ["a.wav"],
            "crop_start_frame": [0],
            "crop_end_frame": [8],
            "crop_ok": [True],
        }
    )

    with pytest.raises(RuntimeError, match="allow_full_extraction=True"):
        build_candidate_feature_cache(
            manifest,
            data_root=tmp_path,
            cache_root=tmp_path / "cache",
            split="Training",
            feature_names=["rms"],
            source_signature="synthetic-source",
            allow_full_extraction=False,
        )


def test_candidate_cache_miss_never_starts_extraction(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "split": ["Training"],
            "record_key": ["Training|call-a|0"],
        }
    )

    with pytest.raises(FileNotFoundError, match="automatic extraction is disabled"):
        load_candidate_feature_cache(
            manifest,
            cache_root=tmp_path / "missing-cache",
            split="Training",
            feature_names=["rms"],
            source_signature="synthetic-source",
        )


def test_stage2_rejects_official_validation_before_any_fit() -> None:
    validation = _aligned_split("Validation")
    candidate_frame = pd.DataFrame(
        {"record_key": validation.record_keys, "rms": np.linspace(0.1, 0.6, 6)}
    )

    with pytest.raises(AssertionError, match="Training only"):
        stage2_training_group_screen(
            validation,
            candidate_frame,
            {"candidate-rms": ["rms"]},
            minimum_training_rows=1,
        )


def test_stage2_selection_lock_requires_exact_training_result() -> None:
    stage2_result = {
        "scope": {"source_split": "Training", "official_validation_used": False},
        "candidate_performance": pd.DataFrame(
            [{"candidate_name": "candidate-rms", "added_features": ["rms"]}]
        ),
    }

    locked = lock_candidate_selection(
        stage2_result,
        selection_name="candidate-rms",
        feature_names=["rms"],
    )

    assert locked.feature_names == ("rms",)
    assert locked.evidence_stage == "full_or_larger_training_call_id_group_split"


def test_frozen_stage2_precommit_accepts_actual_training_evidence() -> None:
    result = _synthetic_stage2_result("zcr", ("zcr",))
    record = freeze_stage2_precommit(
        result, experiment_id="E03", candidate_name="zcr",
        feature_names=("zcr",), config=_stage2_config(),
        source_signature="synthetic-v2-training",
    )
    assert require_stage2_precommit(
        record, experiment_id="E03", candidate_name="zcr",
        feature_names=("zcr",), config=_stage2_config(),
        source_signature="synthetic-v2-training",
    ) is record
    assert record.accuracy == pytest.approx(0.81)
    assert record.accuracy_delta_vs_mfcc40 == pytest.approx(0.01)
    assert record.train_rows == 8 and record.call_id_overlap == 0


def test_frozen_stage2_precommit_rejects_candidate_mismatch() -> None:
    record = freeze_stage2_precommit(
        _synthetic_stage2_result("zcr", ("zcr",)),
        experiment_id="E03", candidate_name="zcr", feature_names=("zcr",),
        config=_stage2_config(), source_signature="synthetic-v2-training",
    )
    with pytest.raises(RuntimeError, match="candidate ID mismatch"):
        require_stage2_precommit(
            record, experiment_id="E04", candidate_name="zcr",
            feature_names=("zcr",), config=_stage2_config(),
            source_signature="synthetic-v2-training",
        )


def test_frozen_stage2_precommit_rejects_feature_mismatch() -> None:
    record = freeze_stage2_precommit(
        _synthetic_stage2_result("zcr", ("zcr",)),
        experiment_id="E03", candidate_name="zcr", feature_names=("zcr",),
        config=_stage2_config(), source_signature="synthetic-v2-training",
    )
    with pytest.raises(RuntimeError, match="feature tuple mismatch"):
        require_stage2_precommit(
            record, experiment_id="E03", candidate_name="zcr",
            feature_names=("spectral_centroid",), config=_stage2_config(),
            source_signature="synthetic-v2-training",
        )


def test_frozen_stage2_precommit_rejects_metric_and_config_fingerprint_mismatch() -> None:
    result = _synthetic_stage2_result("zcr", ("zcr",))
    result["candidate_performance"].loc[0, "accuracy_delta_vs_mfcc40"] = 0.02
    with pytest.raises(AssertionError, match="metric/delta mismatch"):
        freeze_stage2_precommit(
            result, experiment_id="E03", candidate_name="zcr",
            feature_names=("zcr",), config=_stage2_config(),
            source_signature="synthetic-v2-training",
        )
    record = freeze_stage2_precommit(
        _synthetic_stage2_result("zcr", ("zcr",)),
        experiment_id="E03", candidate_name="zcr", feature_names=("zcr",),
        config=_stage2_config(), source_signature="synthetic-v2-training",
    )
    with pytest.raises(RuntimeError, match="metric/provenance fingerprint mismatch"):
        require_stage2_precommit(
            replace(record, accuracy=0.82), experiment_id="E03",
            candidate_name="zcr", feature_names=("zcr",),
            config=_stage2_config(), source_signature="synthetic-v2-training",
        )
    changed_config = {**_stage2_config(), "reference_model": {"max_iter": 2}}
    with pytest.raises(RuntimeError, match="config fingerprint mismatch"):
        require_stage2_precommit(
            record, experiment_id="E03", candidate_name="zcr",
            feature_names=("zcr",), config=changed_config,
            source_signature="synthetic-v2-training",
        )


def test_frozen_stage2_precommit_rejects_missing_record() -> None:
    with pytest.raises(RuntimeError, match="precommit is missing"):
        require_stage2_precommit(
            None, experiment_id="E03", candidate_name="zcr",
            feature_names=("zcr",), config=_stage2_config(),
            source_signature="synthetic-v2-training",
        )


def test_frozen_stage2_precommit_rejects_stale_stage2_provenance() -> None:
    result = _synthetic_stage2_result("zcr", ("zcr",))
    with pytest.raises(AssertionError, match="Training data signature mismatch"):
        freeze_stage2_precommit(
            result, experiment_id="E03", candidate_name="zcr",
            feature_names=("zcr",), config=_stage2_config(),
            source_signature="different-training-data",
        )
    result["provenance"]["config_fingerprint"] = "old-config"
    with pytest.raises(AssertionError, match="config fingerprint mismatch"):
        freeze_stage2_precommit(
            result, experiment_id="E03", candidate_name="zcr",
            feature_names=("zcr",), config=_stage2_config(),
            source_signature="synthetic-v2-training",
        )


def test_notebook_freezes_manual_e03_e04_from_actual_stage2_results() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "mission2_feature_experiments.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }

    namespace = {
        "CANDIDATE_REGISTRY": CANDIDATE_REGISTRY,
        "Path": Path,
        "REPO_ROOT": notebook_path.parents[1],
        "os": __import__("os"),
        "pd": pd,
    }
    exec(
        compile(code_by_id["m2-fe-config"], "m2-fe-config", "exec"),
        namespace,
    )

    assert namespace["PRECOMMITTED_EXPERIMENTS"] == {
        "E03": ("zcr",),
        "E04": ("spectral_centroid",),
    }
    assert namespace["RUN_SELECTED_FEATURE_EXPERIMENT"] is False
    assert namespace["RUN_BASELINE_REPRODUCTION"] is False
    assert namespace["CANDIDATE_SELECTION_LOCKED"] is False
    assert "RUN_E03" not in namespace and "RUN_E04" not in namespace
    for feature_names in namespace["PRECOMMITTED_EXPERIMENTS"].values():
        assert validate_candidate_names(feature_names) == feature_names

    for experiment_id, feature_names in namespace["PRECOMMITTED_EXPERIMENTS"].items():
        selection_namespace = dict(namespace)
        selection_namespace.update(
            {
                "LOCK_STAGE2_SELECTION": True,
                "SELECTED_EXPERIMENT_ID": experiment_id,
                "SELECTED_CANDIDATE_NAME": feature_names[0],
                "SELECTED_FEATURE_NAMES": feature_names,
                "lock_candidate_selection": lock_candidate_selection,
                "freeze_stage2_precommit": freeze_stage2_precommit,
                "stage2_result": _synthetic_stage2_result(feature_names[0], feature_names),
                "MFCC_CONFIG": MFCC_CONFIG,
                "REFERENCE_MODEL_CONFIG": REFERENCE_MODEL_CONFIG,
                "CANDIDATE_EXTRACTION_CONFIG": CANDIDATE_EXTRACTION_CONFIG,
                "eda_manifest_meta": {"source_signature": "synthetic-v2-training"},
            }
        )
        exec(
            compile(code_by_id["m2-fe-cpd-lock"], "m2-fe-cpd-lock", "exec"),
            selection_namespace,
        )
        assert selection_namespace["locked_candidate"].feature_names == feature_names
        assert selection_namespace["CANDIDATE_SELECTION_LOCKED"] is True
        assert selection_namespace["frozen_precommit"].experiment_id == experiment_id

        missing_stage2 = dict(selection_namespace)
        missing_stage2.pop("stage2_result")
        with pytest.raises(RuntimeError, match="run Training-only Stage 2"):
            exec(compile(code_by_id["m2-fe-cpd-lock"], "m2-fe-cpd-lock", "exec"), missing_stage2)

    comparison_namespace = {
        "append_comparison_row": append_comparison_row,
        "display": lambda value: None,
        "empty_experiment_table": empty_experiment_table,
        "SAVE_DECISION_TABLE": False,
    }
    exec(
        compile(code_by_id["m2-fe-cpg-table"], "m2-fe-cpg-table", "exec"),
        comparison_namespace,
    )
    assert comparison_namespace["experiment_table"].empty


def test_notebook_training_only_precombination_diagnostic_contract() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "mission2_feature_experiments.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }

    helper_namespace: dict[str, object] = {}
    exec(
        compile(
            code_by_id["m2-fe-cph-helpers"],
            "m2-fe-cph-helpers",
            "exec",
        ),
        helper_namespace,
    )

    diagnostic_features = (
        "spectral_rolloff",
        "zcr",
        "spectral_centroid",
    )
    diagnostic_pairs = (
        ("spectral_rolloff", "zcr"),
        ("spectral_rolloff", "spectral_centroid"),
        ("zcr", "spectral_centroid"),
    )
    assert helper_namespace["RUN_TRAINING_ONLY_COMBINATION_DIAGNOSTICS"] is False
    assert helper_namespace["COMBINATION_DIAGNOSTIC_FEATURES"] == diagnostic_features
    assert helper_namespace["COMBINATION_DIAGNOSTIC_PAIRS"] == diagnostic_pairs

    fixed_split_audit = {
        "seed": 42,
        "test_size": 0.2,
        "train_rows": 711_467,
        "internal_validation_rows": 177_891,
        "train_calls": 23_360,
        "internal_validation_calls": 5_840,
        "call_id_overlap": 0,
    }
    validate_scope = helper_namespace[
        "validate_training_only_combination_diagnostic_scope"
    ]
    validate_scope(
        source_split="Training",
        source_rows=889_358,
        official_validation_used=False,
        split_audit=fixed_split_audit,
        feature_names=diagnostic_features,
        feature_pairs=diagnostic_pairs,
    )
    with pytest.raises(AssertionError, match="Training-only"):
        validate_scope(
            source_split="Validation",
            source_rows=111_947,
            official_validation_used=True,
            split_audit=fixed_split_audit,
            feature_names=diagnostic_features,
            feature_pairs=diagnostic_pairs,
        )

    candidate_values = pd.DataFrame(
        {
            "spectral_rolloff": [1.0, 2.0, 3.0, 6.0, 5.0, 4.0],
            "zcr": [2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
            "spectral_centroid": [3.0, 2.0, 1.0, 2.0, 1.0, 3.0],
        }
    )
    redundancy = helper_namespace["build_candidate_pair_redundancy_table"](
        candidate_values,
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
    )
    assert list(redundancy.columns) == list(
        helper_namespace["CANDIDATE_PAIR_REDUNDANCY_COLUMNS"]
    )
    assert tuple(
        redundancy[["feature_a", "feature_b"]].itertuples(index=False, name=None)
    ) == diagnostic_pairs

    y_true = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    predictions = {
        "spectral_rolloff": np.array([0, 1, 1, 0, 0], dtype=np.int64),
        "zcr": np.array([0, 0, 0, 0, 1], dtype=np.int64),
        "spectral_centroid": np.array([0, 1, 0, 1, 0], dtype=np.int64),
    }
    complementarity = helper_namespace[
        "build_candidate_pair_complementarity_table"
    ](y_true, predictions)
    assert list(complementarity.columns) == list(
        helper_namespace["CANDIDATE_PAIR_COMPLEMENTARITY_COLUMNS"]
    )
    rolloff_zcr = complementarity.iloc[0]
    assert int(rolloff_zcr["both_correct"]) == 1
    assert int(rolloff_zcr["both_wrong"]) == 1
    assert int(rolloff_zcr["a_correct_b_wrong"]) == 2
    assert int(rolloff_zcr["a_wrong_b_correct"]) == 1
    assert int(rolloff_zcr["prediction_disagreement_count"]) == 3
    assert float(rolloff_zcr["prediction_disagreement_rate"]) == pytest.approx(0.6)
    assert int(rolloff_zcr["error_intersection_count"]) == 1
    assert int(rolloff_zcr["error_union_count"]) == 4
    assert float(rolloff_zcr["error_jaccard"]) == pytest.approx(0.25)

    identical_predictions = {
        feature: predictions["spectral_rolloff"].copy()
        for feature in diagnostic_features
    }
    identical = helper_namespace[
        "build_candidate_pair_complementarity_table"
    ](y_true, identical_predictions)
    assert (identical["prediction_disagreement_count"] == 0).all()
    assert (identical["prediction_disagreement_rate"] == 0.0).all()

    baseline_predictions = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    baseline_changes = helper_namespace["build_baseline_error_change_table"](
        y_true,
        baseline_predictions,
        predictions,
    )
    assert list(baseline_changes.columns) == list(
        helper_namespace["BASELINE_ERROR_CHANGE_COLUMNS"]
    )
    rolloff_change = baseline_changes.loc[
        baseline_changes["candidate"] == "spectral_rolloff"
    ].iloc[0]
    assert int(rolloff_change["baseline_wrong_candidate_correct"]) == 1
    assert int(rolloff_change["baseline_correct_candidate_wrong"]) == 1
    assert int(rolloff_change["net_correct_gain"]) == 0

    training = _aligned_split("Training")
    training_candidates = pd.DataFrame(
        {
            "record_key": training.record_keys,
            "spectral_rolloff": np.linspace(100.0, 600.0, 6),
            "zcr": np.linspace(0.1, 0.6, 6),
            "spectral_centroid": np.linspace(200.0, 700.0, 6),
        }
    )
    diagnostic_baseline, single_results = helper_namespace[
        "fit_training_only_single_feature_predictions"
    ](
        training,
        training_candidates,
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
    )
    assert diagnostic_baseline.feature_names == BASELINE_FEATURE_NAMES
    assert tuple(single_results) == diagnostic_features
    for feature, result in single_results.items():
        assert result.feature_names == (*BASELINE_FEATURE_NAMES, feature)
        assert result.evaluation_shape == (2, 41)
        assert "E05" not in result.experiment_id

    run_tree = ast.parse(code_by_id["m2-fe-cph-run"])
    with pytest.raises(RuntimeError, match="STOP"):
        exec(
            compile(code_by_id["m2-fe-cph-run"], "m2-fe-cph-run", "exec"),
            helper_namespace,
        )
    loaded_names = {
        node.id
        for node in ast.walk(run_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert {
        "aligned_validation",
        "validation_candidate_cache",
        "baseline_report",
        "group_split_indices",
        "run_feature_experiment",
    }.isdisjoint(loaded_names)
    assert not any(
        isinstance(node, ast.Name)
        and (node.id.startswith("E05") or "winner" in node.id.casefold())
        for node in ast.walk(run_tree)
    )


def test_notebook_e05_precommit_and_validated_cache_merge_contract() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "mission2_feature_experiments.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }

    assert {
        "m2-fe-e05-config",
        "m2-fe-e05-lock",
        "m2-fe-e05-cache-merge",
    }.issubset(code_by_id)

    namespace = {
        "CANDIDATE_REGISTRY": CANDIDATE_REGISTRY,
        "Path": Path,
        "REPO_ROOT": notebook_path.parents[1],
        "os": __import__("os"),
        "pd": pd,
    }
    exec(compile(code_by_id["m2-fe-config"], "m2-fe-config", "exec"), namespace)
    exec(
        compile(code_by_id["m2-fe-e05-config"], "m2-fe-e05-config", "exec"),
        namespace,
    )
    expected_features = ("spectral_rolloff", "zcr")
    assert namespace["PRECOMMITTED_EXPERIMENTS"] == {
        "E03": ("zcr",),
        "E04": ("spectral_centroid",),
    }
    assert namespace["PRECOMMITTED_COMBINATION_EXPERIMENTS"] == {
        "E05": expected_features,
    }
    # Committed source gates remain off even though historical outputs remain.
    assert namespace["LOCK_E05_SELECTION"] is False
    assert namespace["RUN_SELECTED_FEATURE_EXPERIMENT"] is False

    stale_namespace = dict(namespace)
    stale_namespace.update(
        {
            "LOCK_E05_SELECTION": True,
            "CANDIDATE_SELECTION_LOCKED": True,
            "locked_candidate": object(),
            "lock_candidate_selection": lock_candidate_selection,
        }
    )
    with pytest.raises(RuntimeError, match="stale"):
        exec(
            compile(code_by_id["m2-fe-e05-lock"], "m2-fe-e05-lock", "exec"),
            stale_namespace,
        )

    clean_namespace = dict(namespace)
    clean_namespace.update(
        {
            "LOCK_E05_SELECTION": True,
            "lock_candidate_selection": lock_candidate_selection,
            "freeze_stage2_precommit": freeze_stage2_precommit,
            "stage2_result": _synthetic_stage2_result(
                "spectral_rolloff+zcr", expected_features
            ),
            "MFCC_CONFIG": MFCC_CONFIG,
            "REFERENCE_MODEL_CONFIG": REFERENCE_MODEL_CONFIG,
            "CANDIDATE_EXTRACTION_CONFIG": CANDIDATE_EXTRACTION_CONFIG,
            "eda_manifest_meta": {"source_signature": "synthetic-v2-training"},
        }
    )
    exec(
        compile(code_by_id["m2-fe-e05-lock"], "m2-fe-e05-lock", "exec"),
        clean_namespace,
    )
    assert clean_namespace["E05_COMBINATION_SELECTION_LOCKED"] is True
    assert clean_namespace["locked_candidate"].feature_names == expected_features
    assert clean_namespace["frozen_precommit"].feature_names == expected_features
    assert clean_namespace["PRECOMMITTED_EXPERIMENTS"] == {
        "E03": ("zcr",),
        "E04": ("spectral_centroid",),
        "E05": expected_features,
    }

    validation = _aligned_split("Validation")
    reversed_keys = validation.record_keys[::-1]
    reversed_calls = validation.tracking["call_id"].to_numpy()[::-1]
    validation_caches = {
        "spectral_rolloff": pd.DataFrame(
            {
                "record_key": reversed_keys,
                "call_id": reversed_calls,
                "spectral_rolloff": np.linspace(100.0, 600.0, 6),
            }
        ),
        "zcr": pd.DataFrame(
            {
                "record_key": reversed_keys,
                "call_id": reversed_calls,
                "zcr": np.linspace(0.1, 0.6, 6),
            }
        ),
    }
    load_calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_load_candidate_feature_cache(
        manifest,
        *,
        cache_root,
        split,
        feature_names,
        source_signature,
    ):
        del manifest, cache_root, source_signature
        names = tuple(feature_names)
        load_calls.append((split, names))
        if split == "Training":
            return pd.DataFrame(
                columns=["record_key", "call_id", *names]
            )
        assert len(names) == 1
        return validation_caches[names[0]].copy()

    cache_namespace = dict(clean_namespace)
    cache_namespace.update(
        {
            "BASELINE_GATE_PASSED": True,
            "EXPECTED_UTTERANCE_COUNTS": {"Validation": 6},
            "aligned_validation": validation,
            "full_eda_manifest": pd.DataFrame(),
            "eda_manifest_meta": {"source_signature": "synthetic-v2-training"},
            "load_candidate_feature_cache": fake_load_candidate_feature_cache,
            "assemble_experiment_matrix": assemble_experiment_matrix,
            "BASELINE_FEATURE_NAMES": BASELINE_FEATURE_NAMES,
            "np": np,
            "require_stage2_precommit": require_stage2_precommit,
        }
    )
    missing_precommit = dict(cache_namespace)
    missing_precommit.pop("frozen_precommit")
    with pytest.raises(RuntimeError, match="precommit is missing"):
        exec(
            compile(code_by_id["m2-fe-e05-cache-merge"], "m2-fe-e05-cache-merge", "exec"),
            missing_precommit,
        )
    assert load_calls == []
    exec(
        compile(
            code_by_id["m2-fe-e05-cache-merge"],
            "m2-fe-e05-cache-merge",
            "exec",
        ),
        cache_namespace,
    )
    assert load_calls[1:] == [
        ("Validation", ("spectral_rolloff",)),
        ("Validation", ("zcr",)),
    ]
    assert ("Validation", expected_features) not in load_calls
    merged = cache_namespace["validation_candidate_cache"]
    assert list(merged.columns) == ["record_key", *expected_features]
    np.testing.assert_array_equal(
        merged["record_key"].to_numpy(dtype=str), validation.record_keys.astype(str)
    )
    matrix = cache_namespace["e05_validation_matrix_preview"]
    assert matrix.shape == (6, 42)
    assert matrix.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(matrix[:, :40], validation.X_mfcc)
    assert cache_namespace["e05_validation_feature_names"] == (
        *BASELINE_FEATURE_NAMES,
        *expected_features,
    )

    comparison_namespace = {
        "append_comparison_row": append_comparison_row,
        "display": lambda value: None,
        "empty_experiment_table": empty_experiment_table,
        "SAVE_DECISION_TABLE": False,
    }
    exec(
        compile(code_by_id["m2-fe-cpg-table"], "m2-fe-cpg-table", "exec"),
        comparison_namespace,
    )
    assert comparison_namespace["experiment_table"].empty
    assert set(namespace["PRECOMMITTED_COMBINATION_EXPERIMENTS"]) == {"E05"}


def test_analysis_signal_validates_and_pads_only_short_waveforms() -> None:
    for invalid in ([], [np.nan], [np.inf], [-np.inf]):
        with pytest.raises(ValueError, match="empty|NaN or inf"):
            candidate_features._analysis_signal(np.asarray(invalid))

    short = candidate_features._analysis_signal(np.array([[0.25, -0.5]]))
    assert short.dtype == np.float32
    assert short.shape == (512,)
    np.testing.assert_array_equal(short[:2], np.array([0.25, -0.5], dtype=np.float32))
    assert np.count_nonzero(short[2:]) == 0

    long = np.linspace(-0.5, 0.5, 513, dtype=np.float32)
    np.testing.assert_array_equal(candidate_features._analysis_signal(long), long)


def test_registered_audio_features_use_only_requested_names_in_order() -> None:
    sample_rate = 8_000
    times = np.arange(1_024) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 1_000 * times)).astype(np.float32)
    names = ("zcr", "peak_amplitude", "rms", "spectral_rolloff", "clipping_ratio")

    observed = candidate_features.extract_registered_audio_features(
        waveform, sample_rate, names
    )

    assert tuple(observed) == names
    assert np.isfinite(list(observed.values())).all()
    assert observed["rms"] == pytest.approx(
        np.sqrt(np.dot(waveform.astype(float), waveform.astype(float)) / len(waveform))
    )
    assert observed["peak_amplitude"] == pytest.approx(float(np.max(np.abs(waveform))))
    assert observed["clipping_ratio"] == pytest.approx(0.0)
    assert 0.20 < observed["zcr"] < 0.30
    assert 700.0 < observed["spectral_rolloff"] < 1_500.0

    with pytest.raises(ValueError, match="unexpected sample rate"):
        candidate_features.extract_registered_audio_features(waveform, 16_000, names)
    with pytest.raises(ValueError, match="unknown or forbidden"):
        candidate_features.extract_registered_audio_features(waveform, sample_rate, ("text",))


def test_short_audio_statistics_ignore_synthetic_fft_padding() -> None:
    waveform = np.array([-1.0, 1.0] * 4, dtype=np.float32)
    names = (
        "silence_ratio", "rms", "rms_frame_std", "peak_amplitude",
        "clipping_ratio", "zcr", "spectral_rolloff",
    )

    observed = candidate_features.extract_registered_audio_features(waveform, 8_000, names)

    assert tuple(observed) == names
    assert observed["rms"] == pytest.approx(1.0)
    assert observed["rms_frame_std"] == pytest.approx(0.0)
    assert observed["silence_ratio"] == pytest.approx(0.0)
    assert observed["peak_amplitude"] == pytest.approx(1.0)
    assert observed["clipping_ratio"] == pytest.approx(1.0)
    assert observed["zcr"] == pytest.approx(1.0)
    assert np.isfinite(observed["spectral_rolloff"])
    assert candidate_features.extract_registered_audio_features(
        np.array([-0.0, 0.5], dtype=np.float32), 8_000, ("zcr",)
    )["zcr"] == 0.0
    assert candidate_features.extract_registered_audio_features(
        np.array([0.5], dtype=np.float32), 8_000, ("zcr",)
    )["zcr"] == 0.0


def test_frame_rms_is_lazy_and_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = np.full(1_024, 0.25, dtype=np.float32)
    original_rms = candidate_features.librosa.feature.rms
    calls = 0

    def counted_rms(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_rms(*args, **kwargs)

    monkeypatch.setattr(candidate_features.librosa.feature, "rms", counted_rms)
    candidate_features.extract_registered_audio_features(waveform, 8_000, ("rms",))
    assert calls == 0
    candidate_features.extract_registered_audio_features(waveform, 8_000, ("rms_frame_std",))
    assert calls == 1
    candidate_features.extract_registered_audio_features(
        waveform, 8_000, ("silence_ratio", "rms_frame_std")
    )
    assert calls == 2


def test_candidate_frame_filters_split_preserves_order_and_applies_crops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = pd.DataFrame(
        [
            ("Training", "Training|a|0", "a", "a.wav", 0, 2, True),
            ("Validation", "Validation|v|0", "v", "missing.wav", 0, 2, True),
            ("Training", "Training|b|0", "b", "b.wav", 1, 3, True),
            ("Training", "Training|a|1", "a", "a.wav", 2, 4, True),
        ],
        columns=["split", "record_key", "call_id", "audio_path", "crop_start_frame", "crop_end_frame", "crop_ok"],
    )
    waveforms = {
        "a.wav": np.array([0.1, 0.1, 0.4, 0.4], dtype=np.float32),
        "b.wav": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    }
    reads: list[str] = []
    callbacks: list[tuple[int, int]] = []

    def fake_read(path: Path):
        reads.append(path.name)
        return waveforms[path.name], 8_000

    monkeypatch.setattr(candidate_features, "read_call_waveform", fake_read)
    frame = candidate_features.extract_candidate_feature_frame(
        manifest, data_root=tmp_path, split="Training", feature_names=("rms", "peak_amplitude"),
        on_call_done=lambda done, total: callbacks.append((done, total)),
    )

    assert reads == ["a.wav", "b.wav"]
    assert callbacks == [(1, 2), (2, 2)]
    assert list(frame.columns) == ["record_key", "call_id", "rms", "peak_amplitude"]
    assert frame["record_key"].tolist() == ["Training|a|0", "Training|b|0", "Training|a|1"]
    np.testing.assert_allclose(frame["rms"], [0.1, np.sqrt((0.3**2 + 0.4**2) / 2), 0.4])
    np.testing.assert_allclose(frame["peak_amplitude"], [0.1, 0.4, 0.4])


def test_candidate_frame_rejects_invalid_crop_and_duplicate_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = pd.DataFrame(
        [("Training", "Training|a|0", "a", "a.wav", 0, 4, True)],
        columns=["split", "record_key", "call_id", "audio_path", "crop_start_frame", "crop_end_frame", "crop_ok"],
    )
    monkeypatch.setattr(
        candidate_features, "read_call_waveform",
        lambda path: (np.ones(3, dtype=np.float32), 8_000),
    )
    with pytest.raises(AssertionError, match="invalid crop"):
        candidate_features.extract_candidate_feature_frame(
            manifest, data_root=tmp_path, split="Training", feature_names=("rms",)
        )
    with pytest.raises(AssertionError, match="record_key is not unique"):
        candidate_features.extract_candidate_feature_frame(
            pd.concat([manifest, manifest], ignore_index=True),
            data_root=tmp_path, split="Training", feature_names=("rms",),
        )


def test_v1_candidate_cache_is_not_accepted_as_current_semantics(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"split": ["Training"], "record_key": ["Training|a|0"]})
    paths = candidate_features.candidate_cache_paths(tmp_path, "Training", ("zcr",))
    expected_metadata = candidate_features.expected_candidate_cache_metadata(
        manifest, split="Training", feature_names=("zcr",), source_signature="synthetic"
    )
    metadata = json.loads(json.dumps(expected_metadata))
    metadata["cache_schema_version"] = "mission2_candidate_audio_features_v1"
    metadata["extraction_config"].pop("short_audio_policy", None)
    paths[0].touch()
    paths[1].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(AssertionError, match="candidate cache metadata mismatch"):
        candidate_features.load_candidate_feature_cache(
            manifest, cache_root=tmp_path, split="Training",
            feature_names=("zcr",), source_signature="synthetic",
        )
    with pytest.raises(FileExistsError, match="candidate cache already exists"):
        candidate_features.build_candidate_feature_cache(
            manifest, data_root=tmp_path, cache_root=tmp_path, split="Training",
            feature_names=("zcr",), source_signature="synthetic",
            allow_full_extraction=True,
        )
    assert paths[0].exists() and paths[1].exists()


def test_outlier_fraction_keeps_three_iqr_and_empty_contract() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    assert candidate_features._outlier_fraction_beyond_iqr(values) == pytest.approx(1 / 7)
    assert candidate_features._outlier_fraction_beyond_iqr(np.array([1.0] * 4)) == 0.0
    assert np.isnan(candidate_features._outlier_fraction_beyond_iqr(np.array([])))
    with pytest.raises(ValueError, match="finite 1D"):
        candidate_features._outlier_fraction_beyond_iqr(np.array([1.0, np.nan]))


def test_e05_code_derives_lock_and_merge_contract_from_one_pair() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "mission2_feature_experiments.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    }
    e05_ids = ("m2-fe-e05-config", "m2-fe-e05-lock", "m2-fe-e05-cache-merge")
    literal_pair = "('spectral_rolloff', 'zcr')"
    assert sum(code_by_id[cell_id].count(literal_pair) for cell_id in e05_ids) == 1

    # Replacing only the canonical tuple must carry through the lock contract.
    changed_config = code_by_id["m2-fe-e05-config"].replace(
        literal_pair, "('zcr', 'spectral_centroid')", 1
    )
    namespace = {
        "CANDIDATE_REGISTRY": CANDIDATE_REGISTRY,
        "Path": Path,
        "REPO_ROOT": notebook_path.parents[1],
        "os": __import__("os"),
        "pd": pd,
        "lock_candidate_selection": lock_candidate_selection,
        "freeze_stage2_precommit": freeze_stage2_precommit,
        "MFCC_CONFIG": MFCC_CONFIG,
        "REFERENCE_MODEL_CONFIG": REFERENCE_MODEL_CONFIG,
        "CANDIDATE_EXTRACTION_CONFIG": CANDIDATE_EXTRACTION_CONFIG,
        "eda_manifest_meta": {"source_signature": "synthetic-v2-training"},
    }
    exec(compile(code_by_id["m2-fe-config"], "m2-fe-config", "exec"), namespace)
    exec(compile(changed_config, "m2-fe-e05-config", "exec"), namespace)
    namespace["LOCK_E05_SELECTION"] = True
    namespace["RUN_SELECTED_FEATURE_EXPERIMENT"] = False
    namespace["stage2_result"] = _synthetic_stage2_result(
        "zcr+spectral_centroid", ("zcr", "spectral_centroid")
    )
    exec(compile(code_by_id["m2-fe-e05-lock"], "m2-fe-e05-lock", "exec"), namespace)

    assert namespace["locked_candidate"].feature_names == ("zcr", "spectral_centroid")
    assert namespace["locked_candidate"].selection_name == "zcr+spectral_centroid"
    assert namespace["E05_TRAINING_ONLY_PRECOMMIT"]["selected_pair"] == (
        "zcr", "spectral_centroid"
    )
