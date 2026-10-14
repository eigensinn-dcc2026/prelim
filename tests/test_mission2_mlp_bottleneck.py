"""Synthetic contract checks for the Mission 2 MLP iteration comparison."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def bottleneck_module():
    try:
        return importlib.import_module("dcc_data.mission2_mlp_bottleneck")
    except ModuleNotFoundError as error:
        if error.name != "dcc_data.mission2_mlp_bottleneck":
            raise
        pytest.fail("MLP bottleneck comparison helper has not been implemented")


def test_configs_change_only_max_iter_and_keep_the_frozen_mlp_settings():
    module = bottleneck_module()

    config_100, config_300 = module.build_mlp_configs()

    assert {name for name in config_100 if config_100[name] != config_300[name]} == {
        "max_iter"
    }
    assert config_100 == {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 1024,
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "max_iter": 100,
        "shuffle": True,
        "random_state": 42,
        "tol": 0.0001,
        "early_stopping": False,
        "n_iter_no_change": 10,
    }
    assert config_300["max_iter"] == 300


def test_clean_59d_scaler_fits_only_the_fixed_internal_training_rows(monkeypatch):
    module = bottleneck_module()
    from dcc_data.mission2_clean_stage4 import clean_59d_feature_names

    monkeypatch.setattr(
        module.models,
        "EXPECTED_INTERNAL_SPLIT",
        {
            "total_rows": 10,
            "train_rows": 8,
            "validation_rows": 2,
            "train_calls": 4,
            "validation_calls": 1,
            "call_overlap": 0,
        },
    )
    matrix = np.zeros((10, 59), dtype=np.float32)
    matrix[2:4, 0] = 10_000
    clean = SimpleNamespace(
        X=matrix,
        y=np.tile([0, 1], 5),
        record_keys=np.asarray([f"key-{i}" for i in range(10)]),
        call_ids=np.repeat(["call-a", "call-b", "call-c", "call-d", "call-e"], 2),
        feature_names=clean_59d_feature_names(),
    )

    prepared = module.prepare_clean_59d(clean)

    assert prepared.validation_idx.tolist() == [2, 3]
    assert prepared.scaler.n_samples_seen_ == 8
    np.testing.assert_array_equal(prepared.scaler.mean_, np.zeros(59))


def test_comparison_delta_is_candidate_minus_100_iteration_reference():
    module = bottleneck_module()
    reference = {
        "max_iter": 100,
        "internal_validation_accuracy": 0.80,
        "train_accuracy": 0.90,
        "train_validation_gap": 0.10,
        "validation_error_count": 20,
    }
    candidate = {
        "max_iter": 300,
        "internal_validation_accuracy": 0.85,
        "train_accuracy": 0.92,
        "train_validation_gap": 0.07,
        "validation_error_count": 15,
    }

    delta = module.build_comparison_delta(reference, candidate)

    assert delta == pytest.approx(
        {
            "validation_accuracy_delta": 0.05,
            "train_accuracy_delta": 0.02,
            "gap_delta": -0.03,
            "error_count_delta": -5,
        }
    )
