"""Minimal helpers for the clean 59D MLP iteration-budget comparison."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from . import mission2_acoustic_feature_expansion as acoustic
from . import mission2_model_experiments as models


def build_mlp_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy the existing M04-100 configuration and change only max_iter."""

    config_100 = {
        **dict(acoustic.M04_ITER100_FEATURE_SCREENING_CONFIG),
        "max_iter": 100,
    }
    config_300 = {**config_100, "max_iter": 300}
    return config_100, config_300


def prepare_clean_59d(clean: Any):
    """Reuse the frozen call_id split and train-only scaler for clean 59D."""

    train_idx, validation_idx, split_audit = models.create_fixed_internal_split(
        clean.call_ids,
        split_function=models.standalone_group_split_indices,
        expected_contract=models.EXPECTED_INTERNAL_SPLIT,
    )
    return acoustic._prepare_representation(
        representation_id="CLEAN_59D_MLP_BOTTLENECK",
        X=clean.X,
        y=clean.y,
        record_keys=clean.record_keys,
        feature_names=clean.feature_names,
        train_idx=train_idx,
        validation_idx=validation_idx,
        split_audit=split_audit,
    )


def fit_mlp_iteration_candidate(
    prepared: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit one configured MLP and report the shared threshold-0.5 metrics."""

    if dict(config) not in build_mlp_configs():
        raise ValueError(
            "only the frozen max_iter=100 and max_iter=300 configs are supported"
        )
    model = MLPClassifier(**dict(config))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_started = perf_counter()
        model.fit(prepared.X_train_scaled, prepared.y_train)
        fit_seconds = perf_counter() - fit_started

    train_probabilities = models.validate_binary_probabilities(
        model.predict_proba(prepared.X_train_scaled),
        expected_rows=len(prepared.y_train),
        classes=model.classes_,
    )
    validation_probabilities = models.validate_binary_probabilities(
        model.predict_proba(prepared.X_evaluation_scaled),
        expected_rows=len(prepared.y_evaluation),
        classes=model.classes_,
    )
    train_prediction = (train_probabilities[:, 1] >= 0.5).astype(np.int64)
    validation_prediction = (validation_probabilities[:, 1] >= 0.5).astype(
        np.int64
    )
    train_metrics = models.calculate_prediction_metrics(
        prepared.y_train, train_prediction
    )
    validation_metrics = models.calculate_prediction_metrics(
        prepared.y_evaluation, validation_prediction
    )
    convergence_messages = tuple(
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    convergence = models.summarize_mlp_convergence(
        model, convergence_warning_messages=convergence_messages
    )
    loss_curve = convergence["loss_curve"]
    loss_curve_start = len(loss_curve) - min(10, len(loss_curve)) + 1

    return {
        "max_iter_setting": int(config["max_iter"]),
        "n_iter_": convergence["n_iter"],
        "convergence_warning_raised": convergence[
            "convergence_warning_raised"
        ],
        "convergence_warning_messages": convergence[
            "convergence_warning_messages"
        ],
        "final_loss": convergence["final_loss"],
        "train_accuracy": train_metrics["accuracy"],
        "internal_validation_accuracy": validation_metrics["accuracy"],
        "train_validation_gap": (
            train_metrics["accuracy"] - validation_metrics["accuracy"]
        ),
        "confusion_matrix": validation_metrics["confusion_matrix"],
        "speaker_0_recall": validation_metrics["recall_speaker_0"],
        "speaker_1_recall": validation_metrics["recall_speaker_1"],
        "validation_error_count": (
            validation_metrics["sample_count"]
            - validation_metrics["correct_count"]
        ),
        "fit_time_seconds": float(fit_seconds),
        "loss_curve_tail": tuple(
            (iteration, loss)
            for iteration, loss in enumerate(
                loss_curve[-min(10, len(loss_curve)) :],
                start=loss_curve_start,
            )
        ),
    }


def build_comparison_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | int]:
    """Return candidate-minus-reference deltas for the requested metrics."""

    return {
        "validation_accuracy_delta": (
            float(candidate["internal_validation_accuracy"])
            - float(reference["internal_validation_accuracy"])
        ),
        "train_accuracy_delta": (
            float(candidate["train_accuracy"]) - float(reference["train_accuracy"])
        ),
        "gap_delta": (
            float(candidate["train_validation_gap"])
            - float(reference["train_validation_gap"])
        ),
        "error_count_delta": int(candidate["validation_error_count"])
        - int(reference["validation_error_count"]),
    }
