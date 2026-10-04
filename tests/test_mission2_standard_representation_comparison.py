"""Synthetic contract checks; these do not establish real-data performance."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def comparison():
    return importlib.import_module("dcc_data.mission2_standard_representation_comparison")


def test_alignment_reorders_by_key_and_rejects_label_call_or_key_mismatch():
    module = comparison()
    clean = SimpleNamespace(record_keys=np.array(["a", "b"]),
                            call_ids=np.array(["c1", "c2"]), y=np.array([0, 1]))
    data = SimpleNamespace(record_keys=np.array(["b", "a"]),
                           call_ids=np.array(["c2", "c1"]), X=np.array([[20], [10]]))
    manifest = pd.DataFrame({"record_key": ["b", "a"], "call_id": ["c2", "c1"],
                             "speaker": [1, 0]})
    matrix, tracking, audit = module.align_inputs(clean, data, manifest)
    assert matrix[:, 0].tolist() == [10, 20]
    assert tracking.speaker.tolist() == [0, 1]
    assert audit["reordered_88d"] is True
    for field, values in [("speaker", [0, 0]), ("call_id", ["wrong", "c1"]),
                          ("record_key", ["a", "a"])]:
        broken = manifest.copy()
        broken[field] = values
        with pytest.raises(ValueError):
            module.align_inputs(clean, data, broken)


def test_reproduction_stops_on_one_changed_prediction_count():
    module = comparison()
    result = SimpleNamespace(metrics={"accuracy": 155434 / 177891,
                                      "confusion_matrix": [[72265, 13086], [9371, 83169]]})
    assert module.verify_reproduction(result)["status"] == "REPRODUCED_METRICS"
    result.metrics["confusion_matrix"] = [[72264, 13087], [9371, 83169]]
    with pytest.raises(RuntimeError, match="REPRODUCTION_STOP"):
        module.verify_reproduction(result)


def test_paired_slices_keep_boundaries_and_empty_groups():
    module = comparison()
    frame = pd.DataFrame({"record_key": list("abcdef"), "call_id": list("abcdef"),
                          "y_true": [0, 0, 1, 1, 0, 0],
                          "pred_59d": [1, 0, 0, 1, 0, 1],
                          "pred_88d": [0, 1, 0, 1, 0, 0],
                          "duration_sec": [.49, .5, 1., 2., 5., .75],
                          "overlap": [True, True, False, False, False, True]})
    table = module.paired_slice_table(frame).set_index("slice")
    assert table.loc["all", ["both_correct", "both_wrong", "correction", "regression", "net_correction"]].tolist() == [2, 1, 2, 1, 1]
    assert table.loc["speaker0_0.5-1s_overlap", ["n", "correction", "regression"]].tolist() == [2, 1, 1]
    assert table.loc["speaker0_1-2s_overlap", "n"] == 0
    assert pd.isna(table.loc["speaker0_1-2s_overlap", "accuracy_59d"])
    assert table.loc["duration_2-5s", "n"] == 1
    assert table.loc["duration_>=5s", "n"] == 1


def test_artifact_writer_preserves_existing_run(tmp_path):
    module = comparison()
    out = tmp_path / "run"
    frame = pd.DataFrame({"record_key": ["a"], "prob1_88d": [.6]})
    module.save_artifacts(out, {"status": "PASS"}, frame, pd.DataFrame(), pd.DataFrame())
    original = (out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.save_artifacts(out, {"status": "changed"}, frame, pd.DataFrame(), pd.DataFrame())
    assert (out / "summary.json").read_bytes() == original
    assert pd.read_parquet(out / "validation_predictions.parquet").record_key.tolist() == ["a"]


def test_candidate_fit_cannot_skip_reference_reproduction():
    module = comparison()
    wrong = SimpleNamespace(metrics={"accuracy": .5, "confusion_matrix": [[1, 1], [1, 1]]})
    with pytest.raises(RuntimeError, match="REPRODUCTION_STOP"):
        module.fit_candidate(None, wrong, allow_fit=True)


def test_pair_scalers_ignore_validation_outliers(monkeypatch):
    module = comparison()
    from dcc_data.mission2_clean_stage4 import clean_59d_feature_names
    monkeypatch.setattr(module.models, "EXPECTED_INTERNAL_SPLIT", {
        "total_rows": 10, "train_rows": 8, "validation_rows": 2,
        "train_calls": 4, "validation_calls": 1, "call_overlap": 0})
    matrix59 = np.zeros((10, 59), dtype=np.float32)
    matrix88 = np.zeros((10, 88), dtype=np.float32)
    matrix59[2:4] = 10000
    matrix88[2:4] = 20000
    clean = SimpleNamespace(X=matrix59, y=np.tile([0, 1], 5),
        record_keys=np.array([f"k{i}" for i in range(10)]),
        call_ids=np.repeat(["a", "b", "c", "d", "e"], 2),
        feature_names=clean_59d_feature_names())
    data = SimpleNamespace(feature_names=tuple(f"f{i}" for i in range(88)))
    reference, candidate, audit = module.prepare_pair(clean, data, matrix88)
    assert reference.validation_idx.tolist() == [2, 3]
    assert np.all(reference.scaler.mean_ == 0)
    assert np.all(candidate.scaler.mean_ == 0)
    assert candidate.scaler.n_samples_seen_ == 8
    assert np.all(candidate.X_evaluation_scaled == 20000)
    assert audit["call_overlap"] == 0


def test_confidence_hce_and_generalization_metrics_use_common_definitions():
    module = comparison()
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([1, 0, 0, 1])
    probability = np.array([[.05, .95], [.6, .4], [.91, .09], [.3, .7]])
    # A fixed probability provider isolates reporting from expensive fitting.
    model = SimpleNamespace(classes_=np.array([0, 1]), n_iter_=7, loss_=.5,
        predict_proba=lambda matrix: np.array([[.8, .2], [.2, .8]]))
    audit = {"train_idx": np.array([0, 1]), "validation_idx": np.array([2, 3, 4, 5]),
        "training_record_keys": np.array(["a", "b"]),
        "validation_record_keys": np.array(["c", "d", "e", "f"]),
        "scaler_fit_scope": "internal_train_only", "convergence_warning_raised": False,
        "warning_messages": (), "fit_seconds": 1., "prediction_seconds": .1,
        **{key: "synthetic" for key in ("feature_fingerprint", "prediction_fingerprint",
            "split_fingerprint", "scaler_fingerprint", "row_fingerprint",
            "model_config_fingerprint", "probability_fingerprint")}}
    result = SimpleNamespace(y_true=truth, y_pred=prediction, probabilities=probability,
        model=model, fit_audit=audit, metrics={"accuracy": .5, "confusion_matrix": [[1, 1], [1, 1]]})
    prepared = SimpleNamespace(y_train=np.array([0, 1]), X_train_scaled=np.zeros((2, 3)),
        X_evaluation_scaled=np.zeros((4, 3)), representation_id="synthetic",
        feature_names=("f1", "f2", "f3"))
    tracking = pd.DataFrame({"record_key": list("abcdef"), "call_id": ["t", "t", "v", "v", "v", "v"],
        "startAt_ms": [0, 1000, 0, 500, 2000, 4000],
        "endAt_ms": [1000, 2000, 1000, 1500, 3000, 5000]})
    report, _ = module.evaluate_result(result, prepared, tracking)
    assert report["train_accuracy"] == 1
    assert report["gap"] == .5
    assert [report[k] for k in ("errors", "FP", "FN", "HCE", "HCE_FP", "HCE_FN")] == [2, 1, 1, 2, 1, 1]
    assert report["mean_confidence"] == pytest.approx(.79)
    assert report["mean_confidence_correct"] == pytest.approx(.65)
    assert report["mean_confidence_error"] == pytest.approx(.93)
    assert report["low_confidence_accuracy"] == 1


def test_default_gate_prevents_model_fit():
    module = comparison()
    with pytest.raises(RuntimeError, match="explicit True"):
        module.fit_prepared(SimpleNamespace(representation_id="synthetic"))
