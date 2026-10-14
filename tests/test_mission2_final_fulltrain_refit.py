from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _final():
    from dcc_data import mission2_final_fulltrain_refit as final

    return final


def _validation_manifest():
    return pd.DataFrame({
        "split": ["Validation", "Validation", "Training"],
        "record_key": ["Validation|call-a|0", "Validation|call-b|0", "Training|old|0"],
        "call_id": ["call-a", "call-b", "old"],
        "annotation_row": [0, 0, 0],
        "speaker": [0, 1, 0],
        "audio_path": ["Validation/a.wav", "Validation/b.wav", "Training/old.wav"],
        "json_path": ["Validation/a.json", "Validation/b.json", "Training/old.json"],
        "crop_start_frame": [800, 1600, 0],
        "crop_end_frame": [1600, 2400, 800],
        "crop_ok": [True, True, True],
        "startAt_ms": [100.0, 200.0, 0.0],
        "endAt_ms": [200.0, 300.0, 100.0],
    })


def _prediction_frame(keys, calls, labels, p1):
    from dcc_data import mission2_acoustic_feature_grid as grid

    _, frame = grid._metric_bundle(labels, p1, keys, calls)
    return frame


def test_manifest_preflight_filters_exact_split_and_preserves_complete_rows():
    final = _final()

    source = final.validate_manifest_rows(
        _validation_manifest(), "Validation", expected_rows=2, expected_calls=2,
    )

    assert source.record_key.tolist() == ["Validation|call-a|0", "Validation|call-b|0"]
    assert source.speaker.tolist() == [0, 1]


def test_validation_identifier_arrays_are_unicode_and_pickle_free(tmp_path):
    final = _final()
    source = final.validate_manifest_rows(
        _validation_manifest(), "Validation", expected_rows=2, expected_calls=2,
    )

    # pandas object columns become object ndarrays with to_numpy(); those cannot
    # be hashed by the final-run contract or loaded from NPZ with allow_pickle=False.
    assert source.call_id.to_numpy().dtype.hasobject
    keys = final._unicode_array(source.record_key)
    calls = final._unicode_array(source.call_id)

    assert keys.dtype.kind == "U"
    assert calls.dtype.kind == "U"
    final._array_fingerprint(keys)
    final._array_fingerprint(calls)

    cache_path = tmp_path / "identifiers.npz"
    np.savez(cache_path, record_keys=keys, call_ids=calls)
    with np.load(cache_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["record_keys"], keys)
        np.testing.assert_array_equal(saved["call_ids"], calls)


def test_manifest_preflight_rejects_duplicate_keys_and_bad_crops():
    final = _final()
    duplicate = _validation_manifest()
    duplicate.loc[1, "record_key"] = duplicate.loc[0, "record_key"]
    with pytest.raises(ValueError, match="duplicate"):
        final.validate_manifest_rows(duplicate, "Validation", expected_rows=2, expected_calls=2)

    invalid_crop = _validation_manifest()
    invalid_crop.loc[0, "crop_end_frame"] = invalid_crop.loc[0, "crop_start_frame"]
    with pytest.raises(ValueError, match="crop"):
        final.validate_manifest_rows(invalid_crop, "Validation", expected_rows=2, expected_calls=2)


def test_joint_matrix_uses_clean_then_f01_then_f02_and_rejects_misalignment():
    final = _final()
    from dcc_data import mission2_acoustic_feature_grid as grid

    clean = np.ones((2, 59), dtype=np.float32)
    f01 = np.full((2, 80), 2.0, dtype=np.float32)
    f02 = np.full((2, 10), 3.0, dtype=np.float32)
    names = tuple(f"clean_{index}" for index in range(59))

    matrix, feature_names = final.build_joint_matrix(clean, names, [0, 1], f01, f02)

    assert matrix.shape == (2, 149)
    assert matrix.dtype == np.float32
    assert feature_names[:59] == names
    assert feature_names[59:139] == grid.BLOCK_BY_ID["F01"].feature_names
    assert feature_names[139:] == grid.BLOCK_BY_ID["F02"].feature_names
    np.testing.assert_array_equal(matrix[0, :59], 1.0)
    np.testing.assert_array_equal(matrix[0, 59:139], 2.0)
    np.testing.assert_array_equal(matrix[0, 139:], 3.0)

    with pytest.raises(ValueError, match="shape"):
        final.build_joint_matrix(clean, names, [0, 1], f01[:1], f02)


def test_refit_epochs_come_from_saved_best_epoch_and_keep_gandalf_history_limit():
    final = _final()
    evidence = {
        "tabm": {
            "metrics": {"best_epoch": 68, "epochs_run": 78, "history_kind": "epoch_history"},
            "history": pd.DataFrame({"epoch": range(1, 79), "best_epoch": [68] * 78}),
        },
        "tabular_resnet": {
            "metrics": {"best_epoch": 16, "epochs_run": 26, "history_kind": "epoch_history"},
            "history": pd.DataFrame({"epoch": range(1, 27), "best_epoch": [16] * 26}),
        },
        "gandalf": {
            "metrics": {"best_epoch": 17, "epochs_run": 18,
                        "history_kind": "terminal_metrics_snapshot"},
            "history": pd.DataFrame([{"train_loss": 0.4}]),
        },
    }

    assert final.derive_fixed_refit_epochs(evidence) == {
        "tabm": 68, "tabular_resnet": 16, "gandalf": 17,
    }

    evidence["tabm"]["history"] = evidence["tabm"]["history"].iloc[:-1]
    with pytest.raises(ValueError, match="history"):
        final.derive_fixed_refit_epochs(evidence)


def test_epoch_batches_keep_every_row_and_merge_singleton_tail():
    final = _final()
    order = np.arange(513, dtype=np.int64)

    batches = final._epoch_batches(order, 256)

    assert [len(batch) for batch in batches] == [256, 257]
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), order)


def test_official_manifest_builder_uses_only_allowed_annotation_fields(tmp_path):
    import json

    final = _final()
    raw_root = tmp_path / "raw_original"
    audio_dir = raw_root / "Validation" / "1.원천데이터" / "VS_서울_구급"
    label_dir = raw_root / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
    audio_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for call_id, utterances in {
        "call-a": [
            {"startAt": 100, "endAt": 200, "speaker": 0, "text": "must not load"},
            {"startAt": 300, "endAt": 450, "speaker": 1, "text": "must not load"},
        ],
        "call-b": [{"startAt": 10, "endAt": 110, "speaker": 1, "text": "must not load"}],
    }.items():
        (audio_dir / f"{call_id}.wav").write_bytes(b"wav-placeholder")
        (label_dir / f"{call_id}.json").write_text(
            json.dumps({"utterances": utterances}, ensure_ascii=False), encoding="utf-8",
        )
    run_root = tmp_path / "run"
    run_root.mkdir()

    built = final.build_official_validation_manifest(
        raw_root, run_root, expected_rows=3, expected_calls=2,
    )

    manifest = built["manifest"]
    assert manifest.record_key.tolist() == [
        "Validation|call-a|0", "Validation|call-a|1", "Validation|call-b|0",
    ]
    assert manifest.speaker.tolist() == [0, 1, 1]
    assert "text" not in manifest.columns
    assert built["metadata"]["text_used_for_model_input"] is False
    assert built["metadata"]["text_persisted"] is False
    assert manifest.crop_start_frame.tolist() == [800, 2400, 80]


def test_official_predictions_keep_filename_time_and_label_identity():
    final = _final()
    manifest = _validation_manifest().iloc[:2].copy()
    predictions = _prediction_frame(
        manifest.record_key.to_numpy(), manifest.call_id.to_numpy(),
        manifest.speaker.to_numpy(), np.asarray([0.2, 0.8]),
    )

    tracked = final.attach_tracking_metadata(predictions, manifest)

    assert tracked.filename.tolist() == ["a.wav", "b.wav"]
    assert tracked.startAt_ms.tolist() == [100.0, 200.0]
    assert tracked.endAt_ms.tolist() == [200.0, 300.0]
    assert tracked.record_key.tolist() == manifest.record_key.tolist()

    reordered = manifest.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="identity"):
        final.attach_tracking_metadata(predictions, reordered)


def test_final_ensemble_reuses_stage14_equal_weight_rule_and_fixed_threshold():
    final = _final()
    manifest = _validation_manifest().iloc[:2].copy()
    keys = manifest.record_key.to_numpy()
    calls = manifest.call_id.to_numpy()
    labels = manifest.speaker.to_numpy()
    members = {
        "tabm_F01_F02": _prediction_frame(keys, calls, labels, np.asarray([0.2, 0.8])),
        "tabular_resnet_F01_F02": _prediction_frame(keys, calls, labels, np.asarray([0.5, 0.2])),
        "gandalf_F01_F02": _prediction_frame(keys, calls, labels, np.asarray([0.8, 0.5])),
    }

    metrics, ensemble = final.build_final_ensemble(members, manifest)

    np.testing.assert_allclose(ensemble.probability_speaker1, [0.5, 0.5], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ensemble.prediction, [1, 1])
    assert metrics["threshold"] == 0.5
    assert metrics["members"] == [
        {"member_id": "tabm_F01_F02", "weight": 1 / 3},
        {"member_id": "tabular_resnet_F01_F02", "weight": 1 / 3},
        {"member_id": "gandalf_F01_F02", "weight": 1 / 3},
    ]
    assert ensemble.filename.tolist() == ["a.wav", "b.wav"]


def test_metrics_can_be_recomputed_from_saved_probabilities():
    final = _final()
    frame = _prediction_frame(
        np.asarray(["k0", "k1", "k2", "k3"]),
        np.asarray(["c0", "c1", "c2", "c3"]),
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.9, 0.7, 0.4]),
    )

    metrics = final.recompute_saved_prediction_metrics(frame)

    assert metrics["validation_accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["recall_0"] == 0.5
    assert metrics["recall_1"] == 0.5
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
