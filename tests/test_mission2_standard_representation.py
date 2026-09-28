from __future__ import annotations

from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from dcc_data import mission2_acoustic_feature_expansion as acoustic
from dcc_data import mission2_model_experiments as models


standard = importlib.import_module("dcc_data.mission2_standard_representation")


def _sine(samples: int = 8_000) -> np.ndarray:
    time = np.arange(samples) / 8_000
    return (0.1 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)


def _small_comparison():
    keys = np.asarray([f"Training|call_{i}|0" for i in range(10)])
    calls = np.asarray([f"call_{i}" for i in range(10)])
    labels = np.asarray([i % 2 for i in range(10)], dtype=np.int64)
    train_idx, val_idx, audit = models.create_fixed_internal_split(
        calls, split_function=models.standalone_group_split_indices
    )
    names_59 = (
        *acoustic.CANONICAL_52D_FEATURE_NAMES,
        *(name for block in acoustic.RESIDUAL_COMBINATION_SPECS["AF10_AF11_COMBINATION"]
          for name in acoustic.BLOCK_BY_ID[block].feature_names),
    )
    matrix_59 = np.arange(10 * 59, dtype=np.float32).reshape(10, 59) / 1000
    prepared = acoustic._prepare_representation(
        representation_id="AF10_AF11_COMBINATION", X=matrix_59, y=labels,
        record_keys=keys, feature_names=names_59, train_idx=train_idx,
        validation_idx=val_idx, split_audit=audit,
    )
    matrix_88 = np.tile(np.arange(88, dtype=np.float32), (10, 1)) + np.arange(10, dtype=np.float32)[:, None]
    return prepared, matrix_88, labels, keys, calls


def test_extract_exact_88d_is_stable_float32_and_waveform_only():
    first, names = standard.extract_egemaps(_sine(), 8_000)
    second, repeated_names = standard.extract_egemaps(_sine(), 8_000)
    assert first.shape == (88,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert names == repeated_names == standard.get_egemaps_feature_names()
    assert len(names) == len(set(names)) == 88


def test_short_signal_fails_without_imputation_and_silence_is_finite():
    with pytest.warns(UserWarning, match="Segment too short"):
        with pytest.raises(ValueError, match="NaN|finite|short"):
            standard.extract_egemaps(_sine(80), 8_000)
    silence, _ = standard.extract_egemaps(np.zeros(8_000, dtype=np.float32), 8_000)
    assert silence.shape == (88,) and np.isfinite(silence).all()
    with pytest.raises(ValueError, match="8 kHz|8000"):
        standard.extract_egemaps(_sine(), 16_000)
    with pytest.raises(ValueError, match="mono|one-dimensional"):
        standard.extract_egemaps(_sine().reshape(1, -1), 8_000)


def test_cache_rejects_record_key_order_and_config_mismatch(tmp_path):
    path = tmp_path / "egemaps.npz"
    names = standard.get_egemaps_feature_names()
    matrix = np.ones((2, 88), dtype=np.float32)
    keys = ("Training|a|0", "Training|b|0")
    calls = ("a", "b")
    standard.build_egemaps_cache(path, matrix, keys, calls, names, source_provenance="synthetic")
    loaded = standard.load_egemaps_cache(path, keys, calls, source_provenance="synthetic")
    assert np.array_equal(loaded.X, matrix)
    with pytest.raises(ValueError, match="record_key"):
        standard.load_egemaps_cache(path, keys[::-1], calls, source_provenance="synthetic")
    with pytest.raises(ValueError, match="config|provenance"):
        standard.load_egemaps_cache(path, keys, calls, source_provenance="changed")
    with pytest.raises(ValueError, match="feature"):
        standard.build_egemaps_cache(tmp_path / "wrong.npz", matrix, keys, calls, names[::-1], source_provenance="synthetic")
    with np.load(path, allow_pickle=False) as original:
        fields = {name: original[name] for name in original.files}
    metadata = json.loads(fields["metadata"].item())
    metadata["opensmile_version"] = "changed"
    fields["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(tmp_path / "stale.npz", **fields)
    with pytest.raises(ValueError, match="config|provenance"):
        standard.load_egemaps_cache(tmp_path / "stale.npz", keys, calls, source_provenance="synthetic")


@pytest.mark.parametrize("change", ["split", "label", "model"])
def test_comparison_rejects_changed_contract_before_fitting(monkeypatch, change):
    prepared, matrix, labels, keys, calls = _small_comparison()
    if change == "split":
        prepared = replace(prepared, train_idx=prepared.train_idx[::-1])
    elif change == "label":
        labels = labels.copy()
        labels[prepared.train_idx[0]] ^= 1
    else:
        monkeypatch.setattr(acoustic, "M04_ITER100_FEATURE_SCREENING_CONFIG", {"max_iter": 99})
    monkeypatch.setattr(acoustic, "_run_prepared_representation", lambda *a, **k: pytest.fail("fitted before validation"))
    data = standard.make_egemaps_data(
        matrix, keys, calls, standard.get_egemaps_feature_names(), source_provenance="synthetic"
    )
    with pytest.raises(ValueError, match="split|label|M04"):
        standard.run_standard_representation_comparison(
            prepared, data, labels, keys, calls, expected_split_contract=None
        )


def test_import_does_not_extract_or_fit(monkeypatch):
    import opensmile
    from sklearn.neural_network import MLPClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("import started extraction or fit")

    monkeypatch.setattr(opensmile.Smile, "process_signal", forbidden)
    monkeypatch.setattr(MLPClassifier, "fit", forbidden)
    importlib.reload(standard)


def test_small_synthetic_comparison_reports_paired_contract():
    prepared, matrix, labels, keys, calls = _small_comparison()
    data = standard.make_egemaps_data(
        matrix, keys, calls, standard.get_egemaps_feature_names(), source_provenance="synthetic"
    )
    result = standard.run_standard_representation_comparison(
        prepared, data, labels, keys, calls, expected_split_contract=None
    )
    assert result["reports"]["59D"]["dimension"] == 59
    assert result["reports"]["eGeMAPSv02"]["dimension"] == 88
    for report in result["reports"].values():
        assert 0 <= report["train_accuracy"] <= 1
        assert 0 <= report["internal_validation_accuracy"] <= 1
        assert sum(map(sum, report["confusion_matrix"])) == len(prepared.validation_idx)
    assert sum(result[key] for key in (
        "both_correct", "both_wrong", "59d_wrong_egemaps_correct",
        "59d_correct_egemaps_wrong",
    )) == len(prepared.validation_idx)
    assert result["net_correct_gain"] == (
        result["59d_wrong_egemaps_correct"] - result["59d_correct_egemaps_wrong"]
    )
