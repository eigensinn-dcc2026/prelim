import importlib
import importlib.util

import numpy as np
import pytest


def _compact_grid():
    spec = importlib.util.find_spec("dcc_data.mission2_acoustic_feature_grid_compact")
    assert spec is not None, "the Colab 3x3 compact grid runner must exist"
    return importlib.import_module("dcc_data.mission2_acoustic_feature_grid_compact")


def test_compact_plan_is_three_models_by_three_named_feature_blocks():
    compact = _compact_grid()

    assert compact.MODEL_IDS == ("tabm", "tabular_resnet", "gandalf")
    assert compact.FEATURE_IDS == ("F01", "F02", "F08")
    assert compact.candidate_experiment_ids() == (
        "tabm_F01", "tabm_F02", "tabm_F08",
        "tabular_resnet_F01", "tabular_resnet_F02", "tabular_resnet_F08",
        "gandalf_F01", "gandalf_F02", "gandalf_F08",
    )


def test_explicit_cpu_device_is_allowed_without_cuda():
    compact = _compact_grid()

    assert compact._validate_device("cpu", cuda_available=False) == "cpu"


def test_cuda_device_still_requires_cuda():
    compact = _compact_grid()

    with pytest.raises(RuntimeError, match="CUDA"):
        compact._validate_device("cuda", cuda_available=False)


def test_voice_quality_selector_uses_exact_cache_columns_and_key_order():
    compact = _compact_grid()
    names = (
        "jitterLocal_sma3nz_amean", "jitterLocal_sma3nz_stddevNorm",
        "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm",
        "HNRdBACF_sma3nz_amean", "HNRdBACF_sma3nz_stddevNorm",
        *(f"unused_{index}" for index in range(82)),
    )
    keys = np.asarray(["Training|call-a|0", "Training|call-b|1"], dtype=str)
    matrix = np.arange(176, dtype=np.float32).reshape(2, 88)

    selected = compact.select_voice_quality_matrix(matrix, names, keys, keys)

    assert selected.shape == (2, 6)
    assert selected.dtype == np.float32
    np.testing.assert_array_equal(selected, matrix[:, :6])
    with pytest.raises(ValueError, match="record_key order"):
        compact.select_voice_quality_matrix(matrix, names, keys[::-1], keys)


def test_voice_quality_selector_rejects_nonfinite_values():
    compact = _compact_grid()
    names = (
        "jitterLocal_sma3nz_amean", "jitterLocal_sma3nz_stddevNorm",
        "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm",
        "HNRdBACF_sma3nz_amean", "HNRdBACF_sma3nz_stddevNorm",
        *(f"unused_{index}" for index in range(82)),
    )
    keys = np.asarray(["Training|call-a|0"], dtype=str)
    matrix = np.zeros((1, 88), dtype=np.float32)
    matrix[0, 4] = np.nan

    with pytest.raises(ValueError, match="NaN/Inf"):
        compact.select_voice_quality_matrix(matrix, names, keys, keys)
