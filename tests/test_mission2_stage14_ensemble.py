from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _stage14():
    from dcc_data import mission2_stage14_ensemble

    return mission2_stage14_ensemble


def _context(rows=3):
    from dcc_data import mission2_acoustic_feature_grid as grid

    return SimpleNamespace(
        clean_X=np.full((rows, 59), 1.0, dtype=np.float32),
        feature_names_59d=tuple(f"clean_{index}" for index in range(59)),
        feature_matrices={
            "F01": np.full((rows, 80), 2.0, dtype=np.float32),
            "F02": np.full((rows, 10), 3.0, dtype=np.float32),
        },
        record_keys=np.asarray([f"Training|call-{index}|{index}" for index in range(rows)]),
        call_ids=np.asarray([f"call-{index}" for index in range(rows)]),
        y=np.asarray([index % 2 for index in range(rows)], dtype=np.int64),
        train_idx=np.arange(rows - 1),
        validation_idx=np.asarray([rows - 1]),
    )


def _prediction_frame(keys, calls, labels, probabilities):
    return pd.DataFrame({
        "record_key": keys,
        "call_id": calls,
        "y_true": labels,
        "probability_speaker1": probabilities,
    })


def test_joint_input_is_clean_then_f01_then_f02_and_149d():
    stage14 = _stage14()
    context = _context()

    matrix, names = stage14.build_joint_input(context)

    assert matrix.shape == (3, 149)
    assert matrix.dtype == np.float32
    assert names[:59] == context.feature_names_59d
    assert names[59:139] == tuple(
        stage14.grid.BLOCK_BY_ID["F01"].feature_names
    )
    assert names[139:] == tuple(stage14.grid.BLOCK_BY_ID["F02"].feature_names)
    assert np.all(matrix[:, :59] == 1.0)
    assert np.all(matrix[:, 59:139] == 2.0)
    assert np.all(matrix[:, 139:] == 3.0)


@pytest.mark.parametrize(
    "feature_id,width,value",
    [("F01", 80, np.nan), ("F02", 10, np.inf)],
)
def test_joint_input_rejects_nonfinite_feature_values(feature_id, width, value):
    stage14 = _stage14()
    context = _context()
    context.feature_matrices[feature_id][0, width - 1] = value

    with pytest.raises(ValueError, match="finite"):
        stage14.build_joint_input(context)


def test_joint_input_rejects_duplicate_feature_names():
    stage14 = _stage14()
    context = _context()
    context.feature_names_59d = (*context.feature_names_59d[:-1],
                                  stage14.grid.BLOCK_BY_ID["F01"].feature_names[0])

    with pytest.raises(ValueError, match="duplicate"):
        stage14.build_joint_input(context)


def test_ensemble_candidates_cover_every_pair_and_the_three_model_average():
    stage14 = _stage14()

    specs = stage14.ensemble_candidate_specs()

    assert set(specs) == {"E_TR", "E_TG", "E_RG", "E_TRG"}
    assert all(sum(weight for _, weight in members) == pytest.approx(1.0)
               for members in specs.values())
    assert specs["E_TR"] == (
        ("tabm_F01_F02", 0.5), ("tabular_resnet_F01_F02", 0.5),
    )
    assert specs["E_TG"] == (
        ("tabm_F01_F02", 0.5), ("gandalf_F01_F02", 0.5),
    )
    assert specs["E_RG"] == (
        ("tabular_resnet_F01_F02", 0.5), ("gandalf_F01_F02", 0.5),
    )
    assert specs["E_TRG"] == (
        ("tabm_F01_F02", 1 / 3),
        ("tabular_resnet_F01_F02", 1 / 3),
        ("gandalf_F01_F02", 1 / 3),
    )


def test_ensemble_uses_aligned_probabilities_and_fixed_half_threshold():
    stage14 = _stage14()
    keys = np.asarray(["k0", "k1", "k2"])
    calls = np.asarray(["c0", "c1", "c2"])
    labels = np.asarray([0, 1, 0])
    members = {
        "tabm_F01_F02": _prediction_frame(keys, calls, labels, [0.2, 0.6, 0.4]),
        "tabular_resnet_F01_F02": _prediction_frame(keys, calls, labels, [0.8, 0.4, 0.6]),
    }

    metrics, combined = stage14.build_ensemble_predictions(members, "E_TR")

    np.testing.assert_array_equal(combined.probability_speaker1, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(combined.prediction, [1, 1, 1])
    assert metrics["threshold"] == 0.5
    assert metrics["confusion_matrix"] == [[0, 2], [0, 1]]


def test_ensemble_rejects_member_prediction_row_misalignment():
    stage14 = _stage14()
    keys = np.asarray(["k0", "k1"])
    calls = np.asarray(["c0", "c1"])
    labels = np.asarray([0, 1])
    members = {
        "tabm_F01_F02": _prediction_frame(keys, calls, labels, [0.2, 0.8]),
        "tabular_resnet_F01_F02": _prediction_frame(
            keys[::-1], calls[::-1], labels[::-1], [0.2, 0.8],
        ),
    }

    with pytest.raises(ValueError, match="identity"):
        stage14.build_ensemble_predictions(members, "E_TR")


def test_joint_feature_loader_reuses_only_hash_verified_f01_f02_cache(tmp_path):
    from dcc_data import mission2_acoustic_feature_grid as grid

    stage14 = _stage14()
    context = _context(rows=2)
    source = tmp_path / "source"
    cache = source / "feature_cache"
    cache.mkdir(parents=True)
    feature_blocks = {}
    for block_id in ("F01", "F02"):
        block = grid.BLOCK_BY_ID[block_id]
        np.savez(
            cache / f"{block_id}.npz",
            X=context.feature_matrices[block_id],
            record_keys=context.record_keys,
            call_ids=context.call_ids,
            feature_names=np.asarray(block.feature_names),
        )
        feature_blocks[block_id] = {
            "artifact_sha256": grid._sha256_file(cache / f"{block_id}.npz"),
        }
    grid._write_json_new(cache / "manifest.json", {
        "schema_version": "mission2_acoustic_feature_grid_cache_v1",
        "source_split": "Training",
        "official_validation_used": False,
        "record_key_signature": grid._record_key_signature(context.record_keys),
        "call_id_hash": grid._array_fingerprint(context.call_ids),
        "labels_hash": grid._array_fingerprint(context.y),
        "feature_blocks": feature_blocks,
    })

    matrices, hashes, _ = stage14.load_joint_feature_blocks(source, context)

    assert set(matrices) == {"F01", "F02"}
    assert hashes == {name: feature_blocks[name]["artifact_sha256"]
                      for name in ("F01", "F02")}

    with np.load(cache / "F01.npz", allow_pickle=False) as saved:
        values = saved["X"]
        values[0, 0] += 1.0
        np.savez(
            cache / "F01.reordered.npz",
            X=values,
            record_keys=context.record_keys,
            call_ids=context.call_ids,
            feature_names=np.asarray(grid.BLOCK_BY_ID["F01"].feature_names),
        )
    (cache / "F01.reordered.npz").replace(cache / "F01.npz")
    with np.load(cache / "F02.npz", allow_pickle=False) as saved:
        np.savez(
            cache / "F02.reordered.npz",
            X=saved["X"], record_keys=context.record_keys,
            call_ids=context.call_ids,
            feature_names=np.asarray(grid.BLOCK_BY_ID["F02"].feature_names),
        )
    (cache / "F02.reordered.npz").replace(cache / "F02.npz")

    with pytest.raises(ValueError, match="artifact.*missing/corrupt"):
        stage14.load_joint_feature_blocks(source, context)
