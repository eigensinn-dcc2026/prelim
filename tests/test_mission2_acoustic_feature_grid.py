import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dcc_data import mission2_model_experiments as models
from dcc_data import mission2_stage13_untested_model_family as stage13
from dcc_data.mission2_acoustic_feature_grid import (
    FEATURE_BLOCKS,
    MODEL_IDS,
    _stage6_validation_snapshots,
    apply_user_scope_exclusions,
    _mark_ledger,
    experiment_ids,
    active_experiment_ids,
    candidate_experiment_ids,
    user_scope_excluded_experiment_ids,
    training_archive_paths,
    resolve_colab_drive_layout,
    _checkpoint_files_from_manifest,
    _resume_ledger_valid,
    load_fixed_internal_split_manifest,
    select_egemaps_blocks,
    validate_frontend_parity,
)


def _fixed_split_fixture():
    calls = np.asarray([f"call-{index}" for index in range(10)], dtype=object)
    keys = np.asarray([f"Training|{call}|0" for call in calls], dtype=object)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
    validation_idx = np.asarray([8, 9], dtype=np.int64)
    validation = pd.DataFrame(
        {
            "record_key": keys[validation_idx],
            "call_id": calls[validation_idx],
            "y_true": labels[validation_idx],
        }
    )
    train_idx = np.asarray([index for index in range(len(keys)) if index not in validation_idx])
    split = {
        "seed": 42,
        "test_size": 0.2,
        "total_rows": 10,
        "train_rows": 8,
        "validation_rows": 2,
        "train_calls": 8,
        "validation_calls": 2,
        "call_overlap": 0,
    }
    identity = {
        "train_keys": models._array_fingerprint(keys[train_idx].astype(str)),
        "validation_keys": models._array_fingerprint(keys[validation_idx].astype(str)),
        "train_y": models._array_fingerprint(labels[train_idx]),
        "validation_y": models._array_fingerprint(labels[validation_idx]),
        "source_y": models._array_fingerprint(labels),
        "source_keys": models._array_fingerprint(keys.astype(str)),
        "source_calls": models._array_fingerprint(calls.astype(str)),
    }
    contract = {
        "source_split": "Training",
        "evaluation_set": "fixed_internal_validation",
        "seed": 42,
        "threshold": 0.5,
        "official_validation_used": False,
        "split": split,
        "identity": identity,
    }
    return keys, calls, labels, validation, contract, split


def test_loads_persisted_validation_membership_and_validates_exact_identity():
    keys, calls, labels, validation, contract, expected = _fixed_split_fixture()

    train_idx, validation_idx, audit = load_fixed_internal_split_manifest(
        keys,
        calls,
        labels,
        (validation, validation.copy()),
        contract,
        expected_split=expected,
    )

    assert train_idx.tolist() == list(range(8))
    assert validation_idx.tolist() == [8, 9]
    assert set(calls[train_idx]).isdisjoint(calls[validation_idx])
    assert audit["seed"] == 42
    assert audit["manifest_rows"] == 2
    assert audit["official_validation_used"] is False


def test_rejects_mismatched_saved_prediction_keys_labels_or_calls():
    keys, calls, labels, validation, contract, expected = _fixed_split_fixture()
    altered = validation.copy()
    altered.loc[0, "call_id"] = "call-other"

    with pytest.raises(ValueError, match="identity|call_id|manifest"):
        load_fixed_internal_split_manifest(
            keys,
            calls,
            labels,
            (validation, altered),
            contract,
            expected_split=expected,
        )


def test_rejects_changed_split_hash_count_seed_or_validation_group_membership():
    keys, calls, labels, validation, contract, expected = _fixed_split_fixture()
    changed_hash = {**contract, "identity": {**contract["identity"], "train_keys": "0" * 64}}
    changed_seed = {**contract, "seed": 43}
    changed_count = {**contract, "split": {**contract["split"], "train_rows": 7}}
    changed_overlap = validation.copy()
    changed_overlap.loc[:, "call_id"] = calls[0]

    for bad_contract, snapshots in (
        (changed_hash, (validation, validation.copy())),
        (changed_seed, (validation, validation.copy())),
        (changed_count, (validation, validation.copy())),
        (contract, (validation, changed_overlap)),
    ):
        with pytest.raises(ValueError, match="identity|seed|count|overlap|snapshot|call_id|hash"):
            load_fixed_internal_split_manifest(
                keys, calls, labels, snapshots, bad_contract, expected_split=expected,
            )


def test_stage6_split_snapshots_follow_pinned_reference_file_names_and_hashes(tmp_path):
    _, _, _, validation, preprocessing, _ = _fixed_split_fixture()
    snapshot_dir = tmp_path / "stage6_reference"
    snapshot_dir.mkdir()
    first_path = snapshot_dir / "mlp_reference_predictions.parquet"
    second_path = snapshot_dir / "stronger_mlp_predictions.parquet"
    validation.to_parquet(first_path, index=False)
    validation.to_parquet(second_path, index=False)

    first_relative = "candidates/mlp_reference/validation_predictions.parquet"
    second_relative = "candidates/stronger_mlp/validation_predictions.parquet"
    pinned_hashes = {
        first_relative: hashlib.sha256(first_path.read_bytes()).hexdigest(),
        second_relative: hashlib.sha256(second_path.read_bytes()).hexdigest(),
    }
    raw_contract = {
        "preprocessing": preprocessing,
        "stage6": {"hashes": pinned_hashes},
    }
    (tmp_path / "contract.json").write_text(json.dumps(raw_contract), encoding="utf-8")
    (tmp_path / "scaler.json").write_text("{}", encoding="utf-8")

    loaded_contract, first, second, audit = _stage6_validation_snapshots(tmp_path)

    assert loaded_contract == preprocessing
    pd.testing.assert_frame_equal(first, validation)
    pd.testing.assert_frame_equal(second, validation)
    assert audit["stage6_reference_a_sha256"] == pinned_hashes[first_relative]
    assert audit["stage6_reference_b_sha256"] == pinned_hashes[second_relative]


def test_feature_grid_keeps_exactly_seven_blocks_and_forty_sequential_runs():
    assert tuple(block.block_id for block in FEATURE_BLOCKS) == tuple(
        f"F0{index}" for index in range(1, 8)
    )
    assert len(MODEL_IDS) == 5
    assert len(experiment_ids()) == 40
    assert experiment_ids()[:5] == tuple(f"baseline_{name}" for name in MODEL_IDS)


def test_gpu_continuation_fits_only_the_28_non_ft_feature_candidates():
    assert len(candidate_experiment_ids()) == 28
    assert candidate_experiment_ids() == tuple(
        f"{model}_F0{feature}"
        for model in ("tabm", "tabular_resnet", "dcnv2", "gandalf")
        for feature in range(1, 8)
    )
    assert not any(name.startswith("baseline_") for name in candidate_experiment_ids())
    assert not any(name.startswith("ft_transformer_") for name in candidate_experiment_ids())


def test_training_archive_resolver_requires_the_exact_visible_folder_and_12_zips(tmp_path):
    archive_root = tmp_path / "prelim" / "data" / "raw_original" / "대학부 데이터"
    with pytest.raises(FileNotFoundError, match="shortcut"):
        training_archive_paths(archive_root)
    archive_root.mkdir(parents=True)
    for number in range(1, 13):
        (archive_root / f"대학부 데이터-20260901T090511Z-1-{number:03d}.zip").touch()
    paths = training_archive_paths(archive_root)
    assert len(paths) == 12
    assert all(path.parent == archive_root for path in paths)
    assert paths[0].name.endswith("-001.zip")
    assert paths[-1].name.endswith("-012.zip")


def test_colab_drive_layout_uses_prelim_shortcut_and_separate_mydrive_output(tmp_path):
    mydrive = tmp_path / "drive" / "MyDrive"
    repo = mydrive / "prelim"
    source_run = repo / "data" / "processed" / "mission2_acoustic_feature_grid" / "20261010T012333Z_8de41728"
    (repo / "src" / "dcc_data").mkdir(parents=True)
    (repo / "src" / "dcc_data" / "mission2_acoustic_feature_grid.py").touch()
    source_run.mkdir(parents=True)
    (source_run / "contract.json").write_text("{}", encoding="utf-8")
    (source_run / "run_ledger.json").write_text("{}", encoding="utf-8")

    layout = resolve_colab_drive_layout(mydrive)

    assert layout["mydrive_root"] == mydrive.resolve()
    assert layout["repo_root"] == repo.resolve()
    assert layout["source_run_root"] == source_run.resolve()
    assert layout["training_archive_root"] == repo / "data" / "raw_original" / "대학부 데이터"
    assert layout["output_root"] == mydrive / "mission2_grid_runs"
    assert not layout["output_root"].exists()


def test_colab_drive_layout_rejects_missing_prelim_shortcut_before_resolving_it(tmp_path):
    mydrive = tmp_path / "drive" / "MyDrive"
    mydrive.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="MyDrive/prelim"):
        resolve_colab_drive_layout(mydrive)


def test_colab_notebook_uses_confirmed_drive_paths_and_one_serial_candidate_controller():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "mission2_acoustic_feature_grid_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]

    assert "MYDRIVE_ROOT / 'prelim'" in all_source
    assert "resolve_colab_drive_layout(MYDRIVE_ROOT)" in all_source
    assert "SOURCE_RUN_ROOT.name == grid.COLAB_SOURCE_RUN_ID" in all_source
    assert "TRAINING_ARCHIVE_ROOT.parts[-2:]" in all_source
    assert "TRAINING_ARCHIVES = grid.training_archive_paths(TRAINING_ARCHIVE_ROOT)" in all_source
    assert "OUTPUT_ROOT.name == grid.COLAB_OUTPUT_DIR_NAME" in all_source
    assert "grid._checkpoint_files_from_manifest(artifact_manifest['files'])" in all_source
    assert "prepare_colab_resume_run(" in all_source
    assert sum("run_remaining_candidate_fits(" in source for source in code_cells) == 1
    assert "grid.run_one(" not in all_source
    assert "extract_training_feature_blocks(" not in all_source


def test_resume_ledger_pins_baselines_and_exclusions_while_candidates_progress():
    statuses = {f"baseline_{model}": "REUSED_VERIFIED" for model in MODEL_IDS}
    statuses.update({name: "SKIPPED_BY_USER_SCOPE" for name in user_scope_excluded_experiment_ids()})
    statuses.update({name: "PENDING" for name in candidate_experiment_ids()})
    ledger = {"experiments": statuses}
    assert _resume_ledger_valid(ledger)
    ledger["experiments"][candidate_experiment_ids()[0]] = "COMPLETE"
    assert _resume_ledger_valid(ledger)
    ledger["experiments"]["baseline_tabm"] = "PENDING"
    assert not _resume_ledger_valid(ledger)
    assert len(set(experiment_ids())) == 40


def test_baseline_checkpoint_inventory_accepts_pytorch_tabular_model_ckpt():
    inventory = {
        "checkpoint/model.ckpt": "sha-model",
        "checkpoint/config.yml": "sha-config",
        "checkpoint/custom_params.sav": "sha-params",
        "checkpoint.pt": "sha-tabm",
        "weights.pth": "sha-pytorch",
    }

    assert _checkpoint_files_from_manifest(inventory) == [
        "checkpoint.pt", "checkpoint/model.ckpt", "weights.pth",
    ]


def test_ft_transformer_feature_runs_are_explicitly_excluded_but_baseline_is_retained():
    expected_excluded = tuple(f"ft_transformer_F0{index}" for index in range(1, 8))
    assert user_scope_excluded_experiment_ids() == expected_excluded
    assert len(active_experiment_ids()) == 33
    assert "baseline_ft_transformer" in active_experiment_ids()
    assert not set(expected_excluded) & set(active_experiment_ids())


def test_scope_exclusion_is_audited_and_preserves_completed_ft_baseline(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    ids = experiment_ids()
    (root / "contract.json").write_text(json.dumps({
        "schema_version": "mission2_acoustic_feature_grid_v1",
        "experiments": list(ids),
    }), encoding="utf-8")
    statuses = {name: "PENDING" for name in ids}
    statuses["baseline_ft_transformer"] = "COMPLETE"
    (root / "run_ledger.json").write_text(json.dumps({
        "run_id": "fixture", "status": "READY", "experiments": statuses,
        "completed_count": 1, "next_experiment": "baseline_tabular_resnet",
    }), encoding="utf-8")

    apply_user_scope_exclusions(root)

    ledger = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    amendment = json.loads((root / "scope_amendment.json").read_text(encoding="utf-8"))
    assert amendment["excluded_experiments"] == list(user_scope_excluded_experiment_ids())
    assert amendment["scheduled_fit_count"] == 33
    assert ledger["experiments"]["baseline_ft_transformer"] == "COMPLETE"
    assert ledger["experiments"]["ft_transformer_F01"] == "SKIPPED_BY_USER_SCOPE"
    assert ledger["skipped_count"] == 7


def test_scope_exclusion_refuses_any_ft_feature_with_existing_artifacts(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    ids = experiment_ids()
    (root / "contract.json").write_text(json.dumps({
        "schema_version": "mission2_acoustic_feature_grid_v1",
        "experiments": list(ids),
    }), encoding="utf-8")
    (root / "run_ledger.json").write_text(json.dumps({
        "run_id": "fixture", "status": "READY",
        "experiments": {name: "PENDING" for name in ids},
    }), encoding="utf-8")
    (root / "ft_transformer_F03").mkdir()

    with pytest.raises(RuntimeError, match="non-pending status or artifacts"):
        apply_user_scope_exclusions(root)

    assert not (root / "scope_amendment.json").exists()


def test_skip_status_cannot_be_added_without_a_matching_scope_amendment(tmp_path):
    ids = experiment_ids()
    statuses = {name: "PENDING" for name in ids}
    statuses["ft_transformer_F01"] = "SKIPPED_BY_USER_SCOPE"
    (tmp_path / "run_ledger.json").write_text(json.dumps({
        "run_id": "fixture", "status": "READY", "experiments": statuses,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="approved scope amendment"):
        _mark_ledger(tmp_path, "baseline_tabm", "COMPLETE")


def test_ledger_completes_with_thirty_three_fits_and_seven_approved_skips(tmp_path):
    ledger_path = tmp_path / "run_ledger.json"
    statuses = {name: "COMPLETE" for name in active_experiment_ids()}
    excluded = user_scope_excluded_experiment_ids()
    statuses.update({name: "SKIPPED_BY_USER_SCOPE" for name in excluded})
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({
        "schema_version": "mission2_acoustic_feature_grid_v1",
        "experiments": list(experiment_ids()),
    }), encoding="utf-8")
    (tmp_path / "scope_amendment.json").write_text(json.dumps({
        "schema_version": "mission2_acoustic_feature_grid_scope_amendment_v1",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "excluded_experiments": list(excluded),
        "scheduled_fit_count": 33,
        "single_feature_candidate_count": 28,
        "baseline_ft_transformer_retained": True,
        "reason": "SKIP FT-Transformer F01-F07 feature additions by user-approved scope; preserve its completed 59D baseline.",
    }), encoding="utf-8")
    ledger_path.write_text(json.dumps({
        "run_id": "fixture", "status": "RUNNING", "experiments": statuses,
    }), encoding="utf-8")

    _mark_ledger(tmp_path, active_experiment_ids()[-1], "COMPLETE")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["completed_count"] == 33
    assert ledger["skipped_count"] == 7
    assert ledger["status"] == "ALL_33_COMPLETE_WITH_7_SKIPPED"


def test_frozen_egemaps_schema_is_exact_and_disjoint():
    assert len(FEATURE_BLOCKS[3].feature_names) == 6
    assert len(FEATURE_BLOCKS[4].feature_names) == 11
    assert not set(FEATURE_BLOCKS[3].feature_names) & set(FEATURE_BLOCKS[4].feature_names)
    assert FEATURE_BLOCKS[3].feature_names == (
        "slopeV0-500_sma3nz_amean",
        "slopeV0-500_sma3nz_stddevNorm",
        "slopeV500-1500_sma3nz_amean",
        "slopeV500-1500_sma3nz_stddevNorm",
        "slopeUV0-500_sma3nz_amean",
        "slopeUV500-1500_sma3nz_amean",
    )
    assert FEATURE_BLOCKS[4].feature_names == (
        "spectralFluxV_sma3nz_amean",
        "spectralFluxV_sma3nz_stddevNorm",
        "mfcc1V_sma3nz_amean",
        "mfcc1V_sma3nz_stddevNorm",
        "mfcc2V_sma3nz_amean",
        "mfcc2V_sma3nz_stddevNorm",
        "mfcc3V_sma3nz_amean",
        "mfcc3V_sma3nz_stddevNorm",
        "mfcc4V_sma3nz_amean",
        "mfcc4V_sma3nz_stddevNorm",
        "spectralFluxUV_sma3nz_amean",
    )


def test_spectral_feature_bands_cover_full_8khz_nyquist():
    contrast = FEATURE_BLOCKS[1]
    power_ratio = FEATURE_BLOCKS[2]
    assert contrast.dimension == 10
    assert contrast.config["bands_hz"] == [
        [0, 250], [250, 500], [500, 1000], [1000, 2000], [2000, 4000],
    ]
    assert power_ratio.dimension == 8
    assert power_ratio.config["bands_hz"] == [
        [0, 500], [500, 1000], [1000, 2000], [2000, 4000],
    ]


def test_af12_original_crop_fallback_is_checked_separately_from_matrix_helper():
    result = validate_frontend_parity()
    checks = {item["samples"]: item for item in result["checks"]}
    assert result["status"] == "PASS"
    assert checks[320]["frames"] == 3
    assert checks[320]["af12_fallback"] is True
    assert checks[480]["af12_fallback"] is False


def test_egemaps_selection_requires_exact_row_order_and_named_columns():
    keys = np.asarray(["Training|call-a|0", "Training|call-b|0"])
    names = (*FEATURE_BLOCKS[3].feature_names, *FEATURE_BLOCKS[4].feature_names,
             *(f"other_{index}" for index in range(88 - 17)))
    matrix = np.arange(2 * 88, dtype=np.float32).reshape(2, 88)
    selected = select_egemaps_blocks(matrix, names, keys, keys)
    assert selected["F04"].shape == (2, 6)
    assert selected["F05"].shape == (2, 11)
    with pytest.raises(ValueError, match="record_key order"):
        select_egemaps_blocks(matrix, names, keys[::-1], keys)


def test_gandalf_logged_metrics_snapshot_keeps_true_epoch_metadata():
    trainer = SimpleNamespace(
        logged_metrics={
            "train_loss": np.float32(0.25),
            "valid_accuracy": np.float32(0.8817),
            "epoch": np.int64(19),
        },
        current_epoch=19,
        callbacks=[SimpleNamespace(
            best_model_path="/content/saved_models/classification-1_epoch=18-valid_loss=0.25.ckpt",
        )],
    )

    history = stage13._gandalf_history_snapshot(trainer)

    assert history.loc[0, "train_loss"] == 0.25
    assert history.loc[0, "valid_accuracy"] == pytest.approx(0.8817)
    assert history.loc[0, "epoch"] == 19
    assert stage13._best_epoch(history) == 19
    assert stage13._epochs_run(history) == 20


def test_gandalf_history_does_not_invent_best_epoch_without_checkpoint_metadata():
    trainer = SimpleNamespace(
        logged_metrics={"valid_accuracy": np.float32(0.88)},
        current_epoch=19,
        callbacks=[],
    )

    history = stage13._gandalf_history_snapshot(trainer)

    assert stage13._best_epoch(history) is None
    assert stage13._epochs_run(history) == 20
