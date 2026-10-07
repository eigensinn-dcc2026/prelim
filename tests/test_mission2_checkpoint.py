"""Tests for dcc_data.mission2_checkpoint: what an L2 checkpoint records about how it was trained, and
the checks scripts/mission2_inference.py runs against it.

A SpeakerCNNLarge-shaped state_dict loads into any SpeakerCNNLarge, whatever it was trained as, so these
checks are the only thing standing between a wrong checkpoint and a silently worse submission.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import torch

from dcc_data.mission2_checkpoint import (
    LEGACY_L2_CHECKPOINTS,
    REQUIRED_METADATA_KEYS,
    CheckpointConfigError,
    build_checkpoint_metadata,
    ENSEMBLE_STATE_DICTS_KEY,
    build_ensemble_metadata,
    checkpoint_sha256,
    l2_member_state_dicts,
    require_l2_inference_config,
    require_resumable_checkpoint,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_features import HOP_LENGTH, N_FFT
from dcc_data.mission2_spectrogram import LABEL_SOURCE_PROVIDED, LABEL_SOURCE_VERIFIED, N_MELS

PROVENANCE = {
    "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT, "sample_rate": 8000,
    "label_source": LABEL_SOURCE_PROVIDED,
}


def _metadata(**overrides) -> dict:
    metadata = build_checkpoint_metadata(
        model_name="large", model_class="SpeakerCNNLarge", index_provenance=PROVENANCE,
        split_seed=42, test_size=0.2, torch_seed=7,
    )
    metadata.update(overrides)
    return metadata


def test_checkpoint_sha256_is_the_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"abc" * 1_000_000)  # spans several read chunks

    assert checkpoint_sha256(path) == hashlib.sha256(b"abc" * 1_000_000).hexdigest()


def test_build_checkpoint_metadata_records_the_training_config() -> None:
    metadata = _metadata()

    assert set(metadata) == set(REQUIRED_METADATA_KEYS)
    assert metadata["model_name"] == "large"
    assert metadata["model_class"] == "SpeakerCNNLarge"
    assert metadata["index_split"] == "Training"
    assert metadata["label_source"] == LABEL_SOURCE_PROVIDED
    assert (metadata["n_mels"], metadata["hop_length"], metadata["n_fft"], metadata["sample_rate"]) == (
        N_MELS, HOP_LENGTH, N_FFT, 8000,
    )
    assert (metadata["split_seed"], metadata["test_size"], metadata["torch_seed"]) == (42, 0.2, 7)


def test_metadata_survives_torch_save_and_a_weights_only_load(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save({"best_model_state_dict": {"w": torch.zeros(1)}, "metadata": _metadata()}, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded["metadata"] == _metadata()


def test_embedded_metadata_is_used_and_needs_no_registry_entry(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"not in the registry")

    metadata, source = resolve_l2_checkpoint_metadata({"metadata": _metadata()}, path)

    assert metadata == _metadata()
    assert "embedded" in source


def test_incomplete_embedded_metadata_is_refused(tmp_path: Path) -> None:
    incomplete = _metadata()
    del incomplete["n_mels"]

    with pytest.raises(CheckpointConfigError, match="incomplete, missing: \\['n_mels'\\]"):
        resolve_l2_checkpoint_metadata({"metadata": incomplete}, tmp_path / "ckpt.pt")


def test_a_checkpoint_with_no_metadata_and_no_registry_entry_is_refused_with_the_way_out(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"some unknown weights")

    with pytest.raises(CheckpointConfigError) as excinfo:
        resolve_l2_checkpoint_metadata({"best_model_state_dict": {}}, path)

    message = str(excinfo.value)
    assert checkpoint_sha256(path) in message
    assert "train_mission2_cnn.py" in message
    assert "LEGACY_L2_CHECKPOINTS" in message


def test_a_registered_legacy_checkpoint_gets_its_registered_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "old.pt"
    path.write_bytes(b"an old checkpoint")
    registered = {"description": "an old file", "metadata": _metadata(torch_seed=None)}
    monkeypatch.setitem(LEGACY_L2_CHECKPOINTS, checkpoint_sha256(path), registered)

    metadata, source = resolve_l2_checkpoint_metadata({"best_model_state_dict": {}}, path)

    assert metadata == registered["metadata"]
    assert "legacy registry" in source and "an old file" in source


def test_a_changed_file_is_no_longer_the_registered_legacy_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "old.pt"
    path.write_bytes(b"an old checkpoint")
    monkeypatch.setitem(
        LEGACY_L2_CHECKPOINTS, checkpoint_sha256(path), {"description": "x", "metadata": _metadata()}
    )
    path.write_bytes(b"an old checkpoint, re-saved with different bytes")

    with pytest.raises(CheckpointConfigError, match="not a registered legacy"):
        resolve_l2_checkpoint_metadata({"best_model_state_dict": {}}, path)


def test_the_config_check_accepts_a_checkpoint_trained_for_this_inference_setup() -> None:
    require_l2_inference_config(_metadata())


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"model_name": "large-masked"}, "model_name: checkpoint recorded 'large-masked'"),
        ({"model_name": "small"}, "model_name: checkpoint recorded 'small'"),
        ({"n_mels": N_MELS * 2}, f"n_mels: checkpoint recorded {N_MELS * 2}"),
        ({"hop_length": HOP_LENGTH // 2}, f"hop_length: checkpoint recorded {HOP_LENGTH // 2}"),
        ({"n_fft": N_FFT * 2}, f"n_fft: checkpoint recorded {N_FFT * 2}"),
        ({"index_split": "Validation"}, "index_split: checkpoint recorded 'Validation'"),
        ({"label_source": "corrected_after_reading_text"}, "label_source: checkpoint recorded 'corrected_after_reading_text'"),
    ],
)
def test_the_config_check_refuses_each_setting_that_differs(override: dict, fragment: str) -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        require_l2_inference_config(_metadata(**override))

    assert fragment in str(excinfo.value)


def test_the_config_check_reports_every_difference_at_once() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        require_l2_inference_config(_metadata(model_name="large-masked", n_mels=80))

    message = str(excinfo.value)
    assert "model_name" in message and "n_mels" in message


def test_every_registered_legacy_checkpoint_entry_is_well_formed_and_runs_under_the_current_defaults() -> None:
    for digest, entry in LEGACY_L2_CHECKPOINTS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert set(REQUIRED_METADATA_KEYS) <= set(entry["metadata"])
        # If a module default ever changes, a registered checkpoint must be refused (loudly), not relabelled:
        # this is the test that fails first.
        require_l2_inference_config(entry["metadata"])


def test_the_checkpoint_trained_on_corrected_labels_is_not_registered_and_is_therefore_refused(tmp_path: Path) -> None:
    # m2_cnn_large.pt (M2-VAL-003) was trained after six Training calls' speaker labels were corrected on the
    # strength of reading their ``text``; the organizers ruled (2026-10-01) that training must use the provided
    # labels as they are. If this hash ever comes back into the registry, someone must have decided on purpose.
    shipped = "04c0cc04e2c9f4033db4f28164b4c3c888b6642588728bc3ea1b3b73ce406189"
    assert shipped not in LEGACY_L2_CHECKPOINTS

    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"stands in for a checkpoint with no recorded config")
    with pytest.raises(CheckpointConfigError, match="records no training config"):
        resolve_l2_checkpoint_metadata({"best_model_state_dict": {}}, path)


def test_the_verified_label_source_is_accepted_too() -> None:
    require_l2_inference_config(_metadata(label_source=LABEL_SOURCE_VERIFIED))


# ---------------------------------------------------------------------------
# build_ensemble_metadata: merging several same-architecture checkpoints'
# metadata into one, for scripts/pack_mission2_l2_ensemble.py.
# ---------------------------------------------------------------------------


def test_build_ensemble_metadata_merges_members_that_only_differ_by_seed() -> None:
    members = [_metadata(torch_seed=1), _metadata(torch_seed=2), _metadata(torch_seed=3)]

    merged = build_ensemble_metadata(members)

    assert merged["torch_seed"] is None
    assert merged["ensemble_member_torch_seeds"] == [1, 2, 3]
    assert merged["ensemble_size"] == 3
    # every other field matches what a single member would have recorded
    for key in REQUIRED_METADATA_KEYS:
        if key == "torch_seed":
            continue
        assert merged[key] == members[0][key]


def test_build_ensemble_metadata_passes_the_config_check_like_a_single_model_does() -> None:
    merged = build_ensemble_metadata([_metadata(torch_seed=1), _metadata(torch_seed=2)])
    require_l2_inference_config(merged)  # must not raise


def test_build_ensemble_metadata_rejects_members_that_disagree_on_training_setup() -> None:
    members = [_metadata(torch_seed=1), _metadata(torch_seed=2, n_mels=80)]

    with pytest.raises(CheckpointConfigError, match="n_mels"):
        build_ensemble_metadata(members)


def test_build_ensemble_metadata_requires_at_least_two_members() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_ensemble_metadata([_metadata()])


def test_build_ensemble_metadata_requires_complete_member_metadata() -> None:
    incomplete = _metadata()
    del incomplete["n_mels"]

    with pytest.raises(CheckpointConfigError):
        build_ensemble_metadata([_metadata(), incomplete])


# ---------------------------------------------------------------------------
# l2_member_state_dicts: the weights stored must be as many as the metadata says
# ---------------------------------------------------------------------------


def _ensemble_checkpoint(n_stored: int, n_declared: int) -> dict:
    metadata = build_ensemble_metadata([_metadata(torch_seed=seed) for seed in range(1, n_declared + 1)]) \
        if n_declared >= 2 else _metadata()
    return {ENSEMBLE_STATE_DICTS_KEY: [{"w": i} for i in range(n_stored)], "metadata": metadata}


def test_a_single_model_checkpoint_gives_its_one_state_dict() -> None:
    assert l2_member_state_dicts({"best_model_state_dict": {"w": 1}}, _metadata()) == [{"w": 1}]


def test_an_ensemble_checkpoint_gives_one_state_dict_per_member() -> None:
    checkpoint = _ensemble_checkpoint(n_stored=3, n_declared=3)

    assert l2_member_state_dicts(checkpoint, checkpoint["metadata"]) == [{"w": 0}, {"w": 1}, {"w": 2}]


@pytest.mark.parametrize(("n_stored", "n_declared"), [(2, 3), (3, 2), (1, 2)])
def test_an_ensemble_whose_weights_and_recorded_size_disagree_is_refused(n_stored: int, n_declared: int) -> None:
    checkpoint = _ensemble_checkpoint(n_stored, n_declared)

    with pytest.raises(CheckpointConfigError, match=f"declares {n_declared} model"):
        l2_member_state_dicts(checkpoint, checkpoint["metadata"])


def test_an_empty_ensemble_is_refused_instead_of_falling_back_to_a_single_model() -> None:
    checkpoint = _ensemble_checkpoint(n_stored=0, n_declared=2)
    checkpoint["best_model_state_dict"] = {"w": "single"}  # must not be used as a silent fallback

    with pytest.raises(CheckpointConfigError, match="holds 0 set"):
        l2_member_state_dicts(checkpoint, checkpoint["metadata"])


def test_a_single_model_checkpoint_labelled_as_an_ensemble_is_refused() -> None:
    metadata = build_ensemble_metadata([_metadata(torch_seed=1), _metadata(torch_seed=2)])

    with pytest.raises(CheckpointConfigError, match="declares 2 model"):
        l2_member_state_dicts({"best_model_state_dict": {"w": 1}}, metadata)


def test_an_ensemble_whose_recorded_seed_list_has_another_length_is_refused() -> None:
    checkpoint = _ensemble_checkpoint(n_stored=2, n_declared=2)
    checkpoint["metadata"]["ensemble_member_torch_seeds"] = [1]

    with pytest.raises(CheckpointConfigError, match="seeds recorded"):
        l2_member_state_dicts(checkpoint, checkpoint["metadata"])


def test_a_checkpoint_with_no_weights_at_all_is_refused_with_a_clear_message() -> None:
    with pytest.raises(CheckpointConfigError, match="no weights to run"):
        l2_member_state_dicts({"metadata": _metadata()}, _metadata())


# ---------------------------------------------------------------------------
# require_resumable_checkpoint: --resume must not re-stamp weights nobody checked
# ---------------------------------------------------------------------------


def _resumable(**metadata_overrides) -> dict:
    return {"last_model_state_dict": {}, "metadata": _metadata(**metadata_overrides)}


def test_a_checkpoint_recording_the_same_setup_may_be_resumed() -> None:
    require_resumable_checkpoint(_resumable(), _metadata(), "ckpt.pt")


def test_a_checkpoint_may_be_resumed_when_it_was_verified_against_the_provided_labels_and_so_is_the_run() -> None:
    both = {"label_source": LABEL_SOURCE_VERIFIED}

    require_resumable_checkpoint(_resumable(**both), _metadata(**both), "ckpt.pt")


@pytest.mark.parametrize(
    "override",
    [
        {"label_source": "corrected_after_reading_text"},
        {"label_source": LABEL_SOURCE_VERIFIED},  # even another *allowed* source: two different indexes are mixed
        {"n_mels": N_MELS * 2},
        {"hop_length": HOP_LENGTH // 2},
        {"n_fft": N_FFT * 2},
        {"sample_rate": 16000},
        {"index_split": "Validation"},
        {"model_name": "large-masked"},
        {"split_seed": 1},
        {"test_size": 0.3},
        {"torch_seed": 8},
    ],
)
def test_resuming_a_checkpoint_recorded_with_any_different_field_is_refused(override: dict) -> None:
    (key,) = override

    with pytest.raises(CheckpointConfigError, match=f"{key}: the checkpoint recorded"):
        require_resumable_checkpoint(_resumable(**override), _metadata(), "ckpt.pt")


def test_the_refusal_lists_every_field_that_differs_and_names_the_file() -> None:
    with pytest.raises(CheckpointConfigError) as excinfo:
        require_resumable_checkpoint(_resumable(n_mels=80, torch_seed=9), _metadata(), "some/ckpt.pt")

    message = str(excinfo.value)
    assert "some/ckpt.pt" in message and "n_mels" in message and "torch_seed" in message


def test_a_checkpoint_without_recorded_metadata_cannot_be_resumed() -> None:
    with pytest.raises(CheckpointConfigError, match="records no training config"):
        require_resumable_checkpoint({"last_model_state_dict": {}}, _metadata(), "ckpt.pt")


def test_a_checkpoint_without_resume_state_cannot_be_resumed_even_with_metadata() -> None:
    with pytest.raises(CheckpointConfigError, match="predates resume support"):
        require_resumable_checkpoint({"metadata": _metadata()}, _metadata(), "ckpt.pt")


def test_a_checkpoint_with_incomplete_metadata_cannot_be_resumed() -> None:
    incomplete = _resumable()
    del incomplete["metadata"]["label_source"]

    with pytest.raises(CheckpointConfigError, match="incomplete, missing: \\['label_source'\\]"):
        require_resumable_checkpoint(incomplete, _metadata(), "ckpt.pt")


def test_a_packed_ensemble_cannot_be_resumed() -> None:
    packed = {ENSEMBLE_STATE_DICTS_KEY: [{}, {}], "metadata": _metadata(), "last_model_state_dict": {}}

    with pytest.raises(CheckpointConfigError, match="packed ensemble"):
        require_resumable_checkpoint(packed, _metadata(), "ckpt.pt")


def test_ensemble_metadata_refuses_a_repeated_or_unknown_member_seed() -> None:
    with pytest.raises(CheckpointConfigError, match="different torch_seed"):
        build_ensemble_metadata([_metadata(torch_seed=1), _metadata(torch_seed=1)])
    with pytest.raises(CheckpointConfigError, match="different torch_seed"):
        build_ensemble_metadata([_metadata(torch_seed=1), _metadata(torch_seed=None)])


def test_an_ensemble_checkpoint_recording_one_seed_twice_is_refused_at_load() -> None:
    metadata = build_ensemble_metadata([_metadata(torch_seed=1), _metadata(torch_seed=2)])
    metadata["ensemble_member_torch_seeds"] = [1, 1]
    checkpoint = {ENSEMBLE_STATE_DICTS_KEY: [{"w": 0}, {"w": 1}], "metadata": metadata}

    with pytest.raises(CheckpointConfigError, match="same torch_seed more than once"):
        l2_member_state_dicts(checkpoint, metadata)
