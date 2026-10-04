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
    build_ensemble_metadata,
    checkpoint_sha256,
    require_l2_inference_config,
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
