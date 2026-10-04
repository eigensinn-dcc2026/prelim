"""What a Mission 2 L2 checkpoint was trained with, recorded next to its weights and checked at inference.

A SpeakerCNNLarge state_dict loads into any SpeakerCNNLarge-shaped model, whatever mel settings it was
trained on and even if it is really a variant with the same modules but a different forward pass (the
model-swap experiments' masked-pooling model shares SpeakerCNNLarge's parameters exactly). Nothing in the
weights says which, so a wrong checkpoint would run without any error and just predict worse.

scripts/train_mission2_cnn.py therefore embeds ``build_checkpoint_metadata(...)`` under
``checkpoint["metadata"]``, and scripts/mission2_inference.py refuses to run a checkpoint whose recorded
config differs from what it would run it with -- or that records nothing at all, unless the file is one of
the known legacy checkpoints registered below by its sha256.

This module deliberately imports no torch: the L1-only inference path must not pay for it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .guards import TRAINING
from .mission2_features import HOP_LENGTH, N_FFT
from .mission2_spectrogram import ALLOWED_LABEL_SOURCES, N_MELS

METADATA_VERSION = 1

# The only L2 model scripts/mission2_inference.py builds (train_mission2_cnn.py's ``--model large``).
INFERENCE_MODEL_NAME = "large"

# Key an ensemble checkpoint (scripts/pack_mission2_l2_ensemble.py) stores its member state_dicts under,
# in place of a single-model checkpoint's "best_model_state_dict". scripts/mission2_inference.py checks
# for this key to decide whether to soft-vote several models instead of running just one.
ENSEMBLE_STATE_DICTS_KEY = "ensemble_model_state_dicts"

REQUIRED_METADATA_KEYS = (
    "metadata_version",
    "model_name",
    "model_class",
    "index_split",
    "label_source",
    "n_mels",
    "hop_length",
    "n_fft",
    "sample_rate",
    "split_seed",
    "test_size",
    "torch_seed",
)


class CheckpointConfigError(ValueError):
    """A checkpoint's recorded training config disagrees with how inference would run it (or is missing).

    A setup error, not a bad utterance: scripts/mission2_inference.py never turns it into a majority-class
    fallback.
    """


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_metadata(
    *,
    model_name: str,
    model_class: str,
    index_provenance: dict,
    split_seed: int,
    test_size: float,
    torch_seed: int | None,
) -> dict:
    """The dict train_mission2_cnn.py stores as ``checkpoint["metadata"]``.

    ``index_provenance`` is what dcc_data.mission2_spectrogram.assert_training_spectrogram_index returned
    for the index the run trained on. Plain Python values only, so the checkpoint still loads with
    ``torch.load(..., weights_only=True)``.
    """

    return {
        "metadata_version": METADATA_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "index_split": index_provenance["split"],
        "label_source": index_provenance["label_source"],
        "n_mels": index_provenance["n_mels"],
        "hop_length": index_provenance["hop_length"],
        "n_fft": index_provenance["n_fft"],
        "sample_rate": index_provenance["sample_rate"],
        "split_seed": split_seed,
        "test_size": test_size,
        "torch_seed": torch_seed,
    }


# Checkpoints trained before metadata existed, keyed by the sha256 of the file, with the settings that were in
# force written out literally (not read from the module defaults), so that changing a default later makes
# inference refuse the old file instead of silently re-labelling it.
#
# Deliberately EMPTY. The one checkpoint that used to be registered here (m2_cnn_large.pt, the 90.89%
# official-Validation result of M2-VAL-003, sha256 04c0cc04e2c9f4033db4f28164b4c3c888b6642588728bc3ea1b3b73ce406189)
# was trained on an index whose speaker labels had been corrected for six Training calls after reading their
# ``text``. The organizers ruled (2026-10-01) that training must use the provided labels as they are, so that
# file must not be submitted and is no longer accepted. Only register a file here when you know -- not
# assume -- that it was trained on the provided labels as they are; otherwise retrain with
# scripts/train_mission2_cnn.py, which records ``label_source`` in the checkpoint itself.
LEGACY_L2_CHECKPOINTS: dict[str, dict] = {}


def _require_complete(metadata: dict, where: str) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise CheckpointConfigError(f"checkpoint metadata ({where}) is incomplete, missing: {missing}")


def resolve_l2_checkpoint_metadata(checkpoint: dict, ckpt_path: str | Path) -> tuple[dict, str]:
    """(metadata, where it came from) for a loaded L2 checkpoint, or CheckpointConfigError.

    Embedded metadata wins; a checkpoint without any is accepted only if its sha256 is registered in
    LEGACY_L2_CHECKPOINTS.
    """

    embedded = checkpoint.get("metadata")
    if embedded is not None:
        _require_complete(embedded, "embedded in the checkpoint")
        return dict(embedded), "embedded in the checkpoint"

    digest = checkpoint_sha256(ckpt_path)
    legacy = LEGACY_L2_CHECKPOINTS.get(digest)
    if legacy is None:
        raise CheckpointConfigError(
            f"{ckpt_path} records no training config and its sha256 ({digest}) is not a registered legacy "
            "checkpoint, so nothing says which model or mel settings it was trained with (a SpeakerCNNLarge-"
            "shaped state_dict loads into the wrong variant without any error). Retrain with the current "
            "scripts/train_mission2_cnn.py, which embeds the config. (dcc_data.mission2_checkpoint."
            "LEGACY_L2_CHECKPOINTS is for files known to have been trained on the provided labels as they are; "
            "the earlier m2_cnn_large.pt was trained on corrected labels and is deliberately not in it.)"
        )
    _require_complete(legacy["metadata"], "legacy registry")
    return dict(legacy["metadata"]), f"legacy registry: {legacy['description']}"


def build_ensemble_metadata(member_metadata: list[dict]) -> dict:
    """Merge several same-architecture L2 checkpoints' metadata into one ensemble checkpoint's metadata.

    The members must agree on every ``REQUIRED_METADATA_KEYS`` field except ``torch_seed`` -- that is
    exactly the one thing a seed ensemble is supposed to vary, but everything else (mel settings, label
    source, split) must match or the members are not fair to average together. The merged dict replaces
    ``torch_seed`` with ``None`` (no single seed applies) and records every member's seed and count instead.
    """

    if len(member_metadata) < 2:
        raise ValueError(f"an ensemble needs at least 2 member checkpoints, got {len(member_metadata)}")
    for metadata in member_metadata:
        _require_complete(metadata, "ensemble member")

    merged = dict(member_metadata[0])
    for key in REQUIRED_METADATA_KEYS:
        if key == "torch_seed":
            continue
        values = {metadata[key] for metadata in member_metadata}
        if len(values) != 1:
            raise CheckpointConfigError(
                f"ensemble members disagree on {key!r}: {sorted(map(str, values))} -- they must be the same "
                "architecture/training setup, varied only by torch_seed, to be averaged together"
            )
    merged["torch_seed"] = None
    merged["ensemble_member_torch_seeds"] = [metadata["torch_seed"] for metadata in member_metadata]
    merged["ensemble_size"] = len(member_metadata)
    return merged


def l2_member_state_dicts(checkpoint: dict, metadata: dict) -> list[dict]:
    """The state_dict(s) scripts/mission2_inference.py runs: one for a single model, one per member for a packed ensemble.

    The weights stored must be as many as the metadata says (``ensemble_size``, 1 when it records none), so a
    file whose metadata and weights disagree -- an ensemble missing a member, a single model labelled as an
    ensemble, or the reverse -- is refused instead of being averaged over whatever happens to be inside.
    """

    if ENSEMBLE_STATE_DICTS_KEY in checkpoint:
        state_dicts = list(checkpoint[ENSEMBLE_STATE_DICTS_KEY])
    elif "best_model_state_dict" in checkpoint:
        state_dicts = [checkpoint["best_model_state_dict"]]
    else:
        raise CheckpointConfigError(
            f"checkpoint has neither {ENSEMBLE_STATE_DICTS_KEY!r} nor 'best_model_state_dict', so there are no weights to run"
        )
    declared = metadata.get("ensemble_size", 1)
    seeds = metadata.get("ensemble_member_torch_seeds")
    if declared != len(state_dicts) or (seeds is not None and len(seeds) != declared):
        raise CheckpointConfigError(
            f"checkpoint metadata declares {declared} model(s) (seeds recorded: {seeds}) but the file holds "
            f"{len(state_dicts)} set(s) of weights -- refusing to run a checkpoint whose weights and recorded config disagree"
        )
    return state_dicts


def require_resumable_checkpoint(checkpoint: dict, expected_metadata: dict, ckpt_path: str | Path) -> None:
    """Refuse to continue training a checkpoint unless it provably came from this same setup.

    ``--resume`` loads the old weights, keeps training and then saves a checkpoint whose ``metadata`` is
    rebuilt from the *current* run. If the old file was never checked, a model trained on other labels (or
    other mel settings) would come out the other end stamped as if it had been trained on today's index --
    which is exactly what scripts/mission2_inference.py trusts. So the file being resumed must record a
    complete config and every field of it must equal what this run is about to stamp (``torch_seed``
    included: resuming with another seed would mix two shuffle/initialisation histories under one label).
    A checkpoint that records nothing (every file written before the config was stored, whatever its
    weights were trained on) cannot be shown to match and is refused, as is a packed ensemble file, which
    holds no optimiser state to resume from.
    """

    if ENSEMBLE_STATE_DICTS_KEY in checkpoint:
        raise CheckpointConfigError(
            f"{ckpt_path} is a packed ensemble (scripts/pack_mission2_l2_ensemble.py): it keeps only each member's "
            "best weights, no optimiser state, so it cannot be resumed -- resume one of the member checkpoints instead"
        )
    recorded = checkpoint.get("metadata")
    if recorded is None or "last_model_state_dict" not in checkpoint:
        raise CheckpointConfigError(
            f"cannot resume {ckpt_path}: it records no training config (or predates resume support), so nothing "
            "shows which labels and settings its weights were trained with, and the checkpoint saved after "
            "resuming would claim this run's. Start a new run with the current scripts/train_mission2_cnn.py instead"
        )
    _require_complete(recorded, f"in the checkpoint being resumed, {ckpt_path}")
    differences = [
        f"{key}: the checkpoint recorded {recorded[key]!r}, this run uses {expected_metadata[key]!r}"
        for key in REQUIRED_METADATA_KEYS
        if recorded[key] != expected_metadata[key]
    ]
    if differences:
        raise CheckpointConfigError(
            f"cannot resume {ckpt_path}: it was trained with a different setup -- " + "; ".join(differences)
        )


def require_l2_inference_config(metadata: dict) -> None:
    """Raise CheckpointConfigError unless the checkpoint suits what scripts/mission2_inference.py runs.

    It builds SpeakerCNNLarge and the log-mel spectrogram with the current module defaults, and the
    checkpoint must have been trained on Training only, on the speaker labels exactly as provided.
    """

    expected = {
        "model_name": INFERENCE_MODEL_NAME,
        "index_split": TRAINING,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
    }
    mismatches = [
        f"{key}: checkpoint recorded {metadata[key]!r}, inference uses {value!r}"
        for key, value in expected.items()
        if metadata[key] != value
    ]
    if metadata["label_source"] not in ALLOWED_LABEL_SOURCES:
        mismatches.append(
            f"label_source: checkpoint recorded {metadata['label_source']!r}, but Mission 2 training must use "
            f"the provided speaker labels as they are (one of {', '.join(ALLOWED_LABEL_SOURCES)})"
        )
    if mismatches:
        raise CheckpointConfigError(
            "checkpoint was not trained for this inference setup -- " + "; ".join(mismatches)
        )
