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
