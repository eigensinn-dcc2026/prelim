"""Tests for scripts/train_mission2_cnn.py's input guard.

Real training runs on Colab GPU; these run ``main()`` for one epoch of the smallest model on a tiny
synthetic spectrogram cache, because what matters here is not the accuracy but *which index the
script is willing to train on*: the official Validation folder must never reach the loss/backprop,
even when someone points ``--index`` at the wrong cache.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

train_mission2_cnn = importlib.import_module("train_mission2_cnn")

from dcc_data.guards import SplitContaminationError  # noqa: E402
from dcc_data.mission2_features import HOP_LENGTH, N_FFT  # noqa: E402
from dcc_data.mission2_spectrogram import N_MELS  # noqa: E402

N_CALLS = 10
UTTERANCES_PER_CALL = 2


def _write_cache(tmp_path: Path, *, provenance: dict | None) -> tuple[Path, Path]:
    """A tiny spectrogram cache: 10 calls x 2 utterances. ``provenance`` is the extra index columns
    (None = an index built before provenance existed)."""

    spectrogram_dir = tmp_path / "specs"
    rows = []
    for call in range(N_CALLS):
        for utt in range(UTTERANCES_PER_CALL):
            n_frames = 8 + 3 * utt
            spec = np.random.default_rng(call * 10 + utt).uniform(-40, 0, size=(N_MELS, n_frames)).astype(np.float32)
            npy_path = f"call{call}/{utt}.npy"
            (spectrogram_dir / npy_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(spectrogram_dir / npy_path, spec)
            rows.append(
                {
                    "call_id": f"call{call}", "utt_idx": utt, "speaker": (call + utt) % 2,
                    "duration_sec": 0.5 + utt, "n_frames": n_frames, "npy_path": npy_path,
                }
            )
    frame = pd.DataFrame(rows)
    if provenance is not None:
        frame = frame.assign(**provenance)
    index_path = tmp_path / "index.parquet"
    frame.to_parquet(index_path, index=False)
    return index_path, spectrogram_dir


TRAINING_PROVENANCE = {
    "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT, "sample_rate": 8000,
    "label_source": "provided_json",
}


def _run_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index_path: Path, spectrogram_dir: Path, *, model: str = "small"
) -> Path:
    model_output = tmp_path / "out" / "m2_cnn.pt"
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(spectrogram_dir),
            "--model", model, "--epochs", "1", "--batch-size", "4", "--model-output", str(model_output),
        ],
    )
    assert train_mission2_cnn.main() == 0
    return model_output


def test_training_runs_on_an_index_stamped_as_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)

    model_output = _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    assert model_output.exists()


def test_training_refuses_an_index_stamped_as_validation_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance={**TRAINING_PROVENANCE, "split": "Validation"})

    with pytest.raises(SplitContaminationError, match="non-Training"):
        _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    assert not (tmp_path / "out").exists()


def test_training_refuses_an_index_with_no_provenance_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=None)

    with pytest.raises(SplitContaminationError, match="no build provenance"):
        _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    assert not (tmp_path / "out").exists()


def test_training_refuses_an_index_that_does_not_say_its_labels_are_the_provided_ones_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dcc_data.mission2_spectrogram import LabelProvenanceError

    index_path, spectrogram_dir = _write_cache(
        tmp_path, provenance={**TRAINING_PROVENANCE, "label_source": "corrected_after_reading_text"}
    )

    with pytest.raises(LabelProvenanceError, match="provided speaker labels"):
        _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    assert not (tmp_path / "out").exists()


def test_training_refuses_an_index_stamped_before_label_provenance_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the five columns the first version of the stamp had -- everything but ``label_source``
    old_stamp = {name: value for name, value in TRAINING_PROVENANCE.items() if name != "label_source"}
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=old_stamp)

    with pytest.raises(SplitContaminationError, match="label_source"):
        _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# The checkpoint records what it was trained with
# ---------------------------------------------------------------------------


def test_the_checkpoint_records_the_model_the_mel_settings_the_split_and_the_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)

    model_output = _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)

    checkpoint = torch.load(model_output, map_location="cpu", weights_only=True)  # plain values only
    assert checkpoint["metadata"] == {
        "metadata_version": 1,
        "model_name": "small",
        "model_class": "SpeakerCNN",
        "index_split": "Training",
        "label_source": "provided_json",
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "sample_rate": 8000,
        "split_seed": 42,
        "test_size": 0.2,
        "torch_seed": 42,  # --torch-seed defaults to --seed
    }
    assert checkpoint["best_epoch"] == 1 and checkpoint["total_epochs_trained"] == 1


def test_a_large_checkpoint_from_the_trainer_loads_into_the_submission_predictor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mission2_inference = importlib.import_module("mission2_inference")
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)

    model_output = _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir, model="large")

    predict, fallback_speaker = mission2_inference.load_predictor(model_output)  # no registry entry needed
    segment = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    assert predict(segment, 8000, 0.5) in (0, 1)
    assert fallback_speaker in (0, 1)


def test_a_checkpoint_of_another_model_from_the_trainer_is_refused_by_the_submission_predictor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    mission2_inference = importlib.import_module("mission2_inference")
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)

    model_output = _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir, model="small")

    with pytest.raises(CheckpointConfigError, match="model_name: checkpoint recorded 'small'"):
        mission2_inference.load_predictor(model_output)
