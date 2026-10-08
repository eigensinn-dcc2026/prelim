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


# ---------------------------------------------------------------------------
# --resume must prove the checkpoint it continues from was trained with the same setup
#
# The checkpoint a resumed run saves is stamped from the *current* index. Without a check, weights trained on
# other labels would come out stamped with today's provenance and pass scripts/mission2_inference.py.
# ---------------------------------------------------------------------------


def _resume_train(
    monkeypatch: pytest.MonkeyPatch, model_output: Path, index_path: Path, spectrogram_dir: Path, *extra: str,
    model: str = "small",
) -> int:
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(spectrogram_dir),
            "--model", model, "--epochs", "1", "--batch-size", "4", "--model-output", str(model_output),
            "--resume", *extra,
        ],
    )
    return train_mission2_cnn.main()


def _trained_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    model_output = _run_train(monkeypatch, tmp_path, index_path, spectrogram_dir)
    return index_path, spectrogram_dir, model_output


def _rewrite_checkpoint(path: Path, edit) -> bytes:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    edit(checkpoint)
    torch.save(checkpoint, path)
    return path.read_bytes()


def test_resume_continues_a_checkpoint_made_with_the_same_setup_and_keeps_its_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    before = torch.load(model_output, map_location="cpu")["metadata"]

    assert _resume_train(monkeypatch, model_output, index_path, spectrogram_dir) == 0

    after = torch.load(model_output, map_location="cpu")
    assert after["total_epochs_trained"] == 2
    assert after["metadata"] == before


def test_resume_refuses_a_checkpoint_whose_recorded_label_source_is_not_the_one_of_this_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    # weights that were trained on corrected labels, being continued on a clean provided-labels index
    bytes_before = _rewrite_checkpoint(
        model_output, lambda ck: ck["metadata"].update(label_source="corrected_after_reading_text")
    )

    with pytest.raises(CheckpointConfigError, match="label_source: the checkpoint recorded 'corrected_after_reading_text'"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir)

    assert model_output.read_bytes() == bytes_before  # nothing was overwritten with a fresh stamp


def test_resume_refuses_a_checkpoint_that_records_no_config_and_does_not_restamp_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    bytes_before = _rewrite_checkpoint(model_output, lambda ck: ck.pop("metadata"))

    with pytest.raises(CheckpointConfigError, match="records no training config"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir)

    assert model_output.read_bytes() == bytes_before


def test_resume_refuses_a_checkpoint_from_before_resume_support_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    from dcc_data.mission2_checkpoint import CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    old_format = {"model_state_dict": torch.load(model_output, map_location="cpu")["best_model_state_dict"],
                  "val_acc": 0.5, "epoch": 3}
    torch.save(old_format, model_output)
    bytes_before = model_output.read_bytes()

    with pytest.raises(CheckpointConfigError, match="predates resume support"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir)

    assert model_output.read_bytes() == bytes_before


@pytest.mark.parametrize(
    ("edit", "recorded_field"),
    [
        (lambda ck: ck["metadata"].update(n_mels=80), "n_mels"),
        (lambda ck: ck["metadata"].update(index_split="Validation"), "index_split"),
        (lambda ck: ck["metadata"].update(split_seed=7), "split_seed"),
        (lambda ck: ck["metadata"].update(test_size=0.3), "test_size"),
        (lambda ck: ck["metadata"].update(torch_seed=99), "torch_seed"),
        (lambda ck: ck["metadata"].update(sample_rate=16000), "sample_rate"),
    ],
)
def test_resume_refuses_a_checkpoint_recorded_with_any_other_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, edit, recorded_field: str
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    bytes_before = _rewrite_checkpoint(model_output, edit)

    with pytest.raises(CheckpointConfigError, match=f"{recorded_field}: the checkpoint recorded"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir)

    assert model_output.read_bytes() == bytes_before


def test_resume_refuses_a_checkpoint_of_another_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)  # a "small" run

    with pytest.raises(CheckpointConfigError, match="model_name: the checkpoint recorded 'small'"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir, model="large")


def test_resume_refuses_a_packed_ensemble_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    from dcc_data.mission2_checkpoint import ENSEMBLE_STATE_DICTS_KEY, CheckpointConfigError

    index_path, spectrogram_dir, model_output = _trained_checkpoint(monkeypatch, tmp_path)
    member = torch.load(model_output, map_location="cpu")
    torch.save(
        {ENSEMBLE_STATE_DICTS_KEY: [member["best_model_state_dict"]] * 2, "metadata": member["metadata"]}, model_output
    )

    with pytest.raises(CheckpointConfigError, match="packed ensemble"):
        _resume_train(monkeypatch, model_output, index_path, spectrogram_dir)


# ---------------------------------------------------------------------------
# --resume must continue the random sequence, not restart it
#
# Every run used to seed the shuffle generator (and dropout) with torch_seed alone, so a run split into
# chunks replayed the first chunk's data order in every chunk -- and with 1-epoch chunks every epoch would
# see the batches in the same order. A resumed run must pick up where the previous one left off.
# ---------------------------------------------------------------------------


def _train_epochs(
    monkeypatch: pytest.MonkeyPatch, model_output: Path, index_path: Path, spectrogram_dir: Path, epochs: int,
    *, resume: bool,
) -> None:
    argv = [
        "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(spectrogram_dir),
        "--model", "small", "--epochs", str(epochs), "--batch-size", "4", "--model-output", str(model_output),
        "--torch-seed", "7",
    ]
    if resume:
        argv.append("--resume")
    monkeypatch.setattr(sys, "argv", argv)
    assert train_mission2_cnn.main() == 0


def test_two_one_epoch_chunks_train_exactly_like_one_two_epoch_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)

    straight = tmp_path / "straight" / "m2_cnn.pt"
    _train_epochs(monkeypatch, straight, index_path, spectrogram_dir, 2, resume=False)

    chunked = tmp_path / "chunked" / "m2_cnn.pt"
    _train_epochs(monkeypatch, chunked, index_path, spectrogram_dir, 1, resume=False)
    _train_epochs(monkeypatch, chunked, index_path, spectrogram_dir, 1, resume=True)

    straight_state = torch.load(straight, map_location="cpu")["last_model_state_dict"]
    chunked_state = torch.load(chunked, map_location="cpu")["last_model_state_dict"]
    assert straight_state.keys() == chunked_state.keys()
    for name in straight_state:
        assert torch.equal(straight_state[name], chunked_state[name]), f"{name} differs: the resumed run did not continue the random sequence"


def test_resume_from_a_checkpoint_without_saved_random_state_does_not_replay_the_first_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ts1/ts2/ts3 checkpoints predate saving the random state: resuming them must still work, and must not
    simply replay epoch 1's shuffle order (it reseeds with torch_seed + epochs already trained instead)."""

    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    model_output = tmp_path / "out" / "m2_cnn.pt"
    _train_epochs(monkeypatch, model_output, index_path, spectrogram_dir, 1, resume=False)
    _rewrite_checkpoint(model_output, lambda checkpoint: checkpoint.pop("rng_state", None))

    orders: list[list[int]] = []
    original_iter = torch.utils.data.RandomSampler.__iter__

    def recording_iter(sampler):
        order = list(original_iter(sampler))
        orders.append(order)
        return iter(order)

    monkeypatch.setattr(torch.utils.data.RandomSampler, "__iter__", recording_iter)
    fresh = tmp_path / "fresh" / "m2_cnn.pt"
    _train_epochs(monkeypatch, fresh, index_path, spectrogram_dir, 1, resume=False)  # records epoch 1's order
    _train_epochs(monkeypatch, model_output, index_path, spectrogram_dir, 1, resume=True)  # records the resumed order

    assert len(orders) == 2
    assert orders[0] != orders[1]
    assert torch.load(model_output, map_location="cpu")["total_epochs_trained"] == 2


# ---------------------------------------------------------------------------
# Speed options for the full-dataset rerun: a packed float16 cache, padding to a multiple, early stopping
#
# None of them may change what the model learns from, beyond float16 rounding of the input (which --amp's first
# conv does anyway) and a few extra silent frames.
# ---------------------------------------------------------------------------


def _round_cache_to_float16(index_path: Path, spectrogram_dir: Path) -> None:
    for npy_path in pd.read_parquet(index_path)["npy_path"]:
        spec = np.load(spectrogram_dir / npy_path)
        np.save(spectrogram_dir / npy_path, spec.astype(np.float16).astype(np.float32))


def _pack(tmp_path: Path, index_path: Path, spectrogram_dir: Path) -> tuple[Path, Path]:
    pack_spectrograms = importlib.import_module("pack_mission2_spectrograms").pack_spectrograms

    packed_path = tmp_path / "packed.npy"
    packed_index_path = tmp_path / "packed_index.parquet"
    pack_spectrograms(pd.read_parquet(index_path), spectrogram_dir, packed_path).to_parquet(packed_index_path, index=False)
    return packed_path, packed_index_path


def test_the_packed_cache_holds_every_spectrogram_at_its_offset(tmp_path: Path) -> None:
    import torch

    from dcc_data.mission2_cnn import PackedSpectrogramDataset, SpectrogramDataset

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    shuffled = pd.read_parquet(index_path).sample(frac=1.0, random_state=0)  # as a parallel build may write it
    shuffled.to_parquet(index_path, index=False)
    packed_path, packed_index_path = _pack(tmp_path, index_path, spectrogram_dir)

    packed_index = pd.read_parquet(packed_index_path)
    expected = shuffled.sort_values(["call_id", "utt_idx"]).reset_index(drop=True)
    assert packed_index.drop(columns="frame_offset").equals(expected)
    from_files = SpectrogramDataset(packed_index, spectrogram_dir)
    from_pack = PackedSpectrogramDataset(packed_index, packed_path)
    assert len(from_pack) == len(from_files)
    for i in range(len(from_files)):
        spec, speaker = from_files[i]
        packed_spec, packed_speaker = from_pack[i]
        assert packed_speaker == speaker
        assert packed_spec.dtype == torch.float32
        assert torch.equal(packed_spec, spec.half().float())


def test_packing_refuses_a_file_whose_shape_does_not_match_its_index_row(tmp_path: Path) -> None:
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    first = pd.read_parquet(index_path)["npy_path"].iloc[0]
    np.save(spectrogram_dir / first, np.zeros((N_MELS, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match its index row"):
        _pack(tmp_path, index_path, spectrogram_dir)

    assert not list(tmp_path.glob("packed*.npy"))


def test_training_on_the_packed_cache_matches_training_on_the_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    _round_cache_to_float16(index_path, spectrogram_dir)  # so float16 packing loses nothing here
    packed_path, packed_index_path = _pack(tmp_path, index_path, spectrogram_dir)

    from_files = tmp_path / "files" / "m2_cnn.pt"
    _train_epochs(monkeypatch, from_files, index_path, spectrogram_dir, 2, resume=False)
    from_pack = tmp_path / "pack" / "m2_cnn.pt"
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(packed_index_path), "--packed-spectrograms", str(packed_path),
            "--model", "small", "--epochs", "2", "--batch-size", "4", "--model-output", str(from_pack),
            "--torch-seed", "7",
        ],
    )
    assert train_mission2_cnn.main() == 0

    files_checkpoint = torch.load(from_files, map_location="cpu")
    pack_checkpoint = torch.load(from_pack, map_location="cpu")
    assert pack_checkpoint["metadata"] == files_checkpoint["metadata"]
    assert pack_checkpoint["training_options"]["packed_spectrograms"] is True
    for name, tensor in files_checkpoint["last_model_state_dict"].items():
        assert torch.equal(tensor, pack_checkpoint["last_model_state_dict"][name]), name


def test_pad_to_multiple_rounds_the_batch_length_up_with_silence() -> None:
    import torch

    from dcc_data.mission2_cnn import collate_batch
    from dcc_data.mission2_spectrogram import SPEC_FLOOR_DB

    batch = [(torch.zeros(N_MELS, 9), 0), (torch.zeros(N_MELS, 4), 1)]

    exact, _ = collate_batch(batch)
    rounded, labels = collate_batch(batch, pad_to_multiple=8)

    assert exact.shape == (2, 1, N_MELS, 9)
    assert rounded.shape == (2, 1, N_MELS, 16)
    assert torch.equal(rounded[..., :9], exact)
    assert bool((rounded[..., 9:] == SPEC_FLOOR_DB).all())
    assert labels.tolist() == [0, 1]
    assert collate_batch([(torch.zeros(N_MELS, 16), 0)], pad_to_multiple=8)[0].shape[-1] == 16


def test_amp_needs_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    monkeypatch.setattr(
        sys, "argv",
        ["train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(spectrogram_dir), "--amp"],
    )
    with pytest.raises(SystemExit):
        train_mission2_cnn.main()


def _flat_val_acc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every epoch scores val_acc 0.0: a fresh run's first epoch is its best, and nothing beats an already trained one."""

    import dcc_data.mission2_cnn as mission2_cnn

    monkeypatch.setattr(
        mission2_cnn, "evaluate", lambda *args, **kwargs: (0.0, np.array([0, 1]), np.array([1, 0]))
    )


def test_early_stopping_ends_the_run_once_val_acc_stops_improving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    model_output = tmp_path / "out" / "m2_cnn.pt"
    _flat_val_acc(monkeypatch)
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(spectrogram_dir),
            "--model", "small", "--epochs", "10", "--batch-size", "4", "--model-output", str(model_output),
            "--early-stop-patience", "2",
        ],
    )

    assert train_mission2_cnn.main() == 0

    checkpoint = torch.load(model_output, map_location="cpu")
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["total_epochs_trained"] == 3  # epochs 2 and 3 did not improve on epoch 1


def test_early_stopping_counts_epochs_from_earlier_resumed_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    index_path, spectrogram_dir = _write_cache(tmp_path, provenance=TRAINING_PROVENANCE)
    model_output = tmp_path / "out" / "m2_cnn.pt"
    _train_epochs(monkeypatch, model_output, index_path, spectrogram_dir, 1, resume=False)
    _flat_val_acc(monkeypatch)  # from here on nothing beats epoch 1
    resume_args = ("--torch-seed", "7", "--early-stop-patience", "2")

    assert _resume_train(monkeypatch, model_output, index_path, spectrogram_dir, *resume_args) == 0
    assert torch.load(model_output, map_location="cpu")["total_epochs_trained"] == 2
    assert _resume_train(monkeypatch, model_output, index_path, spectrogram_dir, *resume_args) == 0
    assert torch.load(model_output, map_location="cpu")["total_epochs_trained"] == 3

    bytes_before = model_output.read_bytes()
    assert _resume_train(monkeypatch, model_output, index_path, spectrogram_dir, *resume_args) == 0
    assert model_output.read_bytes() == bytes_before  # already stopped: nothing trained, nothing rewritten
