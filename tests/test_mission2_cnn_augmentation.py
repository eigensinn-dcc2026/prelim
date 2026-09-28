"""Tests for train-time SpecAugment, recent-epoch weight averaging, and the
train script's --specaug/--swa-last-k options (tiny CPU-only runs)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from dcc_data.mission2_cnn import (
    AugmentedSpectrogramDataset,
    RecentWeightAverager,
    SpeakerCNNLarge,
    SpectrogramDataset,
    collate_batch,
    recalibrate_batch_norm,
    spec_augment,
)
from dcc_data.mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


def _spec(n_frames: int, value: float = -10.0) -> torch.Tensor:
    return torch.full((N_MELS, n_frames), value, dtype=torch.float32)


def test_spec_augment_keeps_shape_dtype_and_does_not_mutate_the_input() -> None:
    spec = _spec(60)
    original = spec.clone()
    torch.manual_seed(0)
    out = spec_augment(spec)
    assert out.shape == spec.shape and out.dtype == spec.dtype
    assert torch.equal(spec, original)


def test_spec_augment_only_writes_the_floor_value_into_masked_cells() -> None:
    torch.manual_seed(1)
    out = spec_augment(_spec(80), freq_mask_width=10, time_mask_frac=0.2)
    changed = out != -10.0
    assert changed.any()
    assert torch.all(out[changed] == SPEC_FLOOR_DB)


def test_spec_augment_is_reproducible_under_the_global_torch_seed() -> None:
    spec = torch.from_numpy(np.random.default_rng(0).uniform(-60, 0, (N_MELS, 70)).astype(np.float32))
    torch.manual_seed(7)
    a = spec_augment(spec)
    torch.manual_seed(7)
    b = spec_augment(spec)
    assert torch.equal(a, b)


def test_time_masks_are_capped_by_the_utterances_own_length() -> None:
    # 10 frames * 0.15 -> at most 1 frame per mask, 2 masks -> at most 2 blanked frames
    for seed in range(30):
        torch.manual_seed(seed)
        out = spec_augment(_spec(10), freq_mask_width=0, time_mask_frac=0.15, n_time_masks=2)
        assert int((out[0] == SPEC_FLOOR_DB).sum()) <= 2
    # 3 frames * 0.15 -> 0: too short to mask at all
    torch.manual_seed(0)
    assert torch.equal(spec_augment(_spec(3), freq_mask_width=0), _spec(3))


def test_time_mask_never_exceeds_the_absolute_frame_cap_on_long_utterances() -> None:
    for seed in range(30):
        torch.manual_seed(seed)
        out = spec_augment(_spec(1000), freq_mask_width=0, time_mask_frac=0.5, n_time_masks=1, max_time_mask_frames=20)
        assert int((out[0] == SPEC_FLOOR_DB).sum()) <= 20


def test_frequency_masks_blank_whole_mel_bands_and_leave_time_untouched() -> None:
    for seed in range(30):
        torch.manual_seed(seed)
        out = spec_augment(_spec(50), freq_mask_width=8, n_freq_masks=2, time_mask_frac=0.0)
        blanked_rows = (out == SPEC_FLOOR_DB).all(dim=1)
        assert int(blanked_rows.sum()) <= 16
        # a masked band is blank across every time step, an unmasked one is untouched
        assert torch.all((out == SPEC_FLOOR_DB).all(dim=1) | (out == -10.0).all(dim=1))


def _index_with_specs(tmp_path: Path, n_calls: int = 12, per_call: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_calls):
        for u in range(per_call):
            n_frames = int(np.random.default_rng(c * 10 + u).integers(8, 60))
            spec = np.random.default_rng(c * 10 + u).uniform(-70, 0, (N_MELS, n_frames)).astype(np.float32)
            rel = f"call{c}/{u}.npy"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(tmp_path / rel, spec)
            rows.append(
                {
                    "call_id": f"call{c}",
                    "utt_idx": u,
                    "speaker": (c + u) % 2,
                    "n_frames": n_frames,
                    "duration_sec": n_frames * 0.02,
                    "npy_path": rel,
                }
            )
    return pd.DataFrame(rows)


def test_augmented_dataset_matches_spec_augment_and_plain_dataset_is_untouched(tmp_path: Path) -> None:
    index = _index_with_specs(tmp_path)
    plain = SpectrogramDataset(index, tmp_path)
    augmented = AugmentedSpectrogramDataset(index, tmp_path, freq_mask_width=6, time_mask_frac=0.2)
    raw, label = plain[5]
    torch.manual_seed(3)
    got, got_label = augmented[5]
    torch.manual_seed(3)
    assert torch.equal(got, spec_augment(raw, freq_mask_width=6, time_mask_frac=0.2))
    assert got_label == label
    assert torch.equal(plain[5][0], raw)  # plain dataset (validation) never augments


def test_recent_weight_averager_averages_only_the_last_k_epochs() -> None:
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2))
    averager = RecentWeightAverager(2)
    for step in (1.0, 2.0, 4.0):
        with torch.no_grad():
            model[0].weight.fill_(step)
            model[1].running_mean.fill_(step * 10)
            model[1].num_batches_tracked.fill_(int(step))
        averager.add(model)
    assert len(averager) == 2
    state = averager.averaged_state()
    assert torch.allclose(state["0.weight"], torch.full((2, 2), 3.0))  # mean(2, 4), epoch 1 dropped
    assert torch.allclose(state["1.running_mean"], torch.full((2,), 30.0))
    assert int(state["1.num_batches_tracked"]) == 4  # integer entries take the newest value


def test_recent_weight_averager_rejects_bad_k_and_empty_use() -> None:
    with pytest.raises(ValueError):
        RecentWeightAverager(0)
    with pytest.raises(ValueError):
        RecentWeightAverager(3).averaged_state()


def test_recalibrate_batch_norm_updates_running_stats_but_not_weights(tmp_path: Path) -> None:
    index = _index_with_specs(tmp_path)
    loader = torch.utils.data.DataLoader(SpectrogramDataset(index, tmp_path), batch_size=8, collate_fn=collate_batch)
    model = SpeakerCNNLarge()
    weights_before = {k: v.clone() for k, v in model.named_parameters()}
    stats_before = model.features[1].running_mean.clone()
    recalibrate_batch_norm(model, loader, "cpu")
    assert not torch.equal(model.features[1].running_mean, stats_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, weights_before[name])


@pytest.fixture()
def train_script():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("train_mission2_cnn")


def _run_train(train_script, tmp_path: Path, extra: list[str], monkeypatch) -> dict:
    index = _index_with_specs(tmp_path / "specs")
    index_path = tmp_path / "index.parquet"
    index.to_parquet(index_path)
    out = tmp_path / "m.pt"
    argv = [
        "train", "--index", str(index_path), "--spectrogram-dir", str(tmp_path / "specs"),
        "--model", "large", "--epochs", "3", "--batch-size", "8", "--torch-seed", "1",
        "--model-output", str(out), *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train_script.main() == 0
    return torch.load(out, map_location="cpu")


def test_train_script_with_specaug_and_swa_saves_swa_weights_that_load(train_script, tmp_path, monkeypatch) -> None:
    checkpoint = _run_train(train_script, tmp_path, ["--specaug", "--swa-last-k", "2"], monkeypatch)
    assert checkpoint["swa_num_epochs_averaged"] == 2
    assert 0.0 <= checkpoint["swa_val_acc"] <= 1.0
    assert "last_val_acc" in checkpoint
    SpeakerCNNLarge().load_state_dict(checkpoint["swa_model_state_dict"])
    SpeakerCNNLarge().load_state_dict(checkpoint["best_model_state_dict"])  # inference format unchanged


def test_train_script_without_new_flags_keeps_the_old_checkpoint_contents(train_script, tmp_path, monkeypatch) -> None:
    checkpoint = _run_train(train_script, tmp_path, [], monkeypatch)
    assert "swa_model_state_dict" not in checkpoint
    assert {
        "best_model_state_dict", "best_val_acc", "best_epoch", "last_model_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "total_epochs_trained",
    } <= set(checkpoint)


def test_train_script_rejects_specaug_for_models_with_a_different_loader(train_script, tmp_path, monkeypatch) -> None:
    index = _index_with_specs(tmp_path / "specs")
    index.to_parquet(tmp_path / "index.parquet")
    monkeypatch.setattr(
        sys, "argv",
        [
            "train", "--index", str(tmp_path / "index.parquet"), "--spectrogram-dir", str(tmp_path / "specs"),
            "--model", "large-duration", "--specaug", "--model-output", str(tmp_path / "m.pt"),
        ],
    )
    with pytest.raises(SystemExit):
        train_script.main()
