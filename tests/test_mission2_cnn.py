"""Tests for the L2 CNN: dataset/collation (variable-length spectrograms)
and the model's forward pass. Real training runs on Colab GPU (see
scripts/train_mission2_cnn.py); these are CPU-only correctness checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dcc_data.mission2_cnn import (
    SpeakerCNN,
    SpeakerCNNLarge,
    SpeakerCNNLargeMasked,
    SpeakerCNNLargeWithDuration,
    SpeakerResNet,
    SpectrogramDataset,
    SpectrogramWithDurationDataset,
    collate_batch,
    collate_batch_with_duration,
    collate_batch_with_lengths,
    evaluate,
    move_optimizer_state_to_device,
    predict_with_confidence,
    train_one_epoch,
    train_with_best_checkpoint,
)
from dcc_data.mission2_spectrogram import N_MELS, SPEC_FLOOR_DB


def _make_index_and_specs(tmp_path, n_mels=N_MELS):
    frame_counts = [3, 10, 25]
    rows = []
    for i, n_frames in enumerate(frame_counts):
        spec = np.random.default_rng(i).uniform(-40, 0, size=(n_mels, n_frames)).astype(np.float32)
        npy_path = f"call{i}/0.npy"
        full_path = tmp_path / npy_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, spec)
        rows.append({"call_id": f"call{i}", "utt_idx": 0, "speaker": i % 2, "n_frames": n_frames, "npy_path": npy_path})
    return pd.DataFrame(rows), frame_counts


def _make_index_and_specs_with_duration(tmp_path, n_mels=N_MELS):
    index_df, frame_counts = _make_index_and_specs(tmp_path, n_mels=n_mels)
    # distinct, deliberately *not* proportional to n_frames -- duration_sec is
    # this utterance's own startAt/endAt span, not something derivable from
    # the spectrogram's own frame count, so the fixture shouldn't conflate them.
    index_df["duration_sec"] = [0.31, 4.97, 1.2]
    return index_df, frame_counts


def test_spectrogram_dataset_returns_tensor_and_label(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)

    assert len(dataset) == 3
    spec, label = dataset[1]
    assert spec.shape == (N_MELS, frame_counts[1])
    assert label in (0, 1)


def test_collate_batch_pads_to_the_batch_max_with_the_floor_value(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]

    specs, labels = collate_batch(batch)

    assert specs.shape == (3, 1, N_MELS, max(frame_counts))
    assert labels.shape == (3,)
    # the shortest spectrogram's padded tail must read as silence (the floor),
    # not 0.0 -- 0 dB would be that utterance's *loudest* frame instead.
    shortest_idx = frame_counts.index(min(frame_counts))
    padded_tail = specs[shortest_idx, 0, :, frame_counts[shortest_idx] :]
    assert torch.allclose(padded_tail, torch.full_like(padded_tail, SPEC_FLOOR_DB))


def test_spectrogram_with_duration_dataset_returns_tensor_duration_and_label(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)

    assert len(dataset) == 3
    spec, duration, label = dataset[1]
    assert spec.shape == (N_MELS, frame_counts[1])
    assert duration == 4.97
    assert label in (0, 1)


def test_collate_batch_with_duration_pads_specs_and_stacks_durations(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]

    specs, durations, labels = collate_batch_with_duration(batch)

    assert specs.shape == (3, 1, N_MELS, max(frame_counts))
    assert durations.shape == (3,)
    assert torch.allclose(durations, torch.tensor([0.31, 4.97, 1.2]), atol=1e-6)
    assert labels.shape == (3,)
    shortest_idx = frame_counts.index(min(frame_counts))
    padded_tail = specs[shortest_idx, 0, :, frame_counts[shortest_idx] :]
    assert torch.allclose(padded_tail, torch.full_like(padded_tail, SPEC_FLOOR_DB))


def test_speaker_cnn_forward_pass_shape_for_variable_length_batch(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]
    specs, _ = collate_batch(batch)

    model = SpeakerCNN()
    logits = model(specs)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_handles_a_very_short_single_frame_spectrogram() -> None:
    model = SpeakerCNN()
    spec = torch.zeros(1, 1, N_MELS, 1)
    logits = model(spec)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_forward_pass_shape_for_variable_length_batch(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]
    specs, _ = collate_batch(batch)

    model = SpeakerCNNLarge()
    logits = model(specs)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_handles_a_very_short_single_frame_spectrogram() -> None:
    model = SpeakerCNNLarge()
    spec = torch.zeros(1, 1, N_MELS, 1)
    logits = model(spec)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_has_more_parameters_than_the_small_model() -> None:
    small_params = sum(p.numel() for p in SpeakerCNN().parameters())
    large_params = sum(p.numel() for p in SpeakerCNNLarge().parameters())
    assert large_params > small_params * 4


def test_speaker_cnn_large_with_duration_forward_pass_shape_for_variable_length_batch(tmp_path) -> None:
    index_df, _ = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]
    specs, durations, _ = collate_batch_with_duration(batch)

    model = SpeakerCNNLargeWithDuration()
    logits = model(specs, durations)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_with_duration_actually_uses_the_duration_input(tmp_path) -> None:
    index_df, _ = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]
    specs, _, _ = collate_batch_with_duration(batch)

    model = SpeakerCNNLargeWithDuration()
    model.eval()
    short_duration = torch.zeros(3)
    long_duration = torch.full((3,), 10.0)
    with torch.no_grad():
        logits_short = model(specs, short_duration)
        logits_long = model(specs, long_duration)

    # Same spectrograms, only the duration input differs -- if the model
    # weren't actually reading it, these would be identical.
    assert not torch.allclose(logits_short, logits_long)


def test_train_one_epoch_and_evaluate_run_without_crashing(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

    model = SpeakerCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")
    assert np.isfinite(loss)

    acc, preds, labels = evaluate(model, loader, device="cpu")
    assert 0.0 <= acc <= 1.0
    assert preds.shape == labels.shape == (3,)


def test_train_one_epoch_and_evaluate_work_with_a_duration_conditioned_model(tmp_path) -> None:
    # train_one_epoch()/evaluate() unpack loader batches as `*inputs, labels`
    # and call `model(*inputs)`, so the same two functions must also drive a
    # model whose loader yields (specs, duration, label) triples, with no
    # duration-specific branching needed in either function.
    index_df, _ = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch_with_duration)

    model = SpeakerCNNLargeWithDuration()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")
    assert np.isfinite(loss)

    acc, preds, labels = evaluate(model, loader, device="cpu")
    assert 0.0 <= acc <= 1.0
    assert preds.shape == labels.shape == (3,)


def test_collate_batch_with_lengths_returns_each_spectrograms_own_frame_count(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    batch = [dataset[i] for i in range(3)]

    specs, lengths, labels = collate_batch_with_lengths(batch)

    assert specs.shape == (3, 1, N_MELS, max(frame_counts))
    assert lengths.dtype == torch.long
    assert lengths.tolist() == frame_counts
    assert labels.shape == (3,)


def test_speaker_resnet_forward_pass_shape_for_variable_length_batch(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    specs, _ = collate_batch([dataset[i] for i in range(3)])

    model = SpeakerResNet()
    logits = model(specs)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_speaker_resnet_handles_a_very_short_single_frame_spectrogram() -> None:
    model = SpeakerResNet()
    model.eval()
    logits = model(torch.zeros(1, 1, N_MELS, 1))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_masked_forward_pass_shape_for_variable_length_batch(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    specs, lengths, _ = collate_batch_with_lengths([dataset[i] for i in range(3)])

    model = SpeakerCNNLargeMasked()
    logits = model(specs, lengths)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_masked_handles_a_very_short_single_frame_spectrogram() -> None:
    model = SpeakerCNNLargeMasked()
    model.eval()
    logits = model(torch.zeros(1, 1, N_MELS, 1), torch.tensor([1]))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_speaker_cnn_large_masked_ignores_how_much_padding_follows_the_utterance() -> None:
    # The whole point of masked pooling: the same utterance must get the same
    # logits whether it sits in a batch padded to 60 frames or to 100. Padding
    # amounts are chosen far past the 4-layer conv stack's receptive field
    # (~19 frames either side) so conv boundary effects can't reach the valid
    # positions -- any remaining difference would be the pooling itself.
    true_frames = 10
    spec = torch.from_numpy(np.random.default_rng(0).uniform(-40, 0, size=(N_MELS, true_frames)).astype(np.float32))

    def pad_to(total_frames: int) -> torch.Tensor:
        padded = torch.full((1, 1, N_MELS, total_frames), SPEC_FLOOR_DB)
        padded[0, 0, :, :true_frames] = spec
        return padded

    lengths = torch.tensor([true_frames])
    masked = SpeakerCNNLargeMasked()
    masked.eval()
    unmasked = SpeakerCNNLarge()
    unmasked.eval()
    with torch.no_grad():
        masked_short_pad = masked(pad_to(60), lengths)
        masked_long_pad = masked(pad_to(100), lengths)
        unmasked_short_pad = unmasked(pad_to(60))
        unmasked_long_pad = unmasked(pad_to(100))

    assert torch.allclose(masked_short_pad, masked_long_pad, atol=1e-5)
    # Guard: this fixture really is sensitive to padding amount -- the
    # unmasked model, which averages over the padded tail, does not pass it.
    assert not torch.allclose(unmasked_short_pad, unmasked_long_pad, atol=1e-5)


def test_speaker_cnn_large_masked_actually_uses_the_lengths_input(tmp_path) -> None:
    index_df, frame_counts = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    specs, lengths, _ = collate_batch_with_lengths([dataset[i] for i in range(3)])

    model = SpeakerCNNLargeMasked()
    model.eval()
    with torch.no_grad():
        logits_true_lengths = model(specs, lengths)
        logits_full_lengths = model(specs, torch.full((3,), max(frame_counts)))

    # Same tensor, only the lengths differ -- if they weren't read, identical.
    assert not torch.allclose(logits_true_lengths, logits_full_lengths)


def test_speaker_cnn_large_masked_loads_a_speaker_cnn_large_state_dict() -> None:
    # Same conv stack + classifier, only the pooling differs -- so an existing
    # SpeakerCNNLarge checkpoint can be re-scored with masked pooling.
    SpeakerCNNLargeMasked().load_state_dict(SpeakerCNNLarge().state_dict())


def test_train_one_epoch_and_evaluate_work_with_a_length_masked_model(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch_with_lengths)

    model = SpeakerCNNLargeMasked()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")
    assert np.isfinite(loss)

    acc, preds, labels = evaluate(model, loader, device="cpu")
    assert 0.0 <= acc <= 1.0
    assert preds.shape == labels.shape == (3,)


def test_train_one_epoch_and_evaluate_work_with_speaker_resnet(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

    model = SpeakerResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")
    assert np.isfinite(loss)

    acc, preds, labels = evaluate(model, loader, device="cpu")
    assert 0.0 <= acc <= 1.0
    assert preds.shape == labels.shape == (3,)


def test_move_optimizer_state_to_device_moves_every_tensor_and_keeps_values(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)

    model = SpeakerCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_one_epoch(model, loader, optimizer, device="cpu")  # populates Adam's exp_avg/exp_avg_sq state

    before = {
        id(p): {k: v.clone() for k, v in state.items() if torch.is_tensor(v)}
        for p, state in optimizer.state.items()
    }
    assert before  # sanity: Adam actually created state to move

    move_optimizer_state_to_device(optimizer, "cpu")

    for p, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
                assert torch.equal(value, before[id(p)][key])


def test_predict_with_confidence_matches_evaluate_and_is_bounded(tmp_path) -> None:
    index_df, _ = _make_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)

    model = SpeakerCNN()
    acc, eval_preds, eval_labels = evaluate(model, loader, device="cpu")

    preds, labels, confidence = predict_with_confidence(model, loader, device="cpu")

    # Same model, same (unshuffled) loader -- predictions must agree with evaluate().
    assert np.array_equal(preds, eval_preds)
    assert np.array_equal(labels, eval_labels)
    assert float((preds == labels).mean()) == acc
    # A softmax probability for a binary (argmax of 2) classification is
    # always >= 0.5 for the class it picked.
    assert ((confidence >= 0.5) & (confidence <= 1.0)).all()


def test_predict_with_confidence_matches_evaluate_for_a_duration_conditioned_model(tmp_path) -> None:
    # scripts/analyze_mission2_cnn_errors.py's duration-bucket breakdown --
    # the whole point of SpeakerCNNLargeWithDuration -- calls
    # predict_with_confidence() on the checkpoint under analysis, so it must
    # drive a duration-conditioned model exactly like evaluate() does.
    index_df, _ = _make_index_and_specs_with_duration(tmp_path)
    dataset = SpectrogramWithDurationDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch_with_duration)

    model = SpeakerCNNLargeWithDuration()
    acc, eval_preds, eval_labels = evaluate(model, loader, device="cpu")

    preds, labels, confidence = predict_with_confidence(model, loader, device="cpu")

    assert np.array_equal(preds, eval_preds)
    assert np.array_equal(labels, eval_labels)
    assert float((preds == labels).mean()) == acc
    assert ((confidence >= 0.5) & (confidence <= 1.0)).all()


def _make_larger_index_and_specs(tmp_path, n_items=12, n_mels=N_MELS):
    rows = []
    for i in range(n_items):
        n_frames = 3 + (i % 5) * 4
        spec = np.random.default_rng(i).uniform(-40, 0, size=(n_mels, n_frames)).astype(np.float32)
        npy_path = f"call{i}/0.npy"
        full_path = tmp_path / npy_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, spec)
        rows.append({"call_id": f"call{i}", "utt_idx": 0, "speaker": i % 2, "n_frames": n_frames, "npy_path": npy_path})
    return pd.DataFrame(rows)


def test_train_with_best_checkpoint_reports_the_best_epoch_not_the_last(tmp_path) -> None:
    index_df = _make_larger_index_and_specs(tmp_path)
    dataset = SpectrogramDataset(index_df, tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

    torch.manual_seed(0)
    model = SpeakerCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)  # deliberately unstable, like the real bug report

    result = train_with_best_checkpoint(model, loader, loader, optimizer, epochs=6, device="cpu")

    assert result["best_val_acc"] == max(h["val_acc"] for h in result["history"])
    # The stored predictions must actually *be* the best epoch's -- recomputing
    # accuracy from them must reproduce the reported best_val_acc, not just
    # happen to carry some other (e.g. the last, possibly worse) epoch's numbers.
    recomputed_acc = float((result["best_val_preds"] == result["best_val_labels"]).mean())
    assert recomputed_acc == result["best_val_acc"]
