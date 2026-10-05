"""Tests for scripts/check_mission2_selection_padding.py.

The script compares a training checkpoint's best-epoch and last-epoch weights under the trainer's batch padding and
the submission script's per-utterance padding. These tests run it end to end on a tiny synthetic Training set
(random-weight models -- what matters here is that the inputs are rebuilt faithfully, the two paddings are the real
ones, and the script refuses anything it cannot vouch for), plus the two equivalences its numbers rest on.
"""

from __future__ import annotations

import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

check = importlib.import_module("check_mission2_selection_padding")

from dcc_data.guards import SplitContaminationError  # noqa: E402
from dcc_data.manifest import build_manifest  # noqa: E402
from dcc_data.mission2_checkpoint import build_checkpoint_metadata  # noqa: E402
from dcc_data.mission2_cnn import SpeakerCNNLarge, SpectrogramDataset, collate_batch, evaluate  # noqa: E402
from dcc_data.mission2_spectrogram import (  # noqa: E402
    N_MELS,
    assert_training_spectrogram_index,
    build_seoul_training_spectrograms,
)

SAMPLE_RATE = 8000
N_CALLS = 12
UTTERANCES = [(0, 900, 0), (1000, 2100, 1), (2200, 3300, 0)]  # startAt ms, endAt ms, speaker -- different lengths


def _write_call(root: Path, call_id: str, seed: int) -> None:
    stem = f"{call_id}_20230101"
    wav_path = root / "Training" / "1.원천데이터" / "TS_서울_구급" / f"{stem}.wav"
    json_path = root / "Training" / "2.라벨링데이터" / "TL_서울_구급" / f"{stem}.json"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    noise = (np.random.default_rng(seed).uniform(-0.5, 0.5, size=SAMPLE_RATE * 4) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(noise.tobytes())
    json_path.write_text(
        json.dumps({
            "_id": call_id,
            "utterances": [
                {"id": f"u{i}", "startAt": start, "endAt": end, "speaker": speaker, "text": "not read"}
                for i, (start, end, speaker) in enumerate(UTTERANCES)
            ],
        }),
        encoding="utf-8",
    )


@pytest.fixture()
def tiny_training_set(tmp_path: Path):
    root = tmp_path / "data"
    for i in range(N_CALLS):
        _write_call(root, f"tcall{i:02d}", seed=i)
    rows = build_manifest(root)
    index_rows, failures = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)
    assert failures == []
    index_df = pd.DataFrame(index_rows)
    index_path = tmp_path / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    return root, index_df, index_path


def _save_training_checkpoint(path: Path, index_df: pd.DataFrame, *, torch_seed: int, with_last: bool = True) -> Path:
    torch.manual_seed(torch_seed)
    best = SpeakerCNNLarge().state_dict()
    torch.manual_seed(torch_seed + 100)
    last = SpeakerCNNLarge().state_dict()
    checkpoint = {
        "best_model_state_dict": best,
        "best_val_acc": 0.9,
        "best_epoch": 4,
        "total_epochs_trained": 7,
        "metadata": build_checkpoint_metadata(
            model_name="large", model_class="SpeakerCNNLarge", index_provenance=assert_training_spectrogram_index(index_df),
            split_seed=42, test_size=0.2, torch_seed=torch_seed,
        ),
    }
    if with_last:
        checkpoint["last_model_state_dict"] = last
    torch.save(checkpoint, path)
    return path


def _run(monkeypatch: pytest.MonkeyPatch, index_path: Path, root: Path, ckpts: list[Path], output: Path, *extra: str) -> int:
    argv = ["check_mission2_selection_padding.py", "--index", str(index_path), "--data-root", str(root),
            "--output", str(output), "--n-calls", "0", *extra]
    for ckpt in ckpts:
        argv += ["--ckpt-path", str(ckpt)]
    monkeypatch.setattr(sys, "argv", argv)
    return check.main()


def test_it_measures_best_and_last_weights_under_both_paddings_on_rebuilt_training_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_training_set
) -> None:
    root, index_df, index_path = tiny_training_set
    ckpts = [_save_training_checkpoint(tmp_path / f"ts{seed}.pt", index_df, torch_seed=seed) for seed in (1, 2)]
    output = tmp_path / "out" / "check.json"

    assert _run(monkeypatch, index_path, root, ckpts, output) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    n = summary["n_utterances_measured"]
    assert 0 < n <= N_CALLS * len(UTTERANCES) and summary["n_calls_measured"] * len(UTTERANCES) == n
    assert summary["rebuilt_inputs_match_index"] is True
    assert "Official Validation not used" in summary["data_used"] and "no checkpoint was changed" in summary["reports_only"]
    assert [m["checkpoint"] for m in summary["checkpoints"]] == ["ts1.pt", "ts2.pt"]
    for member in summary["checkpoints"]:
        assert (member["best_epoch"], member["total_epochs_trained"]) == (4, 7)
        assert member["label_source"] == "provided_json"
        for state in check.STATES:
            for mode in check.MODES:
                entry = member["states"][state][mode]
                assert entry["n"] == n and 0 <= entry["correct"] <= n and entry["accuracy"] == entry["correct"] / n
        for mode in check.MODES:
            paired = member["paired_best_vs_last"][mode]
            correct_gap = member["states"]["best"][mode]["correct"] - member["states"]["last"][mode]["correct"]
            assert paired["best_right_last_wrong"] - paired["best_wrong_last_right"] == correct_gap
    assert summary["batched_vs_one_at_a_time"]["n_prediction_differences"] == 0


def test_the_batch_padding_numbers_are_what_the_trainers_own_evaluate_gives(tmp_path: Path, tiny_training_set) -> None:
    _, index_df, _ = tiny_training_set
    spec_dir = tmp_path / "specs"
    dataset = SpectrogramDataset(index_df, spec_dir)
    specs = [dataset[i][0].numpy() for i in range(len(dataset))]
    torch.manual_seed(5)
    model = SpeakerCNNLarge().eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

    _, trainer_predictions, _ = evaluate(model, loader, device="cpu")

    assert check.predict_batch_padding(model, specs, batch_size=4).tolist() == trainer_predictions.tolist()


def test_running_equal_length_utterances_together_gives_the_submission_scripts_one_at_a_time_answers() -> None:
    rng = np.random.default_rng(0)
    # lengths below, at and above L2_MIN_FRAMES (300): the short ones all pad to 300 and share a batch
    specs = [rng.uniform(-80, 0, size=(N_MELS, frames)).astype(np.float32) for frames in (1, 5, 40, 299, 300, 301, 450, 40, 5)]
    torch.manual_seed(7)
    model = SpeakerCNNLarge().eval()

    batched = check.predict_inference_padding(model, specs, chunk_size=4)
    one_at_a_time = check.predict_one_at_a_time(model, specs, list(range(len(specs))))

    assert batched.tolist() == one_at_a_time.tolist()


def test_the_inference_padding_is_the_submission_scripts_own_function(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    original = check.pad_spectrogram_to_min_frames
    monkeypatch.setattr(check, "pad_spectrogram_to_min_frames", lambda s: seen.append(s.shape[1]) or original(s))
    specs = [np.zeros((N_MELS, 10), dtype=np.float32), np.zeros((N_MELS, 400), dtype=np.float32)]

    check.predict_inference_padding(SpeakerCNNLarge().eval(), specs)

    assert seen == [10, 400]


def test_it_stops_when_a_rebuilt_spectrogram_does_not_match_the_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_training_set
) -> None:
    root, index_df, index_path = tiny_training_set
    index_df.loc[index_df.index[0], "n_frames"] += 3
    index_df.loc[index_df.index[1], "speaker"] = 1 - index_df.loc[index_df.index[1], "speaker"]
    index_df.to_parquet(index_path, index=False)
    ckpt = _save_training_checkpoint(tmp_path / "ts1.pt", index_df, torch_seed=1)
    output = tmp_path / "check.json"

    with pytest.raises(ValueError, match="do not match the index"):
        _run(monkeypatch, index_path, root, [ckpt], output)

    assert not output.exists()


def test_it_refuses_an_index_built_from_the_official_validation_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_training_set
) -> None:
    root, index_df, index_path = tiny_training_set
    ckpt = _save_training_checkpoint(tmp_path / "ts1.pt", index_df, torch_seed=1)
    index_df.assign(split="Validation").to_parquet(index_path, index=False)

    with pytest.raises(SplitContaminationError, match="non-Training"):
        _run(monkeypatch, index_path, root, [ckpt], tmp_path / "check.json")


def test_it_refuses_a_checkpoint_that_keeps_no_last_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_training_set
) -> None:
    root, index_df, index_path = tiny_training_set
    ckpt = _save_training_checkpoint(tmp_path / "packed_like.pt", index_df, torch_seed=1, with_last=False)

    with pytest.raises(ValueError, match="last_model_state_dict"):
        _run(monkeypatch, index_path, root, [ckpt], tmp_path / "check.json")


def test_it_refuses_checkpoints_trained_with_another_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_training_set
) -> None:
    root, index_df, index_path = tiny_training_set
    first = _save_training_checkpoint(tmp_path / "ts1.pt", index_df, torch_seed=1)
    other = torch.load(first, map_location="cpu")
    other["metadata"]["split_seed"] = 7
    torch.save(other, tmp_path / "ts2.pt")

    with pytest.raises(ValueError, match="another train/held-out split"):
        _run(monkeypatch, index_path, root, [first, tmp_path / "ts2.pt"], tmp_path / "check.json")


def test_a_sampled_run_keeps_the_held_out_rows_in_their_original_order(tiny_training_set) -> None:
    _, index_df, _ = tiny_training_set

    sample, total = check.sample_held_out_rows(index_df, split_seed=42, test_size=0.2, n_calls=2, sample_seed=0)

    assert sample["call_id"].nunique() == 2 and total >= 2
    original_positions = [index_df.index[(index_df["call_id"] == c) & (index_df["utt_idx"] == u)][0]
                          for c, u in zip(sample["call_id"], sample["utt_idx"])]
    assert original_positions == sorted(original_positions)
