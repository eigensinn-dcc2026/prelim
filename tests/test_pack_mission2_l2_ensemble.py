"""Tests for scripts/pack_mission2_l2_ensemble.py: bundling several L2 checkpoints into one.

scripts/mission2_inference.py takes exactly one --ckpt_path (the contest's CLI contract), so an ensemble
has to be packed into a single file -- these tests build real tiny single-model checkpoints with
dcc_data.mission2_cnn.SpeakerCNNLarge and run the packer's actual main() against them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

pack_ensemble = importlib.import_module("pack_mission2_l2_ensemble")

from dcc_data.mission2_checkpoint import (  # noqa: E402
    ENSEMBLE_STATE_DICTS_KEY,
    build_checkpoint_metadata,
)
from dcc_data.mission2_cnn import SpeakerCNNLarge  # noqa: E402
from dcc_data.mission2_features import HOP_LENGTH, N_FFT  # noqa: E402
from dcc_data.mission2_spectrogram import LABEL_SOURCE_PROVIDED, N_MELS  # noqa: E402

SAMPLE_RATE = 8000


def _metadata(torch_seed: int, **overrides) -> dict:
    metadata = build_checkpoint_metadata(
        model_name="large", model_class="SpeakerCNNLarge",
        index_provenance={
            "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT,
            "sample_rate": SAMPLE_RATE, "label_source": LABEL_SOURCE_PROVIDED,
        },
        split_seed=42, test_size=0.2, torch_seed=torch_seed,
    )
    metadata.update(overrides)
    return metadata


def _save_member(path: Path, torch_seed: int, *, best_val_acc: float = 0.9, **metadata_overrides) -> Path:
    torch.save(
        {
            "best_model_state_dict": SpeakerCNNLarge().state_dict(),
            "best_val_acc": best_val_acc,
            "best_epoch": 10,
            "metadata": _metadata(torch_seed, **metadata_overrides),
        },
        path,
    )
    return path


def test_main_packs_members_into_one_file_with_their_seeds_recorded(tmp_path: Path) -> None:
    member_paths = [
        _save_member(tmp_path / f"ts{seed}.pt", seed, best_val_acc=0.9 + seed * 0.01)
        for seed in (1, 2, 3)
    ]
    output_path = tmp_path / "ensemble.pt"

    argv = ["pack_mission2_l2_ensemble.py"]
    for p in member_paths:
        argv += ["--ckpt-path", str(p)]
    argv += ["--output", str(output_path)]
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = argv
    try:
        exit_code = pack_ensemble.main()
    finally:
        _sys.argv = old_argv

    assert exit_code == 0
    packed = torch.load(output_path, map_location="cpu")
    assert len(packed[ENSEMBLE_STATE_DICTS_KEY]) == 3
    assert packed["metadata"]["ensemble_member_torch_seeds"] == [1, 2, 3]
    assert packed["metadata"]["ensemble_size"] == 3
    assert packed["metadata"]["torch_seed"] is None
    assert packed["ensemble_member_paths"] == [str(p) for p in member_paths]
    assert packed["ensemble_member_best_val_acc"] == [0.91, 0.92, 0.93]

    # every packed state_dict loads cleanly into the model it came from
    for state_dict in packed[ENSEMBLE_STATE_DICTS_KEY]:
        SpeakerCNNLarge().load_state_dict(state_dict)


def test_main_requires_at_least_two_members(tmp_path: Path) -> None:
    member_path = _save_member(tmp_path / "ts1.pt", 1)
    argv = [
        "pack_mission2_l2_ensemble.py", "--ckpt-path", str(member_path), "--output", str(tmp_path / "out.pt"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            pack_ensemble.parse_args()
    finally:
        sys.argv = old_argv


def test_main_refuses_members_trained_with_different_mel_settings(tmp_path: Path) -> None:
    good = _save_member(tmp_path / "ts1.pt", 1)
    mismatched = _save_member(tmp_path / "ts2.pt", 2, n_mels=80)
    argv = [
        "pack_mission2_l2_ensemble.py",
        "--ckpt-path", str(good), "--ckpt-path", str(mismatched), "--output", str(tmp_path / "out.pt"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(Exception, match="n_mels"):
            pack_ensemble.main()
    finally:
        sys.argv = old_argv
    assert not (tmp_path / "out.pt").exists()


def _pack(monkeypatch: pytest.MonkeyPatch, members: list[Path], output: Path) -> int:
    argv = ["pack_mission2_l2_ensemble.py"]
    for member in members:
        argv += ["--ckpt-path", str(member)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])
    return pack_ensemble.main()


def test_main_refuses_the_same_file_packed_twice_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    member = _save_member(tmp_path / "ts1.pt", 1)

    with pytest.raises(CheckpointConfigError, match="different torch_seed"):
        _pack(monkeypatch, [member, member], tmp_path / "out.pt")

    assert not (tmp_path / "out.pt").exists()


def test_main_refuses_a_copy_of_a_member_recorded_under_another_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _save_member(tmp_path / "ts1.pt", 1)
    copy = tmp_path / "ts2.pt"
    checkpoint = torch.load(first, map_location="cpu")
    checkpoint["metadata"]["torch_seed"] = 2  # same weights, relabelled as another seed's run
    torch.save(checkpoint, copy)

    with pytest.raises(SystemExit, match="identical weights"):
        _pack(monkeypatch, [first, copy], tmp_path / "out.pt")

    assert not (tmp_path / "out.pt").exists()
