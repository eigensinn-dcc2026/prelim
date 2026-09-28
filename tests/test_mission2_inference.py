"""Tests for scripts/mission2_inference.py's L1/L2 model-selection logic.

The contest-mandated CLI (--audio_dir/--label_dir/--ckpt_path/--output) never
gained a new flag for choosing the model -- ``--ckpt_path``'s file extension
picks it instead (see mission2_inference.load_predictor). These tests build
real, tiny checkpoints of each format (a fitted sklearn-style classifier for
L1, a randomly-initialized SpeakerCNNLarge for L2 -- prediction *correctness*
isn't the point here, only that the inference script's plumbing produces a
valid, complete output row per utterance for either checkpoint format) and
run scripts/mission2_inference.py's actual main() against synthetic
audio/label folders.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

mission2_inference = importlib.import_module("mission2_inference")

from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import load_model, save_model  # noqa: E402

SAMPLE_RATE = 8000


def _write_wav(path: Path, duration_s: float, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(n) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _write_call(audio_dir: Path, label_dir: Path, stem: str, utterances_ms, *, seed: int) -> None:
    call_duration_s = max(end for _, end in utterances_ms) / 1000.0 + 0.5
    _write_wav(audio_dir / f"{stem}.wav", call_duration_s, seed=seed)
    document = {
        "_id": stem,
        "utterances": [{"startAt": start, "endAt": end} for start, end in utterances_ms],
    }
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{stem}.json").write_text(json.dumps(document), encoding="utf-8")


def _read_rows(output_path: Path) -> list[dict]:
    with output_path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _run_inference(audio_dir: Path, label_dir: Path, ckpt_path: Path, output_path: Path) -> None:
    import contextlib

    old_argv = sys.argv
    sys.argv = [
        "mission2_inference.py",
        "--audio_dir", str(audio_dir),
        "--label_dir", str(label_dir),
        "--ckpt_path", str(ckpt_path),
        "--output", str(output_path),
    ]
    try:
        with contextlib.redirect_stderr(sys.stderr):
            assert mission2_inference.main() == 0
    finally:
        sys.argv = old_argv


@pytest.fixture()
def l1_checkpoint(tmp_path: Path) -> Path:
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=40)
    model = HistGradientBoostingClassifier(random_state=0).fit(X, y)

    ckpt_path = tmp_path / "l1_model.pkl"
    save_model(ckpt_path, model, {"feature_columns": FEATURE_COLUMNS, "majority_class": int(round(y.mean()))})
    return ckpt_path


@pytest.fixture()
def l2_checkpoint(tmp_path: Path) -> Path:
    import torch

    from dcc_data.mission2_cnn import SpeakerCNNLarge

    model = SpeakerCNNLarge()
    ckpt_path = tmp_path / "l2_model.pt"
    torch.save({"best_model_state_dict": model.state_dict(), "best_val_acc": 0.0, "best_epoch": 0}, ckpt_path)
    return ckpt_path


def test_pad_spectrogram_pads_short_utterances_with_the_floor_value_only() -> None:
    from dcc_data.mission2_spectrogram import N_MELS, SPEC_FLOOR_DB

    short = np.full((N_MELS, 40), -12.5, dtype=np.float32)
    padded = mission2_inference.pad_spectrogram_to_min_frames(short, 100)

    assert padded.shape == (N_MELS, 100)
    assert padded.dtype == np.float32
    assert np.array_equal(padded[:, :40], short)  # the utterance's own frames are untouched
    assert np.all(padded[:, 40:] == SPEC_FLOOR_DB)  # same value collate_batch pads with


def test_pad_spectrogram_leaves_long_utterances_unchanged() -> None:
    from dcc_data.mission2_spectrogram import N_MELS

    long_spec = np.zeros((N_MELS, 250), dtype=np.float32)
    assert mission2_inference.pad_spectrogram_to_min_frames(long_spec, 100) is long_spec


def test_pad_spectrogram_ignores_everything_but_the_utterance_itself() -> None:
    # Isolation: the padded result is a pure function of one spectrogram -- padding one utterance
    # must not depend on any other utterance having been processed before it.
    from dcc_data.mission2_spectrogram import N_MELS

    a = np.random.default_rng(0).standard_normal((N_MELS, 30)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((N_MELS, 90)).astype(np.float32)
    alone = mission2_inference.pad_spectrogram_to_min_frames(a)
    mission2_inference.pad_spectrogram_to_min_frames(b)
    assert np.array_equal(alone, mission2_inference.pad_spectrogram_to_min_frames(a))


def test_load_predictor_dispatches_on_ckpt_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(mission2_inference, "_load_l1_predictor", lambda p: calls.append("l1") or (None, 0))
    monkeypatch.setattr(mission2_inference, "_load_l2_predictor", lambda p: calls.append("l2") or (None, 0))

    for suffix in (".pt", ".pth", ".ckpt", ".PT"):
        calls.clear()
        mission2_inference.load_predictor(Path(f"model{suffix}"))
        assert calls == ["l2"], f"suffix {suffix} should route to the CNN loader"

    for suffix in (".pkl", ".pickle", ""):
        calls.clear()
        mission2_inference.load_predictor(Path(f"model{suffix}"))
        assert calls == ["l1"], f"suffix {suffix!r} should route to the L1 loader"


def test_main_l1_checkpoint_produces_one_row_per_utterance(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800), (1900, 3400)], seed=0)
    _write_call(audio_dir, label_dir, "call2", [(0, 500)], seed=1)

    output_path = tmp_path / "out" / "mission2.csv"
    _run_inference(audio_dir, label_dir, l1_checkpoint, output_path)

    rows = _read_rows(output_path)
    assert len(rows) == 4
    assert {row["audio_file_name"] for row in rows} == {"call1.wav", "call2.wav"}
    for row in rows:
        assert row["speaker"] in ("0", "1")


def test_main_l2_checkpoint_produces_one_row_per_utterance(tmp_path: Path, l2_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800), (1900, 3400)], seed=0)
    _write_call(audio_dir, label_dir, "call2", [(0, 500)], seed=1)

    output_path = tmp_path / "out" / "mission2.csv"
    _run_inference(audio_dir, label_dir, l2_checkpoint, output_path)

    rows = _read_rows(output_path)
    assert len(rows) == 4
    assert {row["audio_file_name"] for row in rows} == {"call1.wav", "call2.wav"}
    for row in rows:
        assert row["speaker"] in ("0", "1")


def test_main_falls_back_to_majority_class_when_audio_is_unreadable(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    audio_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    # A .wav that isn't a real WAV file -- read_call_waveform must raise, and
    # every one of this call's utterances must still get an output row.
    (audio_dir / "broken.wav").write_bytes(b"not a real wav file")
    (label_dir / "broken.json").write_text(
        json.dumps({"_id": "broken", "utterances": [{"startAt": 0, "endAt": 500}, {"startAt": 500, "endAt": 900}]}),
        encoding="utf-8",
    )

    output_path = tmp_path / "out" / "mission2.csv"
    _run_inference(audio_dir, label_dir, l1_checkpoint, output_path)

    _, meta = load_model(l1_checkpoint)
    expected_fallback = str(meta["majority_class"])

    rows = _read_rows(output_path)
    assert len(rows) == 2
    for row in rows:
        assert row["speaker"] == expected_fallback
