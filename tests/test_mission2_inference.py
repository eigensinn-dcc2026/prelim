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


def _l2_metadata(**overrides) -> dict:
    """The config train_mission2_cnn.py would have embedded for a default ``--model large`` run."""

    from dcc_data.mission2_checkpoint import build_checkpoint_metadata
    from dcc_data.mission2_features import HOP_LENGTH, N_FFT
    from dcc_data.mission2_spectrogram import N_MELS

    metadata = build_checkpoint_metadata(
        model_name="large",
        model_class="SpeakerCNNLarge",
        index_provenance={
            "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT,
            "sample_rate": SAMPLE_RATE,
        },
        split_seed=42,
        test_size=0.2,
        torch_seed=42,
    )
    metadata.update(overrides)
    return metadata


def _save_l2_checkpoint(ckpt_path: Path, metadata: dict | None):
    import torch

    from dcc_data.mission2_cnn import SpeakerCNNLarge

    checkpoint = {"best_model_state_dict": SpeakerCNNLarge().state_dict(), "best_val_acc": 0.0, "best_epoch": 0}
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


@pytest.fixture()
def l2_checkpoint(tmp_path: Path) -> Path:
    return _save_l2_checkpoint(tmp_path / "l2_model.pt", _l2_metadata())


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


# ---------------------------------------------------------------------------
# Input completeness: a partial CSV must never look like a successful run
# ---------------------------------------------------------------------------


def _run_inference_expecting_failure(audio_dir: Path, label_dir: Path, ckpt_path: Path, output_path: Path):
    old_argv = sys.argv
    sys.argv = [
        "mission2_inference.py",
        "--audio_dir", str(audio_dir),
        "--label_dir", str(label_dir),
        "--ckpt_path", str(ckpt_path),
        "--output", str(output_path),
    ]
    try:
        with pytest.raises(ValueError) as excinfo:
            mission2_inference.main()
    finally:
        sys.argv = old_argv
    return excinfo.value


def _nothing_written(output_path: Path) -> bool:
    return not output_path.exists() and not list(output_path.parent.glob(".*.tmp"))


def test_pair_by_stem_strict_returns_what_pair_by_stem_returns_when_everything_lines_up(tmp_path: Path) -> None:
    from dcc_data.mission2_features import pair_by_stem

    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call_b", [(0, 500)], seed=0)
    _write_call(audio_dir, label_dir, "call_a", [(0, 500)], seed=1)

    assert mission2_inference.pair_by_stem_strict(audio_dir, label_dir) == pair_by_stem(audio_dir, label_dir)


def test_pair_by_stem_strict_accepts_nested_folders_when_the_stems_are_unique(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "part1" / "call1.wav", 1.0, seed=0)
    _write_wav(tmp_path / "audio" / "part2" / "call2.wav", 1.0, seed=1)
    for stem in ("call1", "call2"):
        (tmp_path / "labels" / "x").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "x" / f"{stem}.json").write_text(json.dumps({"utterances": []}), encoding="utf-8")

    pairs = mission2_inference.pair_by_stem_strict(tmp_path / "audio", tmp_path / "labels")

    assert [stem for stem, _, _ in pairs] == ["call1", "call2"]


def test_pair_by_stem_strict_names_a_wav_without_a_label_and_a_label_without_a_wav(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "both", [(0, 500)], seed=0)
    _write_wav(audio_dir / "wav_only.wav", 1.0, seed=1)
    (label_dir / "json_only.json").write_text(json.dumps({"utterances": []}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        mission2_inference.pair_by_stem_strict(audio_dir, label_dir)

    message = str(excinfo.value)
    assert "1 WAV stem(s) have no label JSON: wav_only" in message
    assert "1 label JSON stem(s) have no WAV: json_only" in message


def test_pair_by_stem_strict_rejects_a_repeated_stem_instead_of_letting_one_overwrite_the_other(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 500)], seed=0)
    _write_wav(audio_dir / "elsewhere" / "call1.wav", 1.0, seed=1)
    (label_dir / "elsewhere").mkdir(parents=True)
    (label_dir / "elsewhere" / "call1.json").write_text(json.dumps({"utterances": []}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        mission2_inference.pair_by_stem_strict(audio_dir, label_dir)

    message = str(excinfo.value)
    assert "1 WAV stem(s) appear more than once: call1" in message
    assert "1 label JSON stem(s) appear more than once: call1" in message


def test_pair_by_stem_strict_rejects_empty_inputs(tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "labels").mkdir()

    with pytest.raises(ValueError, match="no .wav files"):
        mission2_inference.pair_by_stem_strict(tmp_path / "audio", tmp_path / "labels")


def test_main_refuses_a_wav_without_its_label_and_writes_no_csv(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800)], seed=0)
    _write_wav(audio_dir / "call2.wav", 1.0, seed=1)  # its label JSON never arrived
    output_path = tmp_path / "out" / "mission2.csv"

    error = _run_inference_expecting_failure(audio_dir, label_dir, l1_checkpoint, output_path)

    assert "call2" in str(error)
    assert _nothing_written(output_path)


def test_main_refuses_a_label_without_its_wav_and_writes_no_csv(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800)], seed=0)
    (label_dir / "call2.json").write_text(json.dumps({"utterances": [{"startAt": 0, "endAt": 500}]}), encoding="utf-8")
    output_path = tmp_path / "out" / "mission2.csv"

    error = _run_inference_expecting_failure(audio_dir, label_dir, l1_checkpoint, output_path)

    assert "call2" in str(error)
    assert _nothing_written(output_path)


def test_main_refuses_a_repeated_stem_and_writes_no_csv(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90)], seed=0)
    _write_wav(audio_dir / "copy" / "call1.wav", 1.0, seed=1)
    output_path = tmp_path / "out" / "mission2.csv"

    error = _run_inference_expecting_failure(audio_dir, label_dir, l1_checkpoint, output_path)

    assert "more than once: call1" in str(error)
    assert _nothing_written(output_path)


def test_main_accepts_a_call_whose_label_has_no_utterances(tmp_path: Path, l1_checkpoint: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800)], seed=0)
    _write_wav(audio_dir / "empty.wav", 1.0, seed=1)
    (label_dir / "empty.json").write_text(json.dumps({"_id": "empty", "utterances": []}), encoding="utf-8")

    output_path = tmp_path / "out" / "mission2.csv"
    _run_inference(audio_dir, label_dir, l1_checkpoint, output_path)

    assert len(_read_rows(output_path)) == 2


def test_verify_output_complete_accepts_a_matching_run_and_refuses_any_shortfall() -> None:
    rows = [{"speaker": 0}] * 4

    mission2_inference.verify_output_complete(expected_calls=2, calls_seen=2, expected_rows=4, rows=rows)
    with pytest.raises(ValueError, match="incomplete"):
        mission2_inference.verify_output_complete(expected_calls=2, calls_seen=2, expected_rows=5, rows=rows)
    with pytest.raises(ValueError, match="incomplete"):
        mission2_inference.verify_output_complete(expected_calls=3, calls_seen=2, expected_rows=4, rows=rows)


def test_main_refuses_to_write_when_the_row_count_disagrees_with_the_labels(
    tmp_path: Path, l1_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800)], seed=0)
    # as if a code change made the loop drop a row: the independent count says 3, the loop produces 2
    monkeypatch.setattr(mission2_inference, "count_label_utterances", lambda paths: 3)
    output_path = tmp_path / "out" / "mission2.csv"

    error = _run_inference_expecting_failure(audio_dir, label_dir, l1_checkpoint, output_path)

    assert "output is incomplete" in str(error)
    assert _nothing_written(output_path)


# ---------------------------------------------------------------------------
# Checkpoint config: a wrong checkpoint must fail, not silently predict worse
# ---------------------------------------------------------------------------


def _one_call(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_call(audio_dir, label_dir, "call1", [(0, 90), (200, 1800)], seed=0)
    return audio_dir, label_dir, tmp_path / "out" / "mission2.csv"


def test_an_l2_checkpoint_without_any_recorded_config_is_refused(tmp_path: Path) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    audio_dir, label_dir, output_path = _one_call(tmp_path)
    ckpt_path = _save_l2_checkpoint(tmp_path / "unknown.pt", None)

    error = _run_inference_expecting_failure(audio_dir, label_dir, ckpt_path, output_path)

    assert isinstance(error, CheckpointConfigError)
    assert "records no training config" in str(error)
    assert _nothing_written(output_path)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"model_name": "large-masked"}, "model_name"),  # same modules as SpeakerCNNLarge: loads without any error
        ({"n_mels": 80}, "n_mels"),
        ({"hop_length": 80}, "hop_length"),
        ({"index_split": "Validation"}, "index_split"),
    ],
)
def test_an_l2_checkpoint_trained_differently_is_refused_even_though_its_weights_would_load(
    tmp_path: Path, override: dict, fragment: str
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    audio_dir, label_dir, output_path = _one_call(tmp_path)
    ckpt_path = _save_l2_checkpoint(tmp_path / "other.pt", _l2_metadata(**override))

    error = _run_inference_expecting_failure(audio_dir, label_dir, ckpt_path, output_path)

    assert isinstance(error, CheckpointConfigError)
    assert fragment in str(error)
    assert _nothing_written(output_path)


def test_a_registered_legacy_l2_checkpoint_runs_with_its_registered_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dcc_data import mission2_checkpoint

    audio_dir, label_dir, output_path = _one_call(tmp_path)
    ckpt_path = _save_l2_checkpoint(tmp_path / "legacy.pt", None)
    monkeypatch.setitem(
        mission2_checkpoint.LEGACY_L2_CHECKPOINTS,
        mission2_checkpoint.checkpoint_sha256(ckpt_path),
        {"description": "test legacy checkpoint", "metadata": _l2_metadata(torch_seed=None)},
    )

    _run_inference(audio_dir, label_dir, ckpt_path, output_path)

    assert len(_read_rows(output_path)) == 2


def test_audio_at_another_sample_rate_stops_the_run_instead_of_falling_back_to_the_majority_class(
    tmp_path: Path,
) -> None:
    from dcc_data.mission2_checkpoint import CheckpointConfigError

    audio_dir, label_dir, output_path = _one_call(tmp_path)  # 8 kHz WAVs
    ckpt_path = _save_l2_checkpoint(tmp_path / "other_rate.pt", _l2_metadata(sample_rate=16000))

    error = _run_inference_expecting_failure(audio_dir, label_dir, ckpt_path, output_path)

    assert isinstance(error, CheckpointConfigError)
    assert "8000 Hz" in str(error) and "16000 Hz" in str(error)
    assert _nothing_written(output_path)


# ---------------------------------------------------------------------------
# Decision rule: class 1 iff P(speaker=1) >= 0.5, so an exact tie is class 1
# ---------------------------------------------------------------------------


def test_speaker_from_logits_follows_p_speaker_1_at_least_one_half() -> None:
    import torch

    assert mission2_inference._speaker_from_logits(torch.tensor([[0.2, 1.7]])) == 1
    assert mission2_inference._speaker_from_logits(torch.tensor([[1.7, 0.2]])) == 0
    assert mission2_inference._speaker_from_logits(torch.tensor([[3.0, 3.0]])) == 1  # tie: P(1) is exactly 0.5
    assert mission2_inference._speaker_from_logits(torch.tensor([[3.0, 3.0]])).__class__ is int
    # argmax would have said 0 here -- the case this rule exists for
    assert int(torch.tensor([[3.0, 3.0]]).argmax(dim=1).item()) == 0


def test_a_model_whose_two_logits_always_tie_predicts_class_1_end_to_end(tmp_path: Path) -> None:
    import torch

    from dcc_data.mission2_cnn import SpeakerCNNLarge

    model = SpeakerCNNLarge()
    with torch.no_grad():
        model.classifier[-1].weight.zero_()  # final Linear: both logits become the same bias for any input
        model.classifier[-1].bias.zero_()
    ckpt_path = tmp_path / "tied.pt"
    torch.save(
        {"best_model_state_dict": model.state_dict(), "best_val_acc": 0.0, "best_epoch": 0, "metadata": _l2_metadata()},
        ckpt_path,
    )

    predict, _ = mission2_inference.load_predictor(ckpt_path)
    segment = np.random.default_rng(0).standard_normal(4000).astype(np.float32)

    assert predict(segment, SAMPLE_RATE, 0.5) == 1
