"""Tests for the L2 (Mel-spectrogram + CNN) input representation.

L1's 53-64 dim per-utterance summary statistics have plateaued (~85-87%)
without closing the duration-bias or overlap gaps -- see
reports/mission2/m2_error_analysis.md and the follow-up feature-engineering
sessions. This keeps the raw time-frequency structure instead of collapsing
it to a handful of stats, on the theory that short/overlapping utterances
lose exactly the information those stats discard.
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

from dcc_data.guards import SplitContaminationError
from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import HOP_LENGTH, N_FFT
from dcc_data.mission2_spectrogram import (
    INDEX_PROVENANCE_COLUMNS,
    LABEL_SOURCE_PROVIDED,
    N_MELS,
    SPEC_FLOOR_DB,
    LabelProvenanceError,
    assert_training_spectrogram_index,
    build_seoul_training_spectrograms,
    build_seoul_validation_spectrograms,
    mel_spectrogram,
)

SAMPLE_RATE = 8000


def _tone(freq_hz: float, duration_s: float, *, sr: int = 8000, amplitude: float = 0.5) -> np.ndarray:
    n = max(1, int(sr * duration_s))
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_mel_spectrogram_has_n_mels_rows() -> None:
    segment = _tone(150.0, 1.0)
    spec = mel_spectrogram(segment, 8000)
    assert spec.shape[0] == N_MELS
    assert spec.shape[1] > 1


def test_mel_spectrogram_is_finite_and_bounded_by_the_floor() -> None:
    segment = _tone(150.0, 1.0)
    spec = mel_spectrogram(segment, 8000)
    assert np.isfinite(spec).all()
    assert spec.max() <= 1e-6  # normalized relative to its own max -> ~0 dB ceiling
    assert spec.min() >= SPEC_FLOOR_DB - 1e-3


def test_mel_spectrogram_handles_segment_shorter_than_n_fft() -> None:
    rng = np.random.default_rng(0)
    short_segment = rng.standard_normal(50).astype(np.float32)
    spec = mel_spectrogram(short_segment, 8000)
    assert spec.shape[0] == N_MELS
    assert np.isfinite(spec).all()


def test_mel_spectrogram_frame_count_grows_with_duration() -> None:
    short_spec = mel_spectrogram(_tone(150.0, 0.5), 8000)
    long_spec = mel_spectrogram(_tone(150.0, 3.0), 8000)
    assert long_spec.shape[1] > short_spec.shape[1]


def test_mel_spectrogram_hop_length_defaults_to_the_shared_constant() -> None:
    # Same output whether HOP_LENGTH is relied on implicitly or passed
    # explicitly -- callers that don't care about resolution (L1's use of
    # HOP_LENGTH, or any existing spectrogram-building call) see no change.
    segment = _tone(150.0, 1.0)
    default_spec = mel_spectrogram(segment, 8000)
    explicit_spec = mel_spectrogram(segment, 8000, hop_length=HOP_LENGTH)
    assert np.array_equal(default_spec, explicit_spec)


def test_mel_spectrogram_smaller_hop_length_yields_more_frames() -> None:
    # The whole point of the experiment: a short utterance survives fewer
    # MaxPool(stride 2) rounds with more frames to start from -- see
    # SpeakerCNN's docstring. Halving hop_length should roughly double the
    # frame count for the same audio.
    segment = _tone(150.0, 1.0)
    default_spec = mel_spectrogram(segment, 8000, hop_length=HOP_LENGTH)
    finer_spec = mel_spectrogram(segment, 8000, hop_length=HOP_LENGTH // 2)
    assert finer_spec.shape[1] > default_spec.shape[1] * 1.5
    assert finer_spec.shape[0] == N_MELS  # only time resolution changes


def test_mel_spectrogram_n_mels_controls_frequency_bins_only() -> None:
    # Mirrors test_mel_spectrogram_smaller_hop_length_yields_more_frames --
    # this is the other axis: n_mels changes frequency resolution (rows),
    # hop_length changes time resolution (columns), independently.
    segment = _tone(150.0, 1.0)
    default_spec = mel_spectrogram(segment, 8000, n_mels=N_MELS)
    finer_spec = mel_spectrogram(segment, 8000, n_mels=N_MELS * 2)
    assert finer_spec.shape[0] == N_MELS * 2
    assert finer_spec.shape[1] == default_spec.shape[1]


def test_mel_spectrogram_is_near_silent_for_pure_silence() -> None:
    segment = np.zeros(8000, dtype=np.float32)
    spec = mel_spectrogram(segment, 8000)
    # ref=np.max degenerates to 0 energy everywhere -> librosa's power_to_db
    # floor kicks in uniformly; just check it doesn't blow up.
    assert np.isfinite(spec).all()


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


def _write_call(root: Path, wav_relative: str, json_relative: str, utterances_ms, *, seed: int) -> None:
    call_duration_s = max(end for _, end, _ in utterances_ms) / 1000.0 + 0.5
    _write_wav(root / wav_relative, call_duration_s, seed=seed)
    document = {
        "_id": Path(json_relative).stem,
        "utterances": [
            {"id": f"u{i}", "startAt": start, "endAt": end, "speaker": speaker, "text": f"secret {i}"}
            for i, (start, end, speaker) in enumerate(utterances_ms)
        ],
    }
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_build_seoul_training_spectrograms_saves_one_npy_per_utterance(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    utterances = [(0, 90, 0), (200, 1800, 1), (1900, 3400, 0)]
    wav_relative = "chunk/Training/1.원천데이터/TS_서울_구급/call1.wav"
    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/call1.json"
    _write_call(root, wav_relative, json_relative, utterances, seed=0)
    row = ManifestRow("Training", "call1", wav_relative, json_relative)

    output_dir = tmp_path / "specs"
    index_rows, failures = build_seoul_training_spectrograms([row], root, output_dir, n_calls=1, seed=42)

    assert failures == []
    assert len(index_rows) == 3
    for entry in index_rows:
        spec = np.load(output_dir / entry["npy_path"])
        assert spec.shape[0] == N_MELS
        assert spec.shape[1] == entry["n_frames"]
        assert np.isfinite(spec).all()
        assert entry["speaker"] in (0, 1)


def test_build_seoul_validation_spectrograms_uses_the_validation_split_only(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    utterances = [(0, 90, 0), (200, 1800, 1)]

    train_wav = "chunk/Training/1.원천데이터/TS_서울_구급/call_train.wav"
    train_json = "chunk/Training/2.라벨링데이터/TL_서울_구급/call_train.json"
    _write_call(root, train_wav, train_json, utterances, seed=0)
    train_row = ManifestRow("Training", "call_train", train_wav, train_json)

    val_wav = "chunk/Validation/1.원천데이터/VS_서울_구급/call_val.wav"
    val_json = "chunk/Validation/2.라벨링데이터/VL_서울_구급/call_val.json"
    _write_call(root, val_wav, val_json, utterances, seed=1)
    val_row = ManifestRow("Validation", "call_val", val_wav, val_json)

    output_dir = tmp_path / "specs"
    index_rows, failures = build_seoul_validation_spectrograms(
        [train_row, val_row], root, output_dir, n_calls=0, seed=42
    )

    assert failures == []
    # Only the Validation call's utterances -- the Training call must not
    # leak in, the same isolation guarantee L1's validation builder has.
    assert {entry["call_id"] for entry in index_rows} == {"call_val"}
    assert len(index_rows) == 2
    for entry in index_rows:
        spec = np.load(output_dir / entry["npy_path"])
        assert spec.shape[0] == N_MELS
        assert np.isfinite(spec).all()


def test_build_seoul_training_spectrograms_hop_length_controls_frame_count(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    utterances = [(0, 90, 0), (200, 1800, 1), (1900, 3400, 0)]
    wav_relative = "chunk/Training/1.원천데이터/TS_서울_구급/call1.wav"
    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/call1.json"
    _write_call(root, wav_relative, json_relative, utterances, seed=0)
    row = ManifestRow("Training", "call1", wav_relative, json_relative)

    default_rows, _ = build_seoul_training_spectrograms(
        [row], root, tmp_path / "specs_default", n_calls=1, seed=42
    )
    finer_rows, _ = build_seoul_training_spectrograms(
        [row], root, tmp_path / "specs_finer", n_calls=1, seed=42, hop_length=HOP_LENGTH // 2
    )

    default_by_utt = {entry["utt_idx"]: entry["n_frames"] for entry in default_rows}
    finer_by_utt = {entry["utt_idx"]: entry["n_frames"] for entry in finer_rows}
    for utt_idx, default_n_frames in default_by_utt.items():
        assert finer_by_utt[utt_idx] > default_n_frames


def test_build_seoul_training_spectrograms_n_mels_controls_frequency_bins(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    utterances = [(0, 90, 0), (200, 1800, 1), (1900, 3400, 0)]
    wav_relative = "chunk/Training/1.원천데이터/TS_서울_구급/call1.wav"
    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/call1.json"
    _write_call(root, wav_relative, json_relative, utterances, seed=0)
    row = ManifestRow("Training", "call1", wav_relative, json_relative)

    default_rows, _ = build_seoul_training_spectrograms(
        [row], root, tmp_path / "specs_default", n_calls=1, seed=42
    )
    wider_rows, _ = build_seoul_training_spectrograms(
        [row], root, tmp_path / "specs_wider", n_calls=1, seed=42, n_mels=N_MELS * 2
    )

    assert len(default_rows) == len(wider_rows) == 3
    for entry in default_rows:
        spec = np.load(tmp_path / "specs_default" / entry["npy_path"])
        assert spec.shape[0] == N_MELS
    for entry in wider_rows:
        spec = np.load(tmp_path / "specs_wider" / entry["npy_path"])
        assert spec.shape[0] == N_MELS * 2
        # n_frames in the index tracks the time axis (shape[1]), unaffected by n_mels
        assert spec.shape[1] == entry["n_frames"]


# ---------------------------------------------------------------------------
# Provenance: which split an index came from, and what it was built with
# ---------------------------------------------------------------------------

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _training_and_validation_rows(tmp_path: Path):
    root = tmp_path / "raw"
    utterances = [(0, 90, 0), (200, 1800, 1)]
    train_wav = "chunk/Training/1.원천데이터/TS_서울_구급/call_train.wav"
    train_json = "chunk/Training/2.라벨링데이터/TL_서울_구급/call_train.json"
    _write_call(root, train_wav, train_json, utterances, seed=0)
    val_wav = "chunk/Validation/1.원천데이터/VS_서울_구급/call_val.wav"
    val_json = "chunk/Validation/2.라벨링데이터/VL_서울_구급/call_val.json"
    _write_call(root, val_wav, val_json, utterances, seed=1)
    rows = [
        ManifestRow("Training", "call_train", train_wav, train_json),
        ManifestRow("Validation", "call_val", val_wav, val_json),
    ]
    return root, rows


def test_training_builder_stamps_every_index_row_with_its_split_and_build_settings(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)

    index_rows, failures = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    assert failures == []
    assert len(index_rows) == 2
    for entry in index_rows:
        assert entry["split"] == "Training"
        assert entry["n_mels"] == N_MELS
        assert entry["hop_length"] == HOP_LENGTH
        assert entry["n_fft"] == N_FFT
        assert entry["sample_rate"] == SAMPLE_RATE


def test_validation_builder_stamps_rows_as_validation(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)

    index_rows, failures = build_seoul_validation_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    assert failures == []
    assert {entry["split"] for entry in index_rows} == {"Validation"}


def test_index_records_non_default_resolution_settings(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)

    index_rows, _ = build_seoul_training_spectrograms(
        rows, root, tmp_path / "specs", n_calls=0, seed=42, hop_length=HOP_LENGTH // 2, n_mels=N_MELS * 2
    )

    assert {(entry["n_mels"], entry["hop_length"]) for entry in index_rows} == {(N_MELS * 2, HOP_LENGTH // 2)}


def test_training_guard_accepts_a_training_built_index_and_returns_its_settings(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    provenance = assert_training_spectrogram_index(pd.DataFrame(index_rows))

    assert provenance == {
        "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT, "sample_rate": SAMPLE_RATE,
        "label_source": LABEL_SOURCE_PROVIDED,
    }
    # plain Python values, so they can go into a checkpoint that torch.load reads with weights_only=True
    assert all(type(value) in (int, str) for value in provenance.values())


def test_training_guard_refuses_a_validation_built_index(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_validation_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    with pytest.raises(SplitContaminationError, match="non-Training"):
        assert_training_spectrogram_index(pd.DataFrame(index_rows))


def test_training_guard_refuses_an_index_that_mixes_training_and_validation_rows(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    train_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs_t", n_calls=0, seed=42)
    val_rows, _ = build_seoul_validation_spectrograms(rows, root, tmp_path / "specs_v", n_calls=0, seed=42)

    with pytest.raises(SplitContaminationError, match="non-Training"):
        assert_training_spectrogram_index(pd.DataFrame(train_rows + val_rows))


def test_training_guard_refuses_an_index_without_provenance_and_says_how_to_fix_it() -> None:
    legacy = pd.DataFrame(
        {"call_id": ["a", "a"], "utt_idx": [0, 1], "speaker": [0, 1], "duration_sec": [1.0, 2.0],
         "n_frames": [10, 20], "npy_path": ["a/0.npy", "a/1.npy"]}
    )

    with pytest.raises(SplitContaminationError, match="no build provenance") as excinfo:
        assert_training_spectrogram_index(legacy)

    message = str(excinfo.value)
    for column in INDEX_PROVENANCE_COLUMNS:
        assert column in message
    assert "build_mission2_spectrograms.py" in message


def test_the_refusal_message_sends_a_legacy_index_through_the_label_check_not_a_bare_restamp() -> None:
    # A bare "stamp it as Training" one-liner would let an index whose labels were edited train as if it were
    # clean; the only advice offered must be the tool that also checks the labels (see the raw-label tests).
    legacy = pd.DataFrame(
        {"call_id": ["a", "a"], "utt_idx": [0, 1], "speaker": [0, 1], "duration_sec": [1.0, 2.0],
         "n_frames": [10, 20], "npy_path": ["a/0.npy", "a/1.npy"]}
    )
    with pytest.raises(SplitContaminationError) as excinfo:
        assert_training_spectrogram_index(legacy)

    message = str(excinfo.value)
    assert "verify_mission2_raw_labels.py" in message and "--stamp-training-defaults" in message
    assert "df.assign" not in message and "to_parquet" not in message


def test_training_guard_refuses_an_index_whose_labels_are_not_the_provided_ones(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)
    frame = pd.DataFrame(index_rows).assign(label_source="corrected_after_reading_text")

    with pytest.raises(LabelProvenanceError, match="provided speaker labels"):
        assert_training_spectrogram_index(frame)


def test_training_guard_refuses_an_index_that_mixes_label_sources(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)
    frame = pd.DataFrame(index_rows)
    frame.loc[frame.index[0], "label_source"] = "corrected_after_reading_text"

    with pytest.raises(ValueError, match="mixes several label_source"):
        assert_training_spectrogram_index(frame)


def test_training_guard_refuses_an_index_with_only_some_provenance_columns(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    with pytest.raises(SplitContaminationError, match="missing column"):
        assert_training_spectrogram_index(pd.DataFrame(index_rows).drop(columns=["split"]))


def test_training_guard_refuses_an_index_built_with_mixed_settings(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)
    frame = pd.DataFrame(index_rows)
    frame.loc[frame.index[0], "hop_length"] = HOP_LENGTH // 2

    with pytest.raises(ValueError, match="mixes several hop_length"):
        assert_training_spectrogram_index(frame)


def test_training_guard_refuses_an_empty_index(tmp_path: Path) -> None:
    root, rows = _training_and_validation_rows(tmp_path)
    index_rows, _ = build_seoul_training_spectrograms(rows, root, tmp_path / "specs", n_calls=0, seed=42)

    with pytest.raises(ValueError, match="empty"):
        assert_training_spectrogram_index(pd.DataFrame(index_rows).iloc[0:0])


def _parse_build_args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    monkeypatch.syspath_prepend(str(SCRIPTS_ROOT))
    build_script = importlib.import_module("build_mission2_spectrograms")
    monkeypatch.setattr(
        sys, "argv", ["build_mission2_spectrograms.py", "--manifest", "m.csv", "--data-root", "raw", *extra]
    )
    return build_script.parse_args()


def test_build_script_keeps_the_training_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse_build_args(monkeypatch)

    assert args.output_dir == Path("data/m2_spectrograms")
    assert args.index_output == Path("data/m2_spectrogram_index.parquet")
    assert args.failures_log == Path("data/m2_spectrogram_build_failures.jsonl")


def test_build_script_defaults_validation_to_paths_that_never_collide_with_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _parse_build_args(monkeypatch, "--split", "training")
    validation = _parse_build_args(monkeypatch, "--split", "validation")

    assert validation.output_dir == Path("data/m2_spectrograms_validation")
    assert validation.index_output == Path("data/m2_spectrogram_index_validation.parquet")
    assert validation.failures_log == Path("data/m2_spectrogram_build_failures_validation.jsonl")
    assert {validation.output_dir, validation.index_output, validation.failures_log}.isdisjoint(
        {training.output_dir, training.index_output, training.failures_log}
    )


def test_build_script_still_honours_explicit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse_build_args(
        monkeypatch, "--split", "validation", "--output-dir", "x/specs", "--index-output", "x/idx.parquet"
    )

    assert args.output_dir == Path("x/specs")
    assert args.index_output == Path("x/idx.parquet")
    assert args.failures_log == Path("data/m2_spectrogram_build_failures_validation.jsonl")
