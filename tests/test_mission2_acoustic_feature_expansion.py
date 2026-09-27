from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import wave

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dcc_data.mission2_acoustic_feature_expansion"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "mission2_acoustic_feature_expansion.ipynb"

EXPECTED_E05 = tuple(
    [f"mfcc_{index:02d}_mean" for index in range(1, 21)]
    + [f"mfcc_{index:02d}_std" for index in range(1, 21)]
    + ["spectral_rolloff", "zcr"]
)


@pytest.fixture
def expansion_module():
    assert importlib.util.find_spec(MODULE_NAME) is not None, (
        "acoustic feature expansion module has not been implemented"
    )
    return importlib.import_module(MODULE_NAME)


def test_shared_array_fingerprint_is_layout_independent_and_content_sensitive(
    expansion_module,
):
    model_module = importlib.import_module("dcc_data.mission2_model_experiments")
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    baseline = model_module._array_fingerprint(values)
    assert expansion_module._array_fingerprint(values) == baseline
    assert expansion_module._array_fingerprint(np.asfortranarray(values)) == baseline
    assert model_module._array_fingerprint(values.reshape(2, 6)) != baseline
    assert model_module._array_fingerprint(values.astype(np.float64)) != baseline
    changed = values.copy()
    changed[0, 0] = 99
    assert model_module._array_fingerprint(changed) != baseline
    assert model_module._array_fingerprint(np.asarray(1)) != model_module._array_fingerprint(
        np.asarray([1])
    )
    with pytest.raises(TypeError, match="object dtype"):
        model_module._array_fingerprint(np.asarray([object()], dtype=object))
    assert expansion_module._legacy_acoustic_array_fingerprint(
        np.arange(12, dtype=np.int64).reshape(3, 4)
    ) == "2dbaee24e99866f445e3be4af710f8075e4fd7ee175d377b4a901f55288c507b"


def test_shared_config_fingerprint_is_strict_and_preserves_pitch_cache_hash(
    expansion_module,
):
    model_module = importlib.import_module("dcc_data.mission2_model_experiments")
    first = {
        "paths": (Path("cache") / "features",),
        "seed": np.int64(42),
        "weight": np.float32(0.5),
    }
    second = {
        "weight": 0.5,
        "seed": 42,
        "paths": [str(Path("cache") / "features")],
    }
    assert model_module.config_fingerprint(first) == expansion_module.config_fingerprint(
        second
    )
    assert model_module.config_fingerprint(first) != model_module.config_fingerprint(
        {**second, "seed": 43}
    )
    with pytest.raises(TypeError, match="unsupported"):
        expansion_module.config_fingerprint({"unsupported": object()})
    with pytest.raises(TypeError, match="string mapping keys"):
        model_module.config_fingerprint({1: "unsupported key"})
    with pytest.raises(ValueError, match="finite"):
        model_module.config_fingerprint({"weight": float("nan")})
    assert expansion_module.config_fingerprint(
        expansion_module.PITCH_EXTRACTION_CONFIG
    ) == "6a3dda4caa74914667125f04ad27eb4802fadb411ce515f6680804f5b9e1cac3"


def test_comparison_functions_have_distinct_public_names(expansion_module):
    model_module = importlib.import_module("dcc_data.mission2_model_experiments")
    assert callable(expansion_module.compare_feature_candidate_to_reference)
    assert callable(model_module.compare_model_candidate_to_reference)
    assert not hasattr(expansion_module, "compare_candidate_to_reference")
    assert not hasattr(model_module, "compare_candidate_to_reference")


def _sine_wave(
    frequency: float = 200.0,
    *,
    seconds: float = 0.8,
    sample_rate: int = 8_000,
    amplitude: float = 0.5,
) -> np.ndarray:
    times = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def _write_pcm16_wav(path: Path, waveform: np.ndarray, *, sample_rate: int = 8_000):
    """Write a real mono WAV so spawned workers exercise the production reader."""

    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _multiprocessing_fixture(tmp_path: Path) -> pd.DataFrame:
    """Return interleaved calls whose canonical order differs from call grouping."""

    for call_index, frequency in enumerate((110.0, 170.0, 230.0, 310.0)):
        _write_pcm16_wav(
            tmp_path / f"call_{call_index}.wav",
            _sine_wave(frequency, seconds=1.2),
        )
    return pd.DataFrame(
        {
            "split": ["Training"] * 8,
            "record_key": [
                "key_c0_0",
                "key_c1_0",
                "key_c2_0",
                "key_c3_0",
                "key_c0_1",
                "key_c1_1",
                "key_c2_1",
                "key_c3_1",
            ],
            "call_id": [
                "call_0",
                "call_1",
                "call_2",
                "call_3",
                "call_0",
                "call_1",
                "call_2",
                "call_3",
            ],
            "audio_path": [
                "call_0.wav",
                "call_1.wav",
                "call_2.wav",
                "call_3.wav",
                "call_0.wav",
                "call_1.wav",
                "call_2.wav",
                "call_3.wav",
            ],
            "crop_start_frame": [0, 0, 0, 0, 3_200, 3_200, 3_200, 3_200],
            "crop_end_frame": [3_200, 3_200, 3_200, 3_200, 8_000, 8_000, 8_000, 8_000],
            "crop_ok": [True] * 8,
        }
    )


def _reference_autocorrelation_pitch_track(
    segment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent copy of the pre-optimization AF02 equation."""

    sample_rate = 8_000
    frame_length = 1_024
    hop_length = 160
    lag_min = int(np.floor(sample_rate / 400.0))
    lag_max = int(np.ceil(sample_rate / 65.0))
    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)
    signal = np.asarray(segment, dtype=np.float32).reshape(-1)
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    padded = np.pad(signal, (frame_length // 2, frame_length // 2))
    frame_starts = np.arange(
        0, len(padded) - frame_length + 1, hop_length, dtype=np.int64
    )
    window = np.hanning(frame_length).astype(np.float64)
    f0 = np.zeros(len(frame_starts), dtype=np.float64)
    periodicity = np.zeros(len(frame_starts), dtype=np.float64)
    for frame_index, start in enumerate(frame_starts):
        frame = padded[start : start + frame_length].astype(np.float64)
        frame = (frame - np.mean(frame)) * window
        if float(np.sqrt(np.mean(np.square(frame)))) < 0.0001:
            continue
        correlation = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        squared = np.square(frame)
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        denominator = np.sqrt(
            cumulative[frame_length - lags]
            * (cumulative[frame_length] - cumulative[lags])
        )
        scores = np.divide(
            correlation[lags],
            denominator,
            out=np.zeros(len(lags), dtype=np.float64),
            where=denominator > np.finfo(np.float64).eps,
        )
        best = int(np.argmax(scores))
        periodicity[frame_index] = float(np.clip(scores[best], 0.0, 1.0))
        f0[frame_index] = sample_rate / float(lags[best])
    return f0, periodicity


def _reference_pitch_features(segment: np.ndarray) -> dict[str, float]:
    f0, periodicity = _reference_autocorrelation_pitch_track(segment)
    valid = (
        np.isfinite(f0)
        & (f0 > 0.0)
        & np.isfinite(periodicity)
        & (periodicity >= 0.6)
    )
    f0_statistics = {
        "f0_semitone_median": 0.0,
        "f0_semitone_iqr": 0.0,
        "f0_semitone_slope": 0.0,
    }
    if int(np.sum(valid)) >= 3:
        semitones = 12.0 * np.log2(f0[valid] / 27.5)
        voiced_times = np.flatnonzero(valid).astype(np.float64) * 160.0 / 8_000.0
        f0_statistics = {
            "f0_semitone_median": float(np.median(semitones)),
            "f0_semitone_iqr": float(
                np.quantile(semitones, 0.75) - np.quantile(semitones, 0.25)
            ),
            "f0_semitone_slope": float(
                np.polyfit(voiced_times, semitones, 1)[0]
            ),
        }
    return {
        **f0_statistics,
        "voiced_fraction": float(np.mean(valid)),
        "periodicity_mean": float(np.mean(periodicity)),
    }


def _assert_pitch_parity(expansion_module, segment: np.ndarray) -> None:
    reference_f0, reference_periodicity = _reference_autocorrelation_pitch_track(
        segment
    )
    observed_f0, observed_periodicity = (
        expansion_module._autocorrelation_pitch_track(segment)
    )
    assert observed_f0.shape == reference_f0.shape
    np.testing.assert_array_equal(observed_f0, reference_f0)
    reference_voiced = (reference_f0 > 0.0) & (reference_periodicity >= 0.6)
    observed_voiced = (observed_f0 > 0.0) & (observed_periodicity >= 0.6)
    np.testing.assert_array_equal(observed_voiced, reference_voiced)
    np.testing.assert_allclose(
        observed_periodicity, reference_periodicity, rtol=0.0, atol=1e-12
    )
    expected = _reference_pitch_features(segment)
    observed = expansion_module.extract_acoustic_blocks(
        segment, block_ids=("AF02_PITCH_VOICING",)
    )
    assert tuple(observed) == tuple(expected)
    np.testing.assert_allclose(
        list(observed.values()), list(expected.values()), rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(list(observed.values()), dtype=np.float32),
        np.asarray(list(expected.values()), dtype=np.float32),
    )


def _synthetic_e05(rows: int = 48):
    row = np.arange(rows, dtype=np.float32)[:, None]
    column = np.arange(42, dtype=np.float32)[None, :]
    X = np.sin(row / 5.0 + column / 7.0).astype(np.float32)
    y = ((np.arange(rows) // 2) % 2).astype(np.int64)
    record_keys = np.asarray(
        [f"Training|call_{index // 4:03d}|{index % 4}" for index in range(rows)]
    )
    call_ids = np.asarray([f"call_{index // 4:03d}" for index in range(rows)])
    tracking = pd.DataFrame(
        {
            "record_key": record_keys,
            "call_id": call_ids,
            "speaker": y,
            "startAt_ms": np.tile([0.0, 400.0, 800.0, 1_200.0], rows // 4),
            "endAt_ms": np.tile([500.0, 900.0, 1_300.0, 1_700.0], rows // 4),
        }
    )
    return X, y, record_keys, call_ids, tracking


def _candidate_frame(expansion_module, record_keys, call_ids):
    values = {}
    for offset, name in enumerate(expansion_module.ALL_CANDIDATE_FEATURE_NAMES):
        values[name] = (
            np.arange(len(record_keys), dtype=np.float32) / np.float32(10.0)
            + np.float32(offset)
        )
    return pd.DataFrame(
        {"record_key": record_keys, "call_id": call_ids, **values}
    ).iloc[::-1].reset_index(drop=True)


def test_precommit_manifest_records_exact_definitions_and_risks(expansion_module):
    table = expansion_module.candidate_manifest_table()
    assert table["block_id"].tolist() == [
        "AF01_ENERGY_DYNAMICS",
        "AF02_PITCH_VOICING",
        "AF03_NORMALIZED_SPECTRAL_FLUX",
        "AF04_CPP",
        "AF05_DELTA_MFCC_DYNAMICS",
        "AF07_MFCC_CROSS_CORRELATION",
        "AF08_COMPETING_PITCH",
        "AF09_MULTIBAND_ENVELOPE_COHERENCE",
        "AF10_SPEECH_ACTIVITY_BOUNDARY",
        "AF11_SPECTRAL_ENTROPY_DYNAMICS",
        "AF12_TEMPORAL_MFCC_CONTRAST",
    ]
    assert table["output_dimension"].tolist() == [3, 5, 3, 2, 2, 2, 2, 3, 4, 3, 3]
    af01_risk = table.loc[
        table["block_id"] == "AF01_ENERGY_DYNAMICS", "diagnostic_risks"
    ].item()
    assert "recording-level" in af01_risk and "gain" in af01_risk
    af03_definition = table.loc[
        table["block_id"] == "AF03_NORMALIZED_SPECTRAL_FLUX", "calculation"
    ].item()
    assert "L2-normalized" in af03_definition
    assert "not librosa.onset_strength" in af03_definition
    af04_definition = table.loc[table["block_id"] == "AF04_CPP", "calculation"].item()
    assert "q=8..128" in af04_definition
    assert "q=20..123" in af04_definition
    af05_definition = table.loc[
        table["block_id"] == "AF05_DELTA_MFCC_DYNAMICS", "calculation"
    ].item()
    assert "MFCC 1..19" in af05_definition
    assert "nearest" in af05_definition


def test_round2_contract_is_fully_precommitted(expansion_module):
    assert expansion_module.ROUND1_BLOCK_IDS == (
        "AF01_ENERGY_DYNAMICS",
        "AF02_PITCH_VOICING",
        "AF03_NORMALIZED_SPECTRAL_FLUX",
    )
    assert expansion_module.ROUND2_BLOCK_IDS == (
        "AF04_CPP",
        "AF05_DELTA_MFCC_DYNAMICS",
    )
    assert expansion_module.ROUND2_REFERENCE_ID == "AF02_47D_REFERENCE"
    assert expansion_module.ROUND2_REFERENCE_FEATURE_NAMES == (
        *EXPECTED_E05,
        "f0_semitone_median",
        "f0_semitone_iqr",
        "f0_semitone_slope",
        "voiced_fraction",
        "periodicity_mean",
    )
    assert expansion_module.ROUND2_M04_ITER100_ACCURACY_TARGET == 0.866266


def test_cpp_periodic_signal_is_finite_deterministic_float32(expansion_module):
    signal = _sine_wave(200.0)
    first = expansion_module.extract_acoustic_blocks(
        signal, block_ids=("AF04_CPP",)
    )
    second = expansion_module.extract_acoustic_blocks(
        signal, block_ids=("AF04_CPP",)
    )
    assert tuple(first) == ("cpp_median_db", "cpp_iqr_db")
    observed = np.asarray(tuple(first.values()))
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    assert first == second
    assert first["cpp_median_db"] > 0.0
    assert first["cpp_iqr_db"] >= 0.0


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(2_000, dtype=np.float32),
        np.ones(2_000, dtype=np.float32),
    ],
)
def test_cpp_silence_and_constant_inputs_use_finite_fallback(
    expansion_module, signal
):
    features = expansion_module.extract_acoustic_blocks(
        signal, block_ids=("AF04_CPP",)
    )
    np.testing.assert_array_equal(
        np.asarray(tuple(features.values())), np.zeros(2, dtype=np.float32)
    )


def test_cpp_short_input_is_padded_and_finite(expansion_module):
    features = expansion_module.extract_acoustic_blocks(
        np.asarray([0.25], dtype=np.float32), block_ids=("AF04_CPP",)
    )
    observed = np.asarray(tuple(features.values()))
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    np.testing.assert_array_equal(observed, np.zeros(2, dtype=np.float32))


def test_cpp_active_frame_threshold_preserves_gain_sensitivity_risk(expansion_module):
    loud = expansion_module.extract_acoustic_blocks(
        _sine_wave(200.0, amplitude=0.5), block_ids=("AF04_CPP",)
    )
    below_gate = expansion_module.extract_acoustic_blocks(
        _sine_wave(200.0, amplitude=1e-5), block_ids=("AF04_CPP",)
    )
    assert loud["cpp_median_db"] > 0.0
    np.testing.assert_array_equal(
        np.asarray(tuple(below_gate.values())), np.zeros(2, dtype=np.float32)
    )


def test_delta_mfcc_explicit_equation_boundary_and_c0_exclusion(expansion_module):
    mfcc = np.zeros((20, 5), dtype=np.float64)
    mfcc[0] = np.asarray([0.0, 100.0, -100.0, 200.0, -200.0])
    mfcc[1] = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    observed = expansion_module._delta_mfcc_matrix(mfcc)
    expected = np.zeros((19, 5), dtype=np.float64)
    expected[0] = np.asarray([0.5, 0.8, 1.0, 0.8, 0.5])
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)

    c0_only = np.zeros((20, 5), dtype=np.float64)
    c0_only[0] = mfcc[0]
    np.testing.assert_array_equal(
        expansion_module._delta_mfcc_matrix(c0_only), np.zeros((19, 5))
    )


def test_delta_mfcc_summary_matches_hand_calculation_and_short_fallback(
    expansion_module,
):
    mfcc = np.zeros((20, 5), dtype=np.float64)
    mfcc[1] = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    features = expansion_module._delta_mfcc_features_from_matrix(mfcc)
    frame_norms = np.asarray([0.5, 0.8, 1.0, 0.8, 0.5]) / np.sqrt(19.0)
    expected = np.asarray(
        [np.mean(frame_norms), np.std(frame_norms, ddof=0)], dtype=np.float32
    )
    np.testing.assert_allclose(
        np.asarray(tuple(features.values())), expected, rtol=1e-6, atol=1e-7
    )

    short = expansion_module._delta_mfcc_features_from_matrix(
        np.zeros((20, 2), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(tuple(short.values())), np.zeros(2, dtype=np.float32)
    )


def test_delta_mfcc_frontend_preserves_canonical_mfcc_moments(expansion_module):
    from dcc_data.mission2_features import acoustic_feature_vector

    rng = np.random.default_rng(2026)
    signal = (
        _sine_wave(180.0, seconds=0.7)
        + rng.normal(0.0, 0.01, 5_600).astype(np.float32)
    ).astype(np.float32)
    frames = expansion_module._canonical_mfcc_frame_matrix(signal)
    observed_moments = np.concatenate(
        (frames.mean(axis=1), frames.std(axis=1))
    ).astype(np.float32)
    canonical = acoustic_feature_vector(signal, 8_000)[:40]
    np.testing.assert_array_equal(observed_moments, canonical)


def test_delta_mfcc_waveform_behavior_is_finite_deterministic_float32(
    expansion_module,
):
    changing = np.concatenate(
        [_sine_wave(120.0, seconds=0.35), _sine_wave(320.0, seconds=0.35)]
    )
    first = expansion_module.extract_acoustic_blocks(
        changing, block_ids=("AF05_DELTA_MFCC_DYNAMICS",)
    )
    second = expansion_module.extract_acoustic_blocks(
        changing, block_ids=("AF05_DELTA_MFCC_DYNAMICS",)
    )
    observed = np.asarray(tuple(first.values()))
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    assert first == second
    assert first["delta_mfcc_norm_mean"] > 0.0


@pytest.mark.parametrize("block_id", ["AF04_CPP", "AF05_DELTA_MFCC_DYNAMICS"])
@pytest.mark.parametrize(
    "signal",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
    ],
)
def test_round2_blocks_fail_fast_on_invalid_waveforms(
    expansion_module, block_id, signal
):
    with pytest.raises(ValueError, match="waveform"):
        expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))


def test_pitch_contract_is_fully_precommitted(expansion_module):
    assert dict(expansion_module.PITCH_EXTRACTION_CONFIG) == {
        "method": "normalized_autocorrelation",
        "fmin_hz": 65.0,
        "fmax_hz": 400.0,
        "frame_length": 1024,
        "hop_length": 160,
        "semitone_reference_hz": 27.5,
        "minimum_voiced_frames": 3,
        "periodicity_threshold": 0.6,
        "minimum_frame_rms": 0.0001,
        "slope_unit": "semitones_per_second",
        "fallback": "zero_f0_statistics_keep_voicing_statistics",
    }


def test_pitch_block_tracks_a_clean_tone_and_remains_finite(expansion_module):
    features = expansion_module.extract_acoustic_blocks(
        _sine_wave(200.0), block_ids=("AF02_PITCH_VOICING",)
    )
    assert tuple(features) == expansion_module.BLOCK_BY_ID[
        "AF02_PITCH_VOICING"
    ].feature_names
    assert np.isfinite(list(features.values())).all()
    expected_semitones = 12.0 * np.log2(200.0 / 27.5)
    assert features["f0_semitone_median"] == pytest.approx(
        expected_semitones, abs=0.75
    )
    assert features["voiced_fraction"] > 0.5


def test_pitch_block_uses_zero_f0_fallback_for_unvoiced_signal(expansion_module):
    features = expansion_module.extract_acoustic_blocks(
        np.zeros(2_000, dtype=np.float32),
        block_ids=("AF02_PITCH_VOICING",),
    )
    assert features["f0_semitone_median"] == 0.0
    assert features["f0_semitone_iqr"] == 0.0
    assert features["f0_semitone_slope"] == 0.0
    assert features["voiced_fraction"] == 0.0
    assert features["periodicity_mean"] == 0.0
    assert np.isfinite(list(features.values())).all()


@pytest.mark.parametrize("frequency", [80.0, 120.0, 200.0, 350.0])
def test_fft_pitch_matches_reference_for_multiple_clean_tones(
    expansion_module, frequency
):
    _assert_pitch_parity(expansion_module, _sine_wave(frequency))


@pytest.mark.parametrize(
    ("case_name", "waveform"),
    [
        ("short", _sine_wave(180.0, seconds=0.08)),
        ("very_short", np.asarray([0.25], dtype=np.float32)),
        ("silence", np.zeros(2_000, dtype=np.float32)),
        ("low_amplitude", _sine_wave(200.0, amplitude=1e-5)),
        (
            "white_noise",
            np.random.default_rng(2026).normal(0.0, 0.1, 6_400).astype(np.float32),
        ),
        ("clipped", np.clip(_sine_wave(200.0) * 4.0, -1.0, 1.0)),
        (
            "voiced_unvoiced_mix",
            np.concatenate(
                [
                    _sine_wave(160.0, seconds=0.25),
                    np.zeros(1_600, dtype=np.float32),
                    _sine_wave(240.0, seconds=0.25),
                ]
            ),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_fft_pitch_matches_reference_for_edge_waveforms(
    expansion_module, case_name, waveform
):
    del case_name
    _assert_pitch_parity(expansion_module, waveform)


def test_pitch_extraction_no_longer_depends_on_full_np_correlate(
    expansion_module, monkeypatch
):
    waveform = _sine_wave(200.0)
    reference_f0, reference_periodicity = _reference_autocorrelation_pitch_track(
        waveform
    )

    def reject_full_correlation(*args, **kwargs):
        raise AssertionError("AF02 must not call full np.correlate after optimization")

    monkeypatch.setattr(np, "correlate", reject_full_correlation)
    observed_f0, observed_periodicity = (
        expansion_module._autocorrelation_pitch_track(waveform)
    )
    np.testing.assert_array_equal(observed_f0, reference_f0)
    np.testing.assert_allclose(
        observed_periodicity, reference_periodicity, rtol=0.0, atol=1e-12
    )


def test_fft_pitch_matches_reference_on_training_sample(expansion_module):
    manifest_path = REPO_ROOT / "tmp" / "mission2_eda" / "utterance_manifest_v1.parquet"
    raw_root = REPO_ROOT / "data" / "raw_original"
    if not manifest_path.is_file() or not raw_root.is_dir():
        pytest.skip("local Training sample is unavailable")
    from dcc_data.audio import read_wav_frames

    manifest = pd.read_parquet(manifest_path)
    training = manifest.loc[
        (manifest["split"].astype(str) == "Training")
        & manifest["crop_ok"].astype(bool)
    ].copy()
    training["duration_frames"] = (
        training["crop_end_frame"].astype(np.int64)
        - training["crop_start_frame"].astype(np.int64)
    )
    ordered = training.sort_values("duration_frames", kind="stable")
    rows = ordered.iloc[
        np.linspace(0, len(ordered) - 1, 12).round().astype(np.int64)
    ]
    for row in rows.itertuples(index=False):
        audio = read_wav_frames(
            raw_root / str(row.audio_path),
            int(row.crop_start_frame),
            int(row.crop_end_frame),
        )
        assert audio.sample_rate == 8_000
        assert audio.waveform.shape[0] == 1
        _assert_pitch_parity(expansion_module, audio.waveform[0])


def test_normalized_spectral_flux_matches_hand_calculation(expansion_module):
    magnitude = np.asarray(
        [[3.0, 0.0, 0.0], [4.0, 5.0, 0.0]], dtype=np.float64
    )
    flux = expansion_module.normalized_spectral_flux(magnitude)
    # Columns normalize to [0.6, 0.8], [0, 1], and [0, 0].
    expected = np.asarray([np.sqrt(0.6**2 + 0.2**2), 1.0])
    np.testing.assert_allclose(flux, expected, rtol=0, atol=1e-12)


def test_all_blocks_preserve_order_and_round2_float32_finiteness(expansion_module):
    frame = expansion_module.extract_acoustic_blocks(_sine_wave())
    assert tuple(frame) == expansion_module.ALL_CANDIDATE_FEATURE_NAMES
    values = np.asarray(list(frame.values()))
    assert values.shape == (len(expansion_module.ALL_CANDIDATE_FEATURE_NAMES),)
    assert np.isfinite(values).all()
    for block_id in expansion_module.ROUND2_BLOCK_IDS:
        for feature_name in expansion_module.BLOCK_BY_ID[block_id].feature_names:
            assert isinstance(frame[feature_name], np.float32)


def test_pitch_small_runtime_benchmark_is_bounded_and_inspectable(expansion_module):
    report = expansion_module.benchmark_pitch_extraction(
        [_sine_wave(180.0, seconds=0.25), _sine_wave(240.0, seconds=0.25)]
    )
    assert report["utterance_count"] == 2
    assert report["audio_seconds"] == pytest.approx(0.5)
    assert 0.0 <= report["runtime_seconds"] < 10.0
    assert report["estimated_seconds_per_utterance"] >= 0.0
    assert report["pitch_config_fingerprint"] == expansion_module.config_fingerprint(
        expansion_module.PITCH_EXTRACTION_CONFIG
    )


def test_call_grouped_extraction_preserves_manifest_order_and_float32(
    expansion_module, monkeypatch
):
    sample_rate = 8_000
    waveform = _sine_wave(seconds=1.0)[None, :]
    read_paths = []

    def fake_read_wav_frames(path):
        read_paths.append(Path(path))
        return SimpleNamespace(waveform=waveform, sample_rate=sample_rate)

    monkeypatch.setattr(expansion_module, "read_wav_frames", fake_read_wav_frames)
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training", "Training"],
            "record_key": ["key_b0", "key_a0", "key_b1"],
            "call_id": ["call_b", "call_a", "call_b"],
            "audio_path": ["b.wav", "a.wav", "b.wav"],
            "crop_start_frame": [0, 0, 1_000],
            "crop_end_frame": [2_000, 2_000, 3_000],
            "crop_ok": [True, True, True],
        }
    )
    frame = expansion_module.extract_candidate_feature_frame(
        manifest, data_root=REPO_ROOT, split="Training"
    )
    assert frame["record_key"].tolist() == manifest["record_key"].tolist()
    assert [path.name for path in read_paths] == ["b.wav", "a.wav"]
    assert all(
        frame[name].dtype == np.float32
        for name in expansion_module.ALL_CANDIDATE_FEATURE_NAMES
    )
    assert np.isfinite(
        frame.loc[:, expansion_module.ALL_CANDIDATE_FEATURE_NAMES].to_numpy()
    ).all()


@pytest.mark.parametrize("worker_count", [2, 4])
def test_parallel_extraction_is_exactly_equal_to_sequential(
    expansion_module, tmp_path, worker_count
):
    """Catch dropped, duplicated, reordered, or numerically changed worker rows."""

    manifest = _multiprocessing_fixture(tmp_path)
    sequential = expansion_module.extract_candidate_feature_frame(
        manifest,
        data_root=tmp_path,
        split="Training",
        worker_count=1,
    )
    parallel = expansion_module.extract_candidate_feature_frame(
        manifest,
        data_root=tmp_path,
        split="Training",
        worker_count=worker_count,
    )

    pd.testing.assert_frame_equal(parallel, sequential, check_exact=True)
    assert parallel["record_key"].tolist() == manifest["record_key"].tolist()
    assert parallel["call_id"].tolist() == manifest["call_id"].tolist()
    assert len(parallel) == len(manifest)
    assert parallel["record_key"].nunique() == len(manifest)
    np.testing.assert_array_equal(
        parallel.loc[:, expansion_module.ALL_CANDIDATE_FEATURE_NAMES].to_numpy(),
        sequential.loc[:, expansion_module.ALL_CANDIDATE_FEATURE_NAMES].to_numpy(),
    )


def test_default_worker_strategy_prefers_benchmarked_sequential_path(
    expansion_module, monkeypatch
):
    """Catch automatic oversubscription despite the 480-row Windows benchmark."""

    monkeypatch.setattr(expansion_module.os, "cpu_count", lambda: 12)
    assert expansion_module._resolve_worker_count(None) == 1
    assert expansion_module._resolve_worker_count(2) == 2
    assert expansion_module._resolve_worker_count(4) == 4
    assert expansion_module._resolve_worker_count(6) == 6


def test_parallel_completion_order_cannot_change_canonical_row_order(
    expansion_module, tmp_path
):
    """Catch accidental concatenation in future-completion rather than source order."""

    _write_pcm16_wav(tmp_path / "slow.wav", _sine_wave(140.0, seconds=1.2))
    _write_pcm16_wav(tmp_path / "fast.wav", _sine_wave(280.0, seconds=1.2))
    slow_rows = 24
    manifest = pd.DataFrame(
        {
            "split": ["Training"] * (slow_rows + 1),
            "record_key": [
                "slow_00",
                "fast_00",
                *[f"slow_{index:02d}" for index in range(1, slow_rows)],
            ],
            "call_id": ["slow", "fast", *(["slow"] * (slow_rows - 1))],
            "audio_path": ["slow.wav", "fast.wav", *(["slow.wav"] * (slow_rows - 1))],
            "crop_start_frame": [0] * (slow_rows + 1),
            "crop_end_frame": [3_200] * (slow_rows + 1),
            "crop_ok": [True] * (slow_rows + 1),
        }
    )
    completion_ids = []
    frame = expansion_module.extract_candidate_feature_frame(
        manifest,
        data_root=tmp_path,
        split="Training",
        worker_count=2,
        on_call_done=lambda _done, _total, call_id: completion_ids.append(call_id),
    )

    assert completion_ids[0] == "fast"
    assert frame["record_key"].tolist() == manifest["record_key"].tolist()
    assert frame["call_id"].tolist() == manifest["call_id"].tolist()


def test_worker_thread_limit_is_initialized_once_not_per_call(
    expansion_module, monkeypatch, tmp_path
):
    """Catch expensive threadpool discovery being repeated for every call task."""

    _write_pcm16_wav(tmp_path / "call.wav", _sine_wave(200.0, seconds=0.5))
    limit_calls = []

    class Guard:
        pass

    def fake_threadpool_limits(*, limits):
        limit_calls.append(limits)
        return Guard()

    monkeypatch.setattr(expansion_module, "_threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(expansion_module, "_WORKER_THREADPOOL_GUARD", None)
    expansion_module._initialize_worker_thread_limits()
    task = expansion_module._CallExtractionTask(
        data_root=str(tmp_path),
        call_id="call",
        audio_path="call.wav",
        rows=((0, "key", "call", 0, 2_000),),
    )

    expansion_module._extract_call_rows(task)
    expansion_module._extract_call_rows(task)

    assert limit_calls == [1]


@pytest.mark.parametrize("fail_extraction", [False, True])
def test_sequential_thread_limit_restores_parent_after_extraction(
    expansion_module, monkeypatch, tmp_path, fail_extraction
):
    events = []

    class Guard:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_error):
            events.append("restore")

    monkeypatch.setattr(
        expansion_module,
        "_threadpool_limits",
        lambda *, limits: (events.append(("limit", limits)), Guard())[1],
    )

    def fake_extract(task):
        events.append("extract")
        if fail_extraction:
            raise RuntimeError("synthetic failure")
        return [
            {
                "source_position": 0,
                "record_key": "key_0",
                "call_id": "call_0",
                "cpp_median_db": np.float32(1.0),
                "cpp_iqr_db": np.float32(2.0),
            }
        ]

    monkeypatch.setattr(expansion_module, "_extract_call_rows", fake_extract)
    manifest = pd.DataFrame(
        {
            "split": ["Training"],
            "record_key": ["key_0"],
            "call_id": ["call_0"],
            "audio_path": ["unused.wav"],
            "crop_start_frame": [0],
            "crop_end_frame": [100],
            "crop_ok": [True],
        }
    )
    if fail_extraction:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            expansion_module.extract_candidate_feature_frame(
                manifest,
                data_root=tmp_path,
                split="Training",
                worker_count=None,
                block_ids=("AF04_CPP",),
            )
    else:
        expansion_module.extract_candidate_feature_frame(
            manifest,
            data_root=tmp_path,
            split="Training",
            worker_count=None,
            block_ids=("AF04_CPP",),
        )
    assert events == [("limit", 1), "enter", "extract", "restore"]


def test_worker_exception_names_call_and_leaves_no_cache_artifact(
    expansion_module, tmp_path
):
    """Catch swallowed worker failures and falsely complete cache pairs."""

    _write_pcm16_wav(tmp_path / "good.wav", _sine_wave(180.0, seconds=0.5))
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training"],
            "record_key": ["good_0", "bad_0"],
            "call_id": ["call_good", "call_bad"],
            "audio_path": ["good.wav", "missing.wav"],
            "crop_start_frame": [0, 0],
            "crop_end_frame": [2_000, 2_000],
            "crop_ok": [True, True],
        }
    )
    cache_root = tmp_path / "cache"
    parquet_path, metadata_path = expansion_module.candidate_cache_paths(
        cache_root, split="Training"
    )

    with pytest.raises(RuntimeError, match="call_bad"):
        expansion_module.build_candidate_feature_cache(
            manifest=manifest,
            data_root=tmp_path,
            cache_root=cache_root,
            split="Training",
            allow_full_extraction=True,
            worker_count=2,
        )

    assert not parquet_path.exists()
    assert not metadata_path.exists()
    assert not list(cache_root.glob("*.tmp")) if cache_root.exists() else True


def test_spawned_process_can_call_parallel_extraction_from_importable_module(
    expansion_module, tmp_path
):
    """Catch worker targets tied to pytest/__main__ or import-time pool creation."""

    del expansion_module
    manifest = _multiprocessing_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path, orient="records")
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import pandas as pd

        sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
        from dcc_data.mission2_acoustic_feature_expansion import extract_candidate_feature_frame

        if __name__ == '__main__':
            manifest = pd.read_json({str(manifest_path)!r}, orient='records')
            frame = extract_candidate_feature_frame(
                manifest,
                data_root=Path({str(tmp_path)!r}),
                split='Training',
                worker_count=2,
            )
            assert frame['record_key'].tolist() == manifest['record_key'].tolist()
            print('SPAWN_PARALLEL_OK')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SPAWN_PARALLEL_OK" in completed.stdout


def test_candidate_alignment_preserves_canonical_keys_and_e05_prefix(expansion_module):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    matrix, feature_names = expansion_module.assemble_single_block_matrix(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        block_id="AF01_ENERGY_DYNAMICS",
    )
    assert feature_names == EXPECTED_E05 + expansion_module.BLOCK_BY_ID[
        "AF01_ENERGY_DYNAMICS"
    ].feature_names
    assert matrix.shape == (len(y), 45)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[:, :42], X)


def test_round2_preparation_uses_af02_47d_reference_and_only_two_candidates(
    expansion_module,
):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_round2_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )
    assert tuple(prepared) == (
        "AF02_47D_REFERENCE",
        "AF04_CPP",
        "AF05_DELTA_MFCC_DYNAMICS",
    )
    assert prepared["AF02_47D_REFERENCE"].feature_names == (
        expansion_module.ROUND2_REFERENCE_FEATURE_NAMES
    )
    assert len(prepared["AF02_47D_REFERENCE"].feature_names) == 47
    assert len(prepared["AF04_CPP"].feature_names) == 49
    assert len(prepared["AF05_DELTA_MFCC_DYNAMICS"].feature_names) == 49
    reference = prepared["AF02_47D_REFERENCE"]
    for payload in prepared.values():
        np.testing.assert_array_equal(payload.train_idx, reference.train_idx)
        np.testing.assert_array_equal(payload.validation_idx, reference.validation_idx)
        assert payload.X_train_scaled.dtype == np.float32
        assert payload.X_evaluation_scaled.dtype == np.float32


def test_round2_cache_contract_keeps_legacy_artifact_separate(expansion_module, tmp_path):
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training"],
            "record_key": ["key_0", "key_1"],
            "call_id": ["call_0", "call_1"],
        }
    )
    round1_paths = expansion_module.candidate_cache_paths(
        tmp_path, split="Training", block_ids=expansion_module.ROUND1_BLOCK_IDS
    )
    round2_paths = expansion_module.candidate_cache_paths(
        tmp_path, split="Training", block_ids=expansion_module.ROUND2_BLOCK_IDS
    )
    assert round1_paths != round2_paths
    assert round1_paths[0].name == "training_acoustic_expansion_2b612b68d75e.parquet"

    metadata = expansion_module._expected_cache_metadata(
        manifest,
        split="Training",
        block_ids=expansion_module.ROUND2_BLOCK_IDS,
    )
    assert metadata["block_ids"] == list(expansion_module.ROUND2_BLOCK_IDS)
    assert metadata["feature_names"] == [
        "cpp_median_db",
        "cpp_iqr_db",
        "delta_mfcc_norm_mean",
        "delta_mfcc_norm_std",
    ]
    assert metadata["feature_dimension"] == 4


def test_round2_reference_contract_and_fit_gates_accept_exact_47d(expansion_module):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_round2_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )
    reference = prepared["AF02_47D_REFERENCE"]
    contract = expansion_module.build_reference_contract(reference)
    assert contract["feature_fingerprint"]
    with pytest.raises(RuntimeError, match="STOP"):
        expansion_module.run_round2_feature_screening_reference(
            reference, allow_fit=False
        )
    with pytest.raises(RuntimeError, match="STOP"):
        expansion_module.run_round2_feature_screening_candidate(
            prepared["AF04_CPP"], block_id="AF04_CPP", allow_fit=False
        )


def test_forbidden_metadata_cannot_enter_model_matrix(expansion_module):
    with pytest.raises(ValueError, match="forbidden"):
        expansion_module.validate_model_feature_names(
            (*EXPECTED_E05, "call_relative_duration")
        )
    with pytest.raises(ValueError, match="forbidden"):
        expansion_module.validate_model_feature_names((*EXPECTED_E05, "startAt_ms"))


def test_fixed_group_split_and_training_only_scaler(expansion_module):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )
    reference = prepared["E05_REFERENCE"]
    train_calls = set(call_ids[reference.train_idx])
    validation_calls = set(call_ids[reference.validation_idx])
    assert train_calls.isdisjoint(validation_calls)
    assert reference.split_audit["call_overlap"] == 0
    np.testing.assert_allclose(
        reference.scaler.mean_,
        X[reference.train_idx].mean(axis=0),
        rtol=1e-6,
        atol=1e-7,
    )
    assert reference.scaler.n_samples_seen_ == len(reference.train_idx)
    for payload in prepared.values():
        np.testing.assert_array_equal(payload.train_idx, reference.train_idx)
        np.testing.assert_array_equal(payload.validation_idx, reference.validation_idx)
        np.testing.assert_array_equal(payload.y_evaluation, y[reference.validation_idx])
        assert payload.X_train_scaled.dtype == np.float32
        assert payload.X_evaluation_scaled.dtype == np.float32


def test_reference_gate_checks_fingerprints_before_accuracy_target(expansion_module):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )["E05_REFERENCE"]
    contract = expansion_module.build_reference_contract(prepared)
    result = expansion_module.run_feature_screening_reference(
        prepared, allow_fit=True
    )
    audit = expansion_module.audit_reference_reproduction(
        result,
        contract,
        accuracy_target=0.859532,
    )
    assert audit["fingerprint_contract_passed"] is True
    assert audit["accuracy_is_reproduction_target_not_gate"] is True
    assert isinstance(audit["accuracy_target_match"], bool)
    assert contract["array_fingerprint_version"] == expansion_module.ARRAY_FINGERPRINT_VERSION
    legacy_contract = dict(contract)
    legacy_contract.pop("array_fingerprint_version")
    legacy_contract.update(
        expansion_module._prepared_array_fingerprints(
            prepared,
            array_fingerprint=expansion_module._legacy_acoustic_array_fingerprint,
        )
    )
    assert expansion_module.audit_reference_reproduction(
        result, legacy_contract
    )["fingerprint_contract_passed"] is True
    historical_audit = dict(result.fit_audit)
    historical_audit.pop("array_fingerprint_version")
    historical_audit.update(
        {
            name: legacy_contract[name]
            for name in ("split_fingerprint", "scaler_fingerprint")
        }
    )
    assert expansion_module.audit_reference_reproduction(
        replace(result, fit_audit=historical_audit), legacy_contract
    )["fingerprint_contract_passed"] is True
    broken_contract = dict(contract)
    broken_contract["split_fingerprint"] = "broken"
    with pytest.raises(ValueError, match="fingerprint"):
        expansion_module.audit_reference_reproduction(
            result, broken_contract, accuracy_target=result.metrics["accuracy"]
        )


def test_same_run_paired_comparison_contract(expansion_module):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )
    reference = expansion_module.run_feature_screening_reference(
        prepared["E05_REFERENCE"], allow_fit=True
    )
    candidate_result = expansion_module.run_feature_screening_candidate(
        prepared["AF01_ENERGY_DYNAMICS"],
        block_id="AF01_ENERGY_DYNAMICS",
        allow_fit=True,
    )
    comparison = expansion_module.compare_feature_candidate_to_reference(
        reference, candidate_result
    )
    paired_total = sum(
        comparison[name]
        for name in (
            "both_correct",
            "both_wrong",
            "reference_wrong_to_candidate_correct",
            "reference_correct_to_candidate_wrong",
        )
    )
    assert paired_total == len(reference.y_true)
    assert comparison["prediction_disagreement_count"] == (
        comparison["reference_wrong_to_candidate_correct"]
        + comparison["reference_correct_to_candidate_wrong"]
    )
    assert comparison["correct_count_delta"] == (
        comparison["reference_wrong_to_candidate_correct"]
        - comparison["reference_correct_to_candidate_wrong"]
    )


def test_competition_scale_actions_require_explicit_switches(expansion_module):
    with pytest.raises(RuntimeError, match="STOP"):
        expansion_module.build_candidate_feature_cache(
            manifest=pd.DataFrame(),
            data_root=REPO_ROOT,
            cache_root=REPO_ROOT / "unused",
            split="Training",
            allow_full_extraction=False,
        )
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidate = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate,
        expected_split_contract=None,
    )["E05_REFERENCE"]
    with pytest.raises(RuntimeError, match="STOP"):
        expansion_module.run_feature_screening_reference(prepared, allow_fit=False)


def test_candidate_cache_partial_and_stale_artifacts_fail_fast(
    expansion_module, tmp_path
):
    manifest = pd.DataFrame(
        {
            "split": ["Training"],
            "record_key": ["Training|call_001|0"],
            "call_id": ["call_001"],
        }
    )
    parquet_path, metadata_path = expansion_module.candidate_cache_paths(
        tmp_path, split="Training"
    )
    parquet_path.write_bytes(b"partial")
    with pytest.raises(FileNotFoundError, match="automatic extraction is disabled"):
        expansion_module.load_candidate_feature_cache(
            manifest=manifest, cache_root=tmp_path, split="Training"
        )
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata differs"):
        expansion_module.load_candidate_feature_cache(
            manifest=manifest, cache_root=tmp_path, split="Training"
        )


def test_candidate_cache_checks_parquet_keys_and_canonical_order(
    expansion_module, tmp_path
):
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training"],
            "record_key": ["key_0", "key_1"],
            "call_id": ["call_0", "call_1"],
        }
    )
    block_ids = ("AF04_CPP",)
    parquet_path, metadata_path = expansion_module.candidate_cache_paths(
        tmp_path, split="Training", block_ids=block_ids
    )
    metadata_path.write_text(
        json.dumps(
            expansion_module._expected_cache_metadata(
                manifest, split="Training", block_ids=block_ids
            )
        ),
        encoding="utf-8",
    )
    good = pd.DataFrame(
        {
            "record_key": ["key_0", "key_1"],
            "call_id": ["call_0", "call_1"],
            "cpp_median_db": np.array([1.0, 2.0], dtype=np.float32),
            "cpp_iqr_db": np.array([3.0, 4.0], dtype=np.float32),
        }
    )
    good.to_parquet(parquet_path, index=False)
    loaded = expansion_module.load_candidate_feature_cache(
        manifest=manifest, cache_root=tmp_path, split="Training", block_ids=block_ids
    )
    assert loaded["record_key"].tolist() == ["key_0", "key_1"]

    changed = good.copy()
    changed.loc[1, "record_key"] = "wrong_key"
    changed.to_parquet(parquet_path, index=False)
    with pytest.raises(ValueError, match="record_key_signature|record_key"):
        expansion_module.load_candidate_feature_cache(
            manifest=manifest, cache_root=tmp_path, split="Training", block_ids=block_ids
        )

    good.iloc[::-1].to_parquet(parquet_path, index=False)
    with pytest.raises(ValueError, match="order|alignment"):
        expansion_module.load_candidate_feature_cache(
            manifest=manifest, cache_root=tmp_path, split="Training", block_ids=block_ids
        )


def test_notebook_is_fresh_kernel_safe_and_all_heavy_switches_default_false():
    assert NOTEBOOK_PATH.is_file(), "acoustic feature expansion Notebook is missing"
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    assert {
        "afe-checkpoint-a",
        "afe-checkpoint-b",
        "afe-checkpoint-c",
        "afe-checkpoint-d",
        "afe-checkpoint-e",
        "afe-checkpoint-f",
        "afe-checkpoint-g",
        "afe-checkpoint-h",
        "afe-checkpoint-i",
        "afe-round2-hypotheses",
    } <= cell_ids
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    ast.parse(code)
    assert "m04_internal_result" not in code
    assert "prepared_internal" not in code
    assert "screening_results" not in code
    assert "m04_iter100_result" not in code
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bool)
    }
    assert assignments["RUN_FEATURE_EXTRACTION"] is False
    assert assignments["RUN_SINGLE_BLOCK_SCREENING"] is False
    run_switches = {name for name in assignments if name.startswith("RUN_")}
    assert run_switches == {
        "RUN_FEATURE_EXTRACTION",
        "RUN_SINGLE_BLOCK_SCREENING",
    }
    assert "prepare_round2_single_block_screening" in code
    assert "ROUND2_BLOCK_IDS" in code


def _fake_feature_result(
    expansion_module,
    *,
    experiment_id,
    feature_names,
    y_true,
    y_pred,
    probabilities,
    record_keys,
):
    row_count = len(y_true)
    indices = np.arange(row_count, dtype=np.int64)
    return SimpleNamespace(
        experiment_id=experiment_id,
        model_config=dict(expansion_module.M04_ITER100_FEATURE_SCREENING_CONFIG),
        feature_names=tuple(feature_names),
        train_shape=(row_count, len(feature_names)),
        evaluation_shape=(row_count, len(feature_names)),
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        metrics={},
        fit_audit={
            "train_idx": indices.copy(),
            "validation_idx": indices.copy(),
            "training_record_keys": np.asarray(record_keys).astype(str),
            "validation_record_keys": np.asarray(record_keys).astype(str),
            "runtime_seconds": 0.0,
        },
    )


def _synthetic_canonical_52d_prepared(expansion_module):
    matrix = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 100.0
    labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
    record_keys = np.asarray([f"key_{index}" for index in range(8)])
    train_idx = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    validation_idx = np.asarray([6, 7], dtype=np.int64)
    split_contract = {
        "total_rows": 8,
        "train_rows": 6,
        "validation_rows": 2,
    }
    prepared = expansion_module._prepare_representation(
        representation_id="AF09_52D_REFERENCE",
        X=matrix,
        y=labels,
        record_keys=record_keys,
        feature_names=expansion_module.CANONICAL_52D_FEATURE_NAMES,
        train_idx=train_idx,
        validation_idx=validation_idx,
        split_audit=split_contract,
    )
    inputs = {
        "canonical_matrix": matrix,
        "canonical_labels": labels,
        "canonical_record_keys": record_keys,
        "canonical_train_idx": train_idx,
        "canonical_validation_idx": validation_idx,
        "expected_split_contract": split_contract,
    }
    return prepared, inputs


def _synthetic_canonical_52d_result(expansion_module):
    prepared, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    result = expansion_module._run_prepared_representation(
        prepared,
        experiment_id=expansion_module.CANONICAL_52D_REFERENCE_ID,
        allow_fit=True,
    )
    return prepared, inputs, result


def test_next_candidate_pool_covers_three_distinct_information_axes(expansion_module):
    assert expansion_module.NEXT_CANDIDATE_BLOCK_IDS == (
        "AF07_MFCC_CROSS_CORRELATION",
        "AF08_COMPETING_PITCH",
        "AF09_MULTIBAND_ENVELOPE_COHERENCE",
    )
    table = expansion_module.candidate_hypothesis_table().set_index("candidate_id")
    assert table.loc["AF07_MFCC_CROSS_CORRELATION", "status"] == "IMPLEMENT"
    assert table.loc["AF08_COMPETING_PITCH", "status"] == "IMPLEMENT"
    assert table.loc["AF09_MULTIBAND_ENVELOPE_COHERENCE", "status"] == "IMPLEMENT"
    assert table.loc["AF05_DELTA_MFCC_DYNAMICS", "status"] == "DROP"
    assert table.loc["AF06_FORMANT_CENTERS", "status"] == "RETIRED"
    assert table.loc["HNR_HARMONICITY", "status"] == "EXCLUDE"
    assert table.loc["MFCC_DIRECTIONAL_DELTA", "status"] == "EXCLUDE"
    assert table.loc["SPECTRAL_LOW_RANK", "status"] == "EXCLUDE"
    assert table["missing_information_axis"].nunique() >= 6
    definitions = expansion_module.implemented_candidate_definition_table()
    assert definitions["candidate_id"].tolist() == list(
        expansion_module.NEXT_CANDIDATE_BLOCK_IDS
    )
    assert definitions["dimension"].tolist() == [2, 2, 3]
    assert definitions["minimum_valid_frames"].tolist() == ["p_eff + 1", 3, 5]
    assert definitions["output_dtype"].tolist() == ["float32"] * 3


@pytest.mark.parametrize(
    ("block_id", "feature_names"),
    [
        (
            "AF07_MFCC_CROSS_CORRELATION",
            ("mfcc_cross_corr_abs_mean", "mfcc_cross_corr_pc1_share"),
        ),
        (
            "AF08_COMPETING_PITCH",
            ("pitch_peak_margin_median", "pitch_peak_margin_iqr"),
        ),
        (
            "AF09_MULTIBAND_ENVELOPE_COHERENCE",
            (
                "subband_envelope_corr_abs_mean",
                "subband_envelope_corr_min",
                "subband_envelope_pc1_share",
            ),
        ),
    ],
)
def test_next_candidate_waveform_outputs_are_deterministic_finite_float32(
    expansion_module, block_id, feature_names
):
    rng = np.random.default_rng(20260917)
    signal = (
        _sine_wave(150.0)
        + 0.35 * _sine_wave(265.0)
        + rng.normal(0.0, 0.01, 6_400).astype(np.float32)
    ).astype(np.float32)
    first = expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))
    second = expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))
    assert tuple(first) == feature_names
    observed = np.asarray(tuple(first.values()))
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    np.testing.assert_array_equal(observed, np.asarray(tuple(second.values())))


@pytest.mark.parametrize("block_id", [
    "AF07_MFCC_CROSS_CORRELATION",
    "AF08_COMPETING_PITCH",
    "AF09_MULTIBAND_ENVELOPE_COHERENCE",
])
@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(2_000, dtype=np.float32),
        np.ones(2_000, dtype=np.float32),
        np.asarray([0.25], dtype=np.float32),
    ],
)
def test_next_candidate_silence_constant_and_short_inputs_use_zero_fallback(
    expansion_module, block_id, signal
):
    block = expansion_module.BLOCK_BY_ID[block_id]
    observed = expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))
    np.testing.assert_array_equal(
        np.asarray(tuple(observed.values())),
        np.zeros(block.output_dimension, dtype=np.float32),
    )


@pytest.mark.parametrize("block_id", [
    "AF07_MFCC_CROSS_CORRELATION",
    "AF08_COMPETING_PITCH",
    "AF09_MULTIBAND_ENVELOPE_COHERENCE",
])
@pytest.mark.parametrize(
    "signal",
    [np.asarray([], dtype=np.float32), np.asarray([np.nan]), np.asarray([np.inf])],
)
def test_next_candidates_fail_closed_on_empty_or_nonfinite_waveforms(
    expansion_module, block_id, signal
):
    with pytest.raises(ValueError, match="waveform"):
        expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))


def test_next_candidate_cache_contract_is_separate_and_exact(
    expansion_module, tmp_path
):
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training"],
            "record_key": ["key_0", "key_1"],
            "call_id": ["call_0", "call_1"],
        }
    )
    next_paths = expansion_module.candidate_cache_paths(
        tmp_path,
        split="Training",
        block_ids=expansion_module.NEXT_CANDIDATE_BLOCK_IDS,
    )
    assert next_paths != expansion_module.candidate_cache_paths(
        tmp_path,
        split="Training",
        block_ids=expansion_module.ROUND2_BLOCK_IDS,
    )
    metadata = expansion_module._expected_cache_metadata(
        manifest,
        split="Training",
        block_ids=expansion_module.NEXT_CANDIDATE_BLOCK_IDS,
    )
    assert metadata["feature_names"] == [
        "mfcc_cross_corr_abs_mean",
        "mfcc_cross_corr_pc1_share",
        "pitch_peak_margin_median",
        "pitch_peak_margin_iqr",
        "subband_envelope_corr_abs_mean",
        "subband_envelope_corr_min",
        "subband_envelope_pc1_share",
    ]
    assert metadata["feature_dimension"] == 7
    assert metadata["dtype"] == "float32"


def test_residual_target_blocks_are_exact_and_use_one_10d_cache(
    expansion_module, tmp_path
):
    assert expansion_module.RESIDUAL_TARGET_BLOCK_IDS == (
        "AF10_SPEECH_ACTIVITY_BOUNDARY",
        "AF11_SPECTRAL_ENTROPY_DYNAMICS",
        "AF12_TEMPORAL_MFCC_CONTRAST",
    )
    manifest = pd.DataFrame(
        {
            "split": ["Training", "Training"],
            "record_key": ["key_0", "key_1"],
            "call_id": ["call_0", "call_1"],
        }
    )
    residual_paths = expansion_module.candidate_cache_paths(
        tmp_path,
        split="Training",
        block_ids=expansion_module.RESIDUAL_TARGET_BLOCK_IDS,
    )
    assert residual_paths != expansion_module.candidate_cache_paths(
        tmp_path,
        split="Training",
        block_ids=expansion_module.NEXT_CANDIDATE_BLOCK_IDS,
    )
    metadata = expansion_module._expected_cache_metadata(
        manifest,
        split="Training",
        block_ids=expansion_module.RESIDUAL_TARGET_BLOCK_IDS,
    )
    assert metadata["feature_names"] == [
        "active_frame_fraction",
        "longest_active_run_fraction",
        "activity_transition_rate",
        "edge_activity_difference",
        "spectral_entropy_mean",
        "spectral_entropy_std",
        "spectral_entropy_iqr",
        "early_middle_distance",
        "middle_late_distance",
        "early_late_distance",
    ]
    assert metadata["feature_dimension"] == 10
    assert metadata["dtype"] == "float32"


def test_speech_activity_boundary_formula_is_hand_checkable(expansion_module):
    active = np.asarray([False, True, True, False, True])
    features = expansion_module._speech_activity_boundary_features_from_mask(active)
    assert features["active_frame_fraction"] == pytest.approx(3.0 / 5.0)
    assert features["longest_active_run_fraction"] == pytest.approx(2.0 / 5.0)
    assert features["activity_transition_rate"] == pytest.approx(3.0 / 4.0)
    assert features["edge_activity_difference"] == pytest.approx(1.0)


def test_spectral_entropy_dynamics_normalizes_and_zero_falls_back(expansion_module):
    power = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    features = expansion_module._spectral_entropy_features_from_power(power)
    assert features["spectral_entropy_mean"] == pytest.approx(1.0 / 3.0)
    assert features["spectral_entropy_std"] == pytest.approx(np.sqrt(2.0) / 3.0)
    assert features["spectral_entropy_iqr"] == pytest.approx(0.5)


def test_temporal_mfcc_contrast_uses_early_middle_late_means(expansion_module):
    mfcc = np.tile(np.arange(6, dtype=np.float64), (20, 1))
    features = expansion_module._temporal_mfcc_contrast_features_from_matrix(mfcc)
    assert features["early_middle_distance"] == pytest.approx(2.0 * np.sqrt(20.0))
    assert features["middle_late_distance"] == pytest.approx(2.0 * np.sqrt(20.0))
    assert features["early_late_distance"] == pytest.approx(4.0 * np.sqrt(20.0))

    short = expansion_module._temporal_mfcc_contrast_features_from_matrix(
        np.zeros((20, 2), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(tuple(short.values())),
        np.zeros(3, dtype=np.float32),
    )
    waveform_short = expansion_module.extract_acoustic_blocks(
        np.asarray([0.25], dtype=np.float32),
        block_ids=("AF12_TEMPORAL_MFCC_CONTRAST",),
    )
    np.testing.assert_array_equal(
        np.asarray(tuple(waveform_short.values())),
        np.zeros(3, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("block_id", "feature_names"),
    [
        (
            "AF10_SPEECH_ACTIVITY_BOUNDARY",
            (
                "active_frame_fraction",
                "longest_active_run_fraction",
                "activity_transition_rate",
                "edge_activity_difference",
            ),
        ),
        (
            "AF11_SPECTRAL_ENTROPY_DYNAMICS",
            (
                "spectral_entropy_mean",
                "spectral_entropy_std",
                "spectral_entropy_iqr",
            ),
        ),
        (
            "AF12_TEMPORAL_MFCC_CONTRAST",
            (
                "early_middle_distance",
                "middle_late_distance",
                "early_late_distance",
            ),
        ),
    ],
)
def test_residual_target_waveform_outputs_are_deterministic_finite_float32(
    expansion_module, block_id, feature_names
):
    signal = np.concatenate(
        [
            np.zeros(1_600, dtype=np.float32),
            _sine_wave(180.0, seconds=0.4),
            np.zeros(1_600, dtype=np.float32),
        ]
    )
    first = expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))
    second = expansion_module.extract_acoustic_blocks(signal, block_ids=(block_id,))
    assert tuple(first) == feature_names
    observed = np.asarray(tuple(first.values()))
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    np.testing.assert_array_equal(observed, np.asarray(tuple(second.values())))


def test_residual_target_extraction_reads_one_wav_for_all_three_blocks(
    expansion_module, monkeypatch
):
    read_count = 0
    waveform = _sine_wave(180.0, seconds=0.8)

    def fake_read_wav_frames(_path):
        nonlocal read_count
        read_count += 1
        return SimpleNamespace(
            sample_rate=8_000,
            waveform=waveform[None, :],
        )

    monkeypatch.setattr(expansion_module, "read_wav_frames", fake_read_wav_frames)
    task = expansion_module._CallExtractionTask(
        data_root="unused",
        call_id="call",
        audio_path="call.wav",
        rows=((0, "key", "call", 0, len(waveform)),),
        block_ids=expansion_module.RESIDUAL_TARGET_BLOCK_IDS,
    )
    rows = expansion_module._extract_call_rows(task)
    assert read_count == 1
    assert tuple(rows[0])[3:] == tuple(
        name
        for block_id in expansion_module.RESIDUAL_TARGET_BLOCK_IDS
        for name in expansion_module.BLOCK_BY_ID[block_id].feature_names
    )


def test_mfcc_cross_correlation_uses_joint_structure_not_coefficient_moments(
    expansion_module,
):
    mfcc = np.zeros((20, 3), dtype=np.float64)
    mfcc[1] = np.asarray([-1.0, 0.0, 1.0])
    mfcc[2] = 2.0 * mfcc[1]
    features = expansion_module._mfcc_cross_correlation_features_from_matrix(mfcc)
    assert features["mfcc_cross_corr_abs_mean"] == pytest.approx(1.0)
    assert features["mfcc_cross_corr_pc1_share"] == pytest.approx(1.0)


def test_mfcc_cross_correlation_falls_back_when_frames_do_not_exceed_p_eff(
    expansion_module,
):
    mfcc = np.zeros((20, 3), dtype=np.float64)
    mfcc[1] = np.asarray([-1.0, 0.0, 1.0])
    mfcc[2] = np.asarray([0.0, 1.0, -1.0])
    mfcc[3] = np.asarray([1.0, -1.0, 0.0])

    features = expansion_module._mfcc_cross_correlation_features_from_matrix(mfcc)

    np.testing.assert_array_equal(
        np.asarray(tuple(features.values())),
        np.zeros(2, dtype=np.float32),
    )


def test_competing_pitch_margin_excludes_primary_octave_equivalents(expansion_module):
    lags = np.asarray([20, 30, 40, 60, 80, 100], dtype=np.int64)
    scores = np.asarray([[0.20, 0.70, 0.95, 0.60, 0.90, 0.55]])
    margins = expansion_module._competing_pitch_margins(scores, lags)
    # Lag 80 is the primary lag's octave equivalent, so lag 30 is the competitor.
    np.testing.assert_allclose(margins, np.asarray([0.25]), rtol=0.0, atol=1e-12)


def test_subband_coherence_formula_detects_identical_envelopes(expansion_module):
    base = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    envelopes = np.vstack([base, 2.0 * base, -base, 0.5 * base])
    features = expansion_module._subband_coherence_features_from_envelopes(envelopes)
    assert features["subband_envelope_corr_abs_mean"] == pytest.approx(1.0)
    assert features["subband_envelope_corr_min"] == pytest.approx(-1.0)
    assert features["subband_envelope_pc1_share"] == pytest.approx(1.0)


def test_frozen_49d_preparation_preserves_prefix_and_appends_one_block_only(
    expansion_module,
):
    X, y, record_keys, call_ids, _ = _synthetic_e05()
    candidates = _candidate_frame(expansion_module, record_keys, call_ids)
    prepared = expansion_module.prepare_frozen_49d_single_block_screening(
        X_e05=X,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidates,
        expected_split_contract=None,
    )
    assert tuple(prepared) == (
        expansion_module.FROZEN_49D_REFERENCE_ID,
        *expansion_module.NEXT_CANDIDATE_BLOCK_IDS,
    )
    reference = prepared[expansion_module.FROZEN_49D_REFERENCE_ID]
    assert reference.feature_names == expansion_module.FROZEN_49D_FEATURE_NAMES
    assert len(reference.feature_names) == 49
    for block_id in expansion_module.NEXT_CANDIDATE_BLOCK_IDS:
        block = expansion_module.BLOCK_BY_ID[block_id]
        payload = prepared[block_id]
        assert payload.feature_names[:49] == expansion_module.FROZEN_49D_FEATURE_NAMES
        assert payload.feature_names[49:] == block.feature_names
        assert len(payload.feature_names) == 49 + block.output_dimension
        np.testing.assert_array_equal(payload.train_idx, reference.train_idx)
        np.testing.assert_array_equal(payload.validation_idx, reference.validation_idx)
        assert payload.scaler.n_samples_seen_ == len(reference.train_idx)
        assert payload.X_train_scaled.dtype == np.float32
        assert payload.X_evaluation_scaled.dtype == np.float32


def test_frozen_49d_residual_analysis_builds_fixed_slices_and_error_ledger(
    expansion_module,
):
    record_keys = np.asarray([f"key_{index}" for index in range(6)])
    tracking = pd.DataFrame(
        {
            "record_key": record_keys,
            "call_id": ["a", "a", "b", "c", "d", "d"],
            "startAt_ms": [0.0, 400.0, 0.0, 0.0, 0.0, 600.0],
            "endAt_ms": [300.0, 1_100.0, 1_500.0, 3_000.0, 5_500.0, 1_400.0],
        }
    )
    truth = np.asarray([0, 1, 0, 1, 0, 1])
    result_47d = _fake_feature_result(
        expansion_module,
        experiment_id="AF02_47D_M04_ITER100_REFERENCE",
        feature_names=expansion_module.ROUND2_REFERENCE_FEATURE_NAMES,
        y_true=truth,
        y_pred=[1, 0, 0, 1, 1, 0],
        probabilities=[
            [0.10, 0.90], [0.90, 0.10], [0.80, 0.20],
            [0.10, 0.90], [0.05, 0.95], [0.95, 0.05],
        ],
        record_keys=record_keys,
    )
    result_49d = _fake_feature_result(
        expansion_module,
        experiment_id="AF04_CPP",
        feature_names=expansion_module.FROZEN_49D_FEATURE_NAMES,
        y_true=truth,
        y_pred=[0, 0, 1, 1, 1, 0],
        probabilities=[
            [0.90, 0.10], [0.95, 0.05], [0.05, 0.95],
            [0.10, 0.90], [0.01, 0.99], [0.99, 0.01],
        ],
        record_keys=record_keys,
    )
    report = expansion_module.build_frozen_49d_residual_analysis(
        result_47d,
        result_49d,
        full_training_tracking=tracking,
    )
    assert report["global_metrics"]["false_positive_count"] == 2
    assert report["global_metrics"]["false_negative_count"] == 2
    assert report["paired_transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "both_wrong": 3,
        "both_correct": 1,
    }
    assert report["fixed_duration_summary"]["slice"].tolist() == [
        "<0.5s", "0.5-1.0s", "1.0-2.0s", "2.0-5.0s", ">=5.0s"
    ]
    ledger = report["error_ledger"]
    assert ledger["record_key"].tolist() == ["key_1", "key_2", "key_4", "key_5"]
    assert set(ledger["transition_type"]) == {"persistent", "regressed"}
    assert {"FP", "FN"} == set(ledger["error_type"])
    assert report["error_group_summary"].set_index("group").loc[
        "short_persistent", "count"
    ] == 2
    assert report["high_confidence_error_count"] == 4


def test_frozen_49d_residual_analysis_rejects_record_key_misalignment(
    expansion_module,
):
    keys = np.asarray(["key_0", "key_1"])
    tracking = pd.DataFrame(
        {
            "record_key": keys,
            "call_id": ["a", "b"],
            "startAt_ms": [0.0, 0.0],
            "endAt_ms": [500.0, 700.0],
        }
    )
    reference = _fake_feature_result(
        expansion_module,
        experiment_id="AF02_47D_M04_ITER100_REFERENCE",
        feature_names=expansion_module.ROUND2_REFERENCE_FEATURE_NAMES,
        y_true=[0, 1], y_pred=[0, 1],
        probabilities=[[0.8, 0.2], [0.2, 0.8]], record_keys=keys,
    )
    candidate = _fake_feature_result(
        expansion_module,
        experiment_id="AF04_CPP",
        feature_names=expansion_module.FROZEN_49D_FEATURE_NAMES,
        y_true=[0, 1], y_pred=[0, 1],
        probabilities=[[0.8, 0.2], [0.2, 0.8]], record_keys=keys[::-1],
    )
    with pytest.raises(ValueError, match="record_key alignment"):
        expansion_module.build_frozen_49d_residual_analysis(
            reference, candidate, full_training_tracking=tracking
        )


def test_notebook_retains_executed_49d_candidate_pool_without_new_boolean_gate():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    assert {
        "afe-49d-residual-analysis",
        "afe-49d-hypothesis-table",
        "afe-next-candidate-extraction",
        "afe-next-candidate-prepare",
        "afe-next-candidate-screening",
        "afe-next-candidate-diagnostics",
    } <= cell_ids
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bool)
    }
    assert {name for name in assignments if name.startswith("RUN_")} == {
        "RUN_FEATURE_EXTRACTION", "RUN_SINGLE_BLOCK_SCREENING"
    }
    assert "NEXT_CANDIDATE_BLOCK_IDS" in code
    assert "build_candidate_feature_cache(" in code
    assert "prepare_frozen_49d_single_block_screening(" in code
    assert "run_frozen_49d_screening_candidate(" in code


def test_notebook_appends_residual_target_round_with_default_off_gates():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell_ids = {cell.get("id") for cell in notebook["cells"]}
    assert {
        "afe-residual-round-intro",
        "afe-residual-round-setup",
        "afe-residual-round-cache",
        "afe-residual-round-prepare",
        "afe-residual-round-screen",
        "afe-residual-round-summary",
        "afe-residual-round-diagnostics",
    } <= cell_ids
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bool)
    }
    assert "ALLOW_RESIDUAL_ROUND_EXTRACTION" not in assignments
    assert assignments["ALLOW_RESIDUAL_ROUND_SCREENING"] is False
    assert "Official Validation" in all_source
    assert "RESIDUAL_TARGET_BLOCK_IDS" in code
    assert "automatic_promotion" in code
    assert "52D_wrong_to_correct" in code
    assert "52D_correct_to_wrong" in code
    assert "speaker0_0.5-1.0s_overlap" in code
    assert "speaker0_1-2s_overlap" in code
    assert "0.5-1.0s_overlap_all_speakers" in code
    assert "overlap_all" in code


def test_canonical_52d_prepared_validator_accepts_complete_source_contract(
    expansion_module,
):
    prepared, inputs = _synthetic_canonical_52d_prepared(expansion_module)

    audit = expansion_module.validate_canonical_52d_prepared(prepared, **inputs)

    assert audit["feature_dimension"] == 52
    assert audit["train_shape"] == (6, 52)
    assert audit["evaluation_shape"] == (2, 52)
    assert audit["scaler_fit_scope"] == "internal_train_only"
    assert audit["feature_fingerprint"]
    assert audit["split_fingerprint"]
    assert audit["scaler_fingerprint"]


@pytest.mark.parametrize(
    "broken_field",
    [
        "feature_names",
        "train_idx",
        "record_keys",
        "labels",
        "scaler",
        "finite_values",
        "train_shape",
    ],
)
def test_canonical_52d_prepared_validator_stops_on_broken_contract(
    expansion_module,
    broken_field,
):
    prepared, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    if broken_field == "feature_names":
        prepared = replace(prepared, feature_names=prepared.feature_names[:-1])
    elif broken_field == "train_idx":
        prepared = replace(prepared, train_idx=prepared.train_idx[::-1])
    elif broken_field == "record_keys":
        prepared = replace(
            prepared,
            training_record_keys=prepared.training_record_keys[::-1],
        )
    elif broken_field == "labels":
        prepared = replace(prepared, y_train=1 - prepared.y_train)
    elif broken_field == "scaler":
        wrong_scaler = StandardScaler().fit(inputs["canonical_matrix"][:2])
        prepared = replace(prepared, scaler=wrong_scaler)
    elif broken_field == "finite_values":
        broken_train = prepared.X_train_scaled.copy()
        broken_train[0, 0] = np.nan
        prepared = replace(prepared, X_train_scaled=broken_train)
    elif broken_field == "train_shape":
        prepared = replace(prepared, X_train_scaled=prepared.X_train_scaled[:-1])

    with pytest.raises(RuntimeError, match="STOP: prepared_52d"):
        expansion_module.validate_canonical_52d_prepared(prepared, **inputs)


def test_canonical_52d_result_is_reused_without_refit(
    expansion_module,
    monkeypatch,
):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)

    def unexpected_refit(*args, **kwargs):
        raise AssertionError("canonical result must not be refit")

    monkeypatch.setattr(
        expansion_module,
        "_run_prepared_representation",
        unexpected_refit,
    )
    resolution = expansion_module.resolve_canonical_52d_reference(
        prepared,
        existing_result=result,
        allow_refit=True,
        expected_accuracy=result.metrics["accuracy"],
        expected_confusion_matrix=result.metrics["confusion_matrix"],
        **inputs,
    )

    assert resolution.result is result
    assert resolution.action == "reused_canonical"
    assert resolution.stale_mismatches == ()


def test_stale_historical_52d_result_is_refit_from_validated_prepared(
    expansion_module,
    monkeypatch,
):
    prepared, inputs, canonical_result = _synthetic_canonical_52d_result(
        expansion_module
    )
    stale_result = replace(
        canonical_result,
        metrics={
            **dict(canonical_result.metrics),
            "accuracy": expansion_module.HISTORICAL_NON_REPRODUCED_52D_ACCURACY,
            "confusion_matrix": [
                list(row)
                for row in expansion_module.HISTORICAL_NON_REPRODUCED_52D_CONFUSION_MATRIX
            ],
        },
    )
    calls = []

    def return_canonical(prepared_payload, *, experiment_id, allow_fit):
        calls.append((prepared_payload, experiment_id, allow_fit))
        return canonical_result

    monkeypatch.setattr(
        expansion_module,
        "_run_prepared_representation",
        return_canonical,
    )
    resolution = expansion_module.resolve_canonical_52d_reference(
        prepared,
        existing_result=stale_result,
        allow_refit=True,
        expected_accuracy=canonical_result.metrics["accuracy"],
        expected_confusion_matrix=canonical_result.metrics["confusion_matrix"],
        **inputs,
    )

    assert resolution.result is canonical_result
    assert resolution.action == "refit_stale_or_missing"
    assert "accuracy" in resolution.stale_mismatches
    assert "confusion_matrix" in resolution.stale_mismatches
    assert calls == [
        (prepared, expansion_module.CANONICAL_52D_REFERENCE_ID, True)
    ]


def test_historical_non_reproduced_52d_cannot_pass_canonical_audit(
    expansion_module,
):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    prepared_audit = expansion_module.validate_canonical_52d_prepared(
        prepared, **inputs
    )
    historical = replace(
        result,
        metrics={
            **dict(result.metrics),
            "accuracy": expansion_module.HISTORICAL_NON_REPRODUCED_52D_ACCURACY,
            "confusion_matrix": [
                list(row)
                for row in expansion_module.HISTORICAL_NON_REPRODUCED_52D_CONFUSION_MATRIX
            ],
        },
    )

    audit = expansion_module.audit_canonical_52d_result(
        historical,
        prepared,
        prepared_audit=prepared_audit,
    )

    assert audit["is_canonical"] is False
    assert "accuracy" in audit["mismatches"]
    assert "confusion_matrix" in audit["mismatches"]


def test_canonical_52d_audit_accepts_one_ulp_accuracy_serialization_rounding(
    expansion_module,
):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    prepared_audit = expansion_module.validate_canonical_52d_prepared(
        prepared, **inputs
    )
    serialized_accuracy = np.nextafter(
        float(result.metrics["accuracy"]),
        0.0,
    )

    audit = expansion_module.audit_canonical_52d_result(
        result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=serialized_accuracy,
        expected_confusion_matrix=result.metrics["confusion_matrix"],
    )

    assert audit["is_canonical"] is True
    assert "accuracy" not in audit["mismatches"]


def test_canonical_52d_audit_uses_counts_for_float32_rate_and_changed_prediction(
    expansion_module,
):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    prepared_audit = expansion_module.validate_canonical_52d_prepared(
        prepared, **inputs
    )
    rounded_accuracy = float(
        np.nextafter(np.float32(result.metrics["accuracy"]), np.float32(1.0))
    )
    accepted = expansion_module.audit_canonical_52d_result(
        result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=rounded_accuracy,
        expected_confusion_matrix=result.metrics["confusion_matrix"],
    )
    assert accepted["is_canonical"] is True

    changed_prediction = result.y_pred.copy()
    changed_prediction[0] = 1 - changed_prediction[0]
    changed = expansion_module.audit_canonical_52d_result(
        replace(result, y_pred=changed_prediction),
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=rounded_accuracy,
        expected_confusion_matrix=result.metrics["confusion_matrix"],
    )
    assert changed["is_canonical"] is False
    assert "prediction_confusion_matrix" in changed["mismatches"]
    assert changed["environment_versions"]
    assert changed["row_fingerprint"]
    with pytest.raises(RuntimeError, match="reproducibility diagnostic") as error:
        expansion_module.resolve_canonical_52d_reference(
            prepared,
            existing_result=replace(result, y_pred=changed_prediction),
            allow_refit=False,
            expected_accuracy=rounded_accuracy,
            expected_confusion_matrix=result.metrics["confusion_matrix"],
            **inputs,
        )
    assert "prediction_fingerprint" in str(error.value)

    invalid_reference = expansion_module.audit_canonical_52d_result(
        result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=rounded_accuracy,
        expected_confusion_matrix=[],
    )
    assert "expected_confusion_matrix" in invalid_reference["mismatches"]


def test_canonical_52d_audit_accepts_explicit_legacy_array_hashes(expansion_module):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    prepared_audit = expansion_module.validate_canonical_52d_prepared(
        prepared, **inputs
    )
    old_audit = dict(result.fit_audit)
    old_audit.pop("array_fingerprint_version")
    old_audit.update(
        expansion_module._prepared_array_fingerprints(
            prepared,
            array_fingerprint=expansion_module._legacy_acoustic_array_fingerprint,
        )
    )
    historical_result = replace(result, fit_audit=old_audit)
    audit = expansion_module.audit_canonical_52d_result(
        historical_result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=result.metrics["accuracy"],
        expected_confusion_matrix=result.metrics["confusion_matrix"],
    )
    assert audit["is_canonical"] is True
    assert audit["array_fingerprint_version"] == "acoustic-array-legacy-v0"
    old_audit["split_fingerprint"] = "corrupted"
    rejected = expansion_module.audit_canonical_52d_result(
        replace(result, fit_audit=old_audit),
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=result.metrics["accuracy"],
        expected_confusion_matrix=result.metrics["confusion_matrix"],
    )
    assert "split_fingerprint" in rejected["mismatches"]


def test_canonical_52d_audit_rejects_invalid_probability_payloads(expansion_module):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    prepared_audit = expansion_module.validate_canonical_52d_prepared(
        prepared, **inputs
    )
    bad_payloads = {
        "nan": np.full_like(result.probabilities, np.nan),
        "row_sum": np.full_like(result.probabilities, 0.2),
        "shape": result.probabilities[:, :1],
        "range": np.tile(np.array([1.1, -0.1]), (len(result.y_true), 1)),
    }
    for name, probabilities in bad_payloads.items():
        audit = expansion_module.audit_canonical_52d_result(
            replace(result, probabilities=probabilities),
            prepared,
            prepared_audit=prepared_audit,
            expected_accuracy=result.metrics["accuracy"],
            expected_confusion_matrix=result.metrics["confusion_matrix"],
        )
        assert "probabilities" in audit["mismatches"], name


@pytest.mark.parametrize("bad_kind", ["classes", "nan", "row_sum", "range"])
def test_prepared_representation_rejects_bad_model_probabilities(
    expansion_module, monkeypatch, bad_kind
):
    prepared, _inputs = _synthetic_canonical_52d_prepared(expansion_module)

    class FakeMLP:
        def __init__(self, **params):
            self.params = params
            self.classes_ = np.array([1, 0] if bad_kind == "classes" else [0, 1])
            self.n_iter_ = 1
            self.loss_curve_ = (0.2,)

        def fit(self, _X, _y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X):
            if bad_kind == "nan":
                return np.full((len(X), 2), np.nan)
            if bad_kind == "row_sum":
                return np.full((len(X), 2), 0.2)
            if bad_kind == "range":
                return np.tile(np.array([1.1, -0.1]), (len(X), 1))
            return np.tile(np.array([0.8, 0.2]), (len(X), 1))

        def get_params(self, deep=False):
            return self.params

    monkeypatch.setattr(expansion_module, "MLPClassifier", FakeMLP)
    with pytest.raises(ValueError, match="classes|predict_proba"):
        expansion_module._run_prepared_representation(
            prepared, experiment_id="SYNTHETIC", allow_fit=True
        )


def test_screening_prepare_signatures_require_explicit_split_contract(expansion_module):
    for name in (
        "prepare_single_block_screening",
        "prepare_round2_single_block_screening",
        "prepare_frozen_49d_single_block_screening",
    ):
        parameter = inspect.signature(getattr(expansion_module, name)).parameters[
            "expected_split_contract"
        ]
        assert parameter.default is inspect.Parameter.empty, name


def test_invalid_prepared_52d_stops_before_stale_refit(
    expansion_module,
    monkeypatch,
):
    prepared, inputs, result = _synthetic_canonical_52d_result(expansion_module)
    broken = replace(prepared, feature_names=prepared.feature_names[:-1])

    def unexpected_refit(*args, **kwargs):
        raise AssertionError("invalid prepared_52d must stop before refit")

    monkeypatch.setattr(
        expansion_module,
        "_run_prepared_representation",
        unexpected_refit,
    )
    with pytest.raises(RuntimeError, match="STOP: prepared_52d"):
        expansion_module.resolve_canonical_52d_reference(
            broken,
            existing_result=result,
            allow_refit=True,
            expected_accuracy=result.metrics["accuracy"],
            expected_confusion_matrix=result.metrics["confusion_matrix"],
            **inputs,
        )


def test_model_fit_audit_stores_lightweight_prediction_provenance(
    expansion_module,
):
    _, _, result = _synthetic_canonical_52d_result(expansion_module)
    audit = result.fit_audit

    assert audit["prediction_shape"] == result.y_pred.shape
    assert audit["prediction_fingerprint"] == expansion_module._array_fingerprint(
        result.y_pred
    )
    assert audit["probability_shape"] == result.probabilities.shape
    assert audit["probability_fingerprint"] == expansion_module._array_fingerprint(
        result.probabilities
    )
    assert audit["confusion_matrix"] == result.metrics["confusion_matrix"]
    assert audit["accuracy"] == result.metrics["accuracy"]
    assert audit["n_iter"] == result.model.n_iter_
    assert audit["loss_curve"] == tuple(float(value) for value in result.model.loss_curve_)
    assert set(audit["environment_versions"]) == {
        "python",
        "scikit_learn",
        "numpy",
        "scipy",
    }
    assert audit["model_params"] == result.model.get_params(deep=False)
    assert {
        "feature_fingerprint",
        "split_fingerprint",
        "scaler_fingerprint",
    } <= set(audit)
    assert set(audit["source_modules"]) == {
        "mission2_acoustic_feature_expansion",
        "mission2_model_experiments",
    }
    for source in audit["source_modules"].values():
        source_path = Path(source["path"])
        assert source_path.is_file()
        assert source["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert "y_pred" not in audit
    assert "probabilities" not in audit


def test_residual_round_combined_cache_is_strict_load_only():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cache_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("id") == "afe-residual-round-cache"
    )
    source = "".join(cache_cell.get("source", []))

    assert "load_candidate_feature_cache(" in source
    assert "build_candidate_feature_cache(" not in source
    assert "ALLOW_RESIDUAL_ROUND_EXTRACTION" not in source


def test_residual_combination_preparation_builds_exact_four_from_52d_prefix(
    expansion_module,
    monkeypatch,
):
    prepared_52d, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    row_count = len(inputs["canonical_record_keys"])
    call_ids = np.asarray([f"call_{index}" for index in range(row_count)])
    combined_feature_names = tuple(
        feature_name
        for block_id in expansion_module.RESIDUAL_TARGET_BLOCK_IDS
        for feature_name in expansion_module.BLOCK_BY_ID[block_id].feature_names
    )
    candidate_frame = pd.DataFrame(
        {
            "record_key": inputs["canonical_record_keys"],
            "call_id": call_ids,
            **{
                feature_name: np.arange(row_count, dtype=np.float32) + offset
                for offset, feature_name in enumerate(combined_feature_names)
            },
        }
    ).iloc[::-1].reset_index(drop=True)

    def extraction_must_not_run(*args, **kwargs):
        raise AssertionError("combination preparation must use only the loaded cache")

    monkeypatch.setattr(
        expansion_module,
        "build_candidate_feature_cache",
        extraction_must_not_run,
    )
    prepared = expansion_module.prepare_residual_combination_screening(
        prepared_52d=prepared_52d,
        X_52d=inputs["canonical_matrix"],
        y=inputs["canonical_labels"],
        record_keys=inputs["canonical_record_keys"],
        call_ids=call_ids,
        candidate_frame=candidate_frame,
        canonical_train_idx=inputs["canonical_train_idx"],
        canonical_validation_idx=inputs["canonical_validation_idx"],
        expected_split_contract=inputs["expected_split_contract"],
    )

    assert tuple(prepared) == (
        "AF10_AF11_COMBINATION",
        "AF10_AF12_COMBINATION",
        "AF11_AF12_COMBINATION",
        "AF10_AF11_AF12_COMBINATION",
    )
    assert [len(payload.feature_names) for payload in prepared.values()] == [
        59,
        59,
        58,
        62,
    ]
    for combination_id, payload in prepared.items():
        block_ids = expansion_module.RESIDUAL_COMBINATION_SPECS[combination_id]
        expected_suffix = tuple(
            feature_name
            for block_id in block_ids
            for feature_name in expansion_module.BLOCK_BY_ID[block_id].feature_names
        )
        assert payload.feature_names[:52] == expansion_module.CANONICAL_52D_FEATURE_NAMES
        assert payload.feature_names[52:] == expected_suffix
        assert np.array_equal(payload.train_idx, prepared_52d.train_idx)
        assert np.array_equal(payload.validation_idx, prepared_52d.validation_idx)
        assert int(payload.scaler.n_samples_seen_) == len(prepared_52d.train_idx)
        assert np.array_equal(
            payload.X_train_scaled[:, :52],
            prepared_52d.X_train_scaled,
        )
        assert np.array_equal(
            payload.X_evaluation_scaled[:, :52],
            prepared_52d.X_evaluation_scaled,
        )


def test_residual_combination_preparation_stops_on_cache_call_misalignment(
    expansion_module,
):
    prepared_52d, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    row_count = len(inputs["canonical_record_keys"])
    call_ids = np.asarray([f"call_{index}" for index in range(row_count)])
    combined_feature_names = tuple(
        feature_name
        for block_id in expansion_module.RESIDUAL_TARGET_BLOCK_IDS
        for feature_name in expansion_module.BLOCK_BY_ID[block_id].feature_names
    )
    candidate_frame = pd.DataFrame(
        {
            "record_key": inputs["canonical_record_keys"],
            "call_id": call_ids,
            **{
                feature_name: np.ones(row_count, dtype=np.float32)
                for feature_name in combined_feature_names
            },
        }
    )
    candidate_frame.loc[0, "call_id"] = "wrong_call"

    with pytest.raises(ValueError, match="call_id alignment mismatch"):
        expansion_module.prepare_residual_combination_screening(
            prepared_52d=prepared_52d,
            X_52d=inputs["canonical_matrix"],
            y=inputs["canonical_labels"],
            record_keys=inputs["canonical_record_keys"],
            call_ids=call_ids,
            candidate_frame=candidate_frame,
            canonical_train_idx=inputs["canonical_train_idx"],
            canonical_validation_idx=inputs["canonical_validation_idx"],
            expected_split_contract=inputs["expected_split_contract"],
        )


@pytest.mark.parametrize("invalid_label", [2, -1, 0.5])
def test_residual_combination_rejects_nonbinary_canonical_labels(
    expansion_module, invalid_label
):
    prepared_52d, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    labels = inputs["canonical_labels"].astype(np.float64)
    labels[0] = invalid_label
    prepared_52d = replace(
        prepared_52d,
        y_train=labels[prepared_52d.train_idx],
        y_evaluation=labels[prepared_52d.validation_idx],
    )
    row_count = len(labels)
    cache_names = tuple(
        name
        for block_id in expansion_module.RESIDUAL_TARGET_BLOCK_IDS
        for name in expansion_module.BLOCK_BY_ID[block_id].feature_names
    )
    candidate = pd.DataFrame(
        {
            "record_key": inputs["canonical_record_keys"],
            "call_id": [f"call_{index}" for index in range(row_count)],
            **{
                name: np.ones(row_count, dtype=np.float32) for name in cache_names
            },
        }
    )
    with pytest.raises(ValueError, match="binary"):
        expansion_module.prepare_residual_combination_screening(
            prepared_52d=prepared_52d,
            X_52d=inputs["canonical_matrix"],
            y=labels,
            record_keys=inputs["canonical_record_keys"],
            call_ids=candidate["call_id"].to_numpy(),
            candidate_frame=candidate,
            canonical_train_idx=inputs["canonical_train_idx"],
            canonical_validation_idx=inputs["canonical_validation_idx"],
            expected_split_contract=inputs["expected_split_contract"],
        )


def test_residual_combination_preparation_stops_on_noncanonical_prepared_split(
    expansion_module,
):
    prepared_52d, inputs = _synthetic_canonical_52d_prepared(expansion_module)
    row_count = len(inputs["canonical_record_keys"])
    call_ids = np.asarray([f"call_{index}" for index in range(row_count)])
    combined_feature_names = tuple(
        feature_name
        for block_id in expansion_module.RESIDUAL_TARGET_BLOCK_IDS
        for feature_name in expansion_module.BLOCK_BY_ID[block_id].feature_names
    )
    candidate_frame = pd.DataFrame(
        {
            "record_key": inputs["canonical_record_keys"],
            "call_id": call_ids,
            **{
                feature_name: np.ones(row_count, dtype=np.float32)
                for feature_name in combined_feature_names
            },
        }
    )
    stale_prepared = replace(
        prepared_52d,
        train_idx=prepared_52d.train_idx[::-1],
    )

    with pytest.raises(RuntimeError, match="STOP: prepared_52d"):
        expansion_module.prepare_residual_combination_screening(
            prepared_52d=stale_prepared,
            X_52d=inputs["canonical_matrix"],
            y=inputs["canonical_labels"],
            record_keys=inputs["canonical_record_keys"],
            call_ids=call_ids,
            candidate_frame=candidate_frame,
            canonical_train_idx=inputs["canonical_train_idx"],
            canonical_validation_idx=inputs["canonical_validation_idx"],
            expected_split_contract=inputs["expected_split_contract"],
        )


def test_notebook_residual_combination_round_is_bounded_and_retains_outputs():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    expected_ids = (
        "afe-residual-combination-prepare",
        "afe-residual-combination-screen",
        "afe-residual-combination-summary",
        "afe-residual-combination-diagnostics",
    )
    cells = {
        cell.get("id"): cell
        for cell in notebook["cells"]
        if cell.get("id") in expected_ids
    }

    assert tuple(cells) == expected_ids
    for cell in cells.values():
        assert isinstance(cell.get("execution_count"), int)
        assert cell["execution_count"] > 0
        assert cell.get("outputs")

    code = "\n".join("".join(cells[cell_id]["source"]) for cell_id in expected_ids)
    assert "ALLOW_RESIDUAL_COMBINATION_SCREENING = False" in code
    assert "prepare_residual_combination_screening(" in code
    assert "build_candidate_feature_cache(" not in code
    assert "read_wav_frames(" not in code
    assert "Official Validation" not in code
    assert "automatic_promotion': False" in code
    for combination_id in (
        "AF10_AF11_COMBINATION",
        "AF10_AF12_COMBINATION",
        "AF11_AF12_COMBINATION",
        "AF10_AF11_AF12_COMBINATION",
    ):
        assert combination_id in code


def test_notebook_52d_reference_is_not_reused_by_global_name_only():
    """A stale in-memory result must go through the canonical contract guard."""

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "if 'result_52d' not in globals():" not in code
    assert "residual_result_52d = result_52d" not in code
    assert code.count("resolve_canonical_52d_reference(") >= 3
