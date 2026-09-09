"""Tests for the pitch/energy-percentile features added on top of the M2
baseline's original 53-dim acoustic vector.

See reports/mission2/m2_error_analysis.md section 4 (다음 방향 제안 #3):
the 대원/신고자 recall gap is largest in the 0.5-1s duration bucket, and the
report's proposed fix is pitch(F0) + energy-percentile + silence-ratio
signals -- none of which the original 53 dims (MFCC/RMS/ZCR/spectral) carry
directly.
"""
from __future__ import annotations

import numpy as np

from dcc_data.mission2_features import (
    N_UTTERANCE_FEATURES,
    HOP_LENGTH,
    N_FFT,
    UTTERANCE_FEATURE_COLUMNS,
    pitch_energy_feature_vector,
    utterance_feature_vector,
)

SAMPLE_RATE = 8000  # matches the team's confirmed real-data format


def _tone(freq_hz: float, duration_s: float, *, sr: int = SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    n = max(1, int(sr * duration_s))
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_pitch_energy_feature_vector_detects_known_f0_in_a_pure_tone() -> None:
    segment = _tone(150.0, 1.0)
    vector = pitch_energy_feature_vector(segment, SAMPLE_RATE)

    f0_mean, f0_std, voiced_ratio, rms_p10, rms_p90 = vector
    assert abs(f0_mean - 150.0) < 5.0
    assert voiced_ratio > 0.8  # a pure tone is "voiced" (energetic) almost everywhere
    # A pure constant-amplitude tone has ~flat frame energy -- p10 == p90 here
    # is correct, not a bug; the p10 < p90 case is covered separately below.
    assert rms_p90 >= rms_p10 >= 0.0


def test_pitch_energy_feature_vector_energy_percentiles_track_varying_loudness() -> None:
    quiet = _tone(150.0, 0.5, amplitude=0.02)
    loud = _tone(150.0, 0.5, amplitude=0.5)
    segment = np.concatenate([quiet, loud])

    _, _, _, rms_p10, rms_p90 = pitch_energy_feature_vector(segment, SAMPLE_RATE)
    assert rms_p90 > rms_p10 >= 0.0


def test_pitch_energy_feature_vector_reports_no_pitch_for_pure_silence() -> None:
    segment = np.zeros(SAMPLE_RATE, dtype=np.float32)
    vector = pitch_energy_feature_vector(segment, SAMPLE_RATE)

    f0_mean, f0_std, voiced_ratio, rms_p10, rms_p90 = vector
    assert f0_mean == 0.0
    assert f0_std == 0.0
    assert voiced_ratio == 0.0
    assert rms_p10 == 0.0
    assert rms_p90 == 0.0


def test_pitch_energy_feature_vector_handles_segment_shorter_than_n_fft() -> None:
    rng = np.random.default_rng(0)
    short_segment = rng.standard_normal(50).astype(np.float32)  # well under N_FFT
    vector = pitch_energy_feature_vector(short_segment, SAMPLE_RATE)

    assert vector.shape == (5,)
    assert np.isfinite(vector).all()


def test_utterance_feature_vector_grew_by_five_pitch_energy_dims() -> None:
    assert N_UTTERANCE_FEATURES == 58  # 53 original + 5 pitch/energy dims
    assert UTTERANCE_FEATURE_COLUMNS[-5:] == ("f0_mean", "f0_std", "voiced_ratio", "rms_p10", "rms_p90")

    segment = _tone(150.0, 1.0)
    vector = utterance_feature_vector(segment, SAMPLE_RATE, duration_sec=1.0)
    assert vector.shape == (N_UTTERANCE_FEATURES,)
    assert np.isfinite(vector).all()
