"""Tests for the harmonicity (overlap-detection) features.

See reports/mission2/m2_error_analysis.md 1-1절: overlapping utterances
(43.9% of val) account for 57.3% of all errors, with an 8.4pp accuracy gap
vs. non-overlapping ones. This targets that gap with a feature computable
from the segment's own audio alone (no other utterance's timestamp needed,
so it doesn't touch the contest's "no order-derived features" rule): two
overlapping voices with different, generally-unrelated F0s degrade a single
frame's autocorrelation peak relative to a lone periodic voice.
"""
from __future__ import annotations

import numpy as np

from dcc_data.mission2_features import (
    FEATURE_COLUMNS,
    N_UTTERANCE_FEATURES,
    harmonicity_feature_vector,
    utterance_feature_vector,
)

SAMPLE_RATE = 8000


def _tone(freq_hz: float, duration_s: float, *, sr: int = SAMPLE_RATE, amplitude: float = 0.35) -> np.ndarray:
    n = max(1, int(sr * duration_s))
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_harmonicity_is_high_for_a_single_pure_tone() -> None:
    segment = _tone(150.0, 1.0, amplitude=0.5)
    mean, min_, std = harmonicity_feature_vector(segment, SAMPLE_RATE)
    assert mean > 0.8
    assert min_ > 0.8


def test_harmonicity_drops_when_two_unrelated_pitches_overlap() -> None:
    single = _tone(150.0, 1.0, amplitude=0.5)
    overlapped = _tone(150.0, 1.0) + _tone(210.0, 1.0)

    single_mean, _, _ = harmonicity_feature_vector(single, SAMPLE_RATE)
    overlap_mean, _, _ = harmonicity_feature_vector(overlapped, SAMPLE_RATE)

    assert overlap_mean < single_mean - 0.1


def test_harmonicity_is_near_zero_for_pure_silence() -> None:
    segment = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mean, min_, std = harmonicity_feature_vector(segment, SAMPLE_RATE)
    assert mean == 0.0
    assert min_ == 0.0
    assert std == 0.0


def test_harmonicity_feature_vector_handles_segment_shorter_than_n_fft() -> None:
    rng = np.random.default_rng(0)
    short_segment = rng.standard_normal(50).astype(np.float32)
    vector = harmonicity_feature_vector(short_segment, SAMPLE_RATE)
    assert vector.shape == (3,)
    assert np.isfinite(vector).all()


def test_utterance_feature_vector_grew_by_three_harmonicity_dims() -> None:
    assert N_UTTERANCE_FEATURES == 61  # 58 + 3 harmonicity dims
    assert FEATURE_COLUMNS[58:61] == ("harmonicity_mean", "harmonicity_min", "harmonicity_std")

    segment = _tone(150.0, 1.0, amplitude=0.5)
    vector = utterance_feature_vector(segment, SAMPLE_RATE, duration_sec=1.0)
    assert vector.shape == (N_UTTERANCE_FEATURES,)
    assert np.isfinite(vector).all()
