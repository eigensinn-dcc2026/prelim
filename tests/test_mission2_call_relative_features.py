"""Tests for call-relative pitch/energy/duration features.

Follow-up to the pitch/energy pilot (see reports/mission2/... local pilot):
absolute f0_mean barely moved accuracy because it already correlates -0.38
with spectral_centroid_mean -- the GBM had a redundant proxy for it. The
between-CALL variance (different person, different mic gain) swamps the
between-SPEAKER signal within a call. z-scoring each utterance's f0_mean/
rms_p90/duration_sec against *that call's own* mean/std removes the
between-call nuisance variance and should isolate "is this speaker higher or
lower than the call's own baseline" -- without ever touching the speaker
label, so it's reproducible at inference time.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import (
    CALL_RELATIVE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    N_FEATURES,
    N_UTTERANCE_FEATURES,
    UTTERANCE_FEATURE_COLUMNS,
    add_call_relative_features,
    build_seoul_training_dataset,
    utterance_feature_vector,
)

SAMPLE_RATE = 8000


def test_feature_columns_grew_by_three_call_relative_dims() -> None:
    assert N_UTTERANCE_FEATURES == 61
    assert N_FEATURES == 64
    assert len(FEATURE_COLUMNS) == N_FEATURES
    assert FEATURE_COLUMNS == UTTERANCE_FEATURE_COLUMNS + CALL_RELATIVE_FEATURE_COLUMNS
    assert CALL_RELATIVE_FEATURE_COLUMNS == (
        "f0_mean_rel_to_call",
        "rms_p90_rel_to_call",
        "duration_sec_rel_to_call",
    )


def test_utterance_feature_vector_still_returns_only_the_context_free_dims() -> None:
    segment = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32)
    vector = utterance_feature_vector(segment, SAMPLE_RATE, duration_sec=1.0)
    assert vector.shape == (N_UTTERANCE_FEATURES,)


def test_add_call_relative_features_zscores_against_the_calls_own_rows() -> None:
    rows = [
        {"f0_mean": 100.0, "rms_p90": 0.1, "duration_sec": 1.0},
        {"f0_mean": 200.0, "rms_p90": 0.2, "duration_sec": 2.0},
        {"f0_mean": 300.0, "rms_p90": 0.3, "duration_sec": 3.0},
    ]

    result = add_call_relative_features(rows)

    for source_col, rel_col in zip(
        ("f0_mean", "rms_p90", "duration_sec"),
        ("f0_mean_rel_to_call", "rms_p90_rel_to_call", "duration_sec_rel_to_call"),
    ):
        values = np.array([row[source_col] for row in rows])
        mean, std = values.mean(), values.std()
        expected = (values - mean) / (std + 1e-6)
        actual = np.array([row[rel_col] for row in result])
        assert np.allclose(actual, expected), rel_col


def test_add_call_relative_features_is_zero_for_a_single_utterance_call() -> None:
    rows = [{"f0_mean": 123.0, "rms_p90": 0.05, "duration_sec": 0.7}]
    result = add_call_relative_features(rows)
    assert result[0]["f0_mean_rel_to_call"] == 0.0
    assert result[0]["rms_p90_rel_to_call"] == 0.0
    assert result[0]["duration_sec_rel_to_call"] == 0.0


def test_add_call_relative_features_handles_empty_list() -> None:
    assert add_call_relative_features([]) == []


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


def test_call_relative_features_are_scoped_per_call_not_global(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    utterances = [(0, 400, 0), (500, 1200, 1), (1300, 2500, 0), (2600, 3400, 1)]
    for i, stem in enumerate(["callA", "callB"]):
        wav_relative = f"chunk/Training/1.원천데이터/TS_서울_구급/{stem}.wav"
        json_relative = f"chunk/Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
        _write_call(root, wav_relative, json_relative, utterances, seed=i)

    rows = [
        ManifestRow(
            "Training", stem,
            f"chunk/Training/1.원천데이터/TS_서울_구급/{stem}.wav",
            f"chunk/Training/2.라벨링데이터/TL_서울_구급/{stem}.json",
        )
        for stem in ["callA", "callB"]
    ]

    _, feature_rows, failures = build_seoul_training_dataset(rows, root, n_calls=100, seed=42)
    assert failures == []

    df = pd.DataFrame(feature_rows)
    for call_id, group in df.groupby("call_id"):
        f0 = group["f0_mean"].to_numpy(dtype=np.float64)
        expected = (f0 - f0.mean()) / (f0.std() + 1e-6)
        actual = group["f0_mean_rel_to_call"].to_numpy(dtype=np.float64)
        assert np.allclose(actual, expected, atol=1e-5), call_id

    for row in feature_rows:
        assert set(FEATURE_COLUMNS) <= set(row)
        assert np.isfinite(row["f0_mean_rel_to_call"])
