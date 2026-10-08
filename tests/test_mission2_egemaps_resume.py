"""Synthetic contracts only; never reads competition audio or runs openSMILE."""

from __future__ import annotations

import json
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

from dcc_data import mission2_egemaps_resume as resume
from dcc_data import mission2_standard_representation as standard


NAMES = tuple(f"feature_{index}" for index in range(88))
KEYS = np.asarray(("k0", "k1", "k2"))
CALLS = np.asarray(("a", "a", "b"))
FINGERPRINT = "test-clean-fingerprint"


def _manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "record_key": KEYS, "call_id": CALLS,
        "audio_path": ("a.wav", "a.wav", "b.wav"),
        "crop_start_frame": (0, 10, 0),
        "crop_end_frame": (10, 20, 10),
    })


def _partial(tmp_path, *, legacy: bool = False):
    root = tmp_path / "partial"
    matrix, mask, _ = resume._create_partial(root, KEYS, CALLS, NAMES, FINGERPRINT)
    matrix[:2] = 1
    mask[:2] = True
    resume._checkpoint(root, matrix, mask, KEYS, NAMES, FINGERPRINT, {})
    if legacy:
        # The actual RAM recovery saved identity arrays with object dtype.
        for name, values in (("record_keys.npy", KEYS), ("call_ids.npy", CALLS)):
            with (root / name).open("wb") as handle:
                np.save(handle, np.asarray(values, dtype=object))
        path = root / "recovery_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        for name in (
            "source_provenance", "short_audio_fallback", "padding_target_samples",
            "sample_rate", "padded_row_count", "padded_record_keys",
            "padded_original_sample_lengths",
        ):
            metadata.pop(name)
        path.write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_finite_short_crop_is_not_padded(monkeypatch):
    lengths = []

    def extract(waveform, sample_rate):
        lengths.append(len(waveform))
        return np.ones(88, dtype=np.float32), NAMES

    monkeypatch.setattr(resume, "extract_egemaps", extract)
    vector, names, padded = resume.extract_with_short_audio_fallback(
        np.zeros(560, dtype=np.float32), 8_000,
    )
    assert lengths == [560]
    assert vector.shape == (88,) and names == NAMES and padded is False


def test_warned_short_nonfinite_retries_once_at_800(monkeypatch):
    lengths = []

    def extract(waveform, sample_rate):
        lengths.append(len(waveform))
        if len(waveform) == 80:
            warnings.warn("Segment too short, filling with NaN.", UserWarning)
            raise ValueError("eGeMAPSv02 returned NaN or inf; audio may be too short")
        assert np.array_equal(waveform[:80], np.ones(80, dtype=np.float32))
        assert np.count_nonzero(waveform[80:]) == 0
        return np.ones(88, dtype=np.float32), NAMES

    monkeypatch.setattr(resume, "extract_egemaps", extract)
    _, _, padded = resume.extract_with_short_audio_fallback(
        np.ones(80, dtype=np.float32), 8_000,
    )
    assert lengths == [80, 800] and padded is True


def test_padded_retry_nonfinite_stops_after_one_attempt(monkeypatch):
    lengths = []

    def extract(waveform, sample_rate):
        lengths.append(len(waveform))
        if len(waveform) == 80:
            warnings.warn("Segment too short, filling with NaN.", UserWarning)
        raise ValueError("eGeMAPSv02 returned NaN or inf; audio may be too short")

    monkeypatch.setattr(resume, "extract_egemaps", extract)
    with pytest.raises(ValueError, match="NaN or inf"):
        resume.extract_with_short_audio_fallback(np.zeros(80, dtype=np.float32), 8_000)
    assert lengths == [80, 800]


@pytest.mark.parametrize("artifact", ("record_keys", "call_ids"))
def test_partial_key_or_call_order_mismatch_stops(tmp_path, artifact):
    root = _partial(tmp_path)
    path = root / f"{artifact}.npy"
    with path.open("wb") as handle:
        np.save(handle, (KEYS if artifact == "record_keys" else CALLS)[::-1])
    with pytest.raises(ValueError, match="record_key|call_id"):
        resume._validate_partial(root, KEYS, CALLS, NAMES, FINGERPRINT)


def test_partial_completed_mask_and_finite_mismatch_stops(tmp_path):
    root = _partial(tmp_path)
    matrix = np.load(root / "matrix_88_partial.npy", mmap_mode="r+")
    matrix[2] = 1
    matrix.flush()
    with pytest.raises(ValueError, match="completed_mask/finite"):
        resume._validate_partial(root, KEYS, CALLS, NAMES, FINGERPRINT)


def test_valid_legacy_partial_skips_completed_call_and_publishes_provenance(
    tmp_path, monkeypatch, capsys,
):
    root = _partial(tmp_path, legacy=True)
    final = tmp_path / "final.npz"
    reads = []
    extracts = []
    monkeypatch.setattr(resume, "get_egemaps_feature_names", lambda: NAMES)
    monkeypatch.setattr(standard, "get_egemaps_feature_names", lambda: NAMES)

    def read(path):
        reads.append(path.name)
        return SimpleNamespace(waveform=np.ones((1, 20), dtype=np.float32), sample_rate=8_000)

    def extract(crop, sample_rate):
        extracts.append(len(crop))
        return np.full(88, 2, dtype=np.float32), NAMES, True

    monkeypatch.setattr(resume, "read_wav_frames", read)
    monkeypatch.setattr(resume, "extract_with_short_audio_fallback", extract)
    result = resume.resume_egemaps_extraction(
        _manifest(), tmp_path, root, final, KEYS, CALLS, NAMES,
        source_provenance=FINGERPRINT, clean_reference_fingerprint=FINGERPRINT,
        expected_rows=3,
    )
    assert "REUSING_PARTIAL_EGEMAPS_CACHE" in capsys.readouterr().out
    assert reads == ["b.wav"] and extracts == [10]
    assert np.all(result.X[:2] == 1) and np.all(result.X[2] == 2)
    assert result.metadata["short_audio_fallback"] == resume.SHORT_AUDIO_FALLBACK
    assert result.metadata["padding_target_samples"] == 800
    assert result.metadata["sample_rate"] == 8_000
    assert result.metadata["padded_row_count"] == 1
    assert result.metadata["padded_record_keys"] == ["k2"]
    assert result.metadata["padded_original_sample_lengths"] == [10]
    assert root.is_dir()  # Recovery artifacts are retained after final publish.


def test_extraction_error_checkpoints_then_resume_without_redoing_rows(
    tmp_path, monkeypatch,
):
    root = _partial(tmp_path)
    final = tmp_path / "final.npz"
    monkeypatch.setattr(resume, "get_egemaps_feature_names", lambda: NAMES)
    monkeypatch.setattr(standard, "get_egemaps_feature_names", lambda: NAMES)
    reads = []

    def read(path):
        reads.append(path.name)
        return SimpleNamespace(waveform=np.ones((1, 20), dtype=np.float32), sample_rate=8_000)

    monkeypatch.setattr(resume, "read_wav_frames", read)

    def fail(crop, sample_rate):
        raise ValueError("still nonfinite")

    monkeypatch.setattr(resume, "extract_with_short_audio_fallback", fail)
    with pytest.raises(ValueError, match="k2"):
        resume.resume_egemaps_extraction(
            _manifest(), tmp_path, root, final, KEYS, CALLS, NAMES,
            source_provenance=FINGERPRINT, clean_reference_fingerprint=FINGERPRINT,
            expected_rows=3,
        )
    assert not final.exists()
    _, mask, _ = resume._validate_partial(root, KEYS, CALLS, NAMES, FINGERPRINT)
    assert mask.tolist() == [True, True, False]
    monkeypatch.setattr(
        resume, "extract_with_short_audio_fallback",
        lambda crop, rate: (np.ones(88, dtype=np.float32), NAMES, False),
    )
    resume.resume_egemaps_extraction(
        _manifest(), tmp_path, root, final, KEYS, CALLS, NAMES,
        source_provenance=FINGERPRINT, clean_reference_fingerprint=FINGERPRINT,
        expected_rows=3,
    )
    assert reads == ["b.wav", "b.wav"]
    assert final.is_file()
