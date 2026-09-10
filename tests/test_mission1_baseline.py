from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from dcc_data.guards import SplitContaminationError
from dcc_data.manifest import ManifestRow
from dcc_data.mission1 import Mission1Dataset
from dcc_data.mission1_features import (
    ACOUSTIC_FEATURE_COLUMNS,
    N_ACOUSTIC_FEATURES,
    acoustic_feature_vector,
    build_mission1_feature_dataset,
    concatenated_caller_waveform,
    extract_call_feature_row,
)
from dcc_data.mission1_train import (
    MAJORITY_BASELINE_ACCURACY,
    assert_training_features,
    evaluate_frame,
    features_and_labels,
    fit_on_training,
    run_internal_sanity,
    save_mission1_models,
    stratified_sanity_split,
)
from dcc_data.mission2 import SPEAKER_CALLER, SPEAKER_OPERATOR
from dcc_data.mission2_features import acoustic_feature_vector as mission2_acoustic_feature_vector

SAMPLE_RATE = 8000


def _write_pcm_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _write_ramp_wav(path: Path, *, frame_count: int = 64) -> None:
    _write_pcm_wav(path, np.arange(frame_count, dtype="<i2"))


def _write_noise_wav(path: Path, duration_s: float, *, seed: int) -> None:
    n = max(1, int(duration_s * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    _write_pcm_wav(path, (rng.standard_normal(n) * 3000).astype("<i2"))


def _write_json(
    path: Path,
    utterances: list[dict[str, object]],
    *,
    gender: object = "F",
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {
        "utterances": utterances,
        "symptom": ["두통"],
        "address": "must-not-be-a-feature",
    }
    if gender is not None:
        document["gender"] = gender
    if extra:
        document.update(extra)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _region_paths(stem: str, *, region: str, split: str) -> tuple[str, str]:
    source_prefix = "TS" if split == "Training" else "VS"
    label_prefix = "TL" if split == "Training" else "VL"
    wav_relative = f"chunk/{split}/1.원천데이터/{source_prefix}_{region}_구급/{stem}.wav"
    json_relative = f"chunk/{split}/2.라벨링데이터/{label_prefix}_{region}_구급/{stem}.json"
    return wav_relative, json_relative


_DEFAULT_UTTERANCES = [
    {"startAt": 0, "endAt": 90, "text": "operator secret", "speaker": SPEAKER_OPERATOR},
    {"startAt": 200, "endAt": 1800, "text": "caller secret", "speaker": SPEAKER_CALLER},
    {"startAt": 1900, "endAt": 3400, "text": "caller two", "speaker": SPEAKER_CALLER},
]


def _build_region_rows(
    root: Path,
    n_calls: int,
    *,
    region: str,
    split: str,
    gender_cycle: tuple[str, ...] = ("F", "M"),
) -> list[ManifestRow]:
    rows = []
    for i in range(n_calls):
        stem = f"{split.lower()}_{region}_{i:03d}"
        wav_relative, json_relative = _region_paths(stem, region=region, split=split)
        _write_noise_wav(root / wav_relative, 4.0, seed=i)
        _write_json(
            root / json_relative,
            _DEFAULT_UTTERANCES,
            gender=gender_cycle[i % len(gender_cycle)],
        )
        rows.append(ManifestRow(split, stem, wav_relative, json_relative))
    return rows


def test_acoustic_feature_vector_is_reused_not_copied() -> None:
    assert acoustic_feature_vector is mission2_acoustic_feature_vector
    assert len(ACOUSTIC_FEATURE_COLUMNS) == N_ACOUSTIC_FEATURES == 52
    assert "duration_sec" not in ACOUSTIC_FEATURE_COLUMNS


def test_concatenated_waveform_matches_mission1_dataset(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    _write_ramp_wav(root / wav_relative)
    _write_json(
        root / json_relative,
        [
            {"startAt": 3, "endAt": 4, "text": "later caller", "speaker": SPEAKER_CALLER},
            {"startAt": 0, "endAt": 1, "text": "operator", "speaker": SPEAKER_OPERATOR},
            {"startAt": 1, "endAt": 2, "text": "earlier caller", "speaker": SPEAKER_CALLER},
        ],
        gender="F",
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)

    dataset_waveform = Mission1Dataset([row], root, "Training")[0]["features"]["waveform"]
    optimized_waveform, sample_rate = concatenated_caller_waveform(row, root)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(optimized_waveform, dataset_waveform)
    expected = np.concatenate(
        [
            np.arange(8, 16, dtype=np.float32).reshape(1, -1) / 32768.0,
            np.arange(24, 32, dtype=np.float32).reshape(1, -1) / 32768.0,
        ],
        axis=1,
    )
    np.testing.assert_array_equal(optimized_waveform, expected)


def test_call_features_match_dataset_waveform_through_shared_extractor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    rows = _build_region_rows(root, 1, region="서울", split="Training")
    row = rows[0]

    dataset_waveform = Mission1Dataset([row], root, "Training")[0]["features"]["waveform"]
    optimized_waveform, sample_rate = concatenated_caller_waveform(row, root)
    np.testing.assert_array_equal(optimized_waveform, dataset_waveform)

    expected = acoustic_feature_vector(dataset_waveform[0], sample_rate)
    feature_row = extract_call_feature_row(row, root)
    actual = np.array([feature_row[column] for column in ACOUSTIC_FEATURE_COLUMNS])

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (52,)
    assert set(feature_row) == {
        "stem",
        "split",
        "gender",
        "n_caller_utterances",
        *ACOUSTIC_FEATURE_COLUMNS,
    }
    assert "duration_sec" not in feature_row
    assert "startAt" not in feature_row
    assert "endAt" not in feature_row
    assert "text" not in feature_row
    assert "symptom" not in feature_row
    assert "secret" not in repr(feature_row)


def test_build_excludes_validation_and_non_seoul_from_training(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    seoul_train = _build_region_rows(root, 6, region="서울", split="Training")
    other_region = _build_region_rows(root, 2, region="경기", split="Training")
    seoul_val = _build_region_rows(root, 2, region="서울", split="Validation")

    feature_rows, failures = build_mission1_feature_dataset(
        seoul_train + other_region + seoul_val,
        root,
        "Training",
        n_calls=100,
        seed=42,
    )

    assert failures == []
    used_stems = {row["stem"] for row in feature_rows}
    assert used_stems == {row.stem for row in seoul_train}
    assert all(row["split"] == "Training" for row in feature_rows)
    assert {row["gender"] for row in feature_rows} <= {"F", "M"}


def test_build_validation_split_excludes_training_rows(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    seoul_train = _build_region_rows(root, 3, region="서울", split="Training")
    seoul_val = _build_region_rows(root, 4, region="서울", split="Validation")

    feature_rows, failures = build_mission1_feature_dataset(
        seoul_train + seoul_val,
        root,
        "Validation",
        n_calls=100,
        seed=42,
    )

    assert failures == []
    assert {row["stem"] for row in feature_rows} == {row.stem for row in seoul_val}
    assert all(row["split"] == "Validation" for row in feature_rows)


def test_fit_on_training_rejects_validation_rows() -> None:
    columns = list(ACOUSTIC_FEATURE_COLUMNS)
    train = pd.DataFrame({column: np.ones(8) for column in columns})
    train["gender"] = ["F"] * 4 + ["M"] * 4
    train["split"] = "Training"
    val = pd.DataFrame({column: np.full(4, 100.0) for column in columns})
    val["gender"] = ["F", "M", "F", "M"]
    val["split"] = "Validation"

    with pytest.raises(SplitContaminationError):
        fit_on_training(val)
    with pytest.raises(SplitContaminationError):
        fit_on_training(pd.concat([train, val], ignore_index=True))
    with pytest.raises(SplitContaminationError):
        stratified_sanity_split(pd.concat([train, val], ignore_index=True))
    with pytest.raises(SplitContaminationError):
        assert_training_features(val)


def test_scaler_and_models_are_not_fit_on_validation() -> None:
    columns = list(ACOUSTIC_FEATURE_COLUMNS)
    train = pd.DataFrame({column: np.ones(20) for column in columns})
    train["gender"] = ["F"] * 10 + ["M"] * 10
    train["split"] = "Training"
    val = pd.DataFrame({column: np.full(20, 100.0) for column in columns})
    val["gender"] = ["F"] * 10 + ["M"] * 10
    val["split"] = "Validation"

    fitted = fit_on_training(train, seed=42)
    scaler = fitted.lr_pipeline.named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    np.testing.assert_allclose(mean_before, 1.0)

    leaked = StandardScaler().fit(
        pd.concat([train, val], ignore_index=True)[columns]
    )
    assert not np.allclose(leaked.mean_, mean_before)

    evaluate_frame(val, fitted)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    assert scaler.n_samples_seen_ == 20


def test_internal_sanity_and_end_to_end_fit_on_extracted_features(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    rows = _build_region_rows(root, 20, region="서울", split="Training")
    feature_rows, failures = build_mission1_feature_dataset(
        rows, root, "Training", n_calls=100, seed=42
    )
    assert failures == []

    df = pd.DataFrame(feature_rows)
    X, y = features_and_labels(df)
    assert X.shape == (20, 52)
    assert set(np.unique(y)) == {0, 1}

    sanity = run_internal_sanity(df, seed=42, test_size=0.25)
    assert sanity["n_train"] + sanity["n_holdout"] == 20
    assert 0.0 <= sanity["holdout"]["gbm"]["accuracy"] <= 1.0
    assert 0.0 <= sanity["holdout"]["lr"]["accuracy"] <= 1.0

    fitted = fit_on_training(df, seed=42)
    gbm_path = tmp_path / "m1_gbm.pkl"
    lr_path = tmp_path / "m1_lr.pkl"
    save_mission1_models(fitted, gbm_path=gbm_path, lr_path=lr_path, seed=42)
    assert gbm_path.is_file()
    assert lr_path.is_file()
    assert pytest.approx(MAJORITY_BASELINE_ACCURACY, rel=1e-6) == 15542 / 29200


def test_invalid_gender_is_logged_not_silently_dropped(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stem = "bad_gender_000"
    wav_relative, json_relative = _region_paths(stem, region="서울", split="Training")
    _write_noise_wav(root / wav_relative, 4.0, seed=0)
    _write_json(root / json_relative, _DEFAULT_UTTERANCES, gender="여")
    row = ManifestRow("Training", stem, wav_relative, json_relative)

    feature_rows, failures = build_mission1_feature_dataset(
        [row], root, "Training", n_calls=1, seed=42
    )
    assert feature_rows == []
    assert len(failures) == 1
    assert failures[0]["stem"] == stem
    assert failures[0]["error_type"] == "ValueError"
