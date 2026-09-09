from __future__ import annotations

import json
import unicodedata
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import (
    FEATURE_COLUMNS,
    N_FEATURES,
    build_seoul_training_dataset,
    is_seoul_path,
    pair_by_stem,
    utterance_feature_vector,
)
from dcc_data.mission2_train import fit_baseline

SAMPLE_RATE = 8000  # matches the team's confirmed real-data format (mono, 8 kHz)


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


def _write_call(
    root: Path,
    wav_relative: str,
    json_relative: str,
    utterances_ms: list[tuple[int, int, int]],
    *,
    seed: int,
) -> None:
    """utterances_ms: list of (startAt_ms, endAt_ms, speaker)."""

    call_duration_s = max(end for _, end, _ in utterances_ms) / 1000.0 + 0.5
    _write_wav(root / wav_relative, call_duration_s, seed=seed)
    document = {
        "_id": Path(json_relative).stem,
        "startAt": 0,
        "endAt": int(call_duration_s * 1000),
        "gender": "F",
        "symptom": [],
        "utterances": [
            {"id": f"u{i}", "startAt": start, "endAt": end, "speaker": speaker, "text": f"secret {i}"}
            for i, (start, end, speaker) in enumerate(utterances_ms)
        ],
    }
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


_DEFAULT_UTTERANCES = [
    (0, 90, 0),  # 90ms: shorter than N_FFT at 8kHz -> exercises zero-padding
    (200, 1800, 1),
    (1900, 3400, 0),
]


def _build_manifest_rows(root: Path, n_calls: int, *, region: str, split: str) -> list[ManifestRow]:
    rows = []
    source_prefix = "TS" if split == "Training" else "VS"
    label_prefix = "TL" if split == "Training" else "VL"
    for i in range(n_calls):
        stem = f"{split.lower()}_{region}_{i:03d}"
        wav_relative = f"chunk/{split}/1.원천데이터/{source_prefix}_{region}_구급/{stem}.wav"
        json_relative = f"chunk/{split}/2.라벨링데이터/{label_prefix}_{region}_구급/{stem}.json"
        _write_call(root, wav_relative, json_relative, _DEFAULT_UTTERANCES, seed=i)
        rows.append(ManifestRow(split, stem, wav_relative, json_relative))
    return rows


def test_is_seoul_path_matches_region_folder_convention() -> None:
    assert is_seoul_path("chunk/Training/1.원천데이터/TS_서울_구급/a.wav")
    assert is_seoul_path("chunk/Training/2.라벨링데이터/TL_서울_구급/a.json")
    assert not is_seoul_path("chunk/Training/1.원천데이터/TS_경기_구급/a.wav")


def test_is_seoul_path_handles_nfd_encoded_korean_folder_names() -> None:
    # The real dataset's Korean folder names arrived NFD-decomposed after
    # passing through a zip archive / Drive sync -- this reproduces that
    # exact bug (Seoul + Training rows: 0 despite paths looking identical
    # on screen) rather than just asserting normalize() works in the abstract.
    nfc_path = "chunk/Training/1.원천데이터/TS_서울_구급/a.wav"
    nfd_path = unicodedata.normalize("NFD", nfc_path)
    assert nfc_path != nfd_path  # sanity: the two forms really are distinct bytes
    assert is_seoul_path(nfd_path)


def test_feature_columns_never_include_order_or_identifier_fields() -> None:
    forbidden = {"utt_idx", "startAt", "endAt", "call_id", "speaker", "text"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)


def test_utterance_feature_vector_handles_very_short_segment() -> None:
    rng = np.random.default_rng(0)
    short_segment = rng.standard_normal(50).astype(np.float32)  # well under N_FFT
    vector = utterance_feature_vector(short_segment, SAMPLE_RATE, duration_sec=0.05)
    assert vector.shape == (N_FEATURES,)
    assert np.isfinite(vector).all()


def test_build_seoul_training_dataset_excludes_validation_and_non_seoul(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    seoul_train_rows = _build_manifest_rows(root, 6, region="서울", split="Training")
    other_region_rows = _build_manifest_rows(root, 2, region="경기", split="Training")
    seoul_val_rows = _build_manifest_rows(root, 2, region="서울", split="Validation")
    all_rows = seoul_train_rows + other_region_rows + seoul_val_rows

    index_records, feature_rows, failures = build_seoul_training_dataset(all_rows, root, n_calls=100, seed=42)

    assert failures == []
    used_stems = {r.wav_stem for r in index_records}
    assert used_stems == {row.stem for row in seoul_train_rows}
    assert len(feature_rows) == len(index_records)  # nothing failed feature extraction
    for row in feature_rows:
        assert set(FEATURE_COLUMNS) <= set(row)
        assert row["speaker"] in (0, 1)


def test_parallel_workers_produce_the_same_aggregate_results_as_sequential(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rows = _build_manifest_rows(root, 8, region="서울", split="Training")

    seq_index, seq_features, seq_failures = build_seoul_training_dataset(
        rows, root, n_calls=100, seed=42, max_workers=1
    )
    par_index, par_features, par_failures = build_seoul_training_dataset(
        rows, root, n_calls=100, seed=42, max_workers=4
    )

    assert seq_failures == par_failures == []
    # Completion order can differ across processes, so compare as sets/sorted,
    # not the raw (order-dependent) lists.
    key = lambda r: (r.call_id, r.utt_idx)  # noqa: E731
    assert sorted(seq_index, key=key) == sorted(par_index, key=key)
    feature_key = lambda r: (r["call_id"], r["utt_idx"])  # noqa: E731
    assert sorted(seq_features, key=feature_key) == sorted(par_features, key=feature_key)


def test_group_shuffle_split_has_no_call_id_leakage(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rows = _build_manifest_rows(root, 20, region="서울", split="Training")
    _, feature_rows, failures = build_seoul_training_dataset(rows, root, n_calls=100, seed=42)
    assert failures == []

    df = pd.DataFrame(feature_rows)
    result = fit_baseline(df, seed=42, test_size=0.25)

    assert result.train_call_ids + result.val_call_ids == df["call_id"].nunique()
    assert 0.0 <= result.group_split_accuracy <= 1.0
    assert 0.0 <= result.random_split_accuracy <= 1.0
    assert len(result.confusion_matrix) == 2 and len(result.confusion_matrix[0]) == 2


def test_invalid_timestamps_are_rejected_at_index_build_time(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stem = "broken_index_000"
    wav_relative = f"chunk/Training/1.원천데이터/TS_서울_구급/{stem}.wav"
    json_relative = f"chunk/Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
    # Zero-length utterance (startAt == endAt) is rejected by
    # build_utterance_records() itself (same rule as Mission2Dataset), before
    # any audio is touched -- the whole call is logged as one "index" stage
    # failure rather than silently dropped.
    _write_call(root, wav_relative, json_relative, [(100, 100, 1)], seed=0)
    row = ManifestRow("Training", stem, wav_relative, json_relative)

    index_records, feature_rows, failures = build_seoul_training_dataset([row], root, n_calls=1, seed=42)

    assert index_records == []
    assert feature_rows == []
    assert len(failures) == 1
    assert failures[0]["call_id"] == stem
    assert failures[0]["stage"] == "index"
    assert failures[0]["error_type"]


def test_feature_extraction_failure_is_logged_not_silently_dropped(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stem = "broken_feature_000"
    wav_relative = f"chunk/Training/1.원천데이터/TS_서울_구급/{stem}.wav"
    json_relative = f"chunk/Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
    # Timestamps are individually valid (startAt >= 0, endAt > startAt) so
    # index building succeeds, but both fall well beyond this WAV's actual
    # length -- clamping collapses start/end to the same frame, which must
    # be caught and logged at the audio/feature stage, not raised.
    _write_wav(root / wav_relative, duration_s=0.01, seed=0)  # ~80 frames at 8kHz
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(
        json.dumps(
            {
                "_id": stem,
                "utterances": [{"id": "u0", "startAt": 5000, "endAt": 5100, "speaker": 1, "text": "secret"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    row = ManifestRow("Training", stem, wav_relative, json_relative)

    index_records, feature_rows, failures = build_seoul_training_dataset([row], root, n_calls=1, seed=42)

    assert len(index_records) == 1  # module A output is unaffected by an audio-side failure
    assert feature_rows == []
    assert len(failures) == 1
    assert failures[0]["call_id"] == stem
    assert failures[0]["utt_idx"] == 0
    assert failures[0]["stage"] == "feature"
    assert failures[0]["error_type"]


def test_pair_by_stem_flat_matching_for_inference(tmp_path: Path) -> None:
    audio_dir = tmp_path / "wavs"
    label_dir = tmp_path / "labels"
    _write_wav(audio_dir / "a.wav", 1.0, seed=1)
    (label_dir).mkdir(parents=True, exist_ok=True)
    (label_dir / "a.json").write_text("{}", encoding="utf-8")
    _write_wav(audio_dir / "b.wav", 1.0, seed=2)  # no matching label -> excluded

    pairs = pair_by_stem(audio_dir, label_dir)
    assert [p[0] for p in pairs] == ["a"]
