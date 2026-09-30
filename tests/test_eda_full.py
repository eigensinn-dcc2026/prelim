from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dcc_data.eda import (
    analyze_split,
    compare_splits,
    fast_acoustic_feature_vector,
    numeric_profile,
    union_duration_ms,
)
from dcc_data.manifest import ManifestRow


def _document(*, identifier: str, gender: str, second_text: str = "[개인정보]") -> dict:
    return {
        "_id": identifier,
        "audioPath": "source.wav",
        "recordId": f"record-{identifier}",
        "status": 12,
        "startAt": 0,
        "endAt": 1000,
        "utterances": [
            {"id": "u0", "startAt": 0, "endAt": 600, "text": "119입니다", "speaker": 0},
            {"id": "u1", "startAt": 500, "endAt": 1000, "text": second_text, "speaker": 1},
        ],
        "mediaType": "Mobile",
        "gender": gender,
        "address": "not-emitted",
        "disasterLarge": "구급",
        "disasterMedium": "질병",
        "urgencyLevel": "중",
        "sentiment": "불안/걱정",
        "symptom": ["고열", "고열", "기타"],
        "triage": "준응급증상",
    }


def _analyze(tmp_path: Path, split: str, document: dict):
    root = tmp_path / split
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "call.json"
    json_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    row = ManifestRow(split, "call", f"{split}/call.wav", f"{split}/call.json")
    headers = {
        (split, "call"): {
            "channels": "1",
            "sample_rate": "8000",
            "sample_width": "2",
            "duration_s": "1.0",
            "error": "",
        }
    }
    return analyze_split([row], tmp_path, headers)


def test_numeric_profile_and_union_duration() -> None:
    profile = numeric_profile([0, 1, 2, 3, 4], "count")
    assert profile["p50"] == 2
    assert profile["zero_count"] == 1
    assert profile["negative_count"] == 0
    assert union_duration_ms([(0, 600), (500, 1000), (1200, 1300)]) == 1100


def test_fast_acoustic_feature_vector_matches_baseline_schema() -> None:
    time = np.arange(8000, dtype=np.float32) / 8000
    waveform = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    vector = fast_acoustic_feature_vector(waveform, 8000)
    assert vector.shape == (52,)
    assert np.isfinite(vector).all()


def test_analyze_split_covers_all_missions(tmp_path: Path) -> None:
    result = _analyze(tmp_path, "Training", _document(identifier="same", gender="F"))
    summary = result.summary

    assert summary["common"]["json_parsed"] == 1
    assert summary["common"]["wav"]["sample_rates_hz"] == {"8000": 1}
    assert summary["common"]["timestamp_issue_counts"][
        "calls_with_overlapping_utterances"
    ] == 1
    assert summary["mission1"]["gender_distribution"]["F"]["count"] == 1
    assert summary["mission1"]["caller_duration_s"]["p50"] == 0.5
    assert summary["mission2"]["utterance_count_by_speaker"]["0"]["count"] == 1
    assert summary["mission2"]["utterance_count_by_speaker"]["1"]["count"] == 1
    assert summary["mission3"]["target_symptom_prevalence"]["고열"]["count"] == 1
    assert summary["mission3"]["duplicate_labels_within_calls"] == 1
    assert summary["mission3"]["unknown_symptom_labels"] == {"기타": 1}
    assert summary["mission3"]["privacy_marker_utterances"] == 1


def test_compare_splits_reports_overlap_and_oov(tmp_path: Path) -> None:
    training = _analyze(
        tmp_path, "Training", _document(identifier="shared", gender="F")
    )
    validation_document = _document(
        identifier="shared", gender="M", second_text="처음보는토큰"
    )
    validation_document["recordId"] = "record-shared"
    validation = _analyze(tmp_path, "Validation", validation_document)

    comparison = compare_splits(training, validation)
    assert comparison["identifier_overlap_count"]["_id"] == 1
    assert comparison["identifier_overlap_count"]["recordId"] == 1
    assert comparison["identifier_overlap_count"]["stem"] == 1
    assert comparison["validation_token_oov"]["oov_occurrences"] == 1
