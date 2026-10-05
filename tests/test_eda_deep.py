from __future__ import annotations

import numpy as np

from dcc_data.eda_deep import (
    bootstrap_median_difference,
    bootstrap_paired_median_difference,
    call_overlap_metrics,
    cliffs_delta,
    paired_probability_difference,
)


def test_call_overlap_metrics_separates_pair_and_time_measures() -> None:
    utterances = [
        {"startAt": 0, "endAt": 1000, "speaker": 0},
        {"startAt": 900, "endAt": 1600, "speaker": 1},
        {"startAt": 1500, "endAt": 2000, "speaker": 1},
    ]

    result = call_overlap_metrics(utterances, 0, 2000)

    assert result["pair_count"] == 2
    assert result["cross_pair_count"] == 1
    assert result["same_pair_count"] == 1
    assert result["micro_pair_count"] == 2
    assert result["any_overlap_s"] == 0.2
    assert result["cross_overlap_s"] == 0.1
    assert result["any_overlap_ratio"] == 0.1


def test_overlap_metrics_detect_range_errors() -> None:
    utterances = [
        {"startAt": -10, "endAt": 100, "speaker": 0},
        {"startAt": 200, "endAt": 100, "speaker": 1},
        {"startAt": 900, "endAt": 1100, "speaker": 1},
    ]
    result = call_overlap_metrics(utterances, 0, 1000)
    assert result["invalid_interval_count"] == 1
    assert result["outside_call_count"] == 2


def test_overlap_time_uses_union_instead_of_pairwise_sum() -> None:
    utterances = [
        {"startAt": 0, "endAt": 1000, "speaker": 0},
        {"startAt": 500, "endAt": 1500, "speaker": 1},
        {"startAt": 750, "endAt": 1250, "speaker": 1},
    ]

    result = call_overlap_metrics(utterances, 0, 1500)

    assert result["pair_count"] == 3
    assert result["any_overlap_s"] == 0.75
    assert result["cross_overlap_s"] == 0.5


def test_paired_probability_difference_uses_aligned_pairs() -> None:
    assert paired_probability_difference([3, 1, 2], [1, 2, 2]) == 0.0
    assert paired_probability_difference([3, 4], [1, 2]) == 1.0


def test_cliffs_delta_has_expected_direction_and_bounds() -> None:
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert cliffs_delta([1, 1], [1, 1]) == 0.0


def test_bootstrap_median_difference_is_deterministic() -> None:
    first = bootstrap_median_difference([3, 4, 5], [0, 1, 2], iterations=100)
    second = bootstrap_median_difference([3, 4, 5], [0, 1, 2], iterations=100)
    assert first == second
    assert first["estimate"] == 3.0
    assert first["iterations"] == 100
    assert first["resampling_unit"] == "observation"
    assert np.isfinite([first["estimate"], first["ci95_low"], first["ci95_high"]]).all()


def test_paired_bootstrap_preserves_call_pairs() -> None:
    first = bootstrap_paired_median_difference(
        [10, 20, 30], [1, 2, 3], iterations=100
    )
    second = bootstrap_paired_median_difference(
        [10, 20, 30], [1, 2, 3], iterations=100
    )

    assert first == second
    assert first["estimate"] == 18.0
    assert first["iterations"] == 100
    assert first["resampling_unit"] == "call_pair"
    assert first["n_pairs"] == 3


# ---------------------------------------------------------------------------
# End-to-end smoke test on synthetic calls (no real data needed)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import pandas as pd

from dcc_data.eda_deep import build_deep_summary, extract_call_record, load_deep_rows
from dcc_data.eda_deep_report import FIGURE_NAMES, build_deep_section, render_deep_figures
from dcc_data.manifest import ManifestRow
from dcc_data.mission3 import SYMPTOM_CLASSES


def _synthetic_document(index: int) -> dict:
    rng = np.random.default_rng(index)
    utterances, cursor = [], 0
    for turn in range(10):
        length = int(rng.integers(600, 3000))
        # every other turn slightly overlaps the previous speaker
        start = cursor - (200 if turn % 2 else 0)
        utterances.append(
            {"speaker": turn % 2, "startAt": start, "endAt": start + length, "text": "x"}
        )
        cursor = start + length + 100
    return {
        "startAt": 0,
        "endAt": cursor,
        "gender": "F" if index % 2 else "M",
        "symptom": [SYMPTOM_CLASSES[index % 3], SYMPTOM_CLASSES[(index + 1) % 4]][: 1 + index % 2],
        "utterances": utterances,
    }


def _build_synthetic(tmp_path: Path):
    rows = []
    for split, count in (("Training", 24), ("Validation", 12)):
        for i in range(count):
            stem = f"{split[0]}{i}"
            rel = f"{split}/{stem}.json"
            (tmp_path / split).mkdir(exist_ok=True)
            (tmp_path / rel).write_text(
                json.dumps(_synthetic_document(i + (100 if split == "Validation" else 0))),
                encoding="utf-8",
            )
            rows.append(ManifestRow(split=split, stem=stem, wav_path=f"{split}/{stem}.wav", json_path=rel))
    records, detail = load_deep_rows(rows, tmp_path)

    rng = np.random.default_rng(0)
    waveform = []
    for row in rows:
        for speaker in (0, 1):
            waveform.append(
                {
                    "split": row.split,
                    "stem": row.stem,
                    "gender": "F" if int(row.stem[1:]) % 2 else "M",
                    "speaker": speaker,
                    "n_utterances": 5,
                    "duration_sec": float(rng.normal(25 + 5 * speaker, 3)),
                    "feature_audio_duration_sec": 15.0,
                    "rms_mean": float(rng.normal(0.06, 0.01)),
                    "zcr_mean": float(rng.normal(0.19 - 0.02 * speaker, 0.01)),
                    "spectral_centroid_mean": float(rng.normal(1000 - 100 * speaker, 50)),
                    "silence_ratio_abs_le_0_01": float(rng.uniform(0.2, 0.5)),
                }
            )
    full_summary = {
        "cross_split": {
            "identifier_overlap_count": {},
            "exact_transcript_hash_overlap_count": 0,
            "validation_token_oov": {},
        },
        "training": {"common": {"wav": {}}},
        "validation": {"common": {"wav": {}}},
    }
    return rows, records, detail, pd.DataFrame(waveform), full_summary


def test_extract_call_record_is_privacy_safe_and_handles_bad_values() -> None:
    document = {
        "startAt": 0,
        "endAt": 10_000,
        "gender": "F",
        "symptom": ["복통", "없는증상", 3],
        "address": "secret",
        "utterances": [
            {"speaker": 1, "startAt": 0, "endAt": 2_000, "text": "secret"},
            {"speaker": 0, "startAt": 2_000, "endAt": 3_000},
            {"speaker": 1, "startAt": float("nan"), "endAt": 5_000},
            "not-a-dict",
        ],
    }
    record = extract_call_record(document, "Training")
    assert record["call_duration_s"] == 10.0
    assert record["caller_speaking_s"] == 2.0
    assert record["caller_utterance_count"] == 2  # counts speaker==1 entries incl. bad interval
    assert record["symptoms"] == ("복통",)
    assert "secret" not in json.dumps(record, ensure_ascii=False)

    missing = extract_call_record({"utterances": None}, "Validation")
    assert np.isnan(missing["call_duration_s"])
    assert missing["gender"] == "__missing__"


def test_deep_summary_section_and_figures_run_end_to_end(tmp_path: Path) -> None:
    _, records, detail, waveform, full_summary = _build_synthetic(tmp_path)
    assert detail["read_failures"] == {}

    summary = build_deep_summary(records, detail, waveform, full_summary)
    assert summary["meta"]["n_training"] == 24
    assert summary["meta"]["n_validation"] == 12
    assert "pca" not in summary["acoustic_sample"]
    json.dumps(summary, allow_nan=False)  # must be JSON-safe
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "T0" not in serialized and "secret" not in serialized  # no stems/identifiers

    section = build_deep_section(summary, 7)
    assert section[0].startswith("## 7. 심화 분석")
    assert any("Mission별 시사점" in line for line in section)

    created = render_deep_figures(summary, records, tmp_path / "figures")
    assert {path.name for path in created} == set(FIGURE_NAMES.values())
    assert all(path.stat().st_size > 0 for path in created)


def test_build_deep_summary_fails_loudly_without_records() -> None:
    import pytest

    with pytest.raises(ValueError, match="no call JSON could be read"):
        build_deep_summary([], {"read_failures": {"FileNotFoundError": 3}}, pd.DataFrame(), {})
