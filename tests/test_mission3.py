from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dcc_data.manifest import ManifestRow
from dcc_data.mission3 import Mission3Dataset, SYMPTOM_CLASSES, encode_symptoms


def write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_encode_symptoms_uses_fixed_order_and_ignores_unknown_labels() -> None:
    assert SYMPTOM_CLASSES == (
        "고열",
        "구토",
        "두통",
        "복통",
        "어지러움",
        "열상",
        "오심",
        "전신쇠약",
        "호흡곤란",
    )
    target = encode_symptoms(["두통", "복통", "찰과상"])
    assert target.dtype == np.float32
    assert target.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_mission3_returns_text_only_and_counts_unknown_symptoms(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    json_relative = "chunk/Training/label/a.json"
    document = {
        "utterances": [
            {"text": "첫 문장", "speaker": 0, "startAt": 1, "endAt": 2},
            {"text": "둘째 문장", "speaker": 1, "startAt": 3, "endAt": 4},
        ],
        "symptom": ["두통", "복통", "찰과상"],
        "gender": "forbidden",
        "address": "forbidden",
    }
    write_json(root / json_relative, document)
    row = ManifestRow(
        "Training", "a", "chunk/Training/audio/a.wav", json_relative
    )
    dataset_a = Mission3Dataset([row], root, "Training")
    sample_a = dataset_a[0]

    assert set(sample_a["features"]) == {"text"}
    assert sample_a["features"]["text"] == "첫 문장\n둘째 문장"
    assert sample_a["target"].shape == (9,)
    assert sample_a["target"].tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert dataset_a.unknown_symptom_counts() == {"찰과상": 1}

    document["gender"] = "changed"
    document["address"] = "changed"
    for utterance in document["utterances"]:
        utterance["speaker"] = 99
        utterance["startAt"] = 999
        utterance["endAt"] = 1000
    write_json(root / json_relative, document)
    dataset_b = Mission3Dataset([row], root, "Training")
    sample_b = dataset_b[0]
    assert sample_a["features"] == sample_b["features"]
    np.testing.assert_array_equal(sample_a["target"], sample_b["target"])
