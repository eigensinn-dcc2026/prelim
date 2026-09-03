from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from dcc_data.manifest import ManifestRow
from dcc_data.mission2 import Mission2Dataset, milliseconds_to_frame


def write_wav(path: Path, *, frame_count: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(frame_count, dtype="<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())


def write_json(path: Path, utterances: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "utterances": utterances,
                "gender": "must-not-be-a-feature",
                "symptom": ["두통"],
                "address": "must-not-be-a-feature",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def signature(dataset: Mission2Dataset) -> list[tuple[object, ...]]:
    values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = sample["metadata"]
        waveform = sample["features"]["waveform"]
        values.append(
            (
                metadata["start_at"],
                metadata["end_at"],
                sample["target"],
                tuple(waveform.ravel().tolist()),
            )
        )
    return sorted(values)


def test_milliseconds_to_frame_uses_explicit_ties_to_even_rounding() -> None:
    assert milliseconds_to_frame(0, 8000) == 0
    assert milliseconds_to_frame(1000, 8000) == 8000
    assert milliseconds_to_frame(1500, 8000) == 12000
    assert milliseconds_to_frame(0.0625, 8000) == 0
    assert milliseconds_to_frame(0.1875, 8000) == 2


def test_mission2_separates_waveform_features_from_metadata_and_is_order_invariant(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    first = [
        {"startAt": 0, "endAt": 1, "text": "secret one", "speaker": 0},
        {"startAt": 1, "endAt": 2, "text": "secret two", "speaker": 1},
    ]
    write_json(root / json_relative, first)
    row = ManifestRow("Training", "a", wav_relative, json_relative)
    dataset_a = Mission2Dataset([row], root, "Training")

    sample = dataset_a[0]
    assert set(sample) == {"features", "target", "metadata"}
    assert set(sample["features"]) == {"waveform"}
    assert sample["target"] == 0
    assert sample["metadata"] == {
        "start_at": 0,
        "end_at": 1,
        "split": "Training",
        "stem": "a",
    }
    assert "text" not in repr(sample)
    assert "gender" not in repr(sample)
    assert "symptom" not in repr(sample)
    assert "utterance_index" not in repr(sample)

    changed_and_reordered = [
        {"startAt": 1, "endAt": 2, "text": "changed", "speaker": 1},
        {"startAt": 0, "endAt": 1, "text": "also changed", "speaker": 0},
    ]
    write_json(root / json_relative, changed_and_reordered)
    dataset_b = Mission2Dataset([row], root, "Training")
    assert signature(dataset_a) == signature(dataset_b)


def test_mission2_crops_the_exact_millisecond_interval(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    write_json(
        root / json_relative,
        [{"startAt": 1, "endAt": 2, "text": "ignored", "speaker": 1}],
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)
    dataset = Mission2Dataset([row], root, "Training")

    waveform = dataset[0]["features"]["waveform"]
    expected = np.arange(8, 16, dtype=np.float32).reshape(1, -1) / 32768.0
    np.testing.assert_array_equal(waveform, expected)


def test_mission2_rejects_an_invalid_speaker(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    write_json(
        root / json_relative,
        [{"startAt": 0, "endAt": 1, "text": "ignored", "speaker": 2}],
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)

    with pytest.raises(ValueError, match="speaker must be 0 or 1"):
        Mission2Dataset([row], root, "Training")


@pytest.mark.parametrize(
    ("start_at", "end_at", "message"),
    [
        (-1, 1, "startAt must be non-negative"),
        (1, 1, "endAt must be greater than startAt"),
        (2, 1, "endAt must be greater than startAt"),
    ],
)
def test_mission2_rejects_an_invalid_interval(
    tmp_path: Path, start_at: int, end_at: int, message: str
) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    write_json(
        root / json_relative,
        [
            {
                "startAt": start_at,
                "endAt": end_at,
                "text": "ignored",
                "speaker": 0,
            }
        ],
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)

    with pytest.raises(ValueError, match=message):
        Mission2Dataset([row], root, "Training")


def test_mission2_reuses_audio_bounds_validation_for_out_of_range_crop(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative, frame_count=16)
    write_json(
        root / json_relative,
        [{"startAt": 0, "endAt": 1000, "text": "ignored", "speaker": 0}],
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)
    dataset = Mission2Dataset([row], root, "Training")

    with pytest.raises(ValueError, match="Invalid frame interval"):
        _ = dataset[0]
