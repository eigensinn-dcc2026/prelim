from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from dcc_data.manifest import ManifestRow
from dcc_data.mission2 import Mission2Dataset


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(20, dtype="<i2")
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
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def signature(dataset: Mission2Dataset) -> list[tuple[object, ...]]:
    values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        features = sample["features"]
        values.append(
            (
                features["start_at"],
                features["end_at"],
                sample["target"],
                tuple(features["waveform"].ravel().tolist()),
            )
        )
    return sorted(values)


def test_mission2_exposes_no_text_or_utterance_index_and_is_order_invariant(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    first = [
        {"startAt": 1, "endAt": 5, "text": "secret one", "speaker": 0},
        {"startAt": 6, "endAt": 10, "text": "secret two", "speaker": 1},
    ]
    write_json(root / json_relative, first)
    row = ManifestRow("Training", "a", wav_relative, json_relative)
    dataset_a = Mission2Dataset(
        [row], root, "Training", time_to_frame=lambda value, _: int(value)
    )

    sample = dataset_a[0]
    assert set(sample["features"]) == {"waveform", "start_at", "end_at"}
    assert sample["target"] == 0
    assert "text" not in repr(sample)
    assert "utterance_index" not in repr(sample)

    changed_and_reordered = [
        {"startAt": 6, "endAt": 10, "text": "changed", "speaker": 1},
        {"startAt": 1, "endAt": 5, "text": "also changed", "speaker": 0},
    ]
    write_json(root / json_relative, changed_and_reordered)
    dataset_b = Mission2Dataset(
        [row], root, "Training", time_to_frame=lambda value, _: int(value)
    )
    assert signature(dataset_a) == signature(dataset_b)
