from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from dcc_data.manifest import ManifestRow
from dcc_data.mission1 import (
    Mission1Dataset,
    concatenate_caller_waveforms,
    extract_caller_segments,
)
from dcc_data.mission2 import SPEAKER_CALLER, SPEAKER_OPERATOR


def write_wav(path: Path, *, frame_count: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(frame_count, dtype="<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())


def write_json(
    path: Path,
    utterances: list[dict[str, object]],
    *,
    gender: object = "여",
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


def make_pair(
    tmp_path: Path,
    utterances: list[dict[str, object]],
    *,
    gender: object = "여",
    extra: dict[str, object] | None = None,
    omit_gender: bool = False,
) -> tuple[Path, ManifestRow]:
    root = tmp_path / "raw"
    wav_relative = "chunk/Training/audio/a.wav"
    json_relative = "chunk/Training/label/a.json"
    write_wav(root / wav_relative)
    payload_gender: object | None = None if omit_gender else gender
    write_json(
        root / json_relative,
        utterances,
        gender=payload_gender,
        extra=extra,
    )
    row = ManifestRow("Training", "a", wav_relative, json_relative)
    return root, row


def pcm_frames(start: int, end: int) -> np.ndarray:
    return np.arange(start, end, dtype=np.float32).reshape(1, -1) / 32768.0


def test_extract_caller_segments_keeps_only_caller_utterances(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": 0,
                "endAt": 1,
                "text": "operator",
                "speaker": SPEAKER_OPERATOR,
            },
            {"startAt": 1, "endAt": 2, "text": "caller one", "speaker": SPEAKER_CALLER},
            {"startAt": 3, "endAt": 4, "text": "caller two", "speaker": SPEAKER_CALLER},
        ],
    )

    segments = extract_caller_segments(row, root)

    assert len(segments) == 2
    assert [segment.start_at for segment in segments] == [1, 3]
    np.testing.assert_array_equal(segments[0].waveform, pcm_frames(8, 16))
    np.testing.assert_array_equal(segments[1].waveform, pcm_frames(24, 32))


def test_extract_caller_segments_excludes_operator_speech(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": 0,
                "endAt": 1,
                "text": "operator",
                "speaker": SPEAKER_OPERATOR,
            },
            {"startAt": 1, "endAt": 2, "text": "caller", "speaker": SPEAKER_CALLER},
        ],
    )

    segments = extract_caller_segments(row, root)
    operator_waveform = pcm_frames(0, 8)

    assert len(segments) == 1
    assert segments[0].start_at == 1
    assert not np.array_equal(segments[0].waveform, operator_waveform)


def test_extract_caller_segments_crops_exact_millisecond_intervals(
    tmp_path: Path,
) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {"startAt": 1, "endAt": 2, "text": "ignored", "speaker": SPEAKER_CALLER},
            {"startAt": 4, "endAt": 5, "text": "ignored", "speaker": SPEAKER_CALLER},
        ],
    )

    segments = extract_caller_segments(row, root)

    np.testing.assert_array_equal(segments[0].waveform, pcm_frames(8, 16))
    np.testing.assert_array_equal(segments[1].waveform, pcm_frames(32, 40))
    assert segments[0].waveform.shape[-1] == 8
    assert segments[1].waveform.shape[-1] == 8


def test_concatenate_caller_waveforms_preserves_startat_order_and_length(
    tmp_path: Path,
) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {"startAt": 3, "endAt": 4, "text": "later", "speaker": SPEAKER_CALLER},
            {
                "startAt": 0,
                "endAt": 1,
                "text": "operator",
                "speaker": SPEAKER_OPERATOR,
            },
            {"startAt": 1, "endAt": 2, "text": "earlier", "speaker": SPEAKER_CALLER},
        ],
    )

    segments = extract_caller_segments(row, root)
    waveform = concatenate_caller_waveforms(segments)

    assert [segment.start_at for segment in segments] == [1, 3]
    assert waveform.shape == (1, 16)
    np.testing.assert_array_equal(
        waveform, np.concatenate([pcm_frames(8, 16), pcm_frames(24, 32)], axis=1)
    )


def test_mission1_returns_gender_as_target_and_call_level_waveform(
    tmp_path: Path,
) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": 0,
                "endAt": 1,
                "text": "operator",
                "speaker": SPEAKER_OPERATOR,
            },
            {"startAt": 1, "endAt": 2, "text": "caller one", "speaker": SPEAKER_CALLER},
            {"startAt": 3, "endAt": 4, "text": "caller two", "speaker": SPEAKER_CALLER},
        ],
        gender="남",
    )
    dataset = Mission1Dataset([row], root, "Training")
    sample = dataset[0]

    assert set(sample) == {"features", "target", "metadata"}
    assert sample["target"] == "남"
    assert sample["metadata"] == {
        "split": "Training",
        "stem": "a",
        "n_caller_utterances": 2,
    }
    np.testing.assert_array_equal(
        sample["features"]["waveform"],
        np.concatenate([pcm_frames(8, 16), pcm_frames(24, 32)], axis=1),
    )


@pytest.mark.parametrize(
    ("start_at", "end_at", "message"),
    [
        (-1, 1, "startAt must be non-negative"),
        (1, 1, "endAt must be greater than startAt"),
        (2, 1, "endAt must be greater than startAt"),
    ],
)
def test_mission1_rejects_an_invalid_interval(
    tmp_path: Path, start_at: int, end_at: int, message: str
) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": start_at,
                "endAt": end_at,
                "text": "ignored",
                "speaker": SPEAKER_CALLER,
            }
        ],
    )

    with pytest.raises(ValueError, match=message):
        Mission1Dataset([row], root, "Training")


def test_mission1_rejects_an_invalid_speaker(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [{"startAt": 0, "endAt": 1, "text": "ignored", "speaker": 2}],
    )

    with pytest.raises(ValueError, match="speaker must be 0 or 1"):
        Mission1Dataset([row], root, "Training")


def test_mission1_rejects_a_call_with_no_caller_utterances(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": 0,
                "endAt": 1,
                "text": "operator only",
                "speaker": SPEAKER_OPERATOR,
            }
        ],
    )

    with pytest.raises(ValueError, match="no caller utterances"):
        Mission1Dataset([row], root, "Training")
    with pytest.raises(ValueError, match="no caller utterances"):
        extract_caller_segments(row, root)


def test_mission1_rejects_missing_gender(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [{"startAt": 0, "endAt": 1, "text": "caller", "speaker": SPEAKER_CALLER}],
        omit_gender=True,
    )

    with pytest.raises(ValueError, match="gender is required"):
        Mission1Dataset([row], root, "Training")


def test_mission1_features_exclude_disallowed_annotations(tmp_path: Path) -> None:
    root, row = make_pair(
        tmp_path,
        [
            {
                "startAt": 1,
                "endAt": 2,
                "text": "secret transcript",
                "speaker": SPEAKER_CALLER,
            }
        ],
        gender="여",
        extra={"symptom": ["두통", "구토"]},
    )
    sample = Mission1Dataset([row], root, "Training")[0]
    features = sample["features"]

    assert set(features) == {"waveform"}
    assert "start_at" not in features
    assert "end_at" not in features
    assert "text" not in repr(features)
    assert "symptom" not in repr(features)
    assert "secret transcript" not in repr(sample)
    assert "두통" not in repr(sample)
