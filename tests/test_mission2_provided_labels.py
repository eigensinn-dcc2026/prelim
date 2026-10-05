"""Mission 2 trains on the speaker labels exactly as provided.

The organizers ruled (2026-10-01) that reading ``text`` to look at Training errors is fine, but correcting
labels (or selecting training data) on the strength of it and then training on the result is not. Earlier
runs corrected six Training calls' labels inside ``build_utterance_records``; these tests pin that it no
longer changes anything, using those very call ids -- the ids are the only thing left of the old correction,
and only here, so that putting one back fails a test instead of passing unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import build_utterance_records

# The six calls whose labels used to be corrected (4 whole-call swaps, 1 swap from utt_idx 5, 1 swap before utt_idx 6).
FORMERLY_CORRECTED_CALL_IDS = [
    "651e5550bfd63782eb3866a5",
    "651e4b1829d82455557a23b2",
    "651e50c72f06ed4a6e31e3d4",
    "651e4c754428de718c773dab",
    "651e4e712e98ee712096903c",
    "651e4c754428de718c773b56",
]


def _write_call_json(root: Path, json_relative: str, call_id: str, speakers: list[int]) -> None:
    document = {
        "_id": call_id,
        "utterances": [
            {"id": f"u{i}", "startAt": i * 1000, "endAt": i * 1000 + 500, "speaker": speaker, "text": "secret"}
            for i, speaker in enumerate(speakers)
        ],
    }
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("call_id", FORMERLY_CORRECTED_CALL_IDS + ["a_call_that_was_never_listed"])
def test_speakers_come_through_exactly_as_provided(tmp_path: Path, call_id: str) -> None:
    speakers = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # long enough to cross both old utt_idx boundaries (5 and 6)
    json_relative = f"chunk/Training/2.라벨링데이터/TL_서울_구급/{call_id}.json"
    _write_call_json(tmp_path, json_relative, call_id, speakers)
    row = ManifestRow("Training", call_id, f"chunk/Training/1.원천데이터/TS_서울_구급/{call_id}.wav", json_relative)

    records = build_utterance_records(row, tmp_path)

    assert [record.speaker for record in records] == speakers
    assert [record.call_id for record in records] == [call_id] * len(speakers)


def test_no_label_correction_API_is_left_in_the_feature_module() -> None:
    import dcc_data.mission2_features as features

    leftovers = [name for name in dir(features) if "corrected_speaker" in name or "LABEL_SWAP" in name.upper()]
    assert leftovers == []
