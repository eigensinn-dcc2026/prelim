"""Tests for the confirmed speaker-label-swap correction.

See reports/mission2/m2_label_swap_audit.md: a handful of calls have their
대원(speaker 0)/신고자(speaker 1) labels reversed at the source. Most are
confirmed whole-call swaps (transcript text -- the dispatcher's fixed
"119입니다" greeting or closing "출동할게요" attributed to the wrong speaker
id for the entire call). A few turned out to be *partial*: the L2 CNN's
held-out predictions agreed with the label up to some utt_idx and then
confidently disagreed from there on -- dcc_data.mission2_analysis.
best_flip_point/find_label_discontinuity_candidates found the exact
boundary (e.g. 651e4c754428de718c773b56 was originally treated as a
whole-call swap, but only utt_idx 6+ was actually mislabeled). Both fixes
are applied here, never by editing the team's copy of the source JSON, so
they stay auditable and reversible.
"""
from __future__ import annotations

import json
from pathlib import Path

from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import (
    CONFIRMED_LABEL_SWAPPED_CALL_IDS,
    PARTIAL_LABEL_SWAP_FLIP_FROM_UTT_IDX,
    build_utterance_records,
)


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


def test_confirmed_swapped_call_has_every_speaker_flipped(tmp_path: Path) -> None:
    assert CONFIRMED_LABEL_SWAPPED_CALL_IDS  # non-empty: at least the 5 audited calls
    swapped_call_id = next(iter(CONFIRMED_LABEL_SWAPPED_CALL_IDS))

    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/x.json"
    _write_call_json(tmp_path, json_relative, swapped_call_id, speakers=[0, 1, 0, 1])
    row = ManifestRow("Training", "x", "chunk/Training/1.원천데이터/TS_서울_구급/x.wav", json_relative)

    records = build_utterance_records(row, tmp_path)

    assert [r.speaker for r in records] == [1, 0, 1, 0]


def test_unlisted_call_speakers_pass_through_unchanged(tmp_path: Path) -> None:
    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/y.json"
    _write_call_json(tmp_path, json_relative, "not_in_the_swap_list", speakers=[0, 1, 0, 1])
    row = ManifestRow("Training", "y", "chunk/Training/1.원천데이터/TS_서울_구급/y.wav", json_relative)

    records = build_utterance_records(row, tmp_path)

    assert [r.speaker for r in records] == [0, 1, 0, 1]


def test_partially_swapped_call_flips_only_from_its_recorded_utt_idx_onward() -> None:
    assert PARTIAL_LABEL_SWAP_FLIP_FROM_UTT_IDX  # non-empty
    assert CONFIRMED_LABEL_SWAPPED_CALL_IDS.isdisjoint(PARTIAL_LABEL_SWAP_FLIP_FROM_UTT_IDX)


def test_partially_swapped_call_flips_only_from_its_recorded_utt_idx_onward_build(tmp_path: Path) -> None:
    call_id, flip_from = next(iter(PARTIAL_LABEL_SWAP_FLIP_FROM_UTT_IDX.items()))
    speakers = [i % 2 for i in range(flip_from + 3)]  # a few utterances on each side of the boundary

    json_relative = "chunk/Training/2.라벨링데이터/TL_서울_구급/z.json"
    _write_call_json(tmp_path, json_relative, call_id, speakers=speakers)
    row = ManifestRow("Training", "z", "chunk/Training/1.원천데이터/TS_서울_구급/z.wav", json_relative)

    records = build_utterance_records(row, tmp_path)

    expected = [s if i < flip_from else 1 - s for i, s in enumerate(speakers)]
    assert [r.speaker for r in records] == expected
