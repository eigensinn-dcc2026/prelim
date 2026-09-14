from __future__ import annotations

import numpy as np
import pandas as pd

from dcc_data.mission2_analysis import (
    best_flip_point,
    bucket_duration,
    compute_overlap_flags,
    find_label_discontinuity_candidates,
    find_label_swap_candidates,
    select_listening_samples,
)


def test_compute_overlap_flags_detects_intersecting_ranges_within_a_call() -> None:
    index_df = pd.DataFrame(
        {
            "call_id": ["c1", "c1", "c1", "c2"],
            "utt_idx": [0, 1, 2, 0],
            "startAt": [0, 50, 200, 0],
            "endAt": [100, 150, 300, 100],
        }
    )

    result = compute_overlap_flags(index_df)
    flags = result.set_index(["call_id", "utt_idx"])["overlaps_other"]

    assert flags[("c1", 0)] == True  # noqa: E712 -- overlaps utt 1 ([0,100) vs [50,150))
    assert flags[("c1", 1)] == True  # noqa: E712 -- overlaps utt 0
    assert flags[("c1", 2)] == False  # noqa: E712 -- [200,300) touches neither
    assert flags[("c2", 0)] == False  # noqa: E712 -- only utterance in its call


def test_bucket_duration_matches_expected_edges() -> None:
    durations = pd.Series([0.2, 0.5, 0.9, 1.5, 3.0, 5.0])
    buckets = bucket_duration(durations).astype(str).tolist()
    assert buckets == ["<0.5s", "0.5-1s", "0.5-1s", "1-2s", "2-4s", ">=4s"]


def test_select_listening_samples_excludes_overlap_only_from_correct_groups() -> None:
    df = pd.DataFrame(
        {
            "speaker": [0, 0, 0, 1, 1, 1, 0, 1],
            "correct": [True, True, True, True, True, True, False, False],
            "overlaps_other": [False, False, True, False, False, True, False, True],
        }
    )

    groups = select_listening_samples(df, n_per_group=5, seed=42)

    # 정답_안겹침 groups must never include an overlapping row, even though
    # one exists in the pool for each speaker -- that's the entire point of
    # this group (hear each role uncontaminated by the other speaker).
    assert not groups["대원_정답_안겹침"]["overlaps_other"].any()
    assert not groups["신고자_정답_안겹침"]["overlaps_other"].any()
    assert len(groups["대원_정답_안겹침"]) == 2  # only 2 non-overlapping correct speaker=0 rows exist
    assert len(groups["신고자_정답_안겹침"]) == 2

    # 오답 groups are not filtered by overlap.
    assert len(groups["대원_오답"]) == 1
    assert len(groups["신고자_오답"]) == 1


def test_select_listening_samples_omits_empty_groups_instead_of_erroring() -> None:
    df = pd.DataFrame(
        {
            "speaker": [0, 0],
            "correct": [True, True],
            "overlaps_other": [False, False],
        }
    )

    groups = select_listening_samples(df, n_per_group=3, seed=42)

    assert set(groups) == {"대원_정답_안겹침"}
    assert len(groups["대원_정답_안겹침"]) == 2


def _utterances(*speakers_and_texts: tuple[int, str]) -> list[dict]:
    return [{"speaker": speaker, "text": text} for speaker, text in speakers_and_texts]


def test_find_label_swap_candidates_flags_greeting_attributed_to_speaker_one() -> None:
    calls = [
        ("swapped", _utterances((1, "네, 119입니다."), (0, "사람이 쓰러졌어요."))),
    ]
    candidates = find_label_swap_candidates(calls)
    assert [c["call_id"] for c in candidates] == ["swapped"]
    assert candidates[0]["anchor_speaker"] == 1
    assert candidates[0]["n_utterances"] == 2


def test_find_label_swap_candidates_does_not_flag_correctly_labeled_calls() -> None:
    calls = [
        ("normal", _utterances((0, "네, 119입니다."), (1, "사람이 쓰러졌어요."))),
    ]
    assert find_label_swap_candidates(calls) == []


def test_find_label_swap_candidates_skips_calls_with_no_greeting_in_first_three() -> None:
    calls = [
        ("unknown", _utterances((1, "여보세요"), (0, "네"), (1, "무슨 일이세요"), (0, "119입니다 뒤늦게 언급"))),
    ]
    assert find_label_swap_candidates(calls) == []


def test_find_label_swap_candidates_excludes_already_fixed_call_ids() -> None:
    calls = [
        ("already_fixed", _utterances((1, "네, 119입니다."), (0, "사람이 쓰러졌어요."))),
    ]
    assert find_label_swap_candidates(calls, exclude_call_ids=frozenset({"already_fixed"})) == []


def test_find_label_swap_candidates_finds_greeting_within_first_three_utterances() -> None:
    calls = [
        ("late_greeting", _utterances((1, "여보세요"), (0, "네"), (1, "119입니다"))),
    ]
    candidates = find_label_swap_candidates(calls)
    assert [c["call_id"] for c in candidates] == ["late_greeting"]


def test_best_flip_point_finds_a_clean_mid_sequence_discontinuity() -> None:
    # Mirrors the real 651e4c754428de718c773b56 finding: correct for the
    # first 6 utterances, then confidently wrong for the rest -- flipping
    # every label from index 6 onward should reproduce perfect agreement.
    match = np.array([True] * 6 + [False] * 15)

    flip_at, corrected_accuracy = best_flip_point(match)

    assert flip_at == 6
    assert corrected_accuracy == 1.0


def test_best_flip_point_reports_full_accuracy_when_already_perfect() -> None:
    match = np.array([True] * 10)
    _, corrected_accuracy = best_flip_point(match)
    assert corrected_accuracy == 1.0


def test_best_flip_point_finds_no_good_split_for_a_genuinely_hard_call() -> None:
    # No single flip point helps a call whose errors are just scattered
    # (mirrors the messy, moderate-confidence real "hard call" cases) --
    # the best any split point can do here is well short of "near-perfect".
    match = np.array([True, False] * 10)
    _, corrected_accuracy = best_flip_point(match)
    assert corrected_accuracy < 0.7


def test_find_label_discontinuity_candidates_flags_only_the_clean_discontinuity() -> None:
    rows = []
    # "swapped_midway": correct 0-5, then a clean flip from utt_idx 6 --
    # raw accuracy is low but a real discontinuity explains all of it.
    for utt_idx in range(20):
        speaker = utt_idx % 2
        predicted = speaker if utt_idx < 6 else 1 - speaker
        rows.append({"call_id": "swapped_midway", "utt_idx": utt_idx, "speaker": speaker, "predicted": predicted})
    # "hard_call": errors alternate every other utterance (same scattered
    # shape verified in test_best_flip_point_finds_no_good_split_for_a_
    # genuinely_hard_call, best possible corrected accuracy 0.55) -- no
    # split point explains it, so it must NOT be flagged.
    for utt_idx in range(20):
        speaker = 0
        predicted = utt_idx % 2
        rows.append({"call_id": "hard_call", "utt_idx": utt_idx, "speaker": speaker, "predicted": predicted})
    df = pd.DataFrame(rows)

    candidates = find_label_discontinuity_candidates(df, min_utterances=5)

    assert [c["call_id"] for c in candidates] == ["swapped_midway"]
    assert candidates[0]["flip_from_utt_idx"] == 6
    assert candidates[0]["corrected_accuracy"] == 1.0


def test_find_label_discontinuity_candidates_skips_calls_below_min_utterances() -> None:
    rows = [
        {"call_id": "too_short", "utt_idx": 0, "speaker": 0, "predicted": 1},
        {"call_id": "too_short", "utt_idx": 1, "speaker": 1, "predicted": 0},
    ]
    df = pd.DataFrame(rows)
    assert find_label_discontinuity_candidates(df, min_utterances=5) == []
