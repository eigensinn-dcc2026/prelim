from __future__ import annotations

import pandas as pd

from dcc_data.mission2_analysis import bucket_duration, compute_overlap_flags, select_listening_samples


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
