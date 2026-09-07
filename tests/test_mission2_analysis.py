from __future__ import annotations

import pandas as pd

from dcc_data.mission2_analysis import bucket_duration, compute_overlap_flags


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
