"""Synthetic interval tests for the Training overlap-audit definition."""

from __future__ import annotations

import pytest

from dcc_data.intervals import (
    caller_operator_overlap_stats,
    intersection_length,
    interval_length,
    merge_intervals,
)


def test_adjacent_half_open_intervals_do_not_overlap_or_merge() -> None:
    merged = merge_intervals([(0.0, 10.0), (10.0, 20.0)])
    assert merged == [(0.0, 10.0), (10.0, 20.0)]
    assert intersection_length([(0.0, 10.0)], [(10.0, 20.0)]) == 0.0
    stats = caller_operator_overlap_stats([(0.0, 10.0)], [(10.0, 20.0)])
    assert stats["overlap"] == 0.0
    assert stats["overlap_ratio"] == 0.0
    assert stats["caller_union"] == 10.0


def test_partial_overlap_length() -> None:
    stats = caller_operator_overlap_stats([(0.0, 10.0)], [(8.0, 15.0)])
    assert stats["overlap"] == pytest.approx(2.0)
    assert stats["caller_union"] == pytest.approx(10.0)
    assert stats["overlap_ratio"] == pytest.approx(0.2)


def test_nested_caller_intervals_use_union_not_sum_as_denominator() -> None:
    caller = [(0.0, 20.0), (5.0, 10.0)]
    operator = [(5.0, 15.0)]
    stats = caller_operator_overlap_stats(caller, operator)

    assert interval_length(caller) == pytest.approx(25.0)
    assert stats["caller_sum"] == pytest.approx(25.0)
    assert stats["caller_union"] == pytest.approx(20.0)
    assert stats["overlap"] == pytest.approx(10.0)
    assert stats["overlap_ratio"] == pytest.approx(0.5)
    assert stats["overlap_ratio"] != pytest.approx(stats["overlap"] / stats["caller_sum"])


def test_self_overlapping_caller_denominator_is_union() -> None:
    caller = [(0.0, 10.0), (5.0, 15.0)]
    operator = [(8.0, 12.0)]
    stats = caller_operator_overlap_stats(caller, operator)

    assert stats["caller_sum"] == pytest.approx(20.0)
    assert stats["caller_union"] == pytest.approx(15.0)
    assert stats["overlap"] == pytest.approx(4.0)
    assert stats["overlap_ratio"] == pytest.approx(4.0 / 15.0)


def test_disjoint_caller_fragments_intersect_operator_without_double_count() -> None:
    caller = [(0.0, 10.0), (20.0, 30.0)]
    operator = [(8.0, 22.0)]
    stats = caller_operator_overlap_stats(caller, operator)

    assert stats["caller_union"] == pytest.approx(20.0)
    assert stats["overlap"] == pytest.approx(4.0)
    assert stats["overlap_ratio"] == pytest.approx(0.2)


def test_empty_operator_has_zero_overlap() -> None:
    stats = caller_operator_overlap_stats([(0.0, 10.0), (12.0, 15.0)], [])
    assert stats["caller_union"] == pytest.approx(13.0)
    assert stats["overlap"] == 0.0
    assert stats["overlap_ratio"] == 0.0


def test_empty_caller_has_zero_ratio() -> None:
    stats = caller_operator_overlap_stats([], [(0.0, 10.0)])
    assert stats["caller_union"] == 0.0
    assert stats["overlap"] == 0.0
    assert stats["overlap_ratio"] == 0.0


def test_overlap_ratio_is_at_most_one() -> None:
    stats = caller_operator_overlap_stats([(0.0, 10.0)], [(0.0, 10.0), (-5.0, 20.0)])
    assert stats["overlap"] == pytest.approx(10.0)
    assert stats["overlap_ratio"] == pytest.approx(1.0)
    assert 0.0 <= stats["overlap_ratio"] <= 1.0
