"""Timestamp interval helpers shared with the Training overlap audit.

Intervals are treated as half-open [start, end). Adjacent endpoints do not
overlap and are not merged. ``overlap_ratio`` uses caller **union** length
as the denominator, never the raw sum of caller intervals.
"""

from __future__ import annotations

from collections.abc import Sequence


Interval = tuple[float, float]


def _as_interval(item: Sequence[float]) -> Interval:
    if len(item) != 2:
        raise ValueError(f"interval must be (start, end), got {item!r}")
    start = float(item[0])
    end = float(item[1])
    if end <= start:
        raise ValueError(f"interval end must be greater than start: {(start, end)}")
    return start, end


def interval_length(intervals: Sequence[Sequence[float]]) -> float:
    """Sum of interval lengths. Overlapping input is counted twice."""

    return float(sum(end - start for start, end in (_as_interval(item) for item in intervals)))


def merge_intervals(intervals: Sequence[Sequence[float]]) -> list[Interval]:
    """Merge overlapping intervals. Touching endpoints stay separate (half-open)."""

    ordered = sorted((_as_interval(item) for item in intervals), key=lambda item: item[0])
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged:
            merged.append([start, end])
            continue
        _previous_start, previous_end = merged[-1]
        if start < previous_end:
            merged[-1][1] = max(previous_end, end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def intersection_length(
    merged_a: Sequence[Sequence[float]], merged_b: Sequence[Sequence[float]]
) -> float:
    """Length of the intersection of two already-merged interval lists."""

    first = merge_intervals(merged_a) if merged_a else []
    second = merge_intervals(merged_b) if merged_b else []
    index_a = 0
    index_b = 0
    total = 0.0
    while index_a < len(first) and index_b < len(second):
        a_start, a_end = first[index_a]
        b_start, b_end = second[index_b]
        left = max(a_start, b_start)
        right = min(a_end, b_end)
        if right > left:
            total += right - left
        if a_end <= b_end:
            index_a += 1
        else:
            index_b += 1
    return float(total)


def caller_operator_overlap_stats(
    caller_intervals: Sequence[Sequence[float]],
    operator_intervals: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Return caller-operator overlap using the Training audit definition.

    ``overlap_ratio = |caller_union ∩ operator_union| / |caller_union|``.
    Length units match the input timestamps (milliseconds in DCC JSON).
    """

    caller_sum = interval_length(caller_intervals) if caller_intervals else 0.0
    caller_union_intervals = merge_intervals(caller_intervals) if caller_intervals else []
    operator_union_intervals = merge_intervals(operator_intervals) if operator_intervals else []
    caller_union = interval_length(caller_union_intervals) if caller_union_intervals else 0.0
    overlap = intersection_length(caller_union_intervals, operator_union_intervals)
    ratio = overlap / caller_union if caller_union > 0.0 else 0.0
    return {
        "caller_sum": caller_sum,
        "caller_union": caller_union,
        "overlap": overlap,
        "overlap_ratio": ratio,
    }
