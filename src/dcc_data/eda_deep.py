"""Deep-dive EDA statistics that quantify what the competition missions rely on.

Complements :mod:`dcc_data.eda` (integrity and descriptive profile) with effect
sizes and distribution-shift measures: Train/Validation shift, overlap time,
gender effects (M1), speaker utterance structure (M2), symptom association
(M3), and speaker-level acoustic effects.

The functions operate on raw call JSON and the local acoustic sample cache, but
only return aggregate, privacy-safe statistics.  Transcript, address, call
identifiers, and absolute source paths are never exported.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .eda import numeric_profile, union_duration_ms
from .manifest import ManifestRow
from .mission3 import SYMPTOM_CLASSES


ACOUSTIC_FEATURES = (
    "duration_sec",
    "rms_mean",
    "zcr_mean",
    "spectral_centroid_mean",
    "silence_ratio_abs_le_0_01",
)


def _valid_interval(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _speaker_union_seconds(utterances: Sequence[object], speaker: int) -> float:
    intervals = [
        (float(u["startAt"]), float(u["endAt"]))
        for u in utterances
        if isinstance(u, dict)
        and u.get("speaker") == speaker
        and _valid_interval(u.get("startAt"))
        and _valid_interval(u.get("endAt"))
        and u["endAt"] > u["startAt"]
    ]
    return union_duration_ms(intervals) / 1000


def extract_call_record(document: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Convert one call JSON into a privacy-safe call-level measurement row."""

    start, end = document.get("startAt"), document.get("endAt")
    duration_s = (
        (float(end) - float(start)) / 1000
        if _valid_interval(start) and _valid_interval(end) and end > start
        else math.nan
    )
    utterances_raw = document.get("utterances")
    utterances = utterances_raw if isinstance(utterances_raw, list) else []
    caller_count = sum(
        isinstance(u, dict) and u.get("speaker") == 1 for u in utterances
    )
    caller_s = _speaker_union_seconds(utterances, 1)
    symptoms_raw = document.get("symptom")
    present = (
        {value for value in symptoms_raw if isinstance(value, str)}
        if isinstance(symptoms_raw, list)
        else set()
    )
    gender = document.get("gender")
    return {
        "split": split,
        "call_duration_s": duration_s,
        "utterance_count": len(utterances),
        "caller_utterance_count": caller_count,
        "caller_speaking_s": caller_s,
        "caller_speaking_share": caller_s / duration_s if duration_s > 0 else math.nan,
        "gender": gender if isinstance(gender, str) else "__missing__",
        "symptoms": tuple(label for label in SYMPTOM_CLASSES if label in present),
    }


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _coverage_at_least_two(intervals: Sequence[tuple[float, float]]) -> float:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    if not events:
        return 0.0
    events.sort()
    active = 0
    previous = events[0][0]
    total = 0.0
    index = 0
    while index < len(events):
        position = events[index][0]
        if active >= 2:
            total += position - previous
        while index < len(events) and events[index][0] == position:
            active += events[index][1]
            index += 1
        previous = position
    return total


def _intersection_duration(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> float:
    a, b = _merge_intervals(left), _merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def call_overlap_metrics(
    utterances: Sequence[object], call_start: float, call_end: float
) -> dict[str, float | int]:
    """Return pair- and time-based overlap metrics for a single call."""

    valid: list[tuple[float, float, int]] = []
    invalid_interval = 0
    outside_call = 0
    for utterance in utterances:
        if not isinstance(utterance, dict):
            invalid_interval += 1
            continue
        start, end, speaker = (
            utterance.get("startAt"),
            utterance.get("endAt"),
            utterance.get("speaker"),
        )
        if not (_valid_interval(start) and _valid_interval(end) and end > start):
            invalid_interval += 1
            continue
        if float(start) < call_start or float(end) > call_end:
            outside_call += 1
        valid.append((float(start), float(end), int(speaker) if speaker in {0, 1} else -1))

    ordered = sorted(valid)
    pair_count = 0
    cross_pair_count = 0
    same_pair_count = 0
    micro_pair_count = 0
    for index, (start, end, speaker) in enumerate(ordered):
        for other_start, other_end, other_speaker in ordered[index + 1 :]:
            if other_start >= end:
                break
            overlap = min(end, other_end) - other_start
            if overlap <= 0:
                continue
            pair_count += 1
            micro_pair_count += overlap <= 100
            if speaker >= 0 and other_speaker >= 0:
                if speaker == other_speaker:
                    same_pair_count += 1
                else:
                    cross_pair_count += 1

    intervals = [(start, end) for start, end, _ in valid]
    speaker_zero = [(start, end) for start, end, speaker in valid if speaker == 0]
    speaker_one = [(start, end) for start, end, speaker in valid if speaker == 1]
    any_overlap_ms = _coverage_at_least_two(intervals)
    cross_overlap_ms = _intersection_duration(speaker_zero, speaker_one)
    duration_ms = max(0.0, call_end - call_start)
    return {
        "pair_count": pair_count,
        "cross_pair_count": cross_pair_count,
        "same_pair_count": same_pair_count,
        "micro_pair_count": micro_pair_count,
        "any_overlap_s": any_overlap_ms / 1000,
        "cross_overlap_s": cross_overlap_ms / 1000,
        "any_overlap_ratio": any_overlap_ms / duration_ms if duration_ms else 0.0,
        "cross_overlap_ratio": cross_overlap_ms / duration_ms if duration_ms else 0.0,
        "cross_share_of_overlap": cross_overlap_ms / any_overlap_ms if any_overlap_ms else 0.0,
        "invalid_interval_count": invalid_interval,
        "outside_call_count": outside_call,
    }


def load_deep_rows(
    rows: Sequence[ManifestRow], data_root: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read raw JSON once and retain only derived call/utterance measurements."""

    root = Path(data_root)
    records: list[dict[str, Any]] = []
    utterance_durations: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    overlap_rows: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    read_failures: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        try:
            with (root / row.json_path).open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            if not isinstance(document, dict):
                raise TypeError("JSON root must be an object")
            record = extract_call_record(document, row.split)
            utterances_raw = document.get("utterances")
            utterances = utterances_raw if isinstance(utterances_raw, list) else []
            call_utterance_durations: dict[int, list[float]] = defaultdict(list)
            for utterance in utterances:
                if not isinstance(utterance, dict):
                    continue
                start, end, speaker = (
                    utterance.get("startAt"),
                    utterance.get("endAt"),
                    utterance.get("speaker"),
                )
                if (
                    _valid_interval(start)
                    and _valid_interval(end)
                    and end > start
                    and speaker in {0, 1}
                ):
                    duration_s = (float(end) - float(start)) / 1000
                    utterance_durations[row.split][int(speaker)].append(duration_s)
                    call_utterance_durations[int(speaker)].append(duration_s)
            for speaker, label in ((0, "dispatcher"), (1, "caller")):
                values = call_utterance_durations[speaker]
                record[f"{label}_utterance_median_s"] = (
                    float(np.median(values)) if values else math.nan
                )
            records.append(record)
            call_start, call_end = document.get("startAt"), document.get("endAt")
            if _valid_interval(call_start) and _valid_interval(call_end) and call_end > call_start:
                overlap_rows[row.split].append(
                    call_overlap_metrics(utterances, float(call_start), float(call_end))
                )
        except Exception as exc:  # noqa: BLE001 - summarized, never exposes paths
            read_failures[type(exc).__name__] += 1
        if index % 5000 == 0 or index == len(rows):
            print(f"  deep-dive JSON {index:,}/{len(rows):,}")

    detail = {
        "utterance_durations": utterance_durations,
        "overlap_rows": overlap_rows,
        "read_failures": dict(sorted(read_failures.items())),
    }
    return records, detail


def cliffs_delta(left: Iterable[float], right: Iterable[float]) -> float:
    """Return Cliff's delta, positive when values in ``left`` tend to be larger."""

    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return 0.0
    combined = pd.Series(np.concatenate([a, b]))
    ranks = combined.rank(method="average").to_numpy(float)
    statistic = float(ranks[: a.size].sum() - a.size * (a.size + 1) / 2)
    return round(float(2 * statistic / (a.size * b.size) - 1), 6)


def paired_probability_difference(
    left: Iterable[float], right: Iterable[float]
) -> float:
    """Return P(left > right) - P(left < right) for aligned pairs."""

    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired effect inputs must have the same shape")
    finite = np.isfinite(a) & np.isfinite(b)
    differences = a[finite] - b[finite]
    if not differences.size:
        return 0.0
    return round(
        float(np.mean(differences > 0) - np.mean(differences < 0)), 6
    )


def bootstrap_median_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    seed: int = 42,
    iterations: int = 1000,
) -> dict[str, float | int | str]:
    """Bootstrap an independent-sample median difference (left minus right)."""

    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        estimates[index] = np.median(rng.choice(a, a.size, replace=True)) - np.median(
            rng.choice(b, b.size, replace=True)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": round(float(np.median(a) - np.median(b)), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
        "iterations": iterations,
        "resampling_unit": "observation",
    }


def bootstrap_paired_median_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    seed: int = 42,
    iterations: int = 5000,
) -> dict[str, float | int | str]:
    """Bootstrap a median difference while preserving aligned call pairs."""

    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap inputs must have the same shape")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if not a.size:
        raise ValueError("paired bootstrap requires at least one finite pair")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, a.size, a.size)
        estimates[index] = np.median(a[sampled]) - np.median(b[sampled])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": round(float(np.median(a) - np.median(b)), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
        "iterations": iterations,
        "resampling_unit": "call_pair",
        "n_pairs": int(a.size),
    }


def _effect_summary(left: Sequence[float], right: Sequence[float], unit: str) -> dict[str, Any]:
    return {
        "left": numeric_profile(left, unit),
        "right": numeric_profile(right, unit),
        "cliffs_delta_left_minus_right": cliffs_delta(left, right),
        "median_difference_bootstrap_95ci": bootstrap_median_difference(left, right),
    }


def _standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    pooled = math.sqrt((float(a.var()) + float(b.var())) / 2)
    return round(float((b.mean() - a.mean()) / pooled), 6) if pooled else 0.0


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.sort(left), np.sort(right)
    points = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.sort(left), np.sort(right)
    points = np.sort(np.unique(np.concatenate([a, b])))
    if points.size < 2:
        return 0.0
    cdf_a = np.searchsorted(a, points[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, points[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(points)))


def _split_comparison(frame: pd.DataFrame, column: str, unit: str) -> dict[str, Any]:
    training = frame.loc[frame["split"] == "Training", column].dropna().to_numpy(float)
    validation = frame.loc[frame["split"] == "Validation", column].dropna().to_numpy(float)
    return {
        "training": numeric_profile(training, unit),
        "validation": numeric_profile(validation, unit),
        "ks_statistic": round(_ks_statistic(training, validation), 6),
        "wasserstein_distance": round(_wasserstein_1d(training, validation), 6),
        "standardized_mean_difference_validation_minus_training": _standardized_mean_difference(
            training, validation
        ),
    }


def _overlap_summary(rows: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    pair_total = int(frame["pair_count"].sum())
    return {
        "n_calls": int(len(frame)),
        "calls_with_overlap": int((frame["pair_count"] > 0).sum()),
        "calls_with_cross_speaker_overlap": int((frame["cross_pair_count"] > 0).sum()),
        "pair_count_per_call": numeric_profile(frame["pair_count"], "count"),
        "overlap_time_s_per_call": numeric_profile(frame["any_overlap_s"], "seconds"),
        "overlap_time_ratio_per_call": numeric_profile(frame["any_overlap_ratio"], "ratio"),
        "cross_overlap_ratio_per_call": numeric_profile(frame["cross_overlap_ratio"], "ratio"),
        "cross_share_eligible_calls": int((frame["any_overlap_s"] > 0).sum()),
        "cross_share_of_overlap_per_call": numeric_profile(
            frame.loc[frame["any_overlap_s"] > 0, "cross_share_of_overlap"], "ratio"
        ),
        "pair_counts": {
            "all": pair_total,
            "cross_speaker": int(frame["cross_pair_count"].sum()),
            "same_speaker": int(frame["same_pair_count"].sum()),
            "micro_le_100ms": int(frame["micro_pair_count"].sum()),
            "micro_le_100ms_rate": round(
                float(frame["micro_pair_count"].sum() / pair_total), 6
            ) if pair_total else 0.0,
        },
        "range_errors": {
            "invalid_or_reversed_intervals": int(frame["invalid_interval_count"].sum()),
            "utterances_outside_call": int(frame["outside_call_count"].sum()),
        },
    }


def _symptom_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    symptoms = [tuple(record["symptoms"]) for record in records]
    n = len(symptoms)
    counts = Counter(label for labels in symptoms for label in labels)
    combinations = Counter(symptoms)
    cardinality = Counter(len(labels) for labels in symptoms)
    pairwise: dict[str, dict[str, Any]] = {}
    for index, left in enumerate(SYMPTOM_CLASSES):
        for right in SYMPTOM_CLASSES[index + 1 :]:
            both = sum(left in labels and right in labels for labels in symptoms)
            union = counts[left] + counts[right] - both
            p_left, p_right, p_both = counts[left] / n, counts[right] / n, both / n
            denominator = math.sqrt(
                p_left * (1 - p_left) * p_right * (1 - p_right)
            )
            phi = (p_both - p_left * p_right) / denominator if denominator else 0.0
            pairwise[f"{left}__{right}"] = {
                "left": left,
                "right": right,
                "count": both,
                "jaccard": round(both / union, 6) if union else 0.0,
                "conditional_right_given_left": round(both / counts[left], 6)
                if counts[left]
                else 0.0,
                "lift": round(p_both / (p_left * p_right), 6)
                if p_left and p_right
                else 0.0,
                "phi": round(phi, 6),
            }
    return {
        "n_calls": n,
        "prevalence": {
            label: {
                "count": counts[label],
                "pct": round(100 * counts[label] / n, 3),
            }
            for label in SYMPTOM_CLASSES
        },
        "cardinality_distribution": {
            str(key): {
                "count": value,
                "pct": round(100 * value / n, 3),
            }
            for key, value in sorted(cardinality.items())
        },
        "top_combinations": [
            {
                "labels": list(labels),
                "count": count,
                "pct": round(100 * count / n, 3),
            }
            for labels, count in combinations.most_common(10)
        ],
        "pairwise": pairwise,
    }


def _acoustic_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"by_split": {}, "caller_gender": {}}
    for split in ("Training", "Validation"):
        result["by_split"][split] = {}
        subset = frame[frame["split"] == split]
        for feature in ACOUSTIC_FEATURES:
            paired = subset.pivot(index="stem", columns="speaker", values=feature).dropna()
            caller = paired[1].to_numpy(float)
            dispatcher = paired[0].to_numpy(float)
            effect = _effect_summary(
                caller, dispatcher, "ratio" if "ratio" in feature else "feature_value"
            )
            effect["median_difference_bootstrap_95ci"] = (
                bootstrap_paired_median_difference(caller, dispatcher)
            )
            effect["paired_probability_difference"] = (
                paired_probability_difference(caller, dispatcher)
            )
            result["by_split"][split][feature] = effect

    caller_frame = frame[frame["speaker"] == 1]
    for feature in ("duration_sec", "rms_mean", "zcr_mean", "spectral_centroid_mean"):
        female = caller_frame.loc[caller_frame["gender"] == "F", feature].dropna().to_numpy(float)
        male = caller_frame.loc[caller_frame["gender"] == "M", feature].dropna().to_numpy(float)
        result["caller_gender"][feature] = _effect_summary(female, male, "feature_value")

    sampled_calls = frame.drop_duplicates(["split", "stem"])
    result["sample_design"] = {
        "method": "proportional gender stratification within each split",
        "weighting": "none",
        "seed": 42,
        "selected_calls": {
            split: int((sampled_calls["split"] == split).sum())
            for split in ("Training", "Validation")
        },
        "gender_counts": {
            split: {
                gender: int(
                    (
                        (sampled_calls["split"] == split)
                        & (sampled_calls["gender"] == gender)
                    ).sum()
                )
                for gender in ("F", "M")
            }
            for split in ("Training", "Validation")
        },
        "paired_calls": {
            split: int(
                frame[frame["split"] == split]
                .pivot(index="stem", columns="speaker", values="duration_sec")
                .dropna()
                .shape[0]
            )
            for split in ("Training", "Validation")
        },
        "exclusions": (
            "invalid or out-of-range utterance intervals, calls without both "
            "speaker feature rows, and waveform read failures"
        ),
    }
    return result


def build_deep_summary(
    records: Sequence[Mapping[str, Any]],
    detail: Mapping[str, Any],
    waveform_frame: pd.DataFrame,
    full_summary: Mapping[str, Any],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the deep-dive statistics from derived call rows and the waveform cache."""

    if not records:
        raise ValueError(
            "no call JSON could be read; check --data-root against the manifest "
            f"(read failures: {dict(detail['read_failures'])})"
        )
    frame = pd.DataFrame(records)
    split_comparison = {
        "call_duration_s": _split_comparison(frame, "call_duration_s", "seconds"),
        "utterance_count": _split_comparison(frame, "utterance_count", "count"),
        "caller_speaking_share": _split_comparison(
            frame, "caller_speaking_share", "ratio"
        ),
    }

    gender: dict[str, Any] = {}
    for column, unit in (
        ("call_duration_s", "seconds"),
        ("caller_speaking_s", "seconds"),
        ("caller_utterance_count", "count"),
        ("caller_speaking_share", "ratio"),
    ):
        female = frame.loc[frame["gender"] == "F", column].dropna().to_numpy(float)
        male = frame.loc[frame["gender"] == "M", column].dropna().to_numpy(float)
        gender[column] = _effect_summary(female, male, unit)

    speaker: dict[str, Any] = {}
    for split in ("Training", "Validation"):
        dispatcher = detail["utterance_durations"][split][0]
        caller = detail["utterance_durations"][split][1]
        call_pairs = frame.loc[
            frame["split"] == split,
            ["caller_utterance_median_s", "dispatcher_utterance_median_s"],
        ].dropna()
        caller_call_median = call_pairs["caller_utterance_median_s"].to_numpy(float)
        dispatcher_call_median = call_pairs[
            "dispatcher_utterance_median_s"
        ].to_numpy(float)
        within_call_difference = caller_call_median - dispatcher_call_median
        speaker[split] = {
            "dispatcher": numeric_profile(dispatcher, "seconds"),
            "caller": numeric_profile(caller, "seconds"),
            "cliffs_delta_caller_minus_dispatcher": cliffs_delta(caller, dispatcher),
            "utterance_weighted_marginal_effect": {
                "estimand": (
                    "P(random caller utterance > random dispatcher utterance) - "
                    "P(random caller utterance < random dispatcher utterance)"
                ),
                "cliffs_delta": cliffs_delta(caller, dispatcher),
            },
            "within_call_median_effect": {
                "n_calls": int(len(call_pairs)),
                "caller_minus_dispatcher_seconds": numeric_profile(
                    within_call_difference, "seconds"
                ),
                "paired_probability_difference": paired_probability_difference(
                    caller_call_median, dispatcher_call_median
                ),
            },
            "duration_bucket_pct": {
                role: {
                    "lt_0_5s": round(100 * np.mean(np.asarray(values) < 0.5), 3),
                    "lt_1s": round(100 * np.mean(np.asarray(values) < 1.0), 3),
                    "ge_4s": round(100 * np.mean(np.asarray(values) >= 4.0), 3),
                }
                for role, values in (("dispatcher", dispatcher), ("caller", caller))
            },
        }

    overlap = {
        split: _overlap_summary(detail["overlap_rows"][split])
        for split in ("Training", "Validation")
    }
    symptoms = {
        scope: _symptom_summary(
            [record for record in records if scope == "Combined" or record["split"] == scope]
        )
        for scope in ("Training", "Validation", "Combined")
    }

    return {
        "meta": {
            "seed": seed,
            "n_calls": len(records),
            "n_training": int((frame["split"] == "Training").sum()),
            "n_validation": int((frame["split"] == "Validation").sum()),
            "raw_transcript_exported": False,
            "address_exported": False,
            "json_read_failures": detail["read_failures"],
        },
        "split_comparison": split_comparison,
        "overlap": overlap,
        "gender": gender,
        "speaker_utterances": speaker,
        "symptoms": symptoms,
        "acoustic_sample": _acoustic_summary(waveform_frame),
        "existing_quality": {
            "identifier_overlap_count": full_summary["cross_split"][
                "identifier_overlap_count"
            ],
            "exact_transcript_hash_overlap_count": full_summary["cross_split"][
                "exact_transcript_hash_overlap_count"
            ],
            "validation_token_oov": full_summary["cross_split"][
                "validation_token_oov"
            ],
            "wav_training": full_summary["training"]["common"]["wav"],
            "wav_validation": full_summary["validation"]["common"]["wav"],
        },
    }
