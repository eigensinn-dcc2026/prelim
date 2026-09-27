"""Reusable full-dataset EDA for the DCC 2026 emergency-call corpus.

The module deliberately separates exhaustive label/header profiling from
sampled waveform decoding.  No raw transcript or address value is written to
an output artifact; only aggregate counts and hashes used for overlap checks
are retained in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .audio import read_wav_frames, read_wav_info
from .manifest import ManifestRow
from .mission2 import milliseconds_to_frame
from .mission2_features import FEATURE_COLUMNS
from .mission3 import SYMPTOM_CLASSES


SYMPTOM_SET = frozenset(SYMPTOM_CLASSES)
ROOT_FIELDS = (
    "_id",
    "audioPath",
    "recordId",
    "status",
    "startAt",
    "endAt",
    "utterances",
    "mediaType",
    "gender",
    "address",
    "disasterLarge",
    "disasterMedium",
    "urgencyLevel",
    "sentiment",
    "symptom",
    "triage",
)
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
PRIVACY_MARKER = "[개인정보]"
DURATION_BUCKETS_MS = (
    ("<0.5s", 0, 500),
    ("0.5-1s", 500, 1000),
    ("1-2s", 1000, 2000),
    ("2-4s", 2000, 4000),
    (">=4s", 4000, math.inf),
)
EDA_FEATURE_MAX_SECONDS = 15.0
EDA_N_FFT = 512
EDA_HOP_LENGTH = 160
EDA_N_MFCC = 20
EDA_N_MELS = 128


@dataclass(frozen=True)
class CallProfile:
    """Small, privacy-safe record retained for cross-split and sampling work."""

    row: ManifestRow
    document_id: str | None
    record_id: str | None
    gender: str
    symptoms: tuple[str, ...]
    transcript_hash: str


@dataclass
class SplitAnalysis:
    summary: dict[str, Any]
    profiles: list[CallProfile]
    token_counts: Counter[str]
    transcript_hashes: Counter[str]
    identifier_counts: dict[str, Counter[str]]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round(value: float | int | None, digits: int = 6) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def numeric_profile(values: Iterable[float], unit: str) -> dict[str, Any]:
    """Return a JSON-safe numeric profile with robust percentiles."""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "std": None,
            "p1": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "zero_count": 0,
            "negative_count": 0,
            "unit": unit,
        }

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = p * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return ordered[low]
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "n": len(ordered),
        "min": _round(ordered[0]),
        "mean": _round(statistics.fmean(ordered)),
        "std": _round(statistics.pstdev(ordered)),
        "p1": _round(percentile(0.01)),
        "p5": _round(percentile(0.05)),
        "p25": _round(percentile(0.25)),
        "p50": _round(percentile(0.50)),
        "p75": _round(percentile(0.75)),
        "p90": _round(percentile(0.90)),
        "p95": _round(percentile(0.95)),
        "p99": _round(percentile(0.99)),
        "max": _round(ordered[-1]),
        "zero_count": sum(value == 0 for value in ordered),
        "negative_count": sum(value < 0 for value in ordered),
        "unit": unit,
    }


def fixed_histogram(
    values: Iterable[float], edges: Iterable[float], unit: str
) -> dict[str, Any]:
    edge_list = [float(edge) for edge in edges]
    counts = [0] * (len(edge_list) - 1)
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            continue
        for index in range(len(edge_list) - 1):
            left, right = edge_list[index], edge_list[index + 1]
            if left <= value < right or (
                index == len(counts) - 1 and value == right
            ):
                counts[index] += 1
                break
    return {"edges": edge_list, "counts": counts, "unit": unit}


def distribution(counter: Counter[Any], total: int | None = None) -> dict[str, Any]:
    denominator = total if total is not None else sum(counter.values())
    return {
        str(key): {
            "count": int(value),
            "pct": round(100 * value / denominator, 3) if denominator else 0.0,
        }
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def union_duration_ms(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if _is_number(start) and _is_number(end) and end > start
    )
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def overlap_pair_count(intervals: Iterable[tuple[float, float]]) -> int:
    ordered = sorted(intervals)
    count = 0
    active_end = -math.inf
    for start, end in ordered:
        if start < active_end:
            count += 1
        active_end = max(active_end, end)
    return count


def _duration_bucket(duration_ms: float) -> str:
    for label, lower, upper in DURATION_BUCKETS_MS:
        if lower <= duration_ms < upper:
            return label
    return ">=4s"


def _duplicate_summary(counter: Counter[str]) -> dict[str, int]:
    duplicate_groups = sum(1 for value in counter.values() if value > 1)
    duplicate_rows = sum(value - 1 for value in counter.values() if value > 1)
    return {
        "non_null_count": sum(counter.values()),
        "distinct_count": len(counter),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows_beyond_first": duplicate_rows,
    }


def _normalised_transcript_hash(utterances: list[object]) -> str:
    parts: list[str] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        speaker = utterance.get("speaker", "?")
        text = utterance.get("text", "")
        normalised = " ".join(str(text).split()).casefold()
        parts.append(f"{speaker}:{normalised}")
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_document(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except Exception as exc:  # noqa: BLE001 - surfaced in aggregate error counts
        return None, type(exc).__name__
    if not isinstance(document, dict):
        return None, "RootNotObject"
    return document, None


def _header_number(header: Mapping[str, str], field: str) -> float | None:
    raw = header.get(field, "")
    if raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def analyze_split(
    rows: list[ManifestRow],
    data_root: Path,
    headers: Mapping[tuple[str, str], Mapping[str, str]],
) -> SplitAnalysis:
    """Profile every provided manifest row and return aggregate-only output."""

    missing_fields: Counter[str] = Counter()
    null_fields: Counter[str] = Counter()
    invalid_types: Counter[str] = Counter()
    read_failures: Counter[str] = Counter()
    timestamp_issues: Counter[str] = Counter()
    categorical: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "mediaType",
            "status",
            "disasterLarge",
            "disasterMedium",
            "urgencyLevel",
            "sentiment",
            "triage",
        )
    }
    identifier_counts = {
        "stem": Counter(row.stem for row in rows),
        "_id": Counter(),
        "recordId": Counter(),
    }
    gender_counter: Counter[str] = Counter()
    symptom_counter: Counter[str] = Counter()
    unknown_symptoms: Counter[str] = Counter()
    symptom_cardinality: list[float] = []
    symptom_cooccurrence: dict[str, Counter[str]] = {
        label: Counter() for label in SYMPTOM_CLASSES
    }
    duplicate_symptom_labels = 0
    zero_known_symptom_calls = 0

    call_duration_s: list[float] = []
    wav_duration_s: list[float] = []
    json_wav_diff_ms: list[float] = []
    utterances_per_call: list[float] = []
    utterance_duration_s: list[float] = []
    utterance_duration_by_speaker: dict[int, list[float]] = defaultdict(list)
    utterance_count_by_speaker: Counter[int] = Counter()
    duration_total_by_speaker_ms: Counter[int] = Counter()
    duration_bucket_by_speaker: dict[int, Counter[str]] = defaultdict(Counter)
    caller_duration_s: list[float] = []
    caller_utterance_count: list[float] = []
    caller_share: list[float] = []
    caller_overlap_ratio: list[float] = []
    speaker_one_share: list[float] = []
    alternation_counts: list[float] = []
    overlap_pair_counts: list[float] = []
    overlap_excess_s: list[float] = []
    gender_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    text_length_chars: list[float] = []
    text_length_by_speaker: dict[int, list[float]] = defaultdict(list)
    token_counts: Counter[str] = Counter()
    blank_text_count = 0
    privacy_utterance_count = 0
    total_utterances = 0

    wav_channels: Counter[str] = Counter()
    wav_sample_rates: Counter[str] = Counter()
    wav_sample_widths: Counter[str] = Counter()
    wav_errors: Counter[str] = Counter()

    profiles: list[CallProfile] = []
    transcript_hashes: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        document, error = _read_document(data_root / row.json_path)
        if document is None:
            read_failures[error or "Unknown"] += 1
            continue

        for field in ROOT_FIELDS:
            if field not in document:
                missing_fields[field] += 1
            elif document[field] is None:
                null_fields[field] += 1

        document_id = document.get("_id")
        record_id = document.get("recordId")
        if isinstance(document_id, str) and document_id:
            identifier_counts["_id"][document_id] += 1
        elif document_id is not None:
            invalid_types["_id"] += 1
        if isinstance(record_id, str) and record_id:
            identifier_counts["recordId"][record_id] += 1
        elif record_id is not None:
            invalid_types["recordId"] += 1

        for field, counter in categorical.items():
            value = document.get(field)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                counter[str(value)] += 1
            elif value is not None:
                invalid_types[field] += 1

        gender = document.get("gender")
        gender_key = gender if isinstance(gender, str) and gender else "__missing__"
        gender_counter[gender_key] += 1
        if gender is not None and not isinstance(gender, str):
            invalid_types["gender"] += 1

        start_at = document.get("startAt")
        end_at = document.get("endAt")
        call_duration_ms: float | None = None
        if _is_number(start_at) and _is_number(end_at) and end_at > start_at:
            call_duration_ms = float(end_at) - float(start_at)
            call_duration_s.append(call_duration_ms / 1000)
            gender_metrics[gender_key]["call_duration_s"].append(call_duration_ms / 1000)
        else:
            timestamp_issues["invalid_call_interval"] += 1

        header = headers.get((row.split, row.stem), {})
        if header.get("error"):
            wav_errors[header.get("error", "Unknown").split(":", 1)[0]] += 1
        else:
            for field, counter in (
                ("channels", wav_channels),
                ("sample_rate", wav_sample_rates),
                ("sample_width", wav_sample_widths),
            ):
                if header.get(field, ""):
                    counter[header[field]] += 1
            wav_duration = _header_number(header, "duration_s")
            if wav_duration is not None:
                wav_duration_s.append(wav_duration)
                if call_duration_ms is not None:
                    json_wav_diff_ms.append(call_duration_ms - wav_duration * 1000)

        utterances_raw = document.get("utterances")
        utterances = utterances_raw if isinstance(utterances_raw, list) else []
        if not isinstance(utterances_raw, list):
            invalid_types["utterances"] += 1
        utterances_per_call.append(float(len(utterances)))
        transcript_hash = _normalised_transcript_hash(utterances)
        transcript_hashes[transcript_hash] += 1

        all_intervals: list[tuple[float, float]] = []
        caller_intervals: list[tuple[float, float]] = []
        per_speaker_duration_ms: Counter[int] = Counter()
        per_speaker_count: Counter[int] = Counter()
        speaker_sequence: list[int] = []
        previous_start: float | None = None

        for utterance in utterances:
            total_utterances += 1
            if not isinstance(utterance, dict):
                invalid_types["utterance_item"] += 1
                continue
            u_start = utterance.get("startAt")
            u_end = utterance.get("endAt")
            speaker = utterance.get("speaker")
            text = utterance.get("text")

            valid_interval = _is_number(u_start) and _is_number(u_end) and u_end > u_start
            if valid_interval:
                u_start_f, u_end_f = float(u_start), float(u_end)
                duration_ms = u_end_f - u_start_f
                all_intervals.append((u_start_f, u_end_f))
                utterance_duration_s.append(duration_ms / 1000)
                if previous_start is not None and u_start_f < previous_start:
                    timestamp_issues["utterances_not_sorted"] += 1
                previous_start = u_start_f
                if call_duration_ms is not None and (
                    u_start_f < float(start_at) or u_end_f > float(end_at)
                ):
                    timestamp_issues["utterance_outside_call"] += 1
                wav_duration = _header_number(header, "duration_s")
                if wav_duration is not None and u_end_f > wav_duration * 1000 + 50:
                    timestamp_issues["utterance_outside_wav_50ms"] += 1
            else:
                timestamp_issues["invalid_utterance_interval"] += 1
                duration_ms = None

            if isinstance(speaker, int) and speaker in {0, 1}:
                speaker_sequence.append(speaker)
                utterance_count_by_speaker[speaker] += 1
                per_speaker_count[speaker] += 1
                if duration_ms is not None:
                    utterance_duration_by_speaker[speaker].append(duration_ms / 1000)
                    duration_total_by_speaker_ms[speaker] += duration_ms
                    per_speaker_duration_ms[speaker] += duration_ms
                    duration_bucket_by_speaker[speaker][_duration_bucket(duration_ms)] += 1
                    if speaker == 1:
                        caller_intervals.append((float(u_start), float(u_end)))
            else:
                invalid_types["speaker"] += 1

            if isinstance(text, str):
                stripped = text.strip()
                text_length_chars.append(float(len(stripped)))
                if isinstance(speaker, int) and speaker in {0, 1}:
                    text_length_by_speaker[speaker].append(float(len(stripped)))
                if not stripped:
                    blank_text_count += 1
                if PRIVACY_MARKER in text:
                    privacy_utterance_count += 1
                token_counts.update(token.casefold() for token in TOKEN_RE.findall(text))
            else:
                invalid_types["text"] += 1

        valid_duration_total = sum(per_speaker_duration_ms.values())
        caller_union = union_duration_ms(caller_intervals)
        caller_sum = float(per_speaker_duration_ms.get(1, 0))
        caller_duration_s.append(caller_union / 1000)
        caller_utterance_count.append(float(per_speaker_count.get(1, 0)))
        caller_share.append(caller_union / call_duration_ms if call_duration_ms else 0.0)
        caller_overlap_ratio.append(
            max(0.0, caller_sum - caller_union) / caller_sum if caller_sum else 0.0
        )
        speaker_one_share.append(
            caller_sum / valid_duration_total if valid_duration_total else 0.0
        )
        alternation_counts.append(
            float(sum(a != b for a, b in zip(speaker_sequence, speaker_sequence[1:])))
        )
        pair_count = overlap_pair_count(all_intervals)
        overlap_pair_counts.append(float(pair_count))
        overlap_excess = max(
            0.0,
            sum(end - start for start, end in all_intervals)
            - union_duration_ms(all_intervals),
        )
        overlap_excess_s.append(overlap_excess / 1000)
        if pair_count:
            timestamp_issues["calls_with_overlapping_utterances"] += 1

        gender_metrics[gender_key]["caller_duration_s"].append(caller_union / 1000)
        gender_metrics[gender_key]["caller_utterance_count"].append(
            float(per_speaker_count.get(1, 0))
        )
        gender_metrics[gender_key]["caller_share"].append(
            caller_union / call_duration_ms if call_duration_ms else 0.0
        )

        symptoms_raw = document.get("symptom")
        symptoms_list = symptoms_raw if isinstance(symptoms_raw, list) else []
        if not isinstance(symptoms_raw, list):
            invalid_types["symptom"] += 1
        string_symptoms = [item for item in symptoms_list if isinstance(item, str)]
        duplicate_symptom_labels += len(string_symptoms) - len(set(string_symptoms))
        unique_known = tuple(label for label in SYMPTOM_CLASSES if label in set(string_symptoms))
        symptom_counter.update(unique_known)
        unknown_symptoms.update(
            label for label in set(string_symptoms) if label not in SYMPTOM_SET
        )
        symptom_cardinality.append(float(len(unique_known)))
        if not unique_known:
            zero_known_symptom_calls += 1
        for left in unique_known:
            for right in unique_known:
                symptom_cooccurrence[left][right] += 1

        profiles.append(
            CallProfile(
                row=row,
                document_id=document_id if isinstance(document_id, str) else None,
                record_id=record_id if isinstance(record_id, str) else None,
                gender=gender_key,
                symptoms=unique_known,
                transcript_hash=transcript_hash,
            )
        )

        if index % 2000 == 0 or index == len(rows):
            print(f"  json profile {row.split}: {index:,}/{len(rows):,}")

    parsed = len(profiles)
    completeness = {
        field: {
            "missing": int(missing_fields[field]),
            "null": int(null_fields[field]),
            "non_null_pct": round(
                100 * (parsed - missing_fields[field] - null_fields[field]) / parsed, 3
            )
            if parsed
            else 0.0,
        }
        for field in ROOT_FIELDS
    }
    gender_breakdown = {
        gender: {
            "n_calls": len(metrics["caller_duration_s"]),
            "call_duration_s": numeric_profile(metrics["call_duration_s"], "seconds"),
            "caller_duration_s": numeric_profile(
                metrics["caller_duration_s"], "seconds"
            ),
            "caller_utterances": numeric_profile(
                metrics["caller_utterance_count"], "count"
            ),
            "caller_share": numeric_profile(metrics["caller_share"], "ratio"),
        }
        for gender, metrics in sorted(gender_metrics.items())
    }

    summary = {
        "common": {
            "manifest_rows": len(rows),
            "json_parsed": parsed,
            "json_read_failures": dict(sorted(read_failures.items())),
            "field_completeness": completeness,
            "invalid_type_counts": dict(sorted(invalid_types.items())),
            "identifiers": {
                name: _duplicate_summary(counter)
                for name, counter in identifier_counts.items()
            },
            "call_duration_s": numeric_profile(call_duration_s, "seconds"),
            "call_duration_histogram": fixed_histogram(
                call_duration_s, [0, 30, 45, 60, 90, 120, 180, 300], "seconds"
            ),
            "utterances_per_call": numeric_profile(utterances_per_call, "count"),
            "utterance_duration_s": numeric_profile(
                utterance_duration_s, "seconds"
            ),
            "wav": {
                "channels": dict(sorted(wav_channels.items())),
                "sample_rates_hz": dict(sorted(wav_sample_rates.items())),
                "sample_width_bytes": dict(sorted(wav_sample_widths.items())),
                "read_errors": dict(sorted(wav_errors.items())),
                "duration_s": numeric_profile(wav_duration_s, "seconds"),
                "json_minus_wav_duration_ms": numeric_profile(
                    json_wav_diff_ms, "milliseconds"
                ),
            },
            "timestamp_issue_counts": dict(sorted(timestamp_issues.items())),
            "context_distributions": {
                field: distribution(counter) for field, counter in categorical.items()
            },
        },
        "mission1": {
            "gender_distribution": distribution(gender_counter, parsed),
            "caller_duration_s": numeric_profile(caller_duration_s, "seconds"),
            "caller_utterances_per_call": numeric_profile(
                caller_utterance_count, "count"
            ),
            "caller_share_of_call": numeric_profile(caller_share, "ratio"),
            "caller_within_speaker_overlap_ratio": numeric_profile(
                caller_overlap_ratio, "ratio"
            ),
            "by_gender": gender_breakdown,
        },
        "mission2": {
            "utterance_count_by_speaker": distribution(utterance_count_by_speaker),
            "utterance_duration_total_by_speaker": distribution(
                duration_total_by_speaker_ms
            ),
            "utterance_duration_s_by_speaker": {
                str(speaker): numeric_profile(values, "seconds")
                for speaker, values in sorted(utterance_duration_by_speaker.items())
            },
            "duration_bucket_count_by_speaker": {
                str(speaker): {
                    label: int(counter.get(label, 0))
                    for label, _, _ in DURATION_BUCKETS_MS
                }
                for speaker, counter in sorted(duration_bucket_by_speaker.items())
            },
            "speaker_1_duration_share_per_call": numeric_profile(
                speaker_one_share, "ratio"
            ),
            "speaker_alternations_per_call": numeric_profile(
                alternation_counts, "count"
            ),
            "overlap_pairs_per_call": numeric_profile(
                overlap_pair_counts, "count"
            ),
            "overlap_excess_s_per_call": numeric_profile(
                overlap_excess_s, "seconds"
            ),
        },
        "mission3": {
            "target_symptom_prevalence": {
                label: {
                    "count": int(symptom_counter[label]),
                    "pct_of_calls": round(100 * symptom_counter[label] / parsed, 3)
                    if parsed
                    else 0.0,
                }
                for label in SYMPTOM_CLASSES
            },
            "label_cardinality": numeric_profile(symptom_cardinality, "labels_per_call"),
            "label_density_mean": round(
                statistics.fmean(symptom_cardinality) / len(SYMPTOM_CLASSES), 6
            )
            if symptom_cardinality
            else 0.0,
            "zero_target_symptom_calls": zero_known_symptom_calls,
            "duplicate_labels_within_calls": duplicate_symptom_labels,
            "unknown_symptom_labels": dict(unknown_symptoms.most_common()),
            "cooccurrence": {
                left: {right: int(symptom_cooccurrence[left][right]) for right in SYMPTOM_CLASSES}
                for left in SYMPTOM_CLASSES
            },
            "text_length_chars": numeric_profile(text_length_chars, "characters"),
            "text_length_chars_by_speaker": {
                str(speaker): numeric_profile(values, "characters")
                for speaker, values in sorted(text_length_by_speaker.items())
            },
            "blank_text_utterances": blank_text_count,
            "privacy_marker_utterances": privacy_utterance_count,
            "privacy_marker_utterance_pct": round(
                100 * privacy_utterance_count / total_utterances, 3
            )
            if total_utterances
            else 0.0,
            "token_count": sum(token_counts.values()),
            "vocabulary_size": len(token_counts),
        },
    }
    return SplitAnalysis(
        summary=summary,
        profiles=profiles,
        token_counts=token_counts,
        transcript_hashes=transcript_hashes,
        identifier_counts=identifier_counts,
    )


def total_variation(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    keys = set(left) | set(right)
    if not left_total or not right_total:
        return 0.0
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def compare_splits(training: SplitAnalysis, validation: SplitAnalysis) -> dict[str, Any]:
    identifier_overlap = {}
    for name in ("stem", "_id", "recordId"):
        left = set(training.identifier_counts[name])
        right = set(validation.identifier_counts[name])
        identifier_overlap[name] = len(left & right)

    train_gender = Counter(profile.gender for profile in training.profiles)
    val_gender = Counter(profile.gender for profile in validation.profiles)
    train_symptoms = Counter(
        symptom for profile in training.profiles for symptom in profile.symptoms
    )
    val_symptoms = Counter(
        symptom for profile in validation.profiles for symptom in profile.symptoms
    )
    train_calls = len(training.profiles)
    val_calls = len(validation.profiles)

    return {
        "identifier_overlap_count": identifier_overlap,
        "exact_transcript_hash_overlap_count": len(
            set(training.transcript_hashes) & set(validation.transcript_hashes)
        ),
        "validation_token_oov": {
            "token_occurrences": sum(validation.token_counts.values()),
            "oov_occurrences": sum(
                count
                for token, count in validation.token_counts.items()
                if token not in training.token_counts
            ),
            "oov_rate": round(
                sum(
                    count
                    for token, count in validation.token_counts.items()
                    if token not in training.token_counts
                )
                / (sum(validation.token_counts.values()) or 1),
                6,
            ),
        },
        "gender_total_variation": round(total_variation(train_gender, val_gender), 6),
        "symptom_prevalence_delta_percentage_points": {
            label: round(
                100 * val_symptoms[label] / (val_calls or 1)
                - 100 * train_symptoms[label] / (train_calls or 1),
                3,
            )
            for label in SYMPTOM_CLASSES
        },
        "gender_delta_percentage_points": {
            label: round(
                100 * val_gender[label] / (val_calls or 1)
                - 100 * train_gender[label] / (train_calls or 1),
                3,
            )
            for label in sorted(set(train_gender) | set(val_gender))
        },
    }


def stratified_sample(
    profiles: list[CallProfile], n: int, seed: int
) -> list[CallProfile]:
    """Proportionally sample calls by gender with deterministic allocation."""

    if n <= 0 or n >= len(profiles):
        return list(profiles)
    groups: dict[str, list[CallProfile]] = defaultdict(list)
    for profile in profiles:
        groups[profile.gender].append(profile)
    target = min(n, len(profiles))
    allocation: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for label, values in sorted(groups.items()):
        exact = target * len(values) / len(profiles)
        allocation[label] = min(len(values), int(math.floor(exact)))
        fractions.append((exact - math.floor(exact), label))
    remaining = target - sum(allocation.values())
    for _, label in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if allocation[label] < len(groups[label]):
            allocation[label] += 1
            remaining -= 1
    rng = random.Random(seed)
    selected: list[CallProfile] = []
    for label, values in sorted(groups.items()):
        selected.extend(rng.sample(values, allocation[label]))
    return sorted(selected, key=lambda profile: (profile.row.split, profile.row.stem))


def _mel_frequency(hz: np.ndarray | float) -> np.ndarray:
    values = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + values / 700.0)


def _inverse_mel_frequency(mel: np.ndarray | float) -> np.ndarray:
    values = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (values / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    mel_points = np.linspace(
        _mel_frequency(0.0),
        _mel_frequency(sample_rate / 2),
        EDA_N_MELS + 2,
    )
    frequencies = _inverse_mel_frequency(mel_points)
    bins = np.floor((EDA_N_FFT + 1) * frequencies / sample_rate).astype(int)
    bins = np.clip(bins, 0, EDA_N_FFT // 2)
    filters = np.zeros((EDA_N_MELS, EDA_N_FFT // 2 + 1), dtype=np.float64)
    for index in range(EDA_N_MELS):
        left, centre, right = bins[index : index + 3]
        if centre <= left:
            centre = min(left + 1, EDA_N_FFT // 2)
        if right <= centre:
            right = min(centre + 1, EDA_N_FFT // 2)
        if centre > left:
            filters[index, left:centre] = (
                np.arange(left, centre) - left
            ) / (centre - left)
        if right > centre:
            filters[index, centre:right] = (
                right - np.arange(centre, right)
            ) / (right - centre)
    return filters


def _dct_basis(n_input: int, n_output: int) -> np.ndarray:
    samples = np.arange(n_input, dtype=np.float64) + 0.5
    frequencies = np.arange(n_output, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / n_input * frequencies * samples)
    basis[0] *= math.sqrt(1 / n_input)
    basis[1:] *= math.sqrt(2 / n_input)
    return basis


def fast_acoustic_feature_vector(
    segment: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Return a bounded-cost 52-d MFCC/RMS/ZCR/spectral EDA vector.

    This keeps the baseline feature schema and definitions, but uses a small
    NumPy implementation instead of invoking librosa's expensive first-call
    JIT path.  It is intended for distributional EDA, not as a drop-in model
    feature replacement.
    """

    signal = np.asarray(segment, dtype=np.float32).reshape(-1)
    max_samples = int(round(EDA_FEATURE_MAX_SECONDS * sample_rate))
    signal = signal[:max_samples]
    if len(signal) < EDA_N_FFT:
        signal = np.pad(signal, (0, EDA_N_FFT - len(signal)))
    frame_count = 1 + max(0, (len(signal) - EDA_N_FFT) // EDA_HOP_LENGTH)
    starts = np.arange(frame_count) * EDA_HOP_LENGTH
    frames = np.stack([signal[start : start + EDA_N_FFT] for start in starts])
    windowed = frames.astype(np.float64) * np.hanning(EDA_N_FFT)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = spectrum**2

    mel_energy = power @ _mel_filterbank(sample_rate).T
    log_mel = 10.0 * np.log10(np.maximum(mel_energy, 1e-12))
    mfcc = log_mel @ _dct_basis(EDA_N_MELS, EDA_N_MFCC).T

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    frequencies = np.fft.rfftfreq(EDA_N_FFT, d=1 / sample_rate)
    magnitude_sum = np.maximum(spectrum.sum(axis=1), 1e-12)
    centroid = (spectrum * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt(
        (spectrum * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / magnitude_sum
    )
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_index = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = frequencies[rolloff_index]
    flatness = np.exp(np.mean(np.log(np.maximum(spectrum, 1e-12)), axis=1)) / np.maximum(
        np.mean(spectrum, axis=1), 1e-12
    )

    parts = [
        mfcc.mean(axis=0),
        mfcc.std(axis=0),
        [rms.mean(), rms.std()],
        [zcr.mean(), zcr.std()],
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [bandwidth.mean(), bandwidth.std()],
        [flatness.mean(), flatness.std()],
    ]
    vector = np.concatenate(
        [np.asarray(part, dtype=np.float64).ravel() for part in parts]
    )
    if vector.shape != (52,):
        raise ValueError(f"expected 52 acoustic features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_waveform_feature_rows(
    profile: CallProfile, data_root: Path
) -> list[dict[str, Any]]:
    """Extract one 53-d feature vector per speaker after concatenating turns."""

    document, error = _read_document(data_root / profile.row.json_path)
    if document is None:
        raise ValueError(f"cannot read JSON: {error}")
    info = read_wav_info(data_root / profile.row.wav_path)
    audio = read_wav_frames(data_root / profile.row.wav_path, 0, info.frame_count)
    if audio.waveform.ndim != 2 or audio.waveform.shape[0] != 1:
        raise ValueError(f"expected mono WAV, got shape {audio.waveform.shape}")

    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for utterance in document.get("utterances", []):
        if not isinstance(utterance, dict):
            continue
        speaker = utterance.get("speaker")
        start_at = utterance.get("startAt")
        end_at = utterance.get("endAt")
        if speaker not in {0, 1} or not _is_number(start_at) or not _is_number(end_at):
            continue
        start = milliseconds_to_frame(float(start_at), info.sample_rate)
        end = milliseconds_to_frame(float(end_at), info.sample_rate)
        if start < 0 or end <= start or end > info.frame_count:
            continue
        intervals[int(speaker)].append((start, end))

    result: list[dict[str, Any]] = []
    for speaker in (0, 1):
        speaker_intervals = intervals.get(speaker, [])
        if not speaker_intervals:
            continue
        waveform = np.concatenate(
            [audio.waveform[0, start:end] for start, end in speaker_intervals]
        ).astype(np.float32, copy=False)
        duration_sec = len(waveform) / info.sample_rate
        feature_waveform = waveform[: int(round(EDA_FEATURE_MAX_SECONDS * info.sample_rate))]
        vector = fast_acoustic_feature_vector(feature_waveform, info.sample_rate)
        feature_values = np.concatenate(
            [np.asarray([duration_sec], dtype=np.float32), vector]
        )
        row = {
            "split": profile.row.split,
            "stem": profile.row.stem,
            "gender": profile.gender,
            "speaker": speaker,
            "n_utterances": len(speaker_intervals),
            "feature_audio_duration_sec": len(feature_waveform) / info.sample_rate,
            "silence_ratio_abs_le_0_01": float(np.mean(np.abs(waveform) <= 0.01)),
            "peak_abs": float(np.max(np.abs(waveform))),
            "clipping_ratio_abs_ge_0_999": float(np.mean(np.abs(waveform) >= 0.999)),
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(FEATURE_COLUMNS, feature_values, strict=True)
            }
        )
        result.append(row)
    return result


def summarize_waveform_features(
    rows: list[dict[str, Any]], failures: list[dict[str, str]], selected: int
) -> dict[str, Any]:
    key_features = (
        "duration_sec",
        "rms_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_flatness_mean",
        "silence_ratio_abs_le_0_01",
        "peak_abs",
        "clipping_ratio_abs_ge_0_999",
    )
    by_speaker: dict[str, Any] = {}
    for speaker in (0, 1):
        subset = [row for row in rows if row["speaker"] == speaker]
        by_speaker[str(speaker)] = {
            "n_feature_rows": len(subset),
            "features": {
                feature: numeric_profile(
                    (float(row[feature]) for row in subset),
                    "ratio"
                    if "ratio" in feature
                    else "seconds"
                    if feature == "duration_sec"
                    else "feature_value",
                )
                for feature in key_features
            },
        }
    caller_rows = [row for row in rows if row["speaker"] == 1]
    caller_by_gender = {}
    for gender in sorted({str(row["gender"]) for row in caller_rows}):
        subset = [row for row in caller_rows if row["gender"] == gender]
        caller_by_gender[gender] = {
            "n": len(subset),
            "rms_mean": numeric_profile(
                (float(row["rms_mean"]) for row in subset), "feature_value"
            ),
            "spectral_centroid_mean": numeric_profile(
                (float(row["spectral_centroid_mean"]) for row in subset),
                "feature_value",
            ),
            "mfcc_mean_0": numeric_profile(
                (float(row["mfcc_mean_0"]) for row in subset), "feature_value"
            ),
        }
    return {
        "selected_calls": selected,
        "successful_calls": len({row["stem"] for row in rows}),
        "feature_rows": len(rows),
        "failures": len(failures),
        "failure_types": dict(Counter(item["error_type"] for item in failures)),
        "feature_dimension": len(FEATURE_COLUMNS),
        "feature_backend": "bounded NumPy MFCC/RMS/ZCR/spectral profile",
        "feature_audio_max_seconds_per_speaker": EDA_FEATURE_MAX_SECONDS,
        "by_speaker": by_speaker,
        "caller_by_gender": caller_by_gender,
    }


def run_waveform_sample(
    profiles: list[CallProfile],
    data_root: Path,
    n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    selected = stratified_sample(profiles, n, seed)
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, profile in enumerate(selected, start=1):
        try:
            feature_rows.extend(extract_waveform_feature_rows(profile, data_root))
        except Exception as exc:  # noqa: BLE001 - recorded in aggregate output
            failures.append(
                {
                    "split": profile.row.split,
                    "stem": profile.row.stem,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        if index % 50 == 0 or index == len(selected):
            print(
                f"  waveform features {profile.row.split}: "
                f"{index:,}/{len(selected):,}"
            )
    return (
        feature_rows,
        failures,
        summarize_waveform_features(feature_rows, failures, len(selected)),
    )
