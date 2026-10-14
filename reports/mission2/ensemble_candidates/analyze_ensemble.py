#!/usr/bin/env python3
"""Read-only H-split ensemble comparison for saved utterance probabilities.

No training, no inference, no checkpoint load, no random numbers.
The only files this script writes are ensemble_candidates.md and
ensemble_candidates.csv in reports/mission2/ensemble_candidates/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

# Fixed before any accuracy is computed.
THRESHOLD = 0.5
EXPECTED_ROWS = 45504
ECE_BINS = 15
N_GROUPS = 5
ACCURACY_MARGIN = 0.002
MCNEMAR_ALPHA = 0.05
MIN_GROUPS_AHEAD = 4
REFERENCE_ROUNDED_BASELINE = 0.9334
BASELINE_NAME = "base+large+CNN"
MODEL_ORDER = ("base", "large", "CNN", "w2v2")


def fail(message: str) -> None:
    raise SystemExit(message)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    return repo_root() / "reports" / "mission2" / "ensemble_candidates"


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_keys(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    keys = [line.strip() for line in text.splitlines() if line.strip()]
    if len(keys) != len(set(keys)):
        fail(f"duplicate row keys: {path}")
    parsed = [parse_key(key) for key in keys]
    if len(parsed) != len(keys):
        fail(f"row key parse failed: {path}")
    return keys


def parse_key(key: str) -> tuple[str, int, int, int]:
    parts = key.split("\t")
    if len(parts) != 4:
        fail(f"row key does not have 4 fields: {key!r}")
    call_id, utt_idx, start_at, end_at = parts
    try:
        start = int(start_at)
        end = int(end_at)
        index = int(utt_idx)
    except ValueError:
        fail(f"row key fields are not integers: {key!r}")
    if end <= start or index < 0:
        fail(f"row key has a non-positive duration or index: {key!r}")
    return call_id, index, start, end


def load_columns(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        fail(f"probability columns are not a list of strings: {path}")
    return value


def load_matrix(path: Path, n_rows: int) -> np.ndarray:
    probs = np.load(path)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.ndim != 2 or probs.shape[0] != n_rows:
        fail(f"probability shape {getattr(probs, 'shape', None)} does not match {n_rows} rows: {path}")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        fail(f"probabilities are outside [0, 1] or not finite: {path}")
    return np.asarray(probs, dtype=np.float64)


def same_vector(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def check_saved_copy(directory: Path, column: np.ndarray, label: str) -> str:
    path = directory / "wavlm_H_probs.npy"
    if not path.exists():
        return f"{label}: wavlm_H_probs.npy absent"
    saved = np.load(path)
    saved = np.asarray(saved, dtype=np.float64).reshape(-1)
    if not same_vector(saved, column.reshape(-1)):
        fail(f"{label}: wavlm_H_probs.npy does not equal the H_probs column")
    return f"{label}: wavlm_H_probs.npy matches the H_probs column"


def align_to(keys: list[str], source_keys: list[str], matrix: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    if set(keys) != set(source_keys):
        missing = len(set(keys) - set(source_keys))
        extra = len(set(source_keys) - set(keys))
        fail(f"{label} row set differs from SSL-003 (missing {missing}, extra {extra})")
    if keys == source_keys:
        return matrix, "same order"
    position = {key: index for index, key in enumerate(source_keys)}
    order = np.array([position[key] for key in keys], dtype=np.int64)
    return matrix[order], "reordered by key"


def check_label_path(label_path: Path) -> None:
    parts = label_path.parts
    if any(part == "Validation" for part in parts):
        fail("label path contains Validation")
    if "Training" not in parts:
        fail("label path is not under Training")


def index_label_names(names: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        if not name.lower().endswith(".json"):
            continue
        prefix = Path(name).stem.split("_", 1)[0]
        if prefix in found:
            fail(f"duplicate label filename prefix: {prefix}")
        found[prefix] = name
    return found


def load_labels(keys: list[str], label_path: Path) -> np.ndarray:
    check_label_path(label_path)
    if label_path.is_file() and label_path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(label_path)
        names = index_label_names(archive.namelist())
        def read_document(call_id: str) -> dict:
            return json.loads(archive.read(names[call_id]).decode("utf-8"))
    elif label_path.is_dir():
        archive = None
        names = index_label_names([path.name for path in label_path.iterdir()])
        def read_document(call_id: str) -> dict:
            return json.loads((label_path / names[call_id]).read_text(encoding="utf-8"))
    else:
        fail("label path is not a Training JSON directory or zip")
    needed = {parse_key(key)[0] for key in keys}
    missing_files = sorted(needed - set(names))
    if missing_files:
        fail(f"label JSON missing for {len(missing_files)} calls, first {missing_files[:5]}")
    documents: dict[str, dict] = {}
    try:
        for call_id in sorted(needed):
            document = read_document(call_id)
            identity = str(document.get("_id") or document.get("call_id") or "")
            if identity != call_id:
                fail(f"label JSON id {identity!r} does not match filename prefix {call_id}")
            documents[call_id] = document
    finally:
        if archive is not None:
            archive.close()
    speakers = np.empty(len(keys), dtype=np.int64)
    for row_index, key in enumerate(keys):
        call_id, utt_idx, start_at, end_at = parse_key(key)
        utterances = documents[call_id].get("utterances")
        if not isinstance(utterances, list) or utt_idx >= len(utterances):
            fail(f"utterance index missing in label JSON: {key}")
        utterance = utterances[utt_idx]
        if int(utterance["startAt"]) != start_at or int(utterance["endAt"]) != end_at:
            fail(f"label start/end does not match row key: {key}")
        speaker = utterance["speaker"]
        if speaker not in (0, 1):
            fail(f"label speaker is not 0 or 1: {key}")
        speakers[row_index] = int(speaker)
    return speakers


def member_probabilities(ssl003: Path, ssl004: Path, ssl005: Path | None, use_w2v2: bool) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    keys = read_keys(ssl003 / "H_rows.txt")
    if len(keys) != EXPECTED_ROWS:
        fail(f"SSL-003 row count is {len(keys)}, expected {EXPECTED_ROWS}")
    columns = load_columns(ssl003 / "H_probs_columns.json")
    matrix = load_matrix(ssl003 / "H_probs.npy", len(keys))
    if matrix.shape[1] != len(columns):
        fail("SSL-003 column count does not match H_probs")
    if not columns or columns[0] != "wavlm" or not columns[1:] or any(not name.startswith("cnn_seed") for name in columns[1:]):
        fail(f"SSL-003 columns are not wavlm plus cnn_seed columns: {columns}")
    notes = [check_saved_copy(ssl003, matrix[:, 0], "SSL-003")]
    members = {
        "base": matrix[:, 0].copy(),
        "CNN": matrix[:, 1:].mean(axis=1),
    }
    notes.append("CNN = unweighted mean of " + ", ".join(columns[1:]))

    large_keys = read_keys(ssl004 / "H_rows.txt")
    large_columns = load_columns(ssl004 / "H_probs_columns.json")
    large_matrix = load_matrix(ssl004 / "H_probs.npy", len(large_keys))
    if large_matrix.shape[1] != len(large_columns) or large_matrix.shape[1] != 1:
        fail(f"SSL-004 is not a single probability column: {large_columns}")
    large_matrix, large_order = align_to(keys, large_keys, large_matrix, "SSL-004")
    notes.append(check_saved_copy(ssl004, large_matrix[:, 0], "SSL-004"))
    notes.append(f"SSL-004 aligned to SSL-003 keys: {large_order}")
    notes.append(f"SSL-004 column: {large_columns[0]}")
    members["large"] = large_matrix[:, 0].copy()

    if use_w2v2:
        if ssl005 is None:
            fail("SSL-005 path is required when w2v2 is included")
        w2v2_keys = read_keys(ssl005 / "H_rows.txt")
        w2v2_columns = load_columns(ssl005 / "H_probs_columns.json")
        w2v2_matrix = load_matrix(ssl005 / "H_probs.npy", len(w2v2_keys))
        if w2v2_matrix.shape[1] != len(w2v2_columns) or w2v2_matrix.shape[1] != 1:
            fail(f"SSL-005 is not a single probability column: {w2v2_columns}")
        w2v2_matrix, w2v2_order = align_to(keys, w2v2_keys, w2v2_matrix, "SSL-005")
        notes.append(f"SSL-005 aligned to SSL-003 keys: {w2v2_order}")
        notes.append(f"SSL-005 column: {w2v2_columns[0]}")
        members["w2v2"] = w2v2_matrix[:, 0].copy()
    return members, keys, notes


def ssl005_enabled(path: Path | None, presence: str) -> tuple[bool, str]:
    if presence == "no" or path is None:
        return False, "SSL-005 미반영"
    required = ("H_rows.txt", "H_probs.npy", "H_probs_columns.json")
    complete = path.is_dir() and all((path / name).exists() for name in required)
    if presence == "yes" and not complete:
        fail("SSL-005 was marked present but H_rows.txt, H_probs.npy, or H_probs_columns.json is missing")
    if complete:
        return True, "SSL-005 반영"
    return False, "SSL-005 미반영"


def ece_15(probability: np.ndarray, labels: np.ndarray) -> float:
    predicted = probability >= THRESHOLD
    correct = predicted == labels
    confidence = np.maximum(probability, 1.0 - probability)
    total = 0.0
    for index in range(ECE_BINS):
        lower = index / ECE_BINS
        upper = (index + 1) / ECE_BINS
        if index < ECE_BINS - 1:
            mask = (confidence >= lower) & (confidence < upper)
        else:
            mask = (confidence >= lower) & (confidence <= 1.0)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def class_recall(labels: np.ndarray, predicted_label: np.ndarray, speaker: int) -> float:
    mask = labels == speaker
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float((predicted_label[mask] == speaker).sum() / count)


def mcnemar(left_correct: np.ndarray, right_correct: np.ndarray) -> tuple[int, int, float]:
    only_left = int((left_correct & ~right_correct).sum())
    only_right = int((~left_correct & right_correct).sum())
    discordant = only_left + only_right
    if discordant == 0:
        return only_left, only_right, 1.0
    p_value = float(binomtest(only_left, discordant, 0.5).pvalue)
    return only_left, only_right, p_value


def call_group_ids(keys: list[str], diagnostics_path: Path | None) -> tuple[list[list[str]], str]:
    if diagnostics_path is not None and diagnostics_path.exists():
        document = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        groups = document.get("call_groups") if isinstance(document, dict) else None
        extracted: list[list[str]] = []
        if isinstance(groups, list) and len(groups) == N_GROUPS:
            for group in groups:
                ids = None
                if isinstance(group, dict):
                    for field in ("call_ids", "members", "ids"):
                        value = group.get(field)
                        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                            ids = value
                            break
                if ids is None:
                    extracted = []
                    break
                extracted.append(list(ids))
        if extracted:
            return extracted, "cnn_diagnostics.json call id lists"
    calls = sorted({parse_key(key)[0] for key in keys})
    counts = [len(calls) // N_GROUPS] * N_GROUPS
    for index in range(len(calls) % N_GROUPS):
        counts[index] += 1
    groups = []
    start = 0
    for count in counts:
        groups.append(calls[start:start + count])
        start += count
    note = "cnn_diagnostics.json에 통화 소속 목록이 없어 call_id 정렬 후 연속 5등분"
    return groups, note


def group_masks(keys: list[str], groups: list[list[str]]) -> list[np.ndarray]:
    call_ids = np.array([parse_key(key)[0] for key in keys])
    seen: set[str] = set()
    masks = []
    for group in groups:
        duplicated = seen.intersection(group)
        if duplicated:
            fail("call group membership overlaps")
        seen.update(group)
        masks.append(np.isin(call_ids, np.array(group, dtype=call_ids.dtype)))
    if seen != set(call_ids.tolist()):
        fail("call groups do not cover every H call")
    return masks


def duration_masks(keys: list[str]) -> list[tuple[str, np.ndarray]]:
    duration = np.array([(parse_key(key)[3] - parse_key(key)[2]) / 1000.0 for key in keys], dtype=np.float64)
    return [
        ("<0.5s", duration < 0.5),
        ("0.5~1s", (duration >= 0.5) & (duration < 1.0)),
        ("1~3s", (duration >= 1.0) & (duration <= 3.0)),
        (">3s", duration > 3.0),
    ]


def combo_name(members: tuple[str, ...]) -> str:
    return "+".join(members)


def cnn_parameter_count() -> int | None:
    """Count SpeakerCNNLarge weights without drawing random initial values."""
    try:
        from torch import nn
        sys.path.insert(0, str(repo_root() / "src"))
        from dcc_data.mission2_cnn import SpeakerCNNLarge
        originals = (nn.Conv2d.reset_parameters, nn.Linear.reset_parameters, nn.BatchNorm2d.reset_parameters)
        nn.Conv2d.reset_parameters = lambda self: None
        nn.Linear.reset_parameters = lambda self: None
        nn.BatchNorm2d.reset_parameters = lambda self: None
        try:
            model = SpeakerCNNLarge()
            return int(sum(parameter.numel() for parameter in model.parameters()))
        finally:
            nn.Conv2d.reset_parameters, nn.Linear.reset_parameters, nn.BatchNorm2d.reset_parameters = originals
    except Exception:
        return None


def cost_fields(name: str, cnn_params: int | None) -> dict[str, int | str]:
    members = tuple(name.split("+"))
    n_large = int("large" in members)
    n_base = int("base" in members)
    n_w2v2 = int("w2v2" in members)
    n_cnn = 3 if "CNN" in members else 0
    cnn_total = "" if cnn_params is None or n_cnn == 0 else str(cnn_params * n_cnn)
    return {
        "n_members": len(members),
        "n_wavlm_large": n_large,
        "n_wavlm_base": n_base,
        "n_w2v2": n_w2v2,
        "n_cnn_seeds": n_cnn,
        "cnn_parameters": cnn_total,
        "cost_key": f"{n_large + n_w2v2},{n_base},{n_cnn},{len(members)}",
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return None
    return float((a * b).sum() / denom)


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".16g")
    return str(value)


def evaluate(members: dict[str, np.ndarray], labels: np.ndarray, keys: list[str], diagnostics_path: Path | None) -> dict:
    present = [name for name in MODEL_ORDER if name in members]
    labels = np.asarray(labels)
    durations = duration_masks(keys)
    groups, group_note = call_group_ids(keys, diagnostics_path)
    masks = group_masks(keys, groups)
    cnn_params = cnn_parameter_count()
    rows = []
    for width in range(1, len(present) + 1):
        for subset in combinations(present, width):
            probability = np.mean(np.vstack([members[name] for name in subset]), axis=0)
            predicted = probability >= THRESHOLD
            predicted_label = predicted.astype(np.int64)
            correct = predicted_label == labels
            record = {
                "kind": "combination",
                "name": combo_name(subset),
                "accuracy": float(correct.mean()),
                "recall_caller": class_recall(labels, predicted_label, 1),
                "recall_dispatcher": class_recall(labels, predicted_label, 0),
                "ece_15": ece_15(probability, labels),
                "brier": float(np.mean((probability - labels) ** 2)),
                "n_correct": int(correct.sum()),
                "n_rows": int(labels.shape[0]),
            }
            record.update(cost_fields(record["name"], cnn_params))
            for index, mask in enumerate(masks, start=1):
                record[f"group_{index}_rows"] = int(mask.sum())
                record[f"group_{index}_calls"] = len(groups[index - 1])
                record[f"group_{index}_accuracy"] = float(correct[mask].mean()) if mask.any() else None
            for label, mask in durations:
                record[f"duration_{label}_rows"] = int(mask.sum())
                record[f"duration_{label}_accuracy"] = float(correct[mask].mean()) if mask.any() else None
            record["correct"] = correct
            rows.append(record)
    rows.sort(key=lambda record: (-record["accuracy"], record["cost_key"], record["name"]))
    by_name = {record["name"]: record for record in rows}
    if BASELINE_NAME not in by_name:
        fail(f"baseline combination {BASELINE_NAME} is not available")
    baseline = by_name[BASELINE_NAME]
    rank_names = [record["name"] for record in rows]
    for record in rows:
        ahead = 0
        behind = 0
        tied = 0
        for index in range(1, N_GROUPS + 1):
            left = record[f"group_{index}_accuracy"]
            right = baseline[f"group_{index}_accuracy"]
            if left > right:
                ahead += 1
            elif left < right:
                behind += 1
            else:
                tied += 1
        only_this, only_base, p_value = mcnemar(record["correct"], baseline["correct"])
        record["groups_ahead"] = ahead
        record["groups_behind"] = behind
        record["groups_tied"] = tied
        record["accuracy_minus_baseline"] = record["accuracy"] - baseline["accuracy"]
        record["mcnemar_vs_baseline_only_this"] = only_this
        record["mcnemar_vs_baseline_only_baseline"] = only_base
        record["mcnemar_vs_baseline_p"] = p_value
        record["meets_accuracy"] = record["accuracy_minus_baseline"] >= (ACCURACY_MARGIN - 1e-12)
        record["meets_mcnemar"] = p_value < MCNEMAR_ALPHA and only_this > only_base
        record["meets_groups"] = ahead >= MIN_GROUPS_AHEAD
        is_baseline = record["name"] == BASELINE_NAME
        record["adopt"] = (not is_baseline) and record["meets_accuracy"] and record["meets_mcnemar"] and record["meets_groups"]
        if is_baseline:
            record["verdict"] = "기준 조합"
        elif record["adopt"]:
            record["verdict"] = "개선 있음"
        else:
            record["verdict"] = "미달"
        for rank_index, rank_name in enumerate(rank_names[:3], start=1):
            prefix = f"mcnemar_vs_rank{rank_index}"
            if record["name"] == rank_name:
                record[f"{prefix}_name"] = rank_name
                record[f"{prefix}_only_this"] = ""
                record[f"{prefix}_only_other"] = ""
                record[f"{prefix}_p"] = ""
                continue
            other = by_name[rank_name]
            only_this, only_other, rank_p = mcnemar(record["correct"], other["correct"])
            record[f"{prefix}_name"] = rank_name
            record[f"{prefix}_only_this"] = only_this
            record[f"{prefix}_only_other"] = only_other
            record[f"{prefix}_p"] = rank_p
    adopted = [record["name"] for record in rows if record["adopt"]]
    overall = "개선 있음" if adopted else "차이 없음, 추론 비용이 적은 쪽 우선"
    pair_rows = []
    for left_name, right_name in combinations(present, 2):
        left_wrong = (members[left_name] >= THRESHOLD).astype(np.int64) != labels
        right_wrong = (members[right_name] >= THRESHOLD).astype(np.int64) != labels
        pair_rows.append({
            "kind": "pair",
            "name": f"{left_name}|{right_name}",
            "both_wrong": int((left_wrong & right_wrong).sum()),
            "only_left_wrong": int((left_wrong & ~right_wrong).sum()),
            "only_right_wrong": int((~left_wrong & right_wrong).sum()),
            "only_one_wrong": int((left_wrong != right_wrong).sum()),
            "error_correlation": pearson(left_wrong, right_wrong),
        })
    return {
        "rows": rows,
        "pairs": pair_rows,
        "groups": groups,
        "group_note": group_note,
        "durations": durations,
        "overall": overall,
        "adopted": adopted,
        "baseline_accuracy": baseline["accuracy"],
        "cnn_parameters_one_seed": cnn_params,
        "rank_names": rank_names,
    }


def csv_fieldnames(result: dict) -> list[str]:
    names = [
        "kind", "name", "accuracy", "recall_caller", "recall_dispatcher", "ece_15", "brier",
        "n_correct", "n_rows", "n_members", "n_wavlm_large", "n_wavlm_base", "n_w2v2",
        "n_cnn_seeds", "cnn_parameters", "cost_key", "accuracy_minus_baseline",
        "groups_ahead", "groups_behind", "groups_tied",
        "mcnemar_vs_baseline_only_this", "mcnemar_vs_baseline_only_baseline", "mcnemar_vs_baseline_p",
        "meets_accuracy", "meets_mcnemar", "meets_groups", "adopt", "verdict",
    ]
    for index in range(1, N_GROUPS + 1):
        names.extend([f"group_{index}_calls", f"group_{index}_rows", f"group_{index}_accuracy"])
    for label, _mask in result["durations"]:
        names.extend([f"duration_{label}_rows", f"duration_{label}_accuracy"])
    for rank_index in range(1, 4):
        names.extend([
            f"mcnemar_vs_rank{rank_index}_name",
            f"mcnemar_vs_rank{rank_index}_only_this",
            f"mcnemar_vs_rank{rank_index}_only_other",
            f"mcnemar_vs_rank{rank_index}_p",
        ])
    names.extend(["both_wrong", "only_left_wrong", "only_right_wrong", "only_one_wrong", "error_correlation"])
    return names


def write_csv(path: Path, result: dict) -> None:
    fields = csv_fieldnames(result)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in result["rows"]:
            writer.writerow({field: fmt(record.get(field, "")) for field in fields})
        for record in result["pairs"]:
            writer.writerow({field: fmt(record.get(field, "")) for field in fields})


def md_table(headers: list[str], records: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for record in records:
        lines.append("| " + " | ".join(record) + " |")
    return "\n".join(lines)


def write_md(path: Path, result: dict, verification: list[str], command: str) -> None:
    rows = result["rows"]
    baseline = next(record for record in rows if record["name"] == BASELINE_NAME)
    metric_headers = ["조합", "accuracy", "신고자 recall", "대원 recall", "ECE", "Brier", "판정"]
    metric_records = [[
        record["name"], fmt(record["accuracy"]), fmt(record["recall_caller"]), fmt(record["recall_dispatcher"]),
        fmt(record["ece_15"]), fmt(record["brier"]), record["verdict"],
    ] for record in rows]
    mcnemar_headers = ["조합", "이 조합만 정답", "기준만 정답", "p", "accuracy-기준", "앞선 묶음", "정확도 +0.2%p", "p<0.05이고 더 맞음", "묶음 4개 이상"]
    mcnemar_records = [[
        record["name"], fmt(record["mcnemar_vs_baseline_only_this"]), fmt(record["mcnemar_vs_baseline_only_baseline"]),
        fmt(record["mcnemar_vs_baseline_p"]), fmt(record["accuracy_minus_baseline"]), fmt(record["groups_ahead"]),
        fmt(record["meets_accuracy"]), fmt(record["meets_mcnemar"]), fmt(record["meets_groups"]),
    ] for record in rows]
    rank_lines = []
    rank_names = result["rank_names"]
    for left_index, right_index in ((0, 1), (0, 2)):
        if right_index >= len(rank_names):
            continue
        left = next(record for record in rows if record["name"] == rank_names[left_index])
        prefix = f"mcnemar_vs_rank{right_index + 1}"
        rank_lines.append([
            f"{left_index + 1}위 {rank_names[left_index]}",
            f"{right_index + 1}위 {rank_names[right_index]}",
            fmt(left.get(f"{prefix}_only_this", "")),
            fmt(left.get(f"{prefix}_only_other", "")),
            fmt(left.get(f"{prefix}_p", "")),
        ])
    group_headers = ["조합"] + [f"묶음{index} ({rows[0][f'group_{index}_calls']}통화/{rows[0][f'group_{index}_rows']}발화)" for index in range(1, N_GROUPS + 1)]
    group_records = [[record["name"]] + [fmt(record[f"group_{index}_accuracy"]) for index in range(1, N_GROUPS + 1)] for record in rows]
    duration_headers = ["조합"] + [f"{label} (n={rows[0][f'duration_{label}_rows']})" for label, _mask in result["durations"]]
    duration_records = [[record["name"]] + [fmt(record[f"duration_{label}_accuracy"]) for label, _mask in result["durations"]] for record in rows]
    cost_order = sorted(rows, key=lambda record: (record["cost_key"], record["name"]))
    cost_headers = ["조합", "멤버 수", "WavLM-large", "WavLM-base-plus", "wav2vec2", "CNN seed", "CNN 파라미터", "비용 키"]
    cost_records = [[
        record["name"], fmt(record["n_members"]), fmt(record["n_wavlm_large"]), fmt(record["n_wavlm_base"]),
        fmt(record["n_w2v2"]), fmt(record["n_cnn_seeds"]), fmt(record["cnn_parameters"]), record["cost_key"],
    ] for record in cost_order]
    pair_headers = ["쌍", "둘 다 틀림", "왼쪽만 틀림", "오른쪽만 틀림", "한쪽만 틀림", "오류 상관"]
    pair_records = [[
        record["name"], fmt(record["both_wrong"]), fmt(record["only_left_wrong"]), fmt(record["only_right_wrong"]),
        fmt(record["only_one_wrong"]), fmt(record["error_correlation"]),
    ] for record in result["pairs"]]
    adopted = ", ".join(result["adopted"]) if result["adopted"] else "없음"
    cnn_params = "미측정" if result["cnn_parameters_one_seed"] is None else fmt(result["cnn_parameters_one_seed"])
    text = "\n".join([
        "# M2 앙상블 후보",
        "",
        "## 재현 명령",
        "",
        "저장소 루트에서 실행한다. 난수는 쓰지 않는다.",
        "",
        "```",
        command,
        "```",
        "",
        "## 검증",
        "",
        *[f"- {line}" for line in verification],
        f"- 재계산한 {BASELINE_NAME} accuracy: {fmt(baseline['accuracy'])}",
        f"- 고정해 둔 반올림 참고값: {fmt(REFERENCE_ROUNDED_BASELINE)}",
        f"- 재계산값과 참고값의 차: {fmt(baseline['accuracy'] - REFERENCE_ROUNDED_BASELINE)}",
        f"- SpeakerCNNLarge 파라미터(빈 모듈, seed 1개): {cnn_params}",
        "",
        "## 채택 기준",
        "",
        "지표를 계산하기 전에 코드에 고정했다. 기준 조합은 같은 행에서 다시 계산한 base+large+CNN이다. 채택 문턱은 그 accuracy보다 0.2%p(0.002) 높다. 0.9334는 반올림 참고값이라 문턱에 더하지 않는다.",
        "",
        "- accuracy가 기준보다 0.002 이상 높을 것",
        "- 기준과의 양측 정확 McNemar p가 0.05 미만이고, 이 조합만 맞은 수가 기준만 맞은 수보다 클 것",
        "- 통화 5묶음 중 4개 이상에서 accuracy가 기준보다 높을 것",
        "- 셋을 모두 만족하면 개선 있음. 아니면 차이 없음, 추론 비용이 적은 쪽 우선",
        "",
        "## 조합별 지표",
        "",
        md_table(metric_headers, metric_records),
        "",
        "## 기준 대비 McNemar",
        "",
        md_table(mcnemar_headers, mcnemar_records),
        "",
        "## 1위와 2위·3위 McNemar",
        "",
        md_table(["왼쪽", "오른쪽", "왼쪽만 정답", "오른쪽만 정답", "p"], rank_lines),
        "",
        "## 조합별 1위·2위·3위 McNemar",
        "",
        md_table(
            ["조합", "상대", "이 조합만 정답", "상대만 정답", "p"],
            [
                [
                    record["name"],
                    str(record.get(f"mcnemar_vs_rank{rank_index}_name", "")),
                    fmt(record.get(f"mcnemar_vs_rank{rank_index}_only_this", "")),
                    fmt(record.get(f"mcnemar_vs_rank{rank_index}_only_other", "")),
                    fmt(record.get(f"mcnemar_vs_rank{rank_index}_p", "")),
                ]
                for record in rows
                for rank_index in range(1, 4)
                if record["name"] != record.get(f"mcnemar_vs_rank{rank_index}_name")
            ],
        ),
        "",
        "## 통화 5묶음 accuracy",
        "",
        result["group_note"] + ".",
        "",
        md_table(group_headers, group_records),
        "",
        "## 발화 길이별 accuracy",
        "",
        "길이는 (endAt-startAt)/1000초이다. 경계는 0.5초 미만, 0.5초 이상 1초 미만, 1초 이상 3초 이하, 3초 초과이다.",
        "",
        md_table(duration_headers, duration_records),
        "",
        "## 추론 비용",
        "",
        "CNN 파라미터는 체크포인트를 열지 않고 SpeakerCNNLarge 빈 모듈의 파라미터 수에 seed 수 3을 곱한 값이다. WavLM과 wav2vec2 파라미터 수는 측정하지 않았다. 비용 키는 대형 SSL 수, WavLM-base-plus 수, CNN seed 수, 멤버 수 순서이며 작을수록 앞이다.",
        "",
        md_table(cost_headers, cost_records),
        "",
        "## 모델 쌍 오류 일치",
        "",
        "쌍 이름의 왼쪽·오른쪽은 위 표의 열과 같다. 오류 상관은 오답 지시값의 피어슨 상관이다.",
        "",
        md_table(pair_headers, pair_records),
        "",
        "## 판정",
        "",
        f"- 채택 기준을 모두 만족한 조합: {adopted}",
        f"- 판정: {result['overall']}",
        "",
        "## 한계",
        "",
        "- H 분할만 사용했고 공식 Validation은 열지 않았다.",
        "- SSL-005 확률 파일이 없으면 w2v2 조합은 빠진다. 파일이 생기면 같은 명령으로 다시 계산한다.",
        "- 통화 5묶음 소속이 진단 파일에 없으면 call_id를 정렬해 연속 5등분하며, 그 묶음은 진단 파일의 기존 묶음과 같다고 두지 않는다.",
        "",
        "## 계산 정의",
        "",
        "- 임계값은 0.5이고 확률 0.5는 클래스 1로 예측한다. 클래스 0은 대원, 클래스 1은 신고자이다.",
        "- 앙상블 확률은 같은 발화의 멤버 확률을 가중치 1로 평균한 값이다.",
        "- ECE는 max(p, 1-p)를 15개 균등 구간으로 나눈 신뢰도 ECE이다. 마지막 구간만 오른쪽 끝을 포함한다.",
        "- Brier는 클래스 1 확률과 라벨 차의 제곱 평균이다.",
        "- McNemar p는 불일치 수에 대한 양측 이항검정이다.",
        "",
    ])
    path.write_text(text, encoding="utf-8", newline="\n")


def quote_arg(value: Path) -> str:
    text = str(value)
    return '"' + text + '"'


def build_command(args: argparse.Namespace) -> str:
    parts = [
        "python reports/mission2/ensemble_candidates/analyze_ensemble.py",
        f"--ssl003 {quote_arg(args.ssl003)}",
        f"--ssl004 {quote_arg(args.ssl004)}",
        f"--labels {quote_arg(args.labels)}",
        f"--ssl005-present {args.ssl005_present}",
    ]
    if args.ssl005:
        parts.append(f"--ssl005 {quote_arg(args.ssl005)}")
    if args.diagnostics:
        parts.append(f"--diagnostics {quote_arg(args.diagnostics)}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl003", type=Path, required=True)
    parser.add_argument("--ssl004", type=Path, required=True)
    parser.add_argument("--ssl005", type=Path, default=None)
    parser.add_argument("--ssl005-present", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    destination = (args.out or output_dir()).resolve()
    if destination != output_dir().resolve():
        fail("refusing to write outside reports/mission2/ensemble_candidates")
    destination.mkdir(parents=True, exist_ok=True)
    use_w2v2, ssl005_status = ssl005_enabled(args.ssl005, args.ssl005_present)
    diagnostics = args.diagnostics
    if diagnostics is None and (args.ssl004 / "cnn_diagnostics.json").exists():
        diagnostics = args.ssl004 / "cnn_diagnostics.json"
    print(ssl005_status, flush=True)
    members, keys, notes = member_probabilities(args.ssl003, args.ssl004, args.ssl005, use_w2v2)
    print(f"rows={len(keys)} aligned", flush=True)
    labels = load_labels(keys, args.labels)
    print("labels matched", flush=True)
    result = evaluate(members, labels, keys, diagnostics)
    verification = [
        f"행 수: {len(keys)} (기대 {EXPECTED_ROWS})",
        f"라벨 경로: {args.labels}",
        f"행 키 SHA-256 (SSL-003 순서): {sha256_lines(keys)}",
        f"고유 통화 수: {len({parse_key(key)[0] for key in keys})}",
        f"라벨 0(대원): {int((labels == 0).sum())}, 라벨 1(신고자): {int((labels == 1).sum())}",
        ssl005_status,
        f"통화 묶음: {result['group_note']}",
        *notes,
    ]
    command = build_command(args)
    write_csv(destination / "ensemble_candidates.csv", result)
    write_md(destination / "ensemble_candidates.md", result, verification, command)
    print(ssl005_status)
    print(f"rows={len(keys)}")
    for line in verification:
        print(line)
    print(f"baseline_accuracy={fmt(result['baseline_accuracy'])}")
    print(f"verdict={result['overall']}")
    print(f"adopted={','.join(result['adopted']) if result['adopted'] else 'none'}")
    print(f"wrote={destination / 'ensemble_candidates.md'}")
    print(f"wrote={destination / 'ensemble_candidates.csv'}")


if __name__ == "__main__":
    main()
