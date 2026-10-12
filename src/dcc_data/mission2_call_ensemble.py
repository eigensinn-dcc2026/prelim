"""Frozen call-subset splits for a Mission 2 ensemble gate.

Pure functions only: no training loop, no audio, no checkpoints. The gate
asks whether two models trained on different calls beat two torch seeds
trained on the same calls. Thresholds are fixed here, before any holdout
score is computed.

scipy is not in requirements.txt, so the exact McNemar binomial tail is
implemented locally. If scipy is added to requirements.txt later,
``mcnemar_exact`` uses ``scipy.stats.binomtest`` instead.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

# Frozen before any holdout score is looked at. Do not lower these after a run.
GATE_HOLDOUT_CALLS = 1500
GATE_SOURCE_CALLS = 6000
GATE_ARM_A_VAL_CALLS = 1200
GATE_B_VAL_CALLS = 1200
GATE_MIN_CALLS = GATE_HOLDOUT_CALLS + GATE_SOURCE_CALLS + GATE_ARM_A_VAL_CALLS
GATE_MIN_ACCURACY_GAIN = 0.003
GATE_MCNEMAR_ALPHA = 0.05
FULL_FOLD_TORCH_SEED = 1

_GATE_SIZES = {
    "holdout": GATE_HOLDOUT_CALLS,
    "source": GATE_SOURCE_CALLS,
    "arm_a_train": GATE_SOURCE_CALLS,
    "arm_a_val": GATE_ARM_A_VAL_CALLS,
    "b0_train": GATE_SOURCE_CALLS - GATE_B_VAL_CALLS,
    "b0_val": GATE_B_VAL_CALLS,
    "b1_train": GATE_SOURCE_CALLS - GATE_B_VAL_CALLS,
    "b1_val": GATE_B_VAL_CALLS,
}


def shuffle_call_ids(call_ids, seed: int) -> list[str]:
    """Sorted unique call ids, shuffled with ``Generator(PCG64(seed))``."""

    unique = sorted({str(call_id) for call_id in call_ids})
    order = np.arange(len(unique))
    np.random.Generator(np.random.PCG64(seed)).shuffle(order)
    return [unique[int(i)] for i in order]


def _disjoint(name_a: str, ids_a: list[str], name_b: str, ids_b: list[str]) -> None:
    overlap = set(ids_a) & set(ids_b)
    if overlap:
        raise AssertionError(f"{name_a} and {name_b} share {len(overlap)} call ids")


def _check_gate(assignments: dict[str, list[str]]) -> None:
    for name, size in _GATE_SIZES.items():
        ids = assignments[name]
        if len(ids) != size or len(set(ids)) != size:
            raise AssertionError(f"{name} must contain {size} distinct call ids, got {len(ids)}")
    _disjoint("holdout", assignments["holdout"], "source", assignments["source"])
    _disjoint("holdout", assignments["holdout"], "arm_a_val", assignments["arm_a_val"])
    _disjoint("source", assignments["source"], "arm_a_val", assignments["arm_a_val"])
    for name in ("arm_a_train", "arm_a_val", "b0_train", "b0_val", "b1_train", "b1_val"):
        _disjoint("holdout", assignments["holdout"], name, assignments[name])
    _disjoint("arm_a_train", assignments["arm_a_train"], "arm_a_val", assignments["arm_a_val"])
    _disjoint("b0_train", assignments["b0_train"], "b0_val", assignments["b0_val"])
    _disjoint("b1_train", assignments["b1_train"], "b1_val", assignments["b1_val"])
    source = assignments["source"]
    if assignments["arm_a_train"] != source:
        raise AssertionError("arm A train must be the full source list, in shuffle order")
    if assignments["b0_val"] != source[:GATE_B_VAL_CALLS] or assignments["b0_train"] != source[GATE_B_VAL_CALLS:]:
        raise AssertionError("B0 must be source without its first 1200 calls, order preserved")
    if assignments["b1_train"] != source[: GATE_SOURCE_CALLS - GATE_B_VAL_CALLS]:
        raise AssertionError("B1 train must be the first 4800 source calls, order preserved")
    if assignments["b1_val"] != source[GATE_SOURCE_CALLS - GATE_B_VAL_CALLS :]:
        raise AssertionError("B1 val must be the last 1200 source calls, order preserved")


def gate_assignments(call_ids, seed: int = 42) -> dict[str, list[str]]:
    """Holdout, source, and the two arms. Raises if fewer than 8,700 distinct call ids.

    Shuffle once. ``holdout`` is the first 1,500 calls (score only). ``source`` is the next
    6,000 (Arm A train). ``arm_a_val`` is the next 1,200 (Arm A early stopping). Arm B splits
    ``source`` only: B0 validates on ``source[:1200]`` and trains on the rest; B1 trains on
    ``source[:4800]`` and validates on the rest. Calls past 8,700 are unused.
    """

    shuffled = shuffle_call_ids(call_ids, seed)
    if len(shuffled) < GATE_MIN_CALLS:
        raise ValueError(
            f"need at least {GATE_MIN_CALLS} distinct call ids for the gate split, got {len(shuffled)}"
        )
    holdout = shuffled[:GATE_HOLDOUT_CALLS]
    source = shuffled[GATE_HOLDOUT_CALLS : GATE_HOLDOUT_CALLS + GATE_SOURCE_CALLS]
    arm_a_val = shuffled[GATE_HOLDOUT_CALLS + GATE_SOURCE_CALLS : GATE_MIN_CALLS]
    b_train = GATE_SOURCE_CALLS - GATE_B_VAL_CALLS
    assignments = {
        "holdout": holdout,
        "source": source,
        "arm_a_train": list(source),
        "arm_a_val": arm_a_val,
        "b0_train": list(source[GATE_B_VAL_CALLS:]),
        "b0_val": list(source[:GATE_B_VAL_CALLS]),
        "b1_train": list(source[:b_train]),
        "b1_val": list(source[b_train:]),
    }
    _check_gate(assignments)
    return assignments


def full_fold_assignments(call_ids, seed: int = 42, n_folds: int = 3) -> dict:
    """Shuffle every call id again and cut it into ``n_folds`` contiguous pieces.

    This calls ``shuffle_call_ids`` on the whole set. It does not reuse gate slices.
    Member ``k`` early-stops on fold ``k`` and trains on the other folds. Every member's
    torch seed is 1.
    """

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    shuffled = shuffle_call_ids(call_ids, seed)
    n_calls = len(shuffled)
    if n_calls < n_folds:
        raise ValueError(f"need at least {n_folds} distinct call ids, got {n_calls}")
    sizes = [n_calls // n_folds + (1 if fold < n_calls % n_folds else 0) for fold in range(n_folds)]
    folds: list[list[str]] = []
    cursor = 0
    for size in sizes:
        folds.append(shuffled[cursor : cursor + size])
        cursor += size
    members = []
    for fold_index, val_ids in enumerate(folds):
        train_ids = [call_id for other, fold in enumerate(folds) if other != fold_index for call_id in fold]
        if set(train_ids) & set(val_ids):
            raise AssertionError(f"member {fold_index} train and val overlap")
        members.append(
            {"fold": fold_index, "torch_seed": FULL_FOLD_TORCH_SEED, "train": train_ids, "val": list(val_ids)}
        )
    assigned = [call_id for member in members for call_id in member["val"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(shuffled):
        raise AssertionError("each call must sit in exactly one member's val fold")
    if any(member["torch_seed"] != FULL_FOLD_TORCH_SEED for member in members):
        raise AssertionError("full-fold torch seed is fixed at 1")
    return {"folds": folds, "members": members}


def soft_vote(member_probs) -> np.ndarray:
    """Class 1 iff the members' mean P(class=1) is at least 0.5.

    ``member_probs`` is one probability vector per member, shape ``(n_members, n_utterances)``.
    An exact 0.5 mean is class 1, matching ``mission2_cnn.predicted_classes``.
    """

    rows = [np.asarray(row, dtype=np.float64) for row in member_probs]
    if not rows:
        raise ValueError("soft_vote needs at least one member")
    stacked = np.stack(rows, axis=0)
    if np.isnan(stacked).any():
        raise ValueError("soft_vote probabilities contain NaN")
    return (stacked.mean(axis=0) >= 0.5).astype(np.int64)


def _exact_binomial_two_sided_half(k: int, n: int) -> float:
    """Two-sided exact binomial p-value at p=0.5.

    Matches ``scipy.stats.binomtest(..., p=0.5, alternative='two-sided')``: twice the lower
    tail, clipped at 1. The null is symmetric, so either discordant count can be ``k``.
    """

    if n == 0:
        return 1.0
    smaller = min(k, n - k)
    log_comb = 0.0
    log_terms = []
    for i in range(smaller + 1):
        if i:
            log_comb += math.log(n - i + 1) - math.log(i)
        log_terms.append(log_comb - n * math.log(2.0))
    peak = max(log_terms)
    tail = math.exp(peak) * sum(math.exp(term - peak) for term in log_terms)
    return min(1.0, 2.0 * tail)


@functools.lru_cache(maxsize=1)
def _requirements_include_scipy() -> bool:
    requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
    if not requirements.is_file():
        return False
    for line in requirements.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip().lower()
        name = name.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
        if name == "scipy":
            return True
    return False


def _binomial_two_sided_p(k: int, n: int) -> float:
    if _requirements_include_scipy():
        from scipy.stats import binomtest

        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    return _exact_binomial_two_sided_half(k, n)


def mcnemar_exact(correct_a, correct_b) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on per-utterance correctness.

    Returns ``(n_a_only, n_b_only, p_value)``. Concordant utterances are ignored.
    ``n_a_only`` is correct under A and wrong under B. No discordant pair yields p=1.
    """

    a = np.asarray(correct_a, dtype=bool).reshape(-1)
    b = np.asarray(correct_b, dtype=bool).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"correctness vectors differ in length: {a.shape[0]} vs {b.shape[0]}")
    n_a_only = int(np.count_nonzero(a & ~b))
    n_b_only = int(np.count_nonzero(~a & b))
    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        return n_a_only, n_b_only, 1.0
    return n_a_only, n_b_only, _binomial_two_sided_p(n_b_only, n_discordant)


def gate_decision(acc_a: float, acc_b: float, p_value: float) -> bool:
    """True only when B beats A by at least 0.003 and the exact McNemar p-value is below 0.05."""

    if not math.isfinite(acc_a) or not math.isfinite(acc_b) or not math.isfinite(p_value):
        return False
    return (float(acc_b) - float(acc_a)) >= GATE_MIN_ACCURACY_GAIN and float(p_value) < GATE_MCNEMAR_ALPHA


def _recall(labels: np.ndarray, preds: np.ndarray, class_id: int) -> float:
    mask = labels == class_id
    if not np.any(mask):
        return 0.0
    return float(np.mean(preds[mask] == class_id))


def build_gate_metrics(labels, group_member_probs: dict) -> dict:
    """Accuracy, per-class recall, McNemar, and the frozen gate decision.

    ``group_member_probs[name]`` is one P(class=1) vector per member. Groups ``a`` and ``b``
    are the two arms. Any other pair leaves ``gate_pass`` false.
    """

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, member_probs in group_member_probs.items():
        member_rows = list(member_probs)
        predicted = soft_vote(member_rows)
        if predicted.shape != y.shape:
            raise ValueError(f"group {name} has {predicted.shape[0]} rows for {y.shape[0]} labels")
        predictions[name] = predicted
        groups[name] = {
            "accuracy": float(np.mean(predicted == y)) if len(y) else 0.0,
            "recall_0": _recall(y, predicted, 0),
            "recall_1": _recall(y, predicted, 1),
            "n_members": len(member_rows),
        }
    metrics: dict = {
        "n_utterances": int(y.shape[0]),
        "groups": groups,
        "mcnemar": None,
        "gate_pass": False,
        "gate_min_accuracy_gain": GATE_MIN_ACCURACY_GAIN,
        "gate_mcnemar_alpha": GATE_MCNEMAR_ALPHA,
    }
    if "a" in predictions and "b" in predictions:
        n_a_only, n_b_only, p_value = mcnemar_exact(predictions["a"] == y, predictions["b"] == y)
        acc_a = groups["a"]["accuracy"]
        acc_b = groups["b"]["accuracy"]
        metrics["mcnemar"] = {"n_a_only": n_a_only, "n_b_only": n_b_only, "p_value": p_value}
        metrics["acc_a"] = acc_a
        metrics["acc_b"] = acc_b
        metrics["acc_gain_b_minus_a"] = acc_b - acc_a
        metrics["gate_pass"] = bool(gate_decision(acc_a, acc_b, p_value))
    return metrics


def load_call_id_list(path: str | Path) -> list[str]:
    """Non-empty call-id file. Blank lines are ignored. Duplicates are an error."""

    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        call_id = line.strip()
        if call_id and not call_id.startswith("#"):
            ids.append(call_id)
    if not ids:
        raise ValueError(f"call-id list is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"call-id list has duplicate ids: {path}")
    return ids


def resolve_call_id_split(df, train_call_ids, val_call_ids):
    """Rows for an explicit train/val call split. Both sets must be non-empty and disjoint.

    Every listed id has to appear in ``df['call_id']``. Returns ``(train_df, val_df)``
    in the index's original row order.
    """

    train_ids = [str(call_id) for call_id in train_call_ids]
    val_ids = [str(call_id) for call_id in val_call_ids]
    if not train_ids or not val_ids:
        raise ValueError("train and val call-id lists must both be non-empty")
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("train and val call-id lists must not contain duplicates")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"train and val call ids overlap ({len(overlap)}): {preview}")
    known = set(df["call_id"].astype(str))
    missing = (set(train_ids) | set(val_ids)) - known
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} call id(s) are not in the index: {preview}")
    call_id = df["call_id"].astype(str)
    train_df = df.loc[call_id.isin(train_ids)]
    val_df = df.loc[call_id.isin(val_ids)]
    if train_df.empty or val_df.empty:
        raise ValueError("train and val call-id lists must both match at least one utterance")
    return train_df, val_df


def assert_not_official_validation_path(path: str | Path, *, label: str) -> None:
    """Refuse a path whose components include the official Validation split name."""

    if any(part == "Validation" for part in Path(path).parts):
        raise ValueError(f"{label} is an official Validation path: {path}")
