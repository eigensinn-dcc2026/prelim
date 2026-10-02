#!/usr/bin/env python3
"""Analyze frozen Mission 1 LR errors on official Validation.

Replays the EXP-001 LR checkpoint (no fit, no scaler update) on the
Validation feature cache, attaches caller-union duration, utterance count,
and caller-operator overlap, then writes row-level and summary reports.

Overlap uses the Training timestamp-audit definition:
``overlap_ratio = |caller_union ∩ operator_union| / |caller_union|``.

Duration and ``n_caller_utterances`` are analysis variables only. They are
not written into the EXP-001 model feature set.

This script does not train EXP-002 and does not extract F0.

Usage:
    python scripts/analyze_mission1_validation_errors.py \\
        --features data/m1_features_validation.parquet \\
        --lr-ckpt checkpoints/mission1/m1_baseline_lr.pkl \\
        --eval-log reports/mission1/m1_val_eval.json \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw_original
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from dcc_data.guards import VALIDATION  # noqa: E402
from dcc_data.intervals import caller_operator_overlap_stats  # noqa: E402
from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission1 import _validate_utterance  # noqa: E402
from dcc_data.mission1_features import ACOUSTIC_FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission1_train import (  # noqa: E402
    GENDER_CLASSES,
    GENDER_TO_LABEL,
    LABEL_TO_GENDER,
    encode_gender,
    load_mission1_lr,
)
from dcc_data.mission2 import SPEAKER_CALLER, SPEAKER_OPERATOR  # noqa: E402

PITCH_RELATED_SPECTRAL_COLUMNS: tuple[str, ...] = (
    "zcr_mean",
    "spectral_centroid_mean",
    "mfcc_mean_0",
    "mfcc_mean_1",
    "mfcc_mean_2",
    "mfcc_mean_3",
)
HIGH_OVERLAP_RATIO = 0.2
LENGTH_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", type=Path, default=Path("data/m1_features_validation.parquet"))
    parser.add_argument("--lr-ckpt", type=Path, default=Path("checkpoints/mission1/m1_baseline_lr.pkl"))
    parser.add_argument("--eval-log", type=Path, default=Path("reports/mission1/m1_val_eval.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/dcc_manifest_v1.csv"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DCC_DATA_ROOT", REPOSITORY_ROOT / "data/raw_original")))
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("reports/mission1/m1_val_error_rows.parquet"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/mission1/m1_val_error_analysis.json"),
    )
    return parser.parse_args()


def _parse_speaker_intervals(
    document: dict[str, object], json_path: Path
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    utterances = document.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"utterances must be a list: {json_path}")
    callers: list[tuple[float, float]] = []
    operators: list[tuple[float, float]] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            raise ValueError(f"utterance must be an object: {json_path}")
        start_at, end_at, speaker = _validate_utterance(utterance, json_path)
        interval = (float(start_at), float(end_at))
        if speaker == SPEAKER_CALLER:
            callers.append(interval)
        elif speaker == SPEAKER_OPERATOR:
            operators.append(interval)
    if not callers:
        raise ValueError(f"no caller utterances: {json_path}")
    return callers, operators


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _error_rate(error: pd.Series, mask: pd.Series | None = None) -> dict[str, float | int | None]:
    selected = error if mask is None else error.loc[mask]
    n = int(len(selected))
    n_error = int(selected.sum()) if n else 0
    return {
        "n": n,
        "n_error": n_error,
        "error_rate": float(n_error / n) if n else None,
    }


def _rates_by_qcut(
    df: pd.DataFrame, column: str, *, q: int = 5
) -> list[dict[str, object]]:
    binned = pd.qcut(df[column], q=q, duplicates="drop")
    rows: list[dict[str, object]] = []
    categories = list(binned.cat.categories)
    for index, category in enumerate(categories, start=1):
        mask = binned == category
        row: dict[str, object] = {
            "bin": f"Q{index}",
            "interval": str(category),
            **_error_rate(df["error"], mask),
            "n_F_to_M": int(((df["error_type"] == "F→M") & mask).sum()),
            "n_M_to_F": int(((df["error_type"] == "M→F") & mask).sum()),
        }
        rows.append(row)
    return rows


def _q_end_minus_start(rows: list[dict[str, object]]) -> float | None:
    if len(rows) < 2:
        return None
    first = rows[0]["error_rate"]
    last = rows[-1]["error_rate"]
    if first is None or last is None:
        return None
    return float(last) - float(first)


def _cohens_d(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return None
    pooled_n = left.size + right.size - 2
    if pooled_n <= 0:
        return None
    pooled_var = (
        (left.size - 1) * np.var(left, ddof=1) + (right.size - 1) * np.var(right, ddof=1)
    ) / pooled_n
    if pooled_var <= 0.0:
        return 0.0 if float(np.mean(left)) == float(np.mean(right)) else None
    return float((np.mean(left) - np.mean(right)) / np.sqrt(pooled_var))


def _closer_label(error_mean: float, same_mean: float, opposite_mean: float) -> str:
    distance_same = abs(error_mean - same_mean)
    distance_opposite = abs(error_mean - opposite_mean)
    if np.isclose(distance_same, distance_opposite):
        return "tie"
    return "opposite" if distance_opposite < distance_same else "same"


def _load_eval_confusion(eval_log: Path) -> tuple[list[list[int]], int]:
    payload = json.loads(eval_log.read_text(encoding="utf-8"))
    matrix = payload["measured"]["lr"]["confusion_matrix"]
    labels = payload["measured"]["lr"]["confusion_matrix_labels"]
    if list(labels) != list(GENDER_CLASSES):
        raise ValueError(f"eval log confusion labels {labels} != {list(GENDER_CLASSES)}")
    n_validation = int(payload["n_validation"])
    return matrix, n_validation


def attach_json_analysis(
    df: pd.DataFrame, *, manifest_path: Path, data_root: Path
) -> pd.DataFrame:
    rows = load_manifest(manifest_path, data_root=data_root, validate_paths=False)
    by_stem = {row.stem: row for row in rows if row.split == VALIDATION}
    missing = sorted(set(df["stem"].astype(str)) - set(by_stem))
    if missing:
        preview = missing[:10]
        raise ValueError(f"{len(missing)} Validation stem(s) missing from manifest: {preview}")

    caller_union_sec = np.empty(len(df), dtype=np.float64)
    caller_sum_sec = np.empty(len(df), dtype=np.float64)
    overlap_ms = np.empty(len(df), dtype=np.float64)
    overlap_ratio = np.empty(len(df), dtype=np.float64)
    json_n_caller = np.empty(len(df), dtype=np.int64)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    iterator = df["stem"].astype(str)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="JSON overlap", unit="call")

    for index, stem in enumerate(iterator):
        json_path = data_root / by_stem[stem].json_path
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        callers, operators = _parse_speaker_intervals(document, json_path)
        stats = caller_operator_overlap_stats(callers, operators)
        if stats["caller_union"] + LENGTH_ATOL < stats["overlap"]:
            raise ValueError(f"overlap exceeds caller union: {stem}")
        if stats["caller_sum"] + LENGTH_ATOL < stats["caller_union"]:
            raise ValueError(f"caller union exceeds caller sum: {stem}")
        if not 0.0 - LENGTH_ATOL <= stats["overlap_ratio"] <= 1.0 + LENGTH_ATOL:
            raise ValueError(f"overlap_ratio out of range for {stem}: {stats['overlap_ratio']}")
        json_n_caller[index] = len(callers)
        caller_union_sec[index] = stats["caller_union"] / 1000.0
        caller_sum_sec[index] = stats["caller_sum"] / 1000.0
        overlap_ms[index] = stats["overlap"]
        overlap_ratio[index] = min(max(stats["overlap_ratio"], 0.0), 1.0)

    out = df.copy()
    out["caller_union_sec"] = caller_union_sec
    out["caller_sum_sec"] = caller_sum_sec
    out["overlap_ms"] = overlap_ms
    out["overlap_ratio"] = overlap_ratio
    out["n_caller_utterances_json"] = json_n_caller
    mismatched = out["n_caller_utterances"].to_numpy() != json_n_caller
    if mismatched.any():
        examples = out.loc[mismatched, "stem"].astype(str).head(10).tolist()
        raise ValueError(
            f"n_caller_utterances parquet/JSON mismatch on {int(mismatched.sum())} row(s): {examples}"
        )
    return out.drop(columns=["n_caller_utterances_json"])


def _feature_block(df: pd.DataFrame, lr_abs: list[dict[str, object]]) -> dict[str, object]:

    cohens: dict[str, dict[str, float | None]] = {}
    for gender in GENDER_CLASSES:
        same = df["gender"] == gender
        correct = same & ~df["error"]
        incorrect = same & df["error"]
        per_feature: dict[str, float | None] = {}
        for column in ACOUSTIC_FEATURE_COLUMNS:
            per_feature[column] = _cohens_d(
                df.loc[incorrect, column].to_numpy(),
                df.loc[correct, column].to_numpy(),
            )
        cohens[gender] = per_feature

    ranked = []
    for gender, per_feature in cohens.items():
        scored = [
            {"gender": gender, "column": column, "cohens_d_error_minus_correct": value}
            for column, value in per_feature.items()
            if value is not None
        ]
        scored.sort(key=lambda item: abs(item["cohens_d_error_minus_correct"]), reverse=True)
        ranked.extend(scored[:10])

    indicators: dict[str, object] = {
        "note": (
            "Pitch-related / spectral indicators: indirect acoustic measurements "
            "that may relate to pitch or gender. Actual F0 is not measured here."
        ),
        "columns": list(PITCH_RELATED_SPECTRAL_COLUMNS),
        "cohens_d_within_true_gender": {
            gender: {column: cohens[gender][column] for column in PITCH_RELATED_SPECTRAL_COLUMNS}
            for gender in GENDER_CLASSES
        },
        "error_mean_vs_correct_gender_means": {},
    }
    closeness: dict[str, dict[str, object]] = {}
    opposite = {"F": "M", "M": "F"}
    for gender in GENDER_CLASSES:
        error_mask = (df["gender"] == gender) & df["error"]
        same_correct = (df["gender"] == gender) & ~df["error"]
        opposite_correct = (df["gender"] == opposite[gender]) & ~df["error"]
        per_column: dict[str, object] = {}
        n_opposite = 0
        n_same = 0
        n_tie = 0
        for column in PITCH_RELATED_SPECTRAL_COLUMNS:
            error_mean = float(df.loc[error_mask, column].mean()) if error_mask.any() else None
            same_mean = float(df.loc[same_correct, column].mean()) if same_correct.any() else None
            opposite_mean = (
                float(df.loc[opposite_correct, column].mean()) if opposite_correct.any() else None
            )
            closer = None
            if error_mean is not None and same_mean is not None and opposite_mean is not None:
                closer = _closer_label(error_mean, same_mean, opposite_mean)
                if closer == "opposite":
                    n_opposite += 1
                elif closer == "same":
                    n_same += 1
                else:
                    n_tie += 1
            per_column[column] = {
                "error_mean": error_mean,
                "same_gender_correct_mean": same_mean,
                "opposite_gender_correct_mean": opposite_mean,
                "closer_to": closer,
            }
        closeness[gender] = {
            "n_error": int(error_mask.sum()),
            "n_indicators_closer_to_opposite": n_opposite,
            "n_indicators_closer_to_same": n_same,
            "n_indicators_tie": n_tie,
            "columns": per_column,
        }
    indicators["error_mean_vs_correct_gender_means"] = closeness

    return {
        "lr_abs_coef_top": lr_abs,
        "cohens_d_error_minus_correct_top_by_true_gender": ranked,
        "pitch_related_spectral_indicators": indicators,
    }


def build_summary(
    df: pd.DataFrame,
    measured_cm: list[list[int]],
    lr_abs: list[dict[str, object]],
) -> dict[str, object]:
    n = len(df)
    n_error = int(df["error"].sum())
    n_f_to_m = int((df["error_type"] == "F→M").sum())
    n_m_to_f = int((df["error_type"] == "M→F").sum())
    n_f = int((df["gender"] == "F").sum())
    n_m = int((df["gender"] == "M").sum())

    duration_bins = _rates_by_qcut(df, "caller_union_sec")
    n_utt_bins = _rates_by_qcut(df, "n_caller_utterances")
    overlap_bins = _rates_by_qcut(df, "overlap_ratio")

    duration_median = float(df["caller_union_sec"].median())
    n_utt_median = float(df["n_caller_utterances"].median())
    short = df["caller_union_sec"] <= duration_median
    few = df["n_caller_utterances"] <= n_utt_median
    overlap_zero = df["overlap_ratio"] <= LENGTH_ATOL
    high_overlap = df["overlap_ratio"] >= HIGH_OVERLAP_RATIO

    features = _feature_block(df, lr_abs)
    indicators = features["pitch_related_spectral_indicators"]
    closeness = indicators["error_mean_vs_correct_gender_means"]

    overlap_q_diff = _q_end_minus_start(overlap_bins)
    duration_q_diff = _q_end_minus_start(duration_bins)
    n_utt_q_diff = _q_end_minus_start(n_utt_bins)
    duration_q1_minus_q5 = None if duration_q_diff is None else float(-duration_q_diff)
    n_utt_q1_minus_q5 = None if n_utt_q_diff is None else float(-n_utt_q_diff)

    return {
        "실험 ID": "M1-VAL-ERR-001",
        "validation_policy": (
            "Official Validation may be used to compare a small number of "
            "meaningful experiments. It must never be used for fit, scaler "
            "statistics, or backprop. This script only predicts with a frozen LR."
        ),
        "n_validation": n,
        "confusion_matrix_labels": list(GENDER_CLASSES),
        "confusion_matrix": measured_cm,
        "error_counts": {
            "F→M": n_f_to_m,
            "M→F": n_m_to_f,
            "total": n_error,
            "error_rate": float(n_error / n),
            "F_error_rate": float(n_f_to_m / n_f) if n_f else None,
            "M_error_rate": float(n_m_to_f / n_m) if n_m else None,
        },
        "duration": {
            "unit": "caller_union_sec",
            "note": (
                "Primary duration is caller union (deduplicated). caller_sum_sec "
                "is stored on each row as a diagnostic of concat length."
            ),
            "correct": _numeric_summary(df.loc[~df["error"], "caller_union_sec"].to_numpy()),
            "incorrect": _numeric_summary(df.loc[df["error"], "caller_union_sec"].to_numpy()),
            "error_rate_by_quintile": duration_bins,
        },
        "n_caller_utterances": {
            "correct": _numeric_summary(
                df.loc[~df["error"], "n_caller_utterances"].to_numpy(dtype=np.float64)
            ),
            "incorrect": _numeric_summary(
                df.loc[df["error"], "n_caller_utterances"].to_numpy(dtype=np.float64)
            ),
            "error_rate_by_quintile": n_utt_bins,
            "short_and_few": {
                "duration_median_sec": duration_median,
                "n_caller_utterances_median": n_utt_median,
                "short_and_few": _error_rate(df["error"], short & few),
                "short_not_few": _error_rate(df["error"], short & ~few),
                "not_short_and_few": _error_rate(df["error"], ~short & few),
                "not_short_not_few": _error_rate(df["error"], ~short & ~few),
            },
        },
        "overlap": {
            "definition": (
                "overlap_ratio = |caller_union ∩ operator_union| / |caller_union| "
                "(Training overlap audit; denominator is union, not interval sum)."
            ),
            "zero_vs_nonzero_reference_only": {
                "note": (
                    "Training audit found overlap>0 on most calls, so this split "
                    "is reference only and is not used to rank hypotheses."
                ),
                "zero": _error_rate(df["error"], overlap_zero),
                "nonzero": _error_rate(df["error"], ~overlap_zero),
            },
            "error_rate_by_quintile": overlap_bins,
            "high_tail_ratio_ge_0.2": {
                "note": (
                    "0.2 is a descriptive cut near the Training-audit high side "
                    "(median ratio ~0.08), not tuned on Validation."
                ),
                "high": _error_rate(df["error"], high_overlap),
                "not_high": _error_rate(df["error"], ~high_overlap),
            },
            "correct": _numeric_summary(df.loc[~df["error"], "overlap_ratio"].to_numpy()),
            "incorrect": _numeric_summary(df.loc[df["error"], "overlap_ratio"].to_numpy()),
        },
        "acoustic_features": features,
        "hypothesis_evidence": {
            "note": (
                "Evidence statistics use different units and are not ranked "
                "automatically. Inspect each block and choose the next experiment "
                "by human judgment. Actual F0 is not measured in this analysis."
            ),
            "overlap": {
                "error_rate_q5_minus_q1": overlap_q_diff,
                "quintiles": overlap_bins,
                "median_overlap_ratio_correct": float(df.loc[~df["error"], "overlap_ratio"].median()),
                "median_overlap_ratio_incorrect": float(df.loc[df["error"], "overlap_ratio"].median())
                if n_error
                else None,
                "high_tail_error_rate": _error_rate(df["error"], high_overlap),
            },
            "observation_shortage": {
                "duration_error_rate_q1_minus_q5": duration_q1_minus_q5,
                "n_caller_utterances_error_rate_q1_minus_q5": n_utt_q1_minus_q5,
                "duration_quintiles": duration_bins,
                "n_caller_utterances_quintiles": n_utt_bins,
                "median_duration_correct_sec": float(df.loc[~df["error"], "caller_union_sec"].median()),
                "median_duration_incorrect_sec": float(df.loc[df["error"], "caller_union_sec"].median())
                if n_error
                else None,
                "short_and_few_error_rate": _error_rate(df["error"], short & few),
            },
            "pitch_related_spectral": {
                "note": (
                    "Indirect pitch-related / spectral indicators "
                    f"{list(PITCH_RELATED_SPECTRAL_COLUMNS)}. Not a measured F0."
                ),
                "cohens_d_within_true_gender": indicators["cohens_d_within_true_gender"],
                "error_mean_closer_to_opposite_gender": {
                    gender: {
                        "n_error": closeness[gender]["n_error"],
                        "n_indicators_closer_to_opposite": closeness[gender][
                            "n_indicators_closer_to_opposite"
                        ],
                        "n_indicators_closer_to_same": closeness[gender]["n_indicators_closer_to_same"],
                        "n_indicators_tie": closeness[gender]["n_indicators_tie"],
                    }
                    for gender in GENDER_CLASSES
                },
            },
        },
    }


def _lr_abs_coefficients(pipeline: Pipeline) -> list[dict[str, object]]:
    classifier = pipeline.named_steps["clf"]
    coef = np.abs(np.asarray(classifier.coef_, dtype=np.float64).ravel())
    if coef.shape[0] != len(ACOUSTIC_FEATURE_COLUMNS):
        raise ValueError("LR coefficient length does not match ACOUSTIC_FEATURE_COLUMNS")
    ranked = sorted(
        (
            {"column": column, "abs_coef": float(value)}
            for column, value in zip(ACOUSTIC_FEATURE_COLUMNS, coef)
        ),
        key=lambda item: item["abs_coef"],
        reverse=True,
    )
    return ranked[:15]


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}; prepare it with scripts/build_manifest.py")
    df = pd.read_parquet(args.features)
    if "split" in df.columns:
        foreign = df.loc[df["split"] != VALIDATION, "split"]
        if len(foreign):
            raise ValueError(
                f"validation analysis expected only Validation rows, found {foreign.unique().tolist()}"
            )
    expected_cm, expected_n = _load_eval_confusion(args.eval_log)
    if len(df) != expected_n:
        raise ValueError(f"feature table has {len(df)} rows, eval log n_validation={expected_n}")

    lr_pipeline, lr_meta = load_mission1_lr(args.lr_ckpt)
    if tuple(lr_meta["feature_columns"]) != ACOUSTIC_FEATURE_COLUMNS:
        raise ValueError("LR checkpoint feature_columns do not match ACOUSTIC_FEATURE_COLUMNS")
    if not isinstance(lr_pipeline, Pipeline):
        raise TypeError("LR checkpoint is not a Pipeline; refusing to analyze")

    missing = [column for column in ACOUSTIC_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if "n_caller_utterances" not in df.columns:
        raise ValueError("feature table is missing n_caller_utterances")

    X = df[list(ACOUSTIC_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = encode_gender(df["gender"])
    y_pred = np.asarray(lr_pipeline.predict(X))
    measured_cm = confusion_matrix(
        y, y_pred, labels=[GENDER_TO_LABEL["F"], GENDER_TO_LABEL["M"]]
    ).tolist()
    if measured_cm != expected_cm:
        raise ValueError(
            f"replayed LR confusion matrix {measured_cm} != eval log {expected_cm}"
        )

    pred_gender = pd.Series([LABEL_TO_GENDER[int(label)] for label in y_pred], index=df.index)
    error_type = np.where(
        df["gender"].to_numpy() == pred_gender.to_numpy(),
        "ok",
        np.where(df["gender"].to_numpy() == "F", "F→M", "M→F"),
    )
    df = df.copy()
    df["pred"] = pred_gender.to_numpy()
    df["error"] = df["gender"].to_numpy() != pred_gender.to_numpy()
    df["error_type"] = error_type
    if hasattr(lr_pipeline, "predict_proba"):
        df["proba_M"] = lr_pipeline.predict_proba(X)[:, GENDER_TO_LABEL["M"]]
    n_f_to_m = int((df["error_type"] == "F→M").sum())
    n_m_to_f = int((df["error_type"] == "M→F").sum())
    if n_f_to_m != expected_cm[0][1] or n_m_to_f != expected_cm[1][0]:
        raise ValueError(
            f"error-type counts F→M={n_f_to_m}, M→F={n_m_to_f} do not match eval log {expected_cm}"
        )

    df = attach_json_analysis(df, manifest_path=args.manifest, data_root=args.data_root)
    summary = build_summary(df, measured_cm, _lr_abs_coefficients(lr_pipeline))
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.rows_output, index=False)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = summary["error_counts"]
    print(f"Validation rows: {summary['n_validation']}")
    print(f"Replayed LR confusion matrix: {measured_cm}")
    print(
        f"Errors: total={counts['total']} F→M={counts['F→M']} M→F={counts['M→F']} "
        f"rate={counts['error_rate']:.4f}"
    )
    print(f"Row table: {args.rows_output}")
    print(f"Summary: {args.summary_output}")
    print(
        "Hypothesis evidence is written as separate blocks (overlap / "
        "observation_shortage / pitch_related_spectral). Units differ; "
        "do not auto-rank. Actual F0 was not measured."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
