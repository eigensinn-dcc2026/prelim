#!/usr/bin/env python3
"""Compare L1 (LightGBM) and L2 (CNN) per-utterance on official Validation.

Both models were evaluated separately (scripts/evaluate_mission2_l1_on_
official_validation.py, scripts/evaluate_mission2_l2_on_official_validation.py)
but neither saved per-utterance predictions, so there was no way to ask
"which utterances did CNN fix that GBM missed" -- only compare aggregate
accuracy. This re-runs both models' inference (read-only, no training/
tuning) and joins the results by (call_id, utt_idx) against the same
overlap/duration breakdown scripts/analyze_mission2_errors.py uses for L1,
so the two models' error patterns are directly comparable.

Usage (L2 inference run locally, e.g. on a GPU box):
    python scripts/compare_mission2_l1_l2_official_validation.py \\
        --l1-ckpt checkpoints/mission2/m2_baseline_lgbm_tuned_postfix2.pkl \\
        --l1-features data/local_cache/m2_features_validation_official.parquet \\
        --l2-ckpt checkpoints/mission2/m2_cnn_large.pt \\
        --l2-index data/local_cache/m2_spectrogram_index_validation.parquet \\
        --l2-spectrogram-dir data/m2_spectrograms_validation \\
        --utterance-index data/local_cache/m2_utterances_validation_official.parquet

Usage (L2 predictions already computed elsewhere, e.g. Colab GPU -- local CNN
inference on CPU over 111,947 utterances was found to be impractically slow,
so the normal path is: run evaluate_mission2_l2_on_official_validation.py's
--predictions-output on Colab, download that CSV, and pass it here instead
of re-running CNN inference locally):
    python scripts/compare_mission2_l1_l2_official_validation.py \\
        --l1-ckpt checkpoints/mission2/m2_baseline_lgbm_tuned_postfix2.pkl \\
        --l1-features data/local_cache/m2_features_validation_official.parquet \\
        --l2-predictions-csv data/local_cache/m2_l2_val_predictions.csv \\
        --utterance-index data/local_cache/m2_utterances_validation_official.parquet \\
        --l1-result-json reports/mission2/m2_l1_official_validation_result_postfix2.json \\
        --l2-result-json reports/mission2/m2_l2_official_validation_result_colab_gpu.json

Nothing is written unless the inputs line up exactly. The L1 / L2 / utterance-index /
overlap-flag tables must each have one row per (call_id, utt_idx), all four must cover the
same utterances, the ground-truth speaker carried by L1, L2 and the utterance index must
agree row by row, every speaker/prediction must be 0 or 1, and no required value may be
missing. The joined result must also be the whole official Validation set: 111,947
utterances from 3,640 calls, checked before anything is written -- inputs that agree with
each other but hold only part of the data are refused. (--expected-utterances/--expected-calls
change that size for a deliberate subset comparison, and a subset result must then be written
to explicit --output/--summary-output paths, never over the official comparison files.)
Pass --l1-result-json / --l2-result-json (the separately saved official Validation evaluation
results) to also require that the joined predictions reproduce their n_utterances and
confusion_matrix exactly.

The local CNN path (--l2-ckpt) scores ONE model and refuses a packed seed ensemble
(scripts/pack_mission2_l2_ensemble.py); to compare an ensemble, pass its per-utterance
predictions with --l2-predictions-csv.

Only the --l2-predictions-csv path runs on the code in this branch. The
evaluate_mission2_l{1,2}_on_official_validation.py scripts and dcc_data.mission2_cnn used by
the other path live on gwanghee/mission2-l1-model-comparison and gwanghee/mission2-l2-cnn-clean,
which are not merged to main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dcc_data.mission2_analysis import bucket_duration, compute_overlap_flags  # noqa: E402
from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import load_model  # noqa: E402

KEY_COLUMNS = ["call_id", "utt_idx"]

# The official Validation set: 3,640 calls / 111,947 utterances (dcc_data.mission2_features'
# EXPECTED_UTTERANCE_COUNTS, and every saved official-Validation result).
OFFICIAL_VALIDATION_UTTERANCES = 111_947
OFFICIAL_VALIDATION_CALLS = 3_640
DEFAULT_OUTPUT = Path("reports/mission2/m2_l1_l2_official_validation_comparison.csv")
DEFAULT_SUMMARY_OUTPUT = Path("reports/mission2/m2_l1_l2_official_validation_comparison.json")

BINARY_COLUMNS = ("speaker", "l2_speaker", "index_speaker", "l1_pred", "l2_pred")
REQUIRED_COLUMNS = (*BINARY_COLUMNS, "duration_sec", "overlaps_other")


def _require_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, KEY_COLUMNS].head(3).values.tolist()
        raise ValueError(
            f"{name}: {int(duplicated.sum())} rows share a duplicate (call_id, utt_idx) key, e.g. {examples}"
        )


def _require_same_keys_as_l1(l1_keys: set, frame: pd.DataFrame, name: str) -> None:
    keys = set(zip(frame["call_id"], frame["utt_idx"]))
    missing, extra = l1_keys - keys, keys - l1_keys
    if missing or extra:
        raise ValueError(
            f"{name} covers different utterances than L1: {len(missing)} missing, {len(extra)} extra "
            f"(e.g. missing {sorted(missing, key=str)[:3]}, extra {sorted(extra, key=str)[:3]})"
        )


def _require_same_speaker(merged: pd.DataFrame, column: str, name: str) -> None:
    mismatched = int((merged["speaker"] != merged[column]).sum())
    if mismatched:
        raise ValueError(f"{name} speaker disagrees with L1 speaker for {mismatched} of {len(merged)} utterances")


def _require_complete_and_binary(merged: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        missing = int(merged[column].isna().sum())
        if missing:
            raise ValueError(f"{column} has {missing} missing value(s) among {len(merged)} joined utterances")
    for column in BINARY_COLUMNS:
        outside = [value.item() if hasattr(value, "item") else value for value in set(merged[column].unique()) - {0, 1}]
        if outside:
            raise ValueError(f"{column} must be 0 or 1, found {sorted(outside, key=str)[:5]}")


def verify_expected_size(merged: pd.DataFrame, expected_utterances: int, expected_calls: int) -> None:
    """Refuse a joined frame that is not exactly the expected number of utterances and calls."""

    utterances, calls = len(merged), int(merged["call_id"].nunique())
    if utterances != expected_utterances or calls != expected_calls:
        raise ValueError(
            f"joined result has {utterances} utterances from {calls} calls, expected {expected_utterances} "
            f"utterances from {expected_calls} calls (the official Validation set is "
            f"{OFFICIAL_VALIDATION_UTTERANCES} / {OFFICIAL_VALIDATION_CALLS}) -- refusing to write a comparison "
            "of only part of it"
        )


def build_comparison_frame(
    l1_result: pd.DataFrame, l2_result: pd.DataFrame, utt_index: pd.DataFrame, overlap_df: pd.DataFrame
) -> pd.DataFrame:
    """Join L1/L2 predictions with duration and overlap flags, or raise.

    Row order follows ``l1_result``. Anything short of an exact one-to-one join (duplicate or
    missing utterances, or ground-truth speakers that disagree between sources) means the two
    models were not scored on the same utterances, so the caller must not write a comparison.
    Every speaker/prediction must also be 0 or 1 and no required value may be missing.
    """

    inputs = {"L1": l1_result, "L2": l2_result, "utterance index": utt_index, "overlap flags": overlap_df}
    for name, frame in inputs.items():
        _require_unique_keys(frame, name)

    l1_keys = set(zip(l1_result["call_id"], l1_result["utt_idx"]))
    for name in ("L2", "utterance index", "overlap flags"):
        _require_same_keys_as_l1(l1_keys, inputs[name], name)

    merged = l1_result.merge(
        l2_result.rename(columns={"speaker": "l2_speaker"})[KEY_COLUMNS + ["l2_speaker", "l2_pred"]],
        on=KEY_COLUMNS, how="inner", validate="one_to_one",
    )
    merged = merged.merge(
        utt_index[KEY_COLUMNS + ["duration_sec", "speaker"]].rename(columns={"speaker": "index_speaker"}),
        on=KEY_COLUMNS, how="left", validate="one_to_one",
    )
    merged = merged.merge(overlap_df[KEY_COLUMNS + ["overlaps_other"]], on=KEY_COLUMNS, how="left", validate="one_to_one")

    _require_complete_and_binary(merged)
    _require_same_speaker(merged, "l2_speaker", "L2")
    _require_same_speaker(merged, "index_speaker", "utterance index")
    return merged.drop(columns=["l2_speaker", "index_speaker"])


def confusion_matrix_2x2(truth: pd.Series, pred: pd.Series) -> list[list[int]]:
    return [[int(((truth == t) & (pred == p)).sum()) for p in (0, 1)] for t in (0, 1)]


# The key scripts/pack_mission2_l2_ensemble.py (PR #32) stores a packed seed ensemble's member weights under,
# instead of a single model's "best_model_state_dict".
PACKED_ENSEMBLE_KEY = "ensemble_model_state_dicts"


def single_model_state_dict(checkpoint: dict, ckpt_path: Path) -> dict:
    """The one model's weights in an L2 training checkpoint, or ValueError.

    The local CNN path below scores a single SpeakerCNNLarge. A packed ensemble holds several models' weights under
    another key, so it must not fall through to a ``KeyError`` (or, worse, be read as one member and reported as
    "the L2 result"): an ensemble is compared through its per-utterance prediction CSV instead.
    """

    if PACKED_ENSEMBLE_KEY in checkpoint:
        raise ValueError(
            f"{ckpt_path} is a packed seed ensemble ({len(checkpoint[PACKED_ENSEMBLE_KEY])} models); the local CNN "
            "path of this script scores one model. Compare an ensemble through its per-utterance predictions "
            "(call_id, utt_idx, speaker, predicted) with --l2-predictions-csv instead."
        )
    for key in ("best_model_state_dict", "model_state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    raise ValueError(f"{ckpt_path} holds no model weights ('best_model_state_dict' or 'model_state_dict')")


def check_against_saved_result(merged: pd.DataFrame, pred_column: str, result_path: Path, name: str) -> None:
    """Require the joined predictions to reproduce a separately saved official-Validation result."""

    saved = json.loads(Path(result_path).read_text(encoding="utf-8"))
    joined = {
        "n_utterances": len(merged),
        "confusion_matrix": confusion_matrix_2x2(merged["speaker"], merged[pred_column]),
    }
    for key, value in joined.items():
        if saved[key] != value:
            raise ValueError(
                f"{name} does not reproduce {Path(result_path).name}: {key} saved={saved[key]} joined={value}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l1-ckpt", type=Path, required=True)
    parser.add_argument("--l1-features", type=Path, required=True)
    parser.add_argument(
        "--l2-predictions-csv", type=Path, default=None,
        help="call_id/utt_idx/speaker/predicted CSV from evaluate_mission2_l2_on_official_validation.py's "
        "--predictions-output (e.g. computed on Colab GPU). Mutually exclusive with --l2-ckpt/--l2-index/"
        "--l2-spectrogram-dir -- prefer this to avoid re-running CNN inference locally on CPU.",
    )
    parser.add_argument("--l2-ckpt", type=Path, default=None)
    parser.add_argument("--l2-index", type=Path, default=None)
    parser.add_argument("--l2-spectrogram-dir", type=Path, default=None)
    parser.add_argument("--utterance-index", type=Path, required=True, help="has startAt/endAt/duration_sec")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--l1-result-json", type=Path, default=None,
        help="saved official-Validation result JSON of the same L1 checkpoint; if given, the joined L1 "
        "predictions must reproduce its n_utterances and confusion_matrix exactly, or nothing is written",
    )
    parser.add_argument(
        "--l2-result-json", type=Path, default=None,
        help="same as --l1-result-json, for the L2 predictions",
    )
    parser.add_argument(
        "--expected-utterances", type=int, default=OFFICIAL_VALIDATION_UTTERANCES,
        help="how many utterances the joined result must have (default: the whole official Validation set, "
        f"{OFFICIAL_VALIDATION_UTTERANCES}); change only for a deliberate subset comparison",
    )
    parser.add_argument(
        "--expected-calls", type=int, default=OFFICIAL_VALIDATION_CALLS,
        help=f"how many calls the joined result must cover (default: {OFFICIAL_VALIDATION_CALLS})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()

    is_subset = (args.expected_utterances, args.expected_calls) != (
        OFFICIAL_VALIDATION_UTTERANCES, OFFICIAL_VALIDATION_CALLS,
    )
    if is_subset and (args.output == DEFAULT_OUTPUT or args.summary_output == DEFAULT_SUMMARY_OUTPUT):
        parser.error(
            "a subset comparison (--expected-utterances/--expected-calls below the official size) must be "
            "written to explicit --output and --summary-output paths, not over the official comparison files"
        )

    uses_csv = args.l2_predictions_csv is not None
    uses_local_ckpt = args.l2_ckpt is not None or args.l2_index is not None or args.l2_spectrogram_dir is not None
    if uses_csv and uses_local_ckpt:
        parser.error("pass either --l2-predictions-csv or --l2-ckpt/--l2-index/--l2-spectrogram-dir, not both")
    if not uses_csv and not (args.l2_ckpt and args.l2_index and args.l2_spectrogram_dir):
        parser.error(
            "need --l2-predictions-csv, or all three of --l2-ckpt/--l2-index/--l2-spectrogram-dir"
        )
    return args


def main() -> int:
    args = parse_args()

    # --- L1 predictions -----------------------------------------------
    l1_model, l1_meta = load_model(args.l1_ckpt)
    l1_feature_columns = tuple(l1_meta["feature_columns"])
    if l1_feature_columns != FEATURE_COLUMNS:
        raise ValueError("L1 checkpoint feature_columns do not match the current feature extractor")
    l1_df = pd.read_parquet(args.l1_features)
    X_l1 = l1_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    l1_pred = l1_model.predict(X_l1)
    l1_result = l1_df[["call_id", "utt_idx", "speaker"]].copy()
    l1_result["l1_pred"] = l1_pred
    print(f"L1: {len(l1_result)} utterances, accuracy={float((l1_result['speaker'] == l1_result['l1_pred']).mean()):.4f}")

    # --- L2 predictions -------------------------------------------------
    # duration_sec always comes from --utterance-index (not the L2 spectrogram
    # index), since the --l2-predictions-csv path never touches the spectrogram
    # index at all.
    utt_index = pd.read_parquet(args.utterance_index).rename(columns={"start_at": "startAt", "end_at": "endAt"})

    if args.l2_predictions_csv is not None:
        l2_pred_df = pd.read_csv(args.l2_predictions_csv, dtype={"call_id": str})
        l2_accuracy = float((l2_pred_df["speaker"] == l2_pred_df["predicted"]).mean())
        print(f"L2 (from {args.l2_predictions_csv.name}): {len(l2_pred_df)} utterances, accuracy={l2_accuracy:.4f}")
        l2_result = l2_pred_df[["call_id", "utt_idx", "speaker"]].copy()
        l2_result["l2_pred"] = l2_pred_df["predicted"]
    else:
        import torch  # local import: only needed for the (slow, CPU-bound) local-inference path

        # Read and check the checkpoint before importing anything that only PR #32 provides, so a packed
        # ensemble is refused with its explanation wherever this script runs.
        checkpoint = torch.load(args.l2_ckpt, map_location="cpu")
        l2_state_dict = single_model_state_dict(checkpoint, args.l2_ckpt)

        from dcc_data.mission2_cnn import SpeakerCNNLarge, SpectrogramDataset, collate_batch, evaluate

        l2_index_df = pd.read_parquet(args.l2_index)
        loader = torch.utils.data.DataLoader(
            SpectrogramDataset(l2_index_df, args.l2_spectrogram_dir),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0,
        )
        l2_model = SpeakerCNNLarge()
        l2_model.load_state_dict(l2_state_dict)
        l2_accuracy, l2_preds, l2_labels = evaluate(l2_model, loader, "cpu")
        print(f"L2: {len(l2_index_df)} utterances, accuracy={l2_accuracy:.4f}")
        if not np.array_equal(l2_labels, l2_index_df["speaker"].to_numpy()):
            raise ValueError("L2 dataset iteration order does not match l2_index_df row order")
        l2_result = l2_index_df[["call_id", "utt_idx", "speaker"]].copy()
        l2_result["l2_pred"] = l2_preds

    # --- overlap flags ----------------------------------------------------
    overlap_df = compute_overlap_flags(utt_index)

    # --- join everything ------------------------------------------------
    merged = build_comparison_frame(l1_result, l2_result, utt_index, overlap_df)
    verify_expected_size(merged, args.expected_utterances, args.expected_calls)
    if args.l1_result_json is not None:
        check_against_saved_result(merged, "l1_pred", args.l1_result_json, "L1")
    if args.l2_result_json is not None:
        check_against_saved_result(merged, "l2_pred", args.l2_result_json, "L2")
    merged["duration_bucket"] = bucket_duration(merged["duration_sec"])
    merged["l1_correct"] = merged["speaker"] == merged["l1_pred"]
    merged["l2_correct"] = merged["speaker"] == merged["l2_pred"]

    def group_stats(by) -> dict:
        table = merged.groupby(by, observed=True)[["l1_correct", "l2_correct"]].agg(["mean", "count"])
        return {
            str(key): {
                "l1_accuracy": float(row[("l1_correct", "mean")]),
                "l2_accuracy": float(row[("l2_correct", "mean")]),
                "n": int(row[("l1_correct", "count")]),
            }
            for key, row in table.iterrows()
        }

    # 2x2: did L1/L2 agree with the truth, independently
    quadrant_counts = (
        merged.groupby(["l1_correct", "l2_correct"], observed=True).size().rename("n").reset_index()
    )
    quadrants = {
        f"l1_{'correct' if row.l1_correct else 'wrong'}_l2_{'correct' if row.l2_correct else 'wrong'}": int(row.n)
        for row in quadrant_counts.itertuples()
    }

    # Same 2x2, but only within the overlap==True subset -- the whole point
    # of L2 existing is to test whether raw time-frequency structure handles
    # overlapping voices better than L1's summary statistics.
    overlap_subset = merged[merged["overlaps_other"] == True]  # noqa: E712
    overlap_quadrant_counts = (
        overlap_subset.groupby(["l1_correct", "l2_correct"], observed=True).size().rename("n").reset_index()
    )
    overlap_quadrants = {
        f"l1_{'correct' if row.l1_correct else 'wrong'}_l2_{'correct' if row.l2_correct else 'wrong'}": int(row.n)
        for row in overlap_quadrant_counts.itertuples()
    }

    summary = {
        "n_utterances": len(merged),
        "l1_accuracy": float(merged["l1_correct"].mean()),
        "l2_accuracy": float(merged["l2_correct"].mean()),
        "by_overlap": group_stats("overlaps_other"),
        "by_duration_bucket": group_stats("duration_bucket"),
        "by_true_class": group_stats("speaker"),
        "agreement_quadrants": quadrants,
        "agreement_quadrants_overlap_only": overlap_quadrants,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"L1 accuracy: {summary['l1_accuracy']:.4f} / L2 accuracy: {summary['l2_accuracy']:.4f}")
    print()
    print("겹침 여부:")
    for key, stats in summary["by_overlap"].items():
        print(f"  overlaps_other={key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("발화 길이 구간:")
    for key, stats in summary["by_duration_bucket"].items():
        print(f"  {key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("실제 speaker:")
    for key, stats in summary["by_true_class"].items():
        print(f"  speaker={key}: L1={stats['l1_accuracy']:.4f} L2={stats['l2_accuracy']:.4f} (n={stats['n']})")
    print()
    print("전체 동의/불일치 구조:", quadrants)
    print("겹침 구간만:", overlap_quadrants)
    print()
    print(f"per-utterance 상세: {args.output}")
    print(f"요약 JSON: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
