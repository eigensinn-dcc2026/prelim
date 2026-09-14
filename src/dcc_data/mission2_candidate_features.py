"""Two-stage, Training-first candidate screening for Mission 2.

Stage 1 reuses only the Raw EDA Training sample for feasibility diagnostics;
it cannot rank or select the next experiment.  Stage 2 requires a larger or
full Training-only candidate cache and evaluates user-defined feature groups
with the existing call_id GroupShuffleSplit.  Official Validation is not an
input to either screening stage.

Full feature extraction is always an explicit call guarded by
``allow_full_extraction=True``.  A cache miss never starts raw-audio work.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import librosa
import numpy as np
import pandas as pd

from .mission2_experiments import (
    BASELINE_FEATURE_NAMES,
    MFCC_CONFIG,
    AlignedSplit,
    align_extra_feature_matrix,
    assert_single_factor_comparison,
    fit_reference_logistic_regression,
    group_split_indices,
    require_columns,
    validate_model_feature_names,
)
# Reuse only the existing audio decoder; none of the 53-feature/GBM contract is imported.
from .mission2_features import read_call_waveform


CANDIDATE_CACHE_SCHEMA_VERSION = "mission2_candidate_audio_features_v1"
CANDIDATE_EXTRACTION_CONFIG = {
    "sample_rate": 8_000,
    "n_fft": 512,
    "hop_length": 160,
    "silence_threshold_dbfs": -40.0,
    "clipping_amplitude": 0.999,
}


@dataclass(frozen=True)
class CandidateFeatureSpec:
    name: str
    family: str
    waveform_only: bool
    rule_compliance: str
    information_difference_from_mfcc40: str
    computation_cost: str


@dataclass(frozen=True)
class CandidateSelection:
    """A user-selected Stage-2 result; no automatic winner is chosen."""

    selection_name: str
    feature_names: tuple[str, ...]
    evidence_stage: str
    selection_fingerprint: str


def _spec(
    name: str,
    family: str,
    information: str,
    cost: str,
) -> CandidateFeatureSpec:
    return CandidateFeatureSpec(
        name=name,
        family=family,
        waveform_only=True,
        rule_compliance=(
            "waveform-derived audio only; no text/order/absolute timestamp/overlap input"
        ),
        information_difference_from_mfcc40=information,
        computation_cost=cost,
    )


CANDIDATE_REGISTRY: dict[str, CandidateFeatureSpec] = {
    "silence_ratio": _spec(
        "silence_ratio",
        "energy_occupancy",
        "summarizes low-energy frame occupancy rather than MFCC coefficient moments",
        "low",
    ),
    "rms": _spec(
        "rms",
        "energy",
        "summarizes utterance-level signal energy",
        "low",
    ),
    "rms_frame_std": _spec(
        "rms_frame_std",
        "energy",
        "summarizes frame-to-frame energy variability",
        "low",
    ),
    "peak_amplitude": _spec(
        "peak_amplitude",
        "amplitude",
        "captures peak level not represented explicitly by MFCC mean/std",
        "low",
    ),
    "amplitude_dynamic_range": _spec(
        "amplitude_dynamic_range",
        "amplitude",
        "captures robust amplitude spread",
        "low",
    ),
    "clipping_ratio": _spec(
        "clipping_ratio",
        "amplitude_quality",
        "tracks waveform saturation as an audio-quality cue",
        "low",
    ),
    "zcr": _spec(
        "zcr",
        "zero_crossing",
        "captures sign-change rate related to noisiness/voicing",
        "low",
    ),
    "spectral_centroid": _spec(
        "spectral_centroid",
        "spectral",
        "explicitly summarizes spectral center of mass",
        "medium",
    ),
    "spectral_bandwidth": _spec(
        "spectral_bandwidth",
        "spectral",
        "explicitly summarizes spectral spread",
        "medium",
    ),
    "spectral_rolloff": _spec(
        "spectral_rolloff",
        "spectral",
        "summarizes the upper spectral energy boundary",
        "medium",
    ),
    "spectral_flatness": _spec(
        "spectral_flatness",
        "spectral",
        "summarizes tone-like versus noise-like spectral shape",
        "medium",
    ),
}


def candidate_registry_table() -> pd.DataFrame:
    return pd.DataFrame([vars(spec) for spec in CANDIDATE_REGISTRY.values()])


def validate_candidate_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("select at least one candidate feature")
    if len(names) != len(set(names)):
        raise ValueError("candidate feature names must be unique")
    unknown = [name for name in names if name not in CANDIDATE_REGISTRY]
    if unknown:
        raise ValueError(f"unknown or forbidden candidate features: {unknown}")
    for name in names:
        spec = CANDIDATE_REGISTRY[name]
        if not spec.waveform_only:
            raise ValueError(f"candidate is not waveform-only: {name}")
    validate_model_feature_names(names)
    return names


def validate_raw_eda_acoustic_metadata(
    acoustic_metadata: Mapping[str, Any],
    utterance_metadata: Mapping[str, Any],
) -> None:
    """Validate the saved sample's extraction/source contract before reuse."""

    required_acoustic = {
        "schema_version",
        "source_signature",
        "sample_key_signature",
        "random_seed",
        "per_stratum",
        "n_fft",
        "hop_length",
        "n_mfcc",
        "silence_threshold_dbfs",
        "clipping_amplitude",
        "include_f0",
        "feature_columns",
    }
    missing = sorted(required_acoustic - set(acoustic_metadata))
    if missing:
        raise AssertionError(f"Raw EDA acoustic metadata missing keys: {missing}")
    if acoustic_metadata["schema_version"] != "mission2_raw_eda_v1":
        raise AssertionError("Raw EDA acoustic schema mismatch")
    if acoustic_metadata["source_signature"] != utterance_metadata.get(
        "source_signature"
    ):
        raise AssertionError("Raw EDA acoustic/manifest source signature mismatch")
    if {
        "n_fft": acoustic_metadata["n_fft"],
        "hop_length": acoustic_metadata["hop_length"],
        "n_mfcc": acoustic_metadata["n_mfcc"],
    } != {
        "n_fft": MFCC_CONFIG["n_fft"],
        "hop_length": MFCC_CONFIG["hop_length"],
        "n_mfcc": MFCC_CONFIG["n_mfcc"],
    }:
        raise AssertionError("Raw EDA acoustic configuration differs from reference")
    if acoustic_metadata["include_f0"] is not False:
        raise AssertionError("current feasibility cache contract expects F0 disabled")
    missing_candidates = sorted(
        set(CANDIDATE_REGISTRY) - set(acoustic_metadata["feature_columns"])
    )
    if missing_candidates:
        raise AssertionError(
            f"Raw EDA acoustic cache lacks registered candidates: {missing_candidates}"
        )


def _standardized_mean_difference(class_0: np.ndarray, class_1: np.ndarray) -> float:
    if len(class_0) < 2 or len(class_1) < 2:
        return np.nan
    pooled_variance = (
        ((len(class_0) - 1) * np.var(class_0, ddof=1))
        + ((len(class_1) - 1) * np.var(class_1, ddof=1))
    ) / (len(class_0) + len(class_1) - 2)
    if pooled_variance <= 0:
        return np.nan
    return float((np.mean(class_1) - np.mean(class_0)) / np.sqrt(pooled_variance))


def stage1_feasibility_diagnostics(
    raw_eda_acoustic_sample: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    expected_training_rows: int = 600,
) -> dict[str, Any]:
    """Describe Training sample feasibility; never rank or select candidates."""

    names = validate_candidate_names(feature_names)
    require_columns(
        raw_eda_acoustic_sample,
        ["split", "speaker", "call_id", "record_key", *names],
        "Raw EDA acoustic sample",
    )
    if set(raw_eda_acoustic_sample["split"].astype(str)) != {"Training"}:
        raise AssertionError("Stage 1 accepts Raw EDA Training rows only")
    if len(raw_eda_acoustic_sample) != expected_training_rows:
        raise AssertionError(
            f"Stage 1 Training sample row mismatch: "
            f"{len(raw_eda_acoustic_sample)} vs {expected_training_rows}"
        )
    if not raw_eda_acoustic_sample["record_key"].is_unique:
        raise AssertionError("Stage 1 record_key is not unique")
    if set(raw_eda_acoustic_sample["speaker"].astype(int)) != {0, 1}:
        raise AssertionError("Stage 1 labels must be 0/1")

    mfcc_columns = [
        column
        for column in raw_eda_acoustic_sample.columns
        if column.startswith("mfcc_")
        and (column.endswith("_mean") or column.endswith("_std"))
    ]
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for name in names:
        numeric = pd.to_numeric(raw_eda_acoustic_sample[name], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        finite = np.isfinite(values)
        finite_values = values[finite]
        if not len(finite_values):
            quantiles = [np.nan] * 7
            outlier_fraction = np.nan
        else:
            quantiles = np.quantile(
                finite_values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            )
            q1, q3 = quantiles[2], quantiles[4]
            iqr = q3 - q1
            outlier_fraction = (
                float(np.mean((finite_values < q1 - 3 * iqr) | (finite_values > q3 + 3 * iqr)))
                if iqr > 0
                else 0.0
            )
        class_values: dict[int, np.ndarray] = {}
        for speaker in [0, 1]:
            speaker_values = numeric.loc[
                raw_eda_acoustic_sample["speaker"].astype(int) == speaker
            ].to_numpy(dtype=float)
            speaker_values = speaker_values[np.isfinite(speaker_values)]
            class_values[speaker] = speaker_values
            class_rows.append(
                {
                    "feature": name,
                    "speaker": speaker,
                    "finite_count": len(speaker_values),
                    "mean": float(np.mean(speaker_values)) if len(speaker_values) else np.nan,
                    "std": float(np.std(speaker_values)) if len(speaker_values) else np.nan,
                    "median": float(np.median(speaker_values)) if len(speaker_values) else np.nan,
                }
            )

        pearson_values: list[float] = []
        spearman_values: list[float] = []
        for mfcc_name in mfcc_columns:
            pair = pd.DataFrame(
                {
                    "candidate": numeric,
                    "mfcc": pd.to_numeric(
                        raw_eda_acoustic_sample[mfcc_name], errors="coerce"
                    ),
                }
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3:
                continue
            pearson = pair.corr(method="pearson").iloc[0, 1]
            spearman = pair.corr(method="spearman").iloc[0, 1]
            if np.isfinite(pearson):
                pearson_values.append(abs(float(pearson)))
            if np.isfinite(spearman):
                spearman_values.append(abs(float(spearman)))
            correlation_rows.append(
                {
                    "candidate_feature": name,
                    "mfcc_feature": mfcc_name,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
        spec = CANDIDATE_REGISTRY[name]
        summary_rows.append(
            {
                "feature": name,
                "dtype": str(raw_eda_acoustic_sample[name].dtype),
                "rows": len(values),
                "finite_count": int(finite.sum()),
                "missing_count": int(pd.isna(numeric).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "p01": quantiles[0],
                "p05": quantiles[1],
                "q1": quantiles[2],
                "median": quantiles[3],
                "q3": quantiles[4],
                "p95": quantiles[5],
                "p99": quantiles[6],
                "class_1_minus_0_smd": _standardized_mean_difference(
                    class_values[0], class_values[1]
                ),
                "max_abs_pearson_with_mfcc40": max(pearson_values, default=np.nan),
                "max_abs_spearman_with_mfcc40": max(spearman_values, default=np.nan),
                "outlier_fraction_beyond_3iqr": outlier_fraction,
                "computation_cost": spec.computation_cost,
                "rule_compliance": spec.rule_compliance,
                "mfcc_information_difference": spec.information_difference_from_mfcc40,
            }
        )

    return {
        "scope": {
            "source": "Raw EDA acoustic cache: Training partition only",
            "rows": len(raw_eda_acoustic_sample),
            "purpose": "feasibility_diagnostic_only",
            "may_rank_or_select_candidates": False,
            "official_validation_used": False,
        },
        "feature_summary": pd.DataFrame(summary_rows),
        "class_distribution": pd.DataFrame(class_rows),
        "candidate_mfcc_correlations": pd.DataFrame(correlation_rows),
    }


def _analysis_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if not len(values):
        raise ValueError("empty waveform crop")
    if not np.isfinite(values).all():
        raise ValueError("waveform contains NaN or inf")
    n_fft = CANDIDATE_EXTRACTION_CONFIG["n_fft"]
    return np.pad(values, (0, n_fft - len(values))) if len(values) < n_fft else values


def extract_registered_audio_features(
    segment: np.ndarray,
    sample_rate: int,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Compute only explicitly selected waveform-derived feature families."""

    names = validate_candidate_names(feature_names)
    if sample_rate != CANDIDATE_EXTRACTION_CONFIG["sample_rate"]:
        raise ValueError(f"unexpected sample rate: {sample_rate}")
    raw_signal = np.asarray(segment, dtype=np.float32).reshape(-1)
    analysis_signal = _analysis_signal(raw_signal)
    n_fft = CANDIDATE_EXTRACTION_CONFIG["n_fft"]
    hop_length = CANDIDATE_EXTRACTION_CONFIG["hop_length"]
    results: dict[str, float] = {}

    energy_names = {
        "rms",
        "rms_frame_std",
        "silence_ratio",
    }
    if set(names) & energy_names:
        frame_rms = librosa.feature.rms(
            y=analysis_signal, frame_length=n_fft, hop_length=hop_length
        )[0]
        if "rms" in names:
            results["rms"] = float(
                np.sqrt(np.mean(np.square(raw_signal, dtype=np.float64)))
            )
        if "rms_frame_std" in names:
            results["rms_frame_std"] = float(np.std(frame_rms))
        if "silence_ratio" in names:
            threshold = 10 ** (
                CANDIDATE_EXTRACTION_CONFIG["silence_threshold_dbfs"] / 20
            )
            results["silence_ratio"] = float(np.mean(frame_rms <= threshold))

    amplitude_names = {
        "peak_amplitude",
        "amplitude_dynamic_range",
        "clipping_ratio",
    }
    if set(names) & amplitude_names:
        absolute = np.abs(raw_signal)
        if "peak_amplitude" in names:
            results["peak_amplitude"] = float(absolute.max())
        if "amplitude_dynamic_range" in names:
            results["amplitude_dynamic_range"] = float(
                np.quantile(absolute, 0.95) - np.quantile(absolute, 0.05)
            )
        if "clipping_ratio" in names:
            results["clipping_ratio"] = float(
                np.mean(
                    absolute
                    >= CANDIDATE_EXTRACTION_CONFIG["clipping_amplitude"]
                )
            )

    if "zcr" in names:
        results["zcr"] = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y=analysis_signal,
                    frame_length=n_fft,
                    hop_length=hop_length,
                )[0]
            )
        )

    spectral_extractors = {
        "spectral_centroid": lambda: librosa.feature.spectral_centroid(
            y=analysis_signal,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        ),
        "spectral_bandwidth": lambda: librosa.feature.spectral_bandwidth(
            y=analysis_signal,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        ),
        "spectral_rolloff": lambda: librosa.feature.spectral_rolloff(
            y=analysis_signal,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        ),
        "spectral_flatness": lambda: librosa.feature.spectral_flatness(
            y=analysis_signal,
            n_fft=n_fft,
            hop_length=hop_length,
        ),
    }
    for name in names:
        if name in spectral_extractors:
            results[name] = float(np.mean(spectral_extractors[name]()))

    if tuple(results) != tuple(name for name in names if name in results):
        results = {name: results[name] for name in names}
    values = np.asarray(list(results.values()), dtype=float)
    if tuple(results) != names or not np.isfinite(values).all():
        raise AssertionError("candidate extraction returned missing/non-finite values")
    return results


def record_key_signature(record_keys: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(map(str, record_keys)):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_cache_paths(
    cache_root: Path, split: str, feature_names: Sequence[str]
) -> tuple[Path, Path]:
    names = validate_candidate_names(feature_names)
    fingerprint = hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()[:12]
    prefix = Path(cache_root) / f"{split.lower()}_candidate_{fingerprint}"
    return prefix.with_suffix(".parquet"), prefix.with_suffix(".json")


def expected_candidate_cache_metadata(
    manifest: pd.DataFrame,
    *,
    split: str,
    feature_names: Sequence[str],
    source_signature: str,
) -> dict[str, Any]:
    if split not in {"Training", "Validation"}:
        raise ValueError(f"unknown source split: {split}")
    names = validate_candidate_names(feature_names)
    split_manifest = manifest.loc[manifest["split"] == split]
    require_columns(split_manifest, ["record_key"], f"{split} manifest")
    if not split_manifest["record_key"].is_unique:
        raise AssertionError(f"{split} manifest record_key is not unique")
    return {
        "cache_schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "source_split": split,
        "source_signature": str(source_signature),
        "record_key_signature": record_key_signature(split_manifest["record_key"]),
        "row_count": len(split_manifest),
        "feature_names": list(names),
        "feature_count": len(names),
        "waveform_only": True,
        "extraction_config": CANDIDATE_EXTRACTION_CONFIG,
    }


def extract_candidate_feature_frame(
    manifest: pd.DataFrame,
    *,
    data_root: Path,
    split: str,
    feature_names: Sequence[str],
    on_call_done: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Read each call once and calculate only registered selected features."""

    names = validate_candidate_names(feature_names)
    required = [
        "split",
        "record_key",
        "call_id",
        "audio_path",
        "crop_start_frame",
        "crop_end_frame",
        "crop_ok",
    ]
    require_columns(manifest, required, "candidate extraction manifest")
    source = manifest.loc[manifest["split"] == split, required].copy()
    if source.empty:
        raise AssertionError(f"no {split} rows for candidate extraction")
    if not source["record_key"].is_unique:
        raise AssertionError(f"{split} extraction record_key is not unique")
    if not source["crop_ok"].astype(bool).all():
        raise AssertionError(f"{split} has invalid crops; do not silently drop rows")
    source["source_position"] = np.arange(len(source), dtype=np.int64)

    rows: list[dict[str, Any]] = []
    grouped = source.groupby(["call_id", "audio_path"], sort=False)
    total_calls = len(grouped)
    for call_number, ((call_id, audio_path), group) in enumerate(grouped, start=1):
        waveform, sample_rate = read_call_waveform(Path(data_root) / str(audio_path))
        if sample_rate != CANDIDATE_EXTRACTION_CONFIG["sample_rate"]:
            raise AssertionError(
                f"unexpected sample rate for {call_id}: {sample_rate}"
            )
        for row in group.itertuples(index=False):
            start_frame = int(row.crop_start_frame)
            end_frame = int(row.crop_end_frame)
            if start_frame < 0 or end_frame <= start_frame or end_frame > len(waveform):
                raise AssertionError(
                    f"invalid crop for {row.record_key}: [{start_frame}, {end_frame})"
                )
            features = extract_registered_audio_features(
                waveform[start_frame:end_frame], sample_rate, names
            )
            rows.append(
                {
                    "source_position": int(row.source_position),
                    "record_key": str(row.record_key),
                    "call_id": str(call_id),
                    **features,
                }
            )
        if on_call_done is not None:
            on_call_done(call_number, total_calls)
    frame = pd.DataFrame(rows).sort_values("source_position", kind="stable")
    frame = frame.drop(columns="source_position").reset_index(drop=True)
    if len(frame) != len(source) or not frame["record_key"].is_unique:
        raise AssertionError("candidate extraction changed row count/record keys")
    if set(frame["record_key"]) != set(source["record_key"]):
        raise AssertionError("candidate extraction record keys differ from manifest")
    if not np.isfinite(frame[list(names)].to_numpy(dtype=float)).all():
        raise AssertionError("candidate feature frame contains NaN or inf")
    return frame


def build_candidate_feature_cache(
    manifest: pd.DataFrame,
    *,
    data_root: Path,
    cache_root: Path,
    split: str,
    feature_names: Sequence[str],
    source_signature: str,
    allow_full_extraction: bool = False,
    on_call_done: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path]:
    """Explicitly build one immutable, atomically-published feature artifact."""

    if not allow_full_extraction:
        raise RuntimeError(
            "STOP: set allow_full_extraction=True only for a deliberate large audio run"
        )
    names = validate_candidate_names(feature_names)
    parquet_path, metadata_path = candidate_cache_paths(cache_root, split, names)
    if parquet_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"candidate cache already exists; validate/reuse it, do not overwrite: "
            f"{parquet_path} / {metadata_path}"
        )
    metadata = expected_candidate_cache_metadata(
        manifest,
        split=split,
        feature_names=names,
        source_signature=source_signature,
    )
    frame = extract_candidate_feature_frame(
        manifest,
        data_root=data_root,
        split=split,
        feature_names=names,
        on_call_done=on_call_done,
    )
    if len(frame) != metadata["row_count"]:
        raise AssertionError("candidate cache row count differs from metadata contract")

    Path(cache_root).mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary_parquet, index=False)
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_parquet, parquet_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        temporary_parquet.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise
    return parquet_path, metadata_path


def load_candidate_feature_cache(
    manifest: pd.DataFrame,
    *,
    cache_root: Path,
    split: str,
    feature_names: Sequence[str],
    source_signature: str,
) -> pd.DataFrame:
    """Load an exact compatible artifact; never build one on cache miss."""

    names = validate_candidate_names(feature_names)
    parquet_path, metadata_path = candidate_cache_paths(cache_root, split, names)
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing candidate cache; automatic extraction is disabled: "
            f"{parquet_path} / {metadata_path}"
        )
    actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = expected_candidate_cache_metadata(
        manifest,
        split=split,
        feature_names=names,
        source_signature=source_signature,
    )
    if actual != expected:
        raise AssertionError(
            f"candidate cache metadata mismatch: actual={actual!r}, expected={expected!r}"
        )
    frame = pd.read_parquet(parquet_path)
    require_columns(frame, ["record_key", "call_id", *names], "candidate cache")
    if len(frame) != expected["row_count"] or not frame["record_key"].is_unique:
        raise AssertionError("candidate cache row count/key uniqueness mismatch")
    if record_key_signature(frame["record_key"]) != expected["record_key_signature"]:
        raise AssertionError("candidate cache record_key signature mismatch")
    if not np.isfinite(frame[list(names)].to_numpy(dtype=float)).all():
        raise AssertionError("candidate cache contains NaN or inf")
    return frame


def _stage2_distribution_table(
    aligned_training: AlignedSplit,
    candidate_frame: pd.DataFrame,
    candidate_values: np.ndarray,
    feature_names: Sequence[str],
    train_idx: np.ndarray,
    internal_validation_idx: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, indices in [
        ("internal_train", train_idx),
        ("internal_validation", internal_validation_idx),
    ]:
        label_slices: list[tuple[str, np.ndarray]] = [("all", indices)]
        label_slices.extend(
            (str(speaker), indices[aligned_training.y[indices] == speaker])
            for speaker in (0, 1)
        )
        for speaker, selected_indices in label_slices:
            for feature_index, feature in enumerate(feature_names):
                values = candidate_values[selected_indices, feature_index]
                if not len(values):
                    raise AssertionError(
                        f"empty Stage 2 class slice: {split_name}, speaker={speaker}"
                    )
                q1, q3 = np.quantile(values, [0.25, 0.75])
                iqr = q3 - q1
                outlier_fraction = (
                    float(np.mean((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)))
                    if iqr > 0
                    else 0.0
                )
                rows.append(
                    {
                        "partition": split_name,
                        "speaker": speaker,
                        "feature": feature,
                        "dtype": str(candidate_frame[feature].dtype),
                        "rows": len(values),
                        "missing_or_inf": int((~np.isfinite(values)).sum()),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "p01": float(np.quantile(values, 0.01)),
                        "q1": float(q1),
                        "median": float(np.median(values)),
                        "q3": float(q3),
                        "p99": float(np.quantile(values, 0.99)),
                        "outlier_fraction_beyond_3iqr": outlier_fraction,
                    }
                )
    return pd.DataFrame(rows)


def _stage2_redundancy_table(
    aligned_training: AlignedSplit,
    candidate_values: np.ndarray,
    feature_names: Sequence[str],
    train_idx: np.ndarray,
    internal_validation_idx: np.ndarray,
    *,
    correlation_sample_limit: int,
) -> pd.DataFrame:
    """Report deterministic Training-only Pearson diagnostics; never filter."""

    if correlation_sample_limit < 3:
        raise ValueError("correlation_sample_limit must be at least 3")
    mfcc_values = aligned_training.X_mfcc
    rows: list[dict[str, Any]] = []
    for partition, indices in [
        ("internal_train", train_idx),
        ("internal_validation", internal_validation_idx),
    ]:
        sample_count = min(len(indices), correlation_sample_limit)
        sample_offsets = np.linspace(
            0, len(indices) - 1, num=sample_count, dtype=np.int64
        )
        sampled_indices = indices[sample_offsets]
        sampled_mfcc = mfcc_values[sampled_indices]
        centered_mfcc = sampled_mfcc - sampled_mfcc.mean(axis=0)
        mfcc_norms = np.linalg.norm(centered_mfcc, axis=0)
        for feature_index, feature in enumerate(feature_names):
            candidate = candidate_values[sampled_indices, feature_index]
            centered_candidate = candidate - candidate.mean()
            candidate_norm = np.linalg.norm(centered_candidate)
            denominator = mfcc_norms * candidate_norm
            correlations = np.full(sampled_mfcc.shape[1], np.nan, dtype=float)
            valid = denominator > 0
            correlations[valid] = (
                centered_mfcc[:, valid].T @ centered_candidate
            ) / denominator[valid]
            absolute = np.abs(correlations)
            if np.isfinite(absolute).any():
                strongest_index = int(np.nanargmax(absolute))
                strongest_name = BASELINE_FEATURE_NAMES[strongest_index]
                strongest_value = float(absolute[strongest_index])
            else:
                strongest_name = None
                strongest_value = np.nan
            rows.append(
                {
                    "partition": partition,
                    "feature": feature,
                    "correlation_rows": sample_count,
                    "sampling": "deterministic_even_positions_within_fixed_group_split",
                    "most_correlated_mfcc": strongest_name,
                    "max_abs_pearson_with_mfcc40": strongest_value,
                    "automatic_filter_applied": False,
                }
            )
    return pd.DataFrame(rows)


def stage2_training_group_screen(
    aligned_training: AlignedSplit,
    candidate_frame: pd.DataFrame,
    candidate_groups: Mapping[str, Sequence[str]],
    *,
    seed: int = 42,
    test_size: float = 0.2,
    minimum_training_rows: int = 601,
    correlation_sample_limit: int = 100_000,
) -> dict[str, Any]:
    """Compare candidates on one fixed full/larger Training call GroupSplit."""

    if aligned_training.split != "Training":
        raise AssertionError("Stage 2 accepts Training only; official Validation is forbidden")
    if len(aligned_training.y) < minimum_training_rows:
        raise AssertionError(
            "Stage 2 requires a larger-than-600 Training cache; "
            f"rows={len(aligned_training.y)}, minimum={minimum_training_rows}"
        )
    if not candidate_groups:
        raise ValueError("candidate_groups must not be empty")
    groups = {
        str(name): validate_candidate_names(features)
        for name, features in candidate_groups.items()
    }
    all_features = tuple(dict.fromkeys(feature for values in groups.values() for feature in values))
    require_columns(candidate_frame, ["record_key", *all_features], "Stage 2 candidate cache")
    aligned_candidate_values = align_extra_feature_matrix(
        aligned_training, candidate_frame, all_features
    )
    train_idx, internal_validation_idx, split_audit = group_split_indices(
        aligned_training.tracking["call_id"], seed=seed, test_size=test_size
    )
    if split_audit["call_id_overlap"] != 0:
        raise AssertionError("Stage 2 group leakage")

    baseline_internal = fit_reference_logistic_regression(
        experiment_id="stage2_reference_only",
        X_train=aligned_training.X_mfcc[train_idx],
        y_train=aligned_training.y[train_idx],
        X_evaluation=aligned_training.X_mfcc[internal_validation_idx],
        y_evaluation=aligned_training.y[internal_validation_idx],
        feature_names=BASELINE_FEATURE_NAMES,
        training_scope="Training/internal-train",
        evaluation_scope="Training/internal-validation",
    )
    performance_rows: list[dict[str, Any]] = []
    for group_name, names in groups.items():
        selected_columns = [all_features.index(name) for name in names]
        matrix = np.column_stack(
            [aligned_training.X_mfcc, aligned_candidate_values[:, selected_columns]]
        )
        feature_names = (*BASELINE_FEATURE_NAMES, *names)
        assert_single_factor_comparison(
            aligned_training.X_mfcc,
            matrix,
            aligned_training.y,
            aligned_training.y,
            BASELINE_FEATURE_NAMES,
            feature_names,
            names,
        )
        candidate_internal = fit_reference_logistic_regression(
            experiment_id=f"stage2::{group_name}",
            X_train=matrix[train_idx],
            y_train=aligned_training.y[train_idx],
            X_evaluation=matrix[internal_validation_idx],
            y_evaluation=aligned_training.y[internal_validation_idx],
            feature_names=feature_names,
            training_scope="Training/internal-train",
            evaluation_scope="Training/internal-validation",
        )
        performance_rows.append(
            {
                "candidate_name": group_name,
                "added_features": list(names),
                "feature_dim": len(feature_names),
                "internal_validation_rows": len(internal_validation_idx),
                "accuracy": candidate_internal.metrics["accuracy"],
                "accuracy_delta_vs_mfcc40": (
                    candidate_internal.metrics["accuracy"]
                    - baseline_internal.metrics["accuracy"]
                ),
                "recall_0": candidate_internal.metrics["recall_0"],
                "recall_1": candidate_internal.metrics["recall_1"],
                "baseline_recall_0": baseline_internal.metrics["recall_0"],
                "baseline_recall_1": baseline_internal.metrics["recall_1"],
                "official_validation_used": False,
            }
        )
    return {
        "scope": {
            "purpose": "stage2_candidate_priority_evidence",
            "source_split": "Training",
            "source_rows": len(aligned_training.y),
            "minimum_training_rows": minimum_training_rows,
            "official_validation_used": False,
            "automatic_winner_selected": False,
        },
        "split_audit": pd.DataFrame([split_audit]),
        "baseline_metrics": baseline_internal.metrics,
        "candidate_performance": pd.DataFrame(performance_rows),
        "distribution_diagnostics": _stage2_distribution_table(
            aligned_training,
            candidate_frame,
            aligned_candidate_values,
            all_features,
            train_idx,
            internal_validation_idx,
        ),
        "redundancy_diagnostics": _stage2_redundancy_table(
            aligned_training,
            aligned_candidate_values,
            all_features,
            train_idx,
            internal_validation_idx,
            correlation_sample_limit=correlation_sample_limit,
        ),
        "train_indices": train_idx,
        "internal_validation_indices": internal_validation_idx,
    }


def lock_candidate_selection(
    stage2_result: Mapping[str, Any],
    *,
    selection_name: str,
    feature_names: Sequence[str],
) -> CandidateSelection:
    """Lock a human choice that is present in Stage 2; never choose automatically."""

    names = validate_candidate_names(feature_names)
    scope = stage2_result.get("scope", {})
    if scope.get("source_split") != "Training" or scope.get(
        "official_validation_used"
    ) is not False:
        raise AssertionError("candidate selection must be supported by Training-only Stage 2")
    performance = stage2_result["candidate_performance"]
    matched = performance.loc[performance["candidate_name"] == selection_name]
    if len(matched) != 1:
        raise AssertionError("selection_name must match exactly one Stage 2 candidate")
    if tuple(matched.iloc[0]["added_features"]) != names:
        raise AssertionError("locked feature names differ from Stage 2 result")
    fingerprint = hashlib.sha256(
        (selection_name + "|" + "|".join(names)).encode("utf-8")
    ).hexdigest()
    return CandidateSelection(
        selection_name=selection_name,
        feature_names=names,
        evidence_stage="full_or_larger_training_call_id_group_split",
        selection_fingerprint=fingerprint,
    )
