"""Gated acoustic-block screening contracts for DCC 2026 Mission 2.

Only the waveform of the current utterance is used to compute model features.
The module never extracts a full cache or fits a model unless its caller passes
an explicit opt-in flag.  Existing E05 cache loading remains owned by
``mission2_model_experiments``; this module adds new block extraction,
alignment, fixed-split preparation, and same-run paired diagnostics.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import warnings

import librosa
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - sklearn normally supplies this dependency.
    _threadpool_limits = None

_WORKER_THREADPOOL_GUARD: Any | None = None

from .audio import read_wav_frames
from . import mission2_model_experiments as _model_experiments_module
from .mission2_model_experiments import (
    ARRAY_FINGERPRINT_VERSION,
    E05_FEATURE_NAMES,
    EXPECTED_INTERNAL_SPLIT,
    M04_MLP_CONFIG,
    ModelExperimentResult,
    _array_fingerprint,
    _rate_matches_count,
    _overlap_mask,
    build_standalone_diagnostic_report,
    calculate_prediction_metrics,
    config_fingerprint,
    create_fixed_internal_split,
    record_key_signature,
    standalone_group_split_indices,
    validate_binary_probabilities,
)


SAMPLE_RATE = 8_000
N_FFT = 512
HOP_LENGTH = 160
LOG_RMS_FLOOR = 1e-7
CACHE_SCHEMA_VERSION = "mission2_acoustic_feature_expansion_v1"
M04_ITER100_ACCURACY_TARGET = 0.859532
ROUND2_M04_ITER100_ACCURACY_TARGET = 0.866266

PITCH_EXTRACTION_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "method": "normalized_autocorrelation",
        "fmin_hz": 65.0,
        "fmax_hz": 400.0,
        "frame_length": 1024,
        "hop_length": 160,
        "semitone_reference_hz": 27.5,
        "minimum_voiced_frames": 3,
        "periodicity_threshold": 0.6,
        "minimum_frame_rms": 0.0001,
        "slope_unit": "semitones_per_second",
        "fallback": "zero_f0_statistics_keep_voicing_statistics",
    }
)

_PITCH_FRAME_LENGTH = int(PITCH_EXTRACTION_CONFIG["frame_length"])
_PITCH_HOP_LENGTH = int(PITCH_EXTRACTION_CONFIG["hop_length"])
_PITCH_LAG_MIN = int(
    np.floor(SAMPLE_RATE / float(PITCH_EXTRACTION_CONFIG["fmax_hz"]))
)
_PITCH_LAG_MAX = int(
    np.ceil(SAMPLE_RATE / float(PITCH_EXTRACTION_CONFIG["fmin_hz"]))
)
_PITCH_LAGS = np.arange(_PITCH_LAG_MIN, _PITCH_LAG_MAX + 1, dtype=np.int64)
_PITCH_WINDOW = np.hanning(_PITCH_FRAME_LENGTH).astype(np.float64)
_PITCH_FFT_LENGTH = 2 * _PITCH_FRAME_LENGTH
_PITCH_FRAME_BATCH_SIZE = 256
_PITCH_LAGS.setflags(write=False)
_PITCH_WINDOW.setflags(write=False)

# The 480-utterance Windows benchmark found process startup and memory costs
# larger than the parallel compute saving. Keep Notebook execution on the
# measured stable path; explicit counts remain available for other runtimes.
_DEFAULT_WORKER_COUNT = 1

M04_ITER100_FEATURE_SCREENING_CONFIG: Mapping[str, Any] = MappingProxyType(
    {**dict(M04_MLP_CONFIG), "max_iter": 100}
)


@dataclass(frozen=True)
class AcousticFeatureBlock:
    """One result-blind acoustic block frozen before full extraction."""

    block_id: str
    feature_names: tuple[str, ...]
    acoustic_meaning: str
    calculation: str
    e05_overlap: str
    mission2_hypothesis: str
    research_basis: str
    rule_compliance: str
    diagnostic_risks: str
    expected_cost: str
    output_dimension: int


FEATURE_BLOCKS = (
    AcousticFeatureBlock(
        block_id="AF01_ENERGY_DYNAMICS",
        feature_names=(
            "log_rms_mean_db",
            "log_rms_std_db",
            "log_rms_p90_minus_p10_db",
        ),
        acoustic_meaning="utterance energy level and frame-to-frame energy dynamics",
        calculation=(
            "centered frame RMS (512/160), 20*log10(max(RMS, 1e-7)); "
            "mean, standard deviation, and P90-P10"
        ),
        e05_overlap=(
            "MFCC moments contain related energy information, but E05 has no "
            "explicit log-RMS contour summary"
        ),
        mission2_hypothesis=(
            "caller and dispatcher speaking style may differ in level variability; "
            "incremental value must be measured rather than assumed"
        ),
        research_basis=(
            "GeMAPS treats energy/amplitude descriptors as a distinct voice-analysis "
            "group; the implemented frame-RMS equation is explicitly precommitted"
        ),
        rule_compliance="current-utterance waveform only; no annotation or call context",
        diagnostic_risks=(
            "log-RMS mean is recording-level and gain sensitive; channel level may be "
            "a nuisance shortcut, so distribution and slice diagnostics are required"
        ),
        expected_cost="low",
        output_dimension=3,
    ),
    AcousticFeatureBlock(
        block_id="AF02_PITCH_VOICING",
        feature_names=(
            "f0_semitone_median",
            "f0_semitone_iqr",
            "f0_semitone_slope",
            "voiced_fraction",
            "periodicity_mean",
        ),
        acoustic_meaning="F0 level, F0 variation, contour direction, and periodic voicing",
        calculation=(
            "normalized autocorrelation at 8 kHz with fmin=65 Hz, fmax=400 Hz, "
            "Hann frame=1024, hop=160, periodicity threshold=0.6, frame RMS "
            "threshold=1e-4; F0=12*log2(Hz/27.5); least-squares slope over "
            "voiced-frame times in semitones/second; fewer than 3 voiced frames "
            "sets the three F0 statistics to zero while retaining voicing statistics"
        ),
        e05_overlap="E05 has no explicit F0 contour or voicing-probability descriptor",
        mission2_hypothesis=(
            "prosodic and voicing patterns may add role-related information above "
            "static MFCC, rolloff, and ZCR summaries"
        ),
        research_basis=(
            "GeMAPS includes logarithmic F0 and voicing-related descriptors; the "
            "implemented normalized autocorrelation is explicitly precommitted"
        ),
        rule_compliance="current-utterance waveform only; no speaker metadata",
        diagnostic_risks=(
            "pitch tracking can fail on short, noisy, clipped, or unvoiced telephone "
            "speech; the pre-extraction small runtime and finite-fallback checks are mandatory"
        ),
        expected_cost="high relative to AF01/AF03; benchmark before full extraction",
        output_dimension=5,
    ),
    AcousticFeatureBlock(
        block_id="AF03_NORMALIZED_SPECTRAL_FLUX",
        feature_names=(
            "normalized_spectral_flux_mean",
            "normalized_spectral_flux_std",
            "normalized_spectral_flux_p90",
        ),
        acoustic_meaning="frame-to-frame spectral-shape change",
        calculation=(
            "L2-normalized magnitude STFT columns (Hann, 512/160); per transition "
            "sqrt(sum((S_norm[t]-S_norm[t-1])**2)); mean, standard deviation, P90. "
            "This is not librosa.onset_strength and not an eGeMAPS-equivalent implementation"
        ),
        e05_overlap=(
            "rolloff and ZCR are static utterance summaries; this block measures "
            "temporal spectral change"
        ),
        mission2_hypothesis=(
            "role-related speaking dynamics may remain after static spectrum summaries"
        ),
        research_basis=(
            "spectral flux is used as a spectral-dynamics descriptor in voice research; "
            "the exact normalized Euclidean definition above is the experiment contract"
        ),
        rule_compliance="current-utterance waveform only; no neighboring utterance context",
        diagnostic_risks=(
            "sensitive to transient noise and very short segments; normalization removes "
            "global level but not channel/noise effects"
        ),
        expected_cost="medium; reuses one STFT per utterance",
        output_dimension=3,
    ),
    AcousticFeatureBlock(
        block_id="AF04_CPP",
        feature_names=("cpp_median_db", "cpp_iqr_db"),
        acoustic_meaning="cepstral peak prominence as a compact voice-quality descriptor",
        calculation=(
            "centered, mean-removed Hann frames (512/160); active-frame RMS >=1e-4; "
            "power spectrum -> log spectrum -> real cepstrum -> 20*log10(abs(cepstrum)); "
            "ordinary-least-squares baseline over q=8..128 samples and maximum "
            "baseline-subtracted prominence over q=20..123; median and IQR; fewer "
            "than 3 active frames returns two zeros"
        ),
        e05_overlap=(
            "MFCC moments encode spectral-envelope information, but not this explicit "
            "cepstral peak-above-baseline voice-quality summary"
        ),
        mission2_hypothesis=(
            "voice quality may add information for remaining overlap, noisy, and "
            "high-confidence errors; it is not assumed to solve overlap"
        ),
        research_basis=(
            "CPP is a common voice-quality measure; this experiment freezes an "
            "explicit dependency-free frame-level definition"
        ),
        rule_compliance="current-utterance waveform only; no call context or annotation",
        diagnostic_risks=(
            "telephone filtering, noise, and MFCC-related redundancy can dominate; "
            "the active-frame fallback and feature distributions must be reported"
        ),
        expected_cost="medium; one 512-point real FFT and inverse real FFT per frame",
        output_dimension=2,
    ),
    AcousticFeatureBlock(
        block_id="AF05_DELTA_MFCC_DYNAMICS",
        feature_names=("delta_mfcc_norm_mean", "delta_mfcc_norm_std"),
        acoustic_meaning="short-term MFCC trajectory magnitude and variability",
        calculation=(
            "canonical 20-coefficient MFCC frontend (8 kHz, n_fft/win_length=512, "
            "hop=160, Hann, center=True, constant padding, power=2, n_mels=128, "
            "fmin=0, fmax=4000, Slaney mel, DCT-II orthonormal); MFCC 1..19 only; "
            "explicit nearest-edge delta (C[t+1]-C[t-1]+2*(C[t+2]-C[t-2]))/10; "
            "per-frame RMS across coefficients, then mean and population std; fewer "
            "than 3 MFCC frames returns two zeros"
        ),
        e05_overlap=(
            "E05 retains MFCC means and standard deviations but discards frame order; "
            "AF05 summarizes first-order temporal dynamics"
        ),
        mission2_hypothesis=(
            "MFCC trajectory dynamics may help the recurrent 0.5-1.0 second weakness; "
            "this differs from AF03 normalized linear-spectrum flux"
        ),
        research_basis=(
            "delta cepstra are standard speech dynamics descriptors; the explicit "
            "nearest-edge equation avoids library-version ambiguity"
        ),
        rule_compliance="current-utterance waveform only; no text or neighboring utterance",
        diagnostic_risks=(
            "strong redundancy with E05 MFCC moments and sensitivity to very short "
            "segments; canonical-frontend parity and fallback rates are mandatory"
        ),
        expected_cost="medium; one canonical MFCC computation per utterance",
        output_dimension=2,
    ),
    AcousticFeatureBlock(
        block_id="AF07_MFCC_CROSS_CORRELATION",
        feature_names=(
            "mfcc_cross_corr_abs_mean",
            "mfcc_cross_corr_pc1_share",
        ),
        acoustic_meaning=(
            "cross-coefficient dependence and low-rank coordination among canonical "
            "MFCC 1..19 frame trajectories"
        ),
        calculation=(
            "canonical MFCC20 frontend; discard coefficient 0; retain frames whose "
            "Hann-windowed RMS is >=1e-4; center and standardize each varying "
            "coefficient; compute the correlation matrix, mean absolute off-diagonal "
            "correlation, and largest non-negative eigenvalue share; after removing "
            "near-zero-variance coefficients, active_frame_count must be at least "
            "p_eff + 1 and p_eff must be at least 2"
        ),
        e05_overlap=(
            "MFCC mean/std retain marginal location and scale but discard covariance "
            "between coefficient trajectories"
        ),
        mission2_hypothesis=(
            "overlap or mixed acoustic structure may change coordinated spectral-envelope "
            "motion even when marginal MFCC moments look familiar"
        ),
        research_basis=(
            "covariance structure is information not recoverable from independent first "
            "and second marginal moments; the exact standardized correlation definition "
            "is frozen here"
        ),
        rule_compliance="current-utterance waveform only; no diagnostic metadata",
        diagnostic_risks=(
            "noise can also increase or reduce cross-coefficient dependence; targeted "
            "overlap and no-overlap diagnostics are required"
        ),
        expected_cost="medium; one canonical MFCC computation per utterance",
        output_dimension=2,
    ),
    AcousticFeatureBlock(
        block_id="AF08_COMPETING_PITCH",
        feature_names=(
            "pitch_peak_margin_median",
            "pitch_peak_margin_iqr",
        ),
        acoustic_meaning=(
            "ambiguity between the strongest normalized-autocorrelation pitch lag and "
            "the strongest non-octave-equivalent competitor"
        ),
        calculation=(
            "AF02-compatible 1024/160 Hann frames and normalized autocorrelation over "
            "65..400 Hz; for frames with primary periodicity >=0.6 exclude lags within "
            "0.08 octaves of any octave-equivalent primary lag, then summarize primary "
            "minus competitor by median and IQR; at least 3 valid frames are required"
        ),
        e05_overlap=(
            "AF02 retains only the winning F0 and primary periodicity strength; AF04 "
            "retains cepstral prominence, neither records competing-lag ambiguity"
        ),
        mission2_hypothesis=(
            "overlapping periodic sources may leave a less decisive pitch winner and "
            "contribute to persistent or high-confidence errors"
        ),
        research_basis=(
            "multi-period competition is distinct from a single best-lag strength; the "
            "octave-equivalence exclusion prevents one harmonic series from being counted "
            "as two sources"
        ),
        rule_compliance="current-utterance waveform only; no speaker or call statistics",
        diagnostic_risks=(
            "telephone noise and octave errors can also narrow the margin; the block is "
            "an ambiguity descriptor, not an overlap detector"
        ),
        expected_cost="high; AF02-sized normalized-autocorrelation computation",
        output_dimension=2,
    ),
    AcousticFeatureBlock(
        block_id="AF09_MULTIBAND_ENVELOPE_COHERENCE",
        feature_names=(
            "subband_envelope_corr_abs_mean",
            "subband_envelope_corr_min",
            "subband_envelope_pc1_share",
        ),
        acoustic_meaning=(
            "temporal coherence of coarse spectral-band energy envelopes"
        ),
        calculation=(
            "centered Hann STFT 512/160; sum power in 0..500, 500..1000, "
            "1000..2000, and 2000..4000 Hz; log-transform active-frame band powers, "
            "standardize each varying band, then compute mean absolute pairwise "
            "correlation, minimum signed pairwise correlation, and leading eigenvalue "
            "share; at least 5 active frames and 2 varying bands are required"
        ),
        e05_overlap=(
            "rolloff and ZCR are static summaries; MFCC moments do not preserve temporal "
            "coherence between fixed coarse-band envelopes"
        ),
        mission2_hypothesis=(
            "single-source and mixed/noisy frames may differ in how independently their "
            "low- and high-frequency envelopes evolve"
        ),
        research_basis=(
            "cross-band envelope coherence is a spectro-temporal dependence measure, not "
            "a static contrast, tilt, entropy, moment, or normalized spectral flux"
        ),
        rule_compliance="current-utterance waveform only; fixed bands use no learned data",
        diagnostic_risks=(
            "stationary noise and handset filtering can affect band correlations; the "
            "feature must be judged against both overlap and no-overlap slices"
        ),
        expected_cost="medium; one 512-point real FFT per frame",
        output_dimension=3,
    ),
    AcousticFeatureBlock(
        block_id="AF10_SPEECH_ACTIVITY_BOUNDARY",
        feature_names=(
            "active_frame_fraction",
            "longest_active_run_fraction",
            "activity_transition_rate",
            "edge_activity_difference",
        ),
        acoustic_meaning="within-utterance speech activity density and boundary asymmetry",
        calculation=(
            "centered Hann frame RMS (512/160); active threshold is max(1e-4, "
            "0.1 * utterance maximum frame RMS); summarize active-frame fraction, "
            "longest contiguous active run divided by frame count, adjacent activity "
            "transition rate, and trailing-minus-leading 20% active fraction"
        ),
        e05_overlap=(
            "E05 retains global MFCC moments, rolloff, and ZCR but no explicit "
            "within-utterance activity-run or edge-boundary summary"
        ),
        mission2_hypothesis=(
            "short, clipped, or low-information utterances may differ in how activity "
            "occupies the current utterance crop"
        ),
        research_basis=(
            "frame RMS is a deterministic speech-activity proxy; the threshold and "
            "edge definition are frozen and do not use labels or other utterances"
        ),
        rule_compliance="current-utterance waveform only; no call-relative information",
        diagnostic_risks=(
            "gain, stationary noise, and crop padding can affect activity; targeted "
            "duration and overlap diagnostics are required"
        ),
        expected_cost="low; one 512/160 frame-RMS pass",
        output_dimension=4,
    ),
    AcousticFeatureBlock(
        block_id="AF11_SPECTRAL_ENTROPY_DYNAMICS",
        feature_names=(
            "spectral_entropy_mean",
            "spectral_entropy_std",
            "spectral_entropy_iqr",
        ),
        acoustic_meaning="frame spectral diffuseness and its within-utterance variation",
        calculation=(
            "centered Hann STFT 512/160; normalize each frame power spectrum to a "
            "probability distribution and divide Shannon entropy by log(number of bins); "
            "silent-frame entropy is zero; summarize mean, population std, and IQR"
        ),
        e05_overlap=(
            "rolloff and MFCC moments summarize spectral location/shape but not the "
            "framewise normalized entropy distribution"
        ),
        mission2_hypothesis=(
            "overlap, clipped speech, and low-information fragments may differ in "
            "spectral diffuseness and temporal entropy variability"
        ),
        research_basis=(
            "normalized power-spectrum entropy is a bounded deterministic descriptor; "
            "silent frames use an explicit zero fallback"
        ),
        rule_compliance="current-utterance waveform only; no annotation or call context",
        diagnostic_risks=(
            "broadband noise can mimic high entropy and silence prevalence affects the "
            "summary; compare overlap and no-overlap slices"
        ),
        expected_cost="medium; one 512-point real FFT per frame",
        output_dimension=3,
    ),
    AcousticFeatureBlock(
        block_id="AF12_TEMPORAL_MFCC_CONTRAST",
        feature_names=(
            "early_middle_distance",
            "middle_late_distance",
            "early_late_distance",
        ),
        acoustic_meaning="coarse early/middle/late spectral-envelope change",
        calculation=(
            "reuse the canonical MFCC20 frame sequence; split frames contiguously into "
            "three nearly equal early/middle/late regions, average each 20D region, and "
            "compute the three Euclidean distances; fewer than three original 160-sample "
            "hops or fewer than 3 MFCC frames returns zeros"
        ),
        e05_overlap=(
            "E05 MFCC mean/std discards coarse temporal location, while this block "
            "contrasts three ordered regions inside only the current utterance"
        ),
        mission2_hypothesis=(
            "short or boundary-clipped utterances may retain asymmetric MFCC change "
            "that global moments miss"
        ),
        research_basis=(
            "segment-level MFCC mean contrasts retain coarse temporal information while "
            "avoiding absolute call time or neighboring utterances"
        ),
        rule_compliance="current-utterance waveform only; no absolute time or sequence proxy",
        diagnostic_risks=(
            "very short utterances and padding can dominate region means; an explicit "
            "short-frame fallback and duration diagnostics are required"
        ),
        expected_cost="medium; one canonical MFCC20 computation per utterance",
        output_dimension=3,
    ),
)

BLOCK_BY_ID: Mapping[str, AcousticFeatureBlock] = MappingProxyType(
    {block.block_id: block for block in FEATURE_BLOCKS}
)
ALL_CANDIDATE_FEATURE_NAMES = tuple(
    name for block in FEATURE_BLOCKS for name in block.feature_names
)
ROUND1_BLOCK_IDS = (
    "AF01_ENERGY_DYNAMICS",
    "AF02_PITCH_VOICING",
    "AF03_NORMALIZED_SPECTRAL_FLUX",
)
ROUND2_BLOCK_IDS = ("AF04_CPP", "AF05_DELTA_MFCC_DYNAMICS")
ROUND2_REFERENCE_ID = "AF02_47D_REFERENCE"
ROUND2_REFERENCE_FEATURE_NAMES = (
    *E05_FEATURE_NAMES,
    *BLOCK_BY_ID["AF02_PITCH_VOICING"].feature_names,
)
FROZEN_49D_REFERENCE_ID = "AF02_AF04_49D_REFERENCE"
FROZEN_49D_FEATURE_NAMES = (
    *ROUND2_REFERENCE_FEATURE_NAMES,
    *BLOCK_BY_ID["AF04_CPP"].feature_names,
)
NEXT_CANDIDATE_BLOCK_IDS = (
    "AF07_MFCC_CROSS_CORRELATION",
    "AF08_COMPETING_PITCH",
    "AF09_MULTIBAND_ENVELOPE_COHERENCE",
)
RESIDUAL_TARGET_BLOCK_IDS = (
    "AF10_SPEECH_ACTIVITY_BOUNDARY",
    "AF11_SPECTRAL_ENTROPY_DYNAMICS",
    "AF12_TEMPORAL_MFCC_CONTRAST",
)
RESIDUAL_COMBINATION_SPECS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "AF10_AF11_COMBINATION": (
            "AF10_SPEECH_ACTIVITY_BOUNDARY",
            "AF11_SPECTRAL_ENTROPY_DYNAMICS",
        ),
        "AF10_AF12_COMBINATION": (
            "AF10_SPEECH_ACTIVITY_BOUNDARY",
            "AF12_TEMPORAL_MFCC_CONTRAST",
        ),
        "AF11_AF12_COMBINATION": (
            "AF11_SPECTRAL_ENTROPY_DYNAMICS",
            "AF12_TEMPORAL_MFCC_CONTRAST",
        ),
        "AF10_AF11_AF12_COMBINATION": RESIDUAL_TARGET_BLOCK_IDS,
    }
)
CANONICAL_52D_REFERENCE_ID = "AF09_52D_M04_ITER100_REFERENCE"
CANONICAL_52D_PREPARED_ID = "AF09_52D_REFERENCE"
CANONICAL_52D_FEATURE_NAMES = (
    *FROZEN_49D_FEATURE_NAMES,
    *BLOCK_BY_ID["AF09_MULTIBAND_ENVELOPE_COHERENCE"].feature_names,
)
CANONICAL_52D_ACCURACY = 0.869532466510391
CANONICAL_52D_CONFUSION_MATRIX = ((71_066, 14_285), (8_924, 83_616))
HISTORICAL_NON_REPRODUCED_52D_ACCURACY = 0.8689703245245685
HISTORICAL_NON_REPRODUCED_52D_CONFUSION_MATRIX = (
    (70_926, 14_425),
    (8_884, 83_656),
)


@dataclass(frozen=True)
class PreparedRepresentation:
    """One representation using the common split and Training-only scaler policy."""

    representation_id: str
    feature_names: tuple[str, ...]
    X_train_scaled: np.ndarray
    X_evaluation_scaled: np.ndarray
    y_train: np.ndarray
    y_evaluation: np.ndarray
    training_record_keys: np.ndarray
    evaluation_record_keys: np.ndarray
    train_idx: np.ndarray
    validation_idx: np.ndarray
    scaler: StandardScaler
    split_audit: Mapping[str, Any]


@dataclass(frozen=True)
class Canonical52DResolution:
    """Canonical result plus whether the existing object was reused or refit."""

    result: ModelExperimentResult
    action: str
    stale_mismatches: tuple[str, ...]
    prepared_audit: Mapping[str, Any]


@dataclass(frozen=True)
class _CallExtractionTask:
    """Serialization-safe payload for one WAV and all of its utterances."""

    data_root: str
    call_id: str
    audio_path: str
    rows: tuple[tuple[int, str, str, int, int], ...]
    block_ids: tuple[str, ...] = tuple(block.block_id for block in FEATURE_BLOCKS)


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_acoustic_array_fingerprint(values: np.ndarray) -> str:
    """Preserve the pre-v1 hash used by frozen acoustic Notebook artifacts."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_file_provenance(path: str | Path) -> dict[str, str]:
    source_path = Path(path).resolve()
    return {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def _runtime_provenance(model: MLPClassifier) -> dict[str, Any]:
    """Capture small reproducibility metadata without duplicating predictions."""

    return {
        "environment_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_modules": {
            "mission2_acoustic_feature_expansion": _source_file_provenance(__file__),
            "mission2_model_experiments": _source_file_provenance(
                _model_experiments_module.__file__
            ),
        },
        "model_params": dict(model.get_params(deep=False)),
    }


def candidate_manifest_table() -> pd.DataFrame:
    """Return the immutable, result-blind candidate precommit table."""

    return pd.DataFrame([vars(block) for block in FEATURE_BLOCKS])


def candidate_hypothesis_table() -> pd.DataFrame:
    """Return the evidence-led pool, including reviewed exclusions.

    Rows describe hypotheses rather than observed effects.  ``IMPLEMENT`` means
    only that a block passed the pre-extraction design screen; it is not a
    performance or promotion decision.
    """

    rows = [
        {
            "candidate_id": "AF07_MFCC_CROSS_CORRELATION",
            "status": "IMPLEMENT",
            "residual_target": "overlap persistent errors; high-confidence errors",
            "missing_information_axis": "MFCC cross-coefficient dependence",
            "difference_from_49d": "49D keeps marginal MFCC mean/std, not covariance",
            "supporting_evidence": "overlap remains weak after AF02+AF04",
            "counter_evidence_uncertainty": "noise can mimic changed covariance",
            "short_8khz_stability": (
                "fixed 8 kHz frontend; fallback unless active_frame_count >= p_eff + 1"
            ),
            "redundancy_risk": "moderate with MFCC moments, but cross terms are absent",
            "cost": "medium",
            "testable_expectation": "incremental overlap gain without no-overlap collapse",
        },
        {
            "candidate_id": "AF08_COMPETING_PITCH",
            "status": "IMPLEMENT",
            "residual_target": "overlap persistent and high-confidence errors",
            "missing_information_axis": "periodicity peak ambiguity",
            "difference_from_49d": "AF02 stores the winning lag, not competitor strength",
            "supporting_evidence": "mixed periodic sources are plausible in overlap",
            "counter_evidence_uncertainty": "noise and octave errors also narrow margins",
            "short_8khz_stability": "65-400 Hz lag range; 3-frame fallback",
            "redundancy_risk": "moderate with AF02/CPP; ambiguity itself is new",
            "cost": "high",
            "testable_expectation": "paired corrections concentrate in voiced overlap rows",
        },
        {
            "candidate_id": "AF09_MULTIBAND_ENVELOPE_COHERENCE",
            "status": "IMPLEMENT",
            "residual_target": "overlap and noisy high-confidence errors",
            "missing_information_axis": "coarse-band temporal coherence",
            "difference_from_49d": "49D has no cross-band envelope correlation",
            "supporting_evidence": "overlap remains weak while static spectrum is present",
            "counter_evidence_uncertainty": "stationary noise and channel filtering can dominate",
            "short_8khz_stability": "four fixed Nyquist-safe bands; 5-frame fallback",
            "redundancy_risk": "low-to-moderate with AF07; coarse energy envelopes differ",
            "cost": "medium",
            "testable_expectation": "overlap slice improves without broad class-recall loss",
        },
        {
            "candidate_id": "AF05_DELTA_MFCC_DYNAMICS",
            "status": "DROP",
            "residual_target": "0.5-1.0 second errors",
            "missing_information_axis": "MFCC derivative magnitude",
            "difference_from_49d": "frame-order derivative summary",
            "supporting_evidence": "original short-utterance hypothesis",
            "counter_evidence_uncertainty": "fixed-duration result did not support the hypothesis",
            "short_8khz_stability": "implemented fallback but empirically unsupported",
            "redundancy_risk": "high with MFCC moments",
            "cost": "medium",
            "testable_expectation": "not reopened without new evidence",
        },
        {
            "candidate_id": "AF06_FORMANT_CENTERS",
            "status": "RETIRED",
            "residual_target": "spectral-envelope role differences",
            "missing_information_axis": "formant locations",
            "difference_from_49d": "explicit resonance centers",
            "supporting_evidence": "possible vocal-tract information",
            "counter_evidence_uncertainty": "telephone/noise tracking failure and content dependence",
            "short_8khz_stability": "insufficient",
            "redundancy_risk": "high with MFCC envelope",
            "cost": "medium",
            "testable_expectation": "retired",
        },
        {
            "candidate_id": "HNR_HARMONICITY",
            "status": "EXCLUDE",
            "residual_target": "overlap/noise",
            "missing_information_axis": "harmonic-to-noise balance",
            "difference_from_49d": "explicit harmonic ratio",
            "supporting_evidence": "voice-quality relevance",
            "counter_evidence_uncertainty": "AF02 periodicity and AF04 CPP already cover it",
            "short_8khz_stability": "noise-sensitive",
            "redundancy_risk": "high",
            "cost": "medium",
            "testable_expectation": "excluded as redundant",
        },
        {
            "candidate_id": "MFCC_DIRECTIONAL_DELTA",
            "status": "EXCLUDE",
            "residual_target": "short utterances",
            "missing_information_axis": "trajectory direction/curvature",
            "difference_from_49d": "direction rather than derivative magnitude",
            "supporting_evidence": "MFCC order is otherwise discarded",
            "counter_evidence_uncertainty": "too close to failed AF05 and unstable when short",
            "short_8khz_stability": "weak",
            "redundancy_risk": "high with AF05 family",
            "cost": "medium",
            "testable_expectation": "excluded pending contrary evidence",
        },
        {
            "candidate_id": "SPECTRAL_LOW_RANK",
            "status": "EXCLUDE",
            "residual_target": "overlap",
            "missing_information_axis": "spectrogram component rank",
            "difference_from_49d": "joint time-frequency subspace",
            "supporting_evidence": "mixtures can increase component rank",
            "counter_evidence_uncertainty": "substantially duplicates AF07 covariance structure",
            "short_8khz_stability": "rank is sample-size-sensitive",
            "redundancy_risk": "high with AF07",
            "cost": "medium",
            "testable_expectation": "excluded to avoid repackaging",
        },
        {
            "candidate_id": "WAVEFORM_CORRUPTION_RATES",
            "status": "HOLD",
            "residual_target": "high-confidence errors",
            "missing_information_axis": "clipping/dropout artifacts",
            "difference_from_49d": "sample-level corruption structure",
            "supporting_evidence": "degradation can create confident mistakes",
            "counter_evidence_uncertainty": "no current residual count links errors to corruption",
            "short_8khz_stability": "stable",
            "redundancy_risk": "low but shortcut risk is high",
            "cost": "low",
            "testable_expectation": "hold until diagnostic evidence exists",
        },
        {
            "candidate_id": "MODULATION_SPECTRUM",
            "status": "EXCLUDE",
            "residual_target": "speaking rhythm",
            "missing_information_axis": "slow envelope modulation",
            "difference_from_49d": "long-range rhythm",
            "supporting_evidence": "potential role-related cadence",
            "counter_evidence_uncertainty": "0.5-1.0 second rows are too short for stable estimates",
            "short_8khz_stability": "insufficient for target slice",
            "redundancy_risk": "moderate with envelope dynamics",
            "cost": "medium",
            "testable_expectation": "excluded for short-window instability",
        },
    ]
    return pd.DataFrame(rows)


def implemented_candidate_definition_table() -> pd.DataFrame:
    """Return exact extraction and fallback contracts for active candidates."""

    return pd.DataFrame(
        [
            {
                "candidate_id": "AF07_MFCC_CROSS_CORRELATION",
                "feature_names": BLOCK_BY_ID[
                    "AF07_MFCC_CROSS_CORRELATION"
                ].feature_names,
                "dimension": 2,
                "frame_length": 512,
                "hop_length": 160,
                "window": "Hann",
                "normalization": (
                    "active frames; MFCC 1..19 coefficient-wise zero mean and "
                    "population-standard-deviation scaling"
                ),
                "minimum_valid_frames": "p_eff + 1",
                "short_fallback": "all-zero 2D block",
                "silence_behavior": "all-zero 2D block",
                "non_finite_behavior": "raise ValueError before extraction/cache",
                "output_dtype": "float32",
            },
            {
                "candidate_id": "AF08_COMPETING_PITCH",
                "feature_names": BLOCK_BY_ID["AF08_COMPETING_PITCH"].feature_names,
                "dimension": 2,
                "frame_length": 1024,
                "hop_length": 160,
                "window": "Hann",
                "normalization": (
                    "lag-wise autocorrelation divided by overlapping left/right "
                    "frame energy; octave-equivalent competitors excluded"
                ),
                "minimum_valid_frames": 3,
                "short_fallback": "all-zero 2D block",
                "silence_behavior": "all-zero 2D block",
                "non_finite_behavior": "raise ValueError before extraction/cache",
                "output_dtype": "float32",
            },
            {
                "candidate_id": "AF09_MULTIBAND_ENVELOPE_COHERENCE",
                "feature_names": BLOCK_BY_ID[
                    "AF09_MULTIBAND_ENVELOPE_COHERENCE"
                ].feature_names,
                "dimension": 3,
                "frame_length": 512,
                "hop_length": 160,
                "window": "Hann",
                "normalization": (
                    "log band power; each varying band zero mean and "
                    "population-standard-deviation scaled"
                ),
                "minimum_valid_frames": 5,
                "short_fallback": "all-zero 3D block",
                "silence_behavior": "all-zero 3D block",
                "non_finite_behavior": "raise ValueError before extraction/cache",
                "output_dtype": "float32",
            },
        ]
    )


def _validate_block_ids(block_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(str(block_id) for block_id in block_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("block_ids must be non-empty and unique")
    unknown = [block_id for block_id in ids if block_id not in BLOCK_BY_ID]
    if unknown:
        raise ValueError(f"unknown acoustic feature block: {unknown}")
    expected_order = tuple(block.block_id for block in FEATURE_BLOCKS if block.block_id in ids)
    if ids != expected_order:
        raise ValueError("candidate block ordering differs from the precommit manifest")
    return ids


def validate_model_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Reject unknown columns so tracking metadata can never become model input."""

    names = tuple(str(name) for name in feature_names)
    allowed = set(E05_FEATURE_NAMES) | set(ALL_CANDIDATE_FEATURE_NAMES)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown or forbidden model features: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError("model feature names must be unique")
    return names


def _validated_signal(segment: np.ndarray, *, minimum_length: int) -> np.ndarray:
    signal = np.asarray(segment, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("waveform segment cannot be empty")
    if not np.isfinite(signal).all():
        raise ValueError("waveform segment contains NaN or inf")
    if len(signal) < minimum_length:
        signal = np.pad(signal, (0, minimum_length - len(signal)))
    return signal.astype(np.float32, copy=False)


def _centered_frames(
    segment: np.ndarray, *, frame_length: int, hop_length: int
) -> np.ndarray:
    """Return zero-centered overlapping frames without a JIT-backed dependency."""

    signal = _validated_signal(segment, minimum_length=frame_length)
    padded = np.pad(signal.astype(np.float64), (frame_length // 2, frame_length // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    if frames.shape[0] == 0:
        raise AssertionError("centered framing unexpectedly produced no frames")
    return frames


def _energy_features(segment: np.ndarray) -> dict[str, float]:
    frames = _centered_frames(
        segment, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    log_rms = 20.0 * np.log10(np.maximum(rms, LOG_RMS_FLOOR))
    return {
        "log_rms_mean_db": float(np.mean(log_rms)),
        "log_rms_std_db": float(np.std(log_rms)),
        "log_rms_p90_minus_p10_db": float(
            np.quantile(log_rms, 0.90) - np.quantile(log_rms, 0.10)
        ),
    }


def _autocorrelation_pitch_track(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return F0 and peak normalized autocorrelation for every analysis frame."""

    config = PITCH_EXTRACTION_CONFIG
    frames = _centered_frames(
        segment,
        frame_length=_PITCH_FRAME_LENGTH,
        hop_length=_PITCH_HOP_LENGTH,
    )
    frames = (frames - np.mean(frames, axis=1, keepdims=True)) * _PITCH_WINDOW
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active_indices = np.flatnonzero(
        frame_rms >= float(config["minimum_frame_rms"])
    )
    f0 = np.zeros(len(frames), dtype=np.float64)
    periodicity = np.zeros(len(frames), dtype=np.float64)

    # FFT zero-padding to 2N produces the same linear, non-negative-lag
    # autocorrelation as np.correlate(frame, frame, mode="full")[N - 1:].
    # Batching bounds peak memory for the longest allowed utterances without
    # changing any frame, lag, threshold, or downstream pitch definition.
    for batch_start in range(0, len(active_indices), _PITCH_FRAME_BATCH_SIZE):
        batch_indices = active_indices[
            batch_start : batch_start + _PITCH_FRAME_BATCH_SIZE
        ]
        batch = frames[batch_indices]
        spectrum = np.fft.rfft(batch, n=_PITCH_FFT_LENGTH, axis=1)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum),
            n=_PITCH_FFT_LENGTH,
            axis=1,
        )[:, _PITCH_LAGS]
        squared = np.square(batch)
        cumulative = np.concatenate(
            (np.zeros((len(batch), 1), dtype=np.float64), np.cumsum(squared, axis=1)),
            axis=1,
        )
        left_energy = cumulative[:, _PITCH_FRAME_LENGTH - _PITCH_LAGS]
        right_energy = (
            cumulative[:, [_PITCH_FRAME_LENGTH]] - cumulative[:, _PITCH_LAGS]
        )
        denominator = np.sqrt(left_energy * right_energy)
        scores = np.divide(
            correlation,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > np.finfo(np.float64).eps,
        )
        best = np.argmax(scores, axis=1)
        periodicity[batch_indices] = np.clip(
            scores[np.arange(len(batch_indices)), best], 0.0, 1.0
        )
        f0[batch_indices] = SAMPLE_RATE / _PITCH_LAGS[best].astype(np.float64)
    return f0, periodicity


def _pitch_features(segment: np.ndarray) -> dict[str, float]:
    config = PITCH_EXTRACTION_CONFIG
    f0, periodicity = _autocorrelation_pitch_track(segment)
    valid = (
        np.isfinite(f0)
        & (f0 > 0.0)
        & np.isfinite(periodicity)
        & (periodicity >= float(config["periodicity_threshold"]))
    )
    voiced_count = int(np.sum(valid))
    f0_statistics = {
        "f0_semitone_median": 0.0,
        "f0_semitone_iqr": 0.0,
        "f0_semitone_slope": 0.0,
    }
    if voiced_count >= int(config["minimum_voiced_frames"]):
        semitones = 12.0 * np.log2(
            f0[valid] / float(config["semitone_reference_hz"])
        )
        voiced_times = (
            np.flatnonzero(valid).astype(np.float64)
            * float(config["hop_length"])
            / SAMPLE_RATE
        )
        slope = float(np.polyfit(voiced_times, semitones, 1)[0])
        f0_statistics = {
            "f0_semitone_median": float(np.median(semitones)),
            "f0_semitone_iqr": float(
                np.quantile(semitones, 0.75) - np.quantile(semitones, 0.25)
            ),
            "f0_semitone_slope": slope,
        }
    result = {
        **f0_statistics,
        "voiced_fraction": float(np.mean(valid)),
        "periodicity_mean": float(np.mean(periodicity)),
    }
    if not np.isfinite(list(result.values())).all():
        raise AssertionError("AF02 pitch/voicing fallback failed the finite contract")
    return result


def normalized_spectral_flux(magnitude_spectrogram: np.ndarray) -> np.ndarray:
    """Compute the frozen L2-normalized Euclidean frame-difference definition."""

    magnitude = np.asarray(magnitude_spectrogram, dtype=np.float64)
    if magnitude.ndim != 2 or magnitude.shape[0] == 0 or magnitude.shape[1] == 0:
        raise ValueError("magnitude_spectrogram must be a non-empty 2D matrix")
    if not np.isfinite(magnitude).all() or np.any(magnitude < 0.0):
        raise ValueError("magnitude_spectrogram must be finite and non-negative")
    norms = np.linalg.norm(magnitude, axis=0, keepdims=True)
    normalized = np.divide(
        magnitude,
        norms,
        out=np.zeros_like(magnitude),
        where=norms > np.finfo(np.float64).eps,
    )
    if normalized.shape[1] < 2:
        return np.asarray([0.0], dtype=np.float64)
    differences = np.diff(normalized, axis=1)
    return np.sqrt(np.sum(np.square(differences), axis=0))


def _spectral_flux_features(segment: np.ndarray) -> dict[str, float]:
    frames = _centered_frames(
        segment, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    windowed = frames * np.hanning(N_FFT)
    magnitude = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)).T
    flux = normalized_spectral_flux(magnitude)
    return {
        "normalized_spectral_flux_mean": float(np.mean(flux)),
        "normalized_spectral_flux_std": float(np.std(flux)),
        "normalized_spectral_flux_p90": float(np.quantile(flux, 0.90)),
    }


def _cpp_features(segment: np.ndarray) -> dict[str, np.float32]:
    """Extract the frozen frame-level cepstral-peak-prominence summary."""

    signal = _validated_signal(segment, minimum_length=N_FFT)
    if float(np.ptp(signal)) <= np.finfo(np.float32).eps:
        return {
            "cpp_median_db": np.float32(0.0),
            "cpp_iqr_db": np.float32(0.0),
        }
    frames = _centered_frames(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    windowed = centered * np.hanning(N_FFT)
    active = np.sqrt(np.mean(np.square(windowed), axis=1)) >= 1e-4
    if int(np.sum(active)) < 3:
        return {
            "cpp_median_db": np.float32(0.0),
            "cpp_iqr_db": np.float32(0.0),
        }

    power = np.square(np.abs(np.fft.rfft(windowed[active], n=N_FFT, axis=1)))
    log_power = np.log(np.maximum(power, 1e-12))
    cepstrum = np.fft.irfft(log_power, n=N_FFT, axis=1)
    cepstrum_db = 20.0 * np.log10(np.maximum(np.abs(cepstrum), 1e-12))

    baseline_q = np.arange(8, 129, dtype=np.float64)
    baseline_values = cepstrum_db[:, 8:129]
    q_centered = baseline_q - np.mean(baseline_q)
    slopes = (baseline_values @ q_centered) / np.sum(np.square(q_centered))
    intercepts = np.mean(baseline_values, axis=1) - slopes * np.mean(baseline_q)

    peak_q = np.arange(20, 124, dtype=np.float64)
    peak_values = cepstrum_db[:, 20:124]
    baselines = intercepts[:, None] + slopes[:, None] * peak_q[None, :]
    prominence = np.max(peak_values - baselines, axis=1)
    result = {
        "cpp_median_db": np.float32(np.median(prominence)),
        "cpp_iqr_db": np.float32(
            np.quantile(prominence, 0.75) - np.quantile(prominence, 0.25)
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF04 CPP failed the finite contract")
    return result


def _canonical_mfcc_frame_matrix(segment: np.ndarray) -> np.ndarray:
    """Return the exact 20-coefficient frontend used by the canonical MFCC feature."""

    signal = _validated_signal(segment, minimum_length=N_FFT)
    return librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=20,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="constant",
        power=2.0,
        n_mels=128,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2.0,
        htk=False,
        mel_norm="slaney",
        dct_type=2,
        norm="ortho",
        lifter=0,
    )


def _mfcc_cross_correlation_features_from_matrix(
    mfcc: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> dict[str, np.float32]:
    """Summarize standardized MFCC 1--19 cross-coefficient dependence."""

    matrix = np.asarray(mfcc, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 20 or matrix.shape[1] == 0:
        raise ValueError("MFCC matrix must have shape (20, non-empty frames)")
    if not np.isfinite(matrix).all():
        raise ValueError("MFCC matrix contains NaN or inf")
    if active_mask is None:
        active = np.ones(matrix.shape[1], dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (matrix.shape[1],):
            raise ValueError("MFCC active-frame mask shape mismatch")
    fallback = {
        "mfcc_cross_corr_abs_mean": np.float32(0.0),
        "mfcc_cross_corr_pc1_share": np.float32(0.0),
    }
    active_frame_count = int(np.sum(active))
    if active_frame_count == 0:
        return fallback
    coefficients = matrix[1:20, active]
    centered = coefficients - np.mean(coefficients, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, ddof=0)
    varying = scale > 1e-6
    p_eff = int(np.sum(varying))
    if p_eff < 2 or active_frame_count < p_eff + 1:
        return fallback
    standardized = centered[varying] / scale[varying, None]
    correlation = standardized @ standardized.T / standardized.shape[1]
    correlation = (correlation + correlation.T) / 2.0
    upper = correlation[np.triu_indices(correlation.shape[0], k=1)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    eigenvalue_sum = float(np.sum(eigenvalues))
    if upper.size == 0 or eigenvalue_sum <= np.finfo(np.float64).eps:
        return fallback
    result = {
        "mfcc_cross_corr_abs_mean": np.float32(np.mean(np.abs(upper))),
        "mfcc_cross_corr_pc1_share": np.float32(
            eigenvalues[-1] / eigenvalue_sum
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF07 MFCC correlation failed the finite contract")
    return result


def _mfcc_cross_correlation_features(
    segment: np.ndarray,
) -> dict[str, np.float32]:
    signal = _validated_signal(segment, minimum_length=N_FFT)
    fallback = {
        "mfcc_cross_corr_abs_mean": np.float32(0.0),
        "mfcc_cross_corr_pc1_share": np.float32(0.0),
    }
    if float(np.ptp(signal)) <= np.finfo(np.float32).eps:
        return fallback
    frames = _centered_frames(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    windowed = frames * np.hanning(N_FFT)
    active = np.sqrt(np.mean(np.square(windowed), axis=1)) >= 1e-4
    mfcc = _canonical_mfcc_frame_matrix(signal)
    if mfcc.shape[1] != len(active):
        raise AssertionError("canonical MFCC and active-frame counts differ")
    return _mfcc_cross_correlation_features_from_matrix(
        mfcc, active_mask=active
    )


def _competing_pitch_margins(
    normalized_autocorrelation_scores: np.ndarray,
    lags: np.ndarray,
) -> np.ndarray:
    """Return primary-minus-competitor margins after octave-equivalence removal."""

    scores = np.asarray(normalized_autocorrelation_scores, dtype=np.float64)
    lag_values = np.asarray(lags, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(lag_values) or len(lag_values) < 2:
        raise ValueError("pitch score matrix and lag vector shapes are incompatible")
    if not np.isfinite(scores).all() or np.any(lag_values <= 0):
        raise ValueError("pitch scores/lags must be finite and positive")
    margins = np.zeros(scores.shape[0], dtype=np.float64)
    for row_index, row in enumerate(scores):
        best_index = int(np.argmax(row))
        primary_lag = float(lag_values[best_index])
        octave_distance = np.abs(
            np.log2(lag_values.astype(np.float64) / primary_lag)
            - np.round(np.log2(lag_values.astype(np.float64) / primary_lag))
        )
        eligible = octave_distance > 0.08
        if not np.any(eligible):
            continue
        primary = float(np.clip(row[best_index], 0.0, 1.0))
        competitor = float(np.clip(np.max(row[eligible]), 0.0, primary))
        margins[row_index] = primary - competitor
    return margins


def _competing_pitch_features(segment: np.ndarray) -> dict[str, np.float32]:
    """Measure AF02-compatible pitch ambiguity without changing frozen AF02."""

    fallback = {
        "pitch_peak_margin_median": np.float32(0.0),
        "pitch_peak_margin_iqr": np.float32(0.0),
    }
    signal = _validated_signal(segment, minimum_length=_PITCH_FRAME_LENGTH)
    if float(np.ptp(signal)) <= np.finfo(np.float32).eps:
        return fallback
    frames = _centered_frames(
        signal,
        frame_length=_PITCH_FRAME_LENGTH,
        hop_length=_PITCH_HOP_LENGTH,
    )
    frames = (frames - np.mean(frames, axis=1, keepdims=True)) * _PITCH_WINDOW
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active_indices = np.flatnonzero(
        frame_rms >= float(PITCH_EXTRACTION_CONFIG["minimum_frame_rms"])
    )
    score_rows: list[np.ndarray] = []
    for batch_start in range(0, len(active_indices), _PITCH_FRAME_BATCH_SIZE):
        batch_indices = active_indices[
            batch_start : batch_start + _PITCH_FRAME_BATCH_SIZE
        ]
        batch = frames[batch_indices]
        spectrum = np.fft.rfft(batch, n=_PITCH_FFT_LENGTH, axis=1)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum),
            n=_PITCH_FFT_LENGTH,
            axis=1,
        )[:, _PITCH_LAGS]
        squared = np.square(batch)
        cumulative = np.concatenate(
            (np.zeros((len(batch), 1)), np.cumsum(squared, axis=1)), axis=1
        )
        denominator = np.sqrt(
            cumulative[:, _PITCH_FRAME_LENGTH - _PITCH_LAGS]
            * (cumulative[:, [_PITCH_FRAME_LENGTH]] - cumulative[:, _PITCH_LAGS])
        )
        score_rows.append(
            np.divide(
                correlation,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > np.finfo(np.float64).eps,
            )
        )
    if not score_rows:
        return fallback
    scores = np.concatenate(score_rows, axis=0)
    primary = np.max(scores, axis=1)
    valid = primary >= float(PITCH_EXTRACTION_CONFIG["periodicity_threshold"])
    if int(np.sum(valid)) < 3:
        return fallback
    margins = _competing_pitch_margins(scores[valid], _PITCH_LAGS)
    result = {
        "pitch_peak_margin_median": np.float32(np.median(margins)),
        "pitch_peak_margin_iqr": np.float32(
            np.quantile(margins, 0.75) - np.quantile(margins, 0.25)
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF08 competing pitch failed the finite contract")
    return result


def _subband_coherence_features_from_envelopes(
    log_band_envelopes: np.ndarray,
) -> dict[str, np.float32]:
    """Summarize standardized temporal coherence among fixed subbands."""

    envelopes = np.asarray(log_band_envelopes, dtype=np.float64)
    if envelopes.ndim != 2 or envelopes.shape[0] != 4 or envelopes.shape[1] == 0:
        raise ValueError("subband envelopes must have shape (4, non-empty frames)")
    if not np.isfinite(envelopes).all():
        raise ValueError("subband envelopes contain NaN or inf")
    fallback = {
        "subband_envelope_corr_abs_mean": np.float32(0.0),
        "subband_envelope_corr_min": np.float32(0.0),
        "subband_envelope_pc1_share": np.float32(0.0),
    }
    if envelopes.shape[1] < 5:
        return fallback
    centered = envelopes - np.mean(envelopes, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, ddof=0)
    varying = scale > 1e-6
    if int(np.sum(varying)) < 2:
        return fallback
    standardized = centered[varying] / scale[varying, None]
    correlation = standardized @ standardized.T / standardized.shape[1]
    correlation = (correlation + correlation.T) / 2.0
    upper = correlation[np.triu_indices(correlation.shape[0], k=1)]
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    eigenvalue_sum = float(np.sum(eigenvalues))
    if upper.size == 0 or eigenvalue_sum <= np.finfo(np.float64).eps:
        return fallback
    result = {
        "subband_envelope_corr_abs_mean": np.float32(np.mean(np.abs(upper))),
        "subband_envelope_corr_min": np.float32(np.min(upper)),
        "subband_envelope_pc1_share": np.float32(
            eigenvalues[-1] / eigenvalue_sum
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF09 subband coherence failed the finite contract")
    return result


def _multiband_envelope_coherence_features(
    segment: np.ndarray,
) -> dict[str, np.float32]:
    signal = _validated_signal(segment, minimum_length=N_FFT)
    fallback = {
        "subband_envelope_corr_abs_mean": np.float32(0.0),
        "subband_envelope_corr_min": np.float32(0.0),
        "subband_envelope_pc1_share": np.float32(0.0),
    }
    if float(np.ptp(signal)) <= np.finfo(np.float32).eps:
        return fallback
    frames = _centered_frames(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    windowed = frames * np.hanning(N_FFT)
    active = np.sqrt(np.mean(np.square(windowed), axis=1)) >= 1e-4
    if int(np.sum(active)) < 5:
        return fallback
    power = np.square(np.abs(np.fft.rfft(windowed[active], n=N_FFT, axis=1)))
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    band_edges = ((0.0, 500.0), (500.0, 1_000.0), (1_000.0, 2_000.0), (2_000.0, 4_000.0))
    band_envelopes = []
    for band_index, (lower, upper) in enumerate(band_edges):
        if band_index == len(band_edges) - 1:
            selected = (frequencies >= lower) & (frequencies <= upper)
        else:
            selected = (frequencies >= lower) & (frequencies < upper)
        band_power = np.sum(power[:, selected], axis=1)
        band_envelopes.append(np.log(np.maximum(band_power, 1e-12)))
    return _subband_coherence_features_from_envelopes(
        np.vstack(band_envelopes)
    )


def _delta_mfcc_matrix(mfcc: np.ndarray) -> np.ndarray:
    """Apply the frozen width-five delta equation to MFCC 1..19."""

    matrix = np.asarray(mfcc, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 20 or matrix.shape[1] == 0:
        raise ValueError("MFCC matrix must have shape (20, non-empty frames)")
    if not np.isfinite(matrix).all():
        raise ValueError("MFCC matrix contains NaN or inf")
    coefficients = matrix[1:20]
    padded = np.pad(coefficients, ((0, 0), (2, 2)), mode="edge")
    frame_count = coefficients.shape[1]
    return (
        padded[:, 3 : 3 + frame_count]
        - padded[:, 1 : 1 + frame_count]
        + 2.0
        * (
            padded[:, 4 : 4 + frame_count]
            - padded[:, 0:frame_count]
        )
    ) / 10.0


def _delta_mfcc_features_from_matrix(mfcc: np.ndarray) -> dict[str, np.float32]:
    """Summarize per-frame MFCC delta RMS, with the frozen short fallback."""

    matrix = np.asarray(mfcc)
    if matrix.ndim != 2 or matrix.shape[0] != 20 or matrix.shape[1] == 0:
        raise ValueError("MFCC matrix must have shape (20, non-empty frames)")
    if not np.isfinite(matrix).all():
        raise ValueError("MFCC matrix contains NaN or inf")
    if matrix.shape[1] < 3:
        return {
            "delta_mfcc_norm_mean": np.float32(0.0),
            "delta_mfcc_norm_std": np.float32(0.0),
        }
    delta = _delta_mfcc_matrix(matrix)
    frame_norm = np.sqrt(np.mean(np.square(delta), axis=0))
    result = {
        "delta_mfcc_norm_mean": np.float32(np.mean(frame_norm)),
        "delta_mfcc_norm_std": np.float32(np.std(frame_norm, ddof=0)),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF05 delta-MFCC failed the finite contract")
    return result


def _delta_mfcc_features(segment: np.ndarray) -> dict[str, np.float32]:
    return _delta_mfcc_features_from_matrix(_canonical_mfcc_frame_matrix(segment))


def _speech_activity_boundary_features_from_mask(
    active_mask: np.ndarray,
) -> dict[str, np.float32]:
    """Summarize a deterministic frame-activity mask without call context."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 1 or active.size == 0:
        raise ValueError("activity mask must be a non-empty one-dimensional array")
    padded = np.pad(active.astype(np.int8), (1, 1))
    change_points = np.flatnonzero(np.diff(padded) != 0)
    run_lengths = change_points[1::2] - change_points[::2]
    longest_run = int(np.max(run_lengths)) if run_lengths.size else 0
    edge_frame_count = max(1, int(np.ceil(0.2 * active.size)))
    result = {
        "active_frame_fraction": np.float32(np.mean(active)),
        "longest_active_run_fraction": np.float32(longest_run / active.size),
        "activity_transition_rate": np.float32(
            np.mean(active[1:] != active[:-1]) if active.size > 1 else 0.0
        ),
        "edge_activity_difference": np.float32(
            np.mean(active[-edge_frame_count:]) - np.mean(active[:edge_frame_count])
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF10 activity-boundary features failed the finite contract")
    return result


def _speech_activity_boundary_features(
    segment: np.ndarray,
) -> dict[str, np.float32]:
    signal = _validated_signal(segment, minimum_length=N_FFT)
    frames = _centered_frames(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    windowed = frames * np.hanning(N_FFT)
    frame_rms = np.sqrt(np.mean(np.square(windowed), axis=1))
    threshold = max(1e-4, 0.1 * float(np.max(frame_rms)))
    return _speech_activity_boundary_features_from_mask(frame_rms >= threshold)


def _spectral_entropy_features_from_power(
    frame_power: np.ndarray,
) -> dict[str, np.float32]:
    """Summarize normalized frame entropy; a zero-power frame contributes zero."""

    power = np.asarray(frame_power, dtype=np.float64)
    if power.ndim != 2 or power.shape[0] == 0 or power.shape[1] < 2:
        raise ValueError("power spectrum must have shape (non-empty frames, >=2 bins)")
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("power spectrum must be finite and non-negative")
    totals = np.sum(power, axis=1, keepdims=True)
    probabilities = np.divide(
        power,
        totals,
        out=np.zeros_like(power),
        where=totals > np.finfo(np.float64).eps,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0.0,
    )
    entropy = -np.sum(probabilities * log_probabilities, axis=1) / np.log(
        power.shape[1]
    )
    result = {
        "spectral_entropy_mean": np.float32(np.mean(entropy)),
        "spectral_entropy_std": np.float32(np.std(entropy, ddof=0)),
        "spectral_entropy_iqr": np.float32(
            np.quantile(entropy, 0.75) - np.quantile(entropy, 0.25)
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF11 spectral-entropy features failed the finite contract")
    return result


def _spectral_entropy_dynamics_features(
    segment: np.ndarray,
) -> dict[str, np.float32]:
    signal = _validated_signal(segment, minimum_length=N_FFT)
    frames = _centered_frames(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)
    windowed = frames * np.hanning(N_FFT)
    power = np.square(np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)))
    return _spectral_entropy_features_from_power(power)


def _temporal_mfcc_contrast_features_from_matrix(
    mfcc: np.ndarray,
) -> dict[str, np.float32]:
    """Compare canonical MFCC means across three contiguous utterance regions."""

    matrix = np.asarray(mfcc, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 20 or matrix.shape[1] == 0:
        raise ValueError("MFCC matrix must have shape (20, non-empty frames)")
    if not np.isfinite(matrix).all():
        raise ValueError("MFCC matrix contains NaN or inf")
    fallback = {
        "early_middle_distance": np.float32(0.0),
        "middle_late_distance": np.float32(0.0),
        "early_late_distance": np.float32(0.0),
    }
    if matrix.shape[1] < 3:
        return fallback
    early, middle, late = np.array_split(matrix, 3, axis=1)
    early_mean = np.mean(early, axis=1)
    middle_mean = np.mean(middle, axis=1)
    late_mean = np.mean(late, axis=1)
    result = {
        "early_middle_distance": np.float32(
            np.linalg.norm(early_mean - middle_mean)
        ),
        "middle_late_distance": np.float32(
            np.linalg.norm(middle_mean - late_mean)
        ),
        "early_late_distance": np.float32(
            np.linalg.norm(early_mean - late_mean)
        ),
    }
    if not np.isfinite(np.asarray(tuple(result.values()))).all():
        raise AssertionError("AF12 temporal-MFCC features failed the finite contract")
    return result


def _temporal_mfcc_contrast_features(
    segment: np.ndarray,
) -> dict[str, np.float32]:
    signal = _validated_signal(segment, minimum_length=1)
    if signal.size < 3 * HOP_LENGTH:
        return {
            "early_middle_distance": np.float32(0.0),
            "middle_late_distance": np.float32(0.0),
            "early_late_distance": np.float32(0.0),
        }
    return _temporal_mfcc_contrast_features_from_matrix(
        _canonical_mfcc_frame_matrix(signal)
    )


def extract_acoustic_blocks(
    segment: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> dict[str, float]:
    """Extract only precommitted blocks from one current-utterance waveform."""

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, received {sample_rate}")
    ids = _validate_block_ids(block_ids)
    extractors: Mapping[str, Callable[[np.ndarray], dict[str, float]]] = {
        "AF01_ENERGY_DYNAMICS": _energy_features,
        "AF02_PITCH_VOICING": _pitch_features,
        "AF03_NORMALIZED_SPECTRAL_FLUX": _spectral_flux_features,
        "AF04_CPP": _cpp_features,
        "AF05_DELTA_MFCC_DYNAMICS": _delta_mfcc_features,
        "AF07_MFCC_CROSS_CORRELATION": _mfcc_cross_correlation_features,
        "AF08_COMPETING_PITCH": _competing_pitch_features,
        "AF09_MULTIBAND_ENVELOPE_COHERENCE": (
            _multiband_envelope_coherence_features
        ),
        "AF10_SPEECH_ACTIVITY_BOUNDARY": _speech_activity_boundary_features,
        "AF11_SPECTRAL_ENTROPY_DYNAMICS": _spectral_entropy_dynamics_features,
        "AF12_TEMPORAL_MFCC_CONTRAST": _temporal_mfcc_contrast_features,
    }
    result: dict[str, float] = {}
    for block_id in ids:
        values = extractors[block_id](segment)
        if tuple(values) != BLOCK_BY_ID[block_id].feature_names:
            raise AssertionError(f"{block_id} returned a different feature order")
        result.update(values)
    expected_names = tuple(
        name for block_id in ids for name in BLOCK_BY_ID[block_id].feature_names
    )
    if tuple(result) != expected_names:
        raise AssertionError("candidate extraction changed precommit feature ordering")
    if not np.isfinite(np.asarray(list(result.values()), dtype=np.float32)).all():
        raise AssertionError("candidate extraction returned NaN or inf")
    return result


def benchmark_pitch_extraction(
    segments: Sequence[np.ndarray], *, sample_rate: int = SAMPLE_RATE
) -> dict[str, Any]:
    """Time a deliberately small AF02 sample before any full extraction."""

    if not segments:
        raise ValueError("benchmark requires at least one waveform segment")
    started = perf_counter()
    for segment in segments:
        extract_acoustic_blocks(
            segment,
            sample_rate=sample_rate,
            block_ids=("AF02_PITCH_VOICING",),
        )
    elapsed = perf_counter() - started
    audio_seconds = sum(len(np.asarray(segment).reshape(-1)) for segment in segments) / sample_rate
    return {
        "utterance_count": len(segments),
        "audio_seconds": float(audio_seconds),
        "runtime_seconds": float(elapsed),
        "estimated_seconds_per_utterance": float(elapsed / len(segments)),
        "pitch_config_fingerprint": config_fingerprint(PITCH_EXTRACTION_CONFIG),
        "scope": "small_runtime_benchmark_only_not_full_extraction",
    }


def _expected_cache_metadata(
    manifest: pd.DataFrame,
    *,
    split: str,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> dict[str, Any]:
    ids = _validate_block_ids(block_ids)
    feature_names = tuple(
        name for block_id in ids for name in BLOCK_BY_ID[block_id].feature_names
    )
    required = {"split", "record_key", "call_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"candidate manifest is missing columns: {missing}")
    selected = manifest.loc[manifest["split"].astype(str) == split]
    if selected.empty or not selected["record_key"].astype(str).is_unique:
        raise ValueError(f"{split} manifest record_key must be non-empty and unique")
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_split": split,
        "row_count": int(len(selected)),
        "record_key_signature": record_key_signature(selected["record_key"].astype(str)),
        "block_ids": list(ids),
        "feature_names": list(feature_names),
        "feature_dimension": len(feature_names),
        "feature_manifest_fingerprint": _json_fingerprint(
            candidate_manifest_table()
            .set_index("block_id")
            .loc[list(ids)]
            .reset_index()
            .to_dict(orient="records")
        ),
        "pitch_config_fingerprint": config_fingerprint(PITCH_EXTRACTION_CONFIG),
        "waveform_only": True,
        "dtype": "float32",
        "automatic_extraction": False,
    }


def candidate_cache_paths(
    cache_root: str | Path,
    *,
    split: str,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> tuple[Path, Path]:
    """Return stable cache paths derived from the precommit manifest."""

    ids = _validate_block_ids(block_ids)
    fingerprint = _json_fingerprint(
        {"blocks": list(ids), "version": CACHE_SCHEMA_VERSION}
    )[:12]
    prefix = Path(cache_root) / f"{split.lower()}_acoustic_expansion_{fingerprint}"
    return prefix.with_suffix(".parquet"), prefix.with_suffix(".json")


def _resolve_worker_count(worker_count: int | None) -> int:
    """Use the benchmarked default unless an internal caller requests a probe."""

    if worker_count is not None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise TypeError("worker_count must be an integer or None")
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        return worker_count
    return _DEFAULT_WORKER_COUNT


def _build_call_extraction_tasks(
    source: pd.DataFrame,
    *,
    data_root: str | Path,
    block_ids: Sequence[str],
) -> tuple[_CallExtractionTask, ...]:
    ids = _validate_block_ids(block_ids)
    tasks = []
    grouped = source.groupby(["call_id", "audio_path"], sort=False)
    for (call_id, audio_path), group in grouped:
        task_rows = tuple(
            (
                int(row.source_position),
                str(row.record_key),
                str(row.call_id),
                int(row.crop_start_frame),
                int(row.crop_end_frame),
            )
            for row in group.itertuples(index=False)
        )
        tasks.append(
            _CallExtractionTask(
                data_root=str(Path(data_root)),
                call_id=str(call_id),
                audio_path=str(audio_path),
                rows=task_rows,
                block_ids=ids,
            )
        )
    return tuple(tasks)


def _initialize_worker_thread_limits() -> None:
    """Limit native numeric pools once when a process worker starts."""

    global _WORKER_THREADPOOL_GUARD
    if _threadpool_limits is not None:
        _WORKER_THREADPOOL_GUARD = _threadpool_limits(limits=1)


def _extract_call_rows(task: _CallExtractionTask) -> list[dict[str, Any]]:
    """Extract one call in an importable worker and name the call on failure."""

    try:
        audio = read_wav_frames(Path(task.data_root) / task.audio_path)
        if audio.waveform.shape[0] != 1:
            raise ValueError(
                f"expected mono WAV, got {audio.waveform.shape[0]} channels"
            )
        if audio.sample_rate != SAMPLE_RATE:
            raise ValueError(f"unexpected sample rate: {audio.sample_rate}")
        waveform = audio.waveform[0]
        result = []
        for source_position, record_key, call_id, start, end in task.rows:
            if start < 0 or end <= start or end > len(waveform):
                raise ValueError(f"invalid crop bounds for {record_key}")
            features = extract_acoustic_blocks(
                waveform[start:end],
                sample_rate=audio.sample_rate,
                block_ids=task.block_ids,
            )
            result.append(
                {
                    "source_position": source_position,
                    "record_key": record_key,
                    "call_id": call_id,
                    **features,
                }
            )
        return result
    except Exception as exc:
        raise RuntimeError(
            f"call-level feature extraction failed for call_id={task.call_id}: {exc}"
        ) from exc


def _extract_tasks_parallel(
    tasks: Sequence[_CallExtractionTask],
    *,
    worker_count: int,
    on_call_done: Callable[[int, int, str], None] | None,
) -> list[dict[str, Any]]:
    """Run a bounded number of call futures and propagate the first failure."""

    executor_options: dict[str, Any] = {
        "max_workers": worker_count,
        "initializer": _initialize_worker_thread_limits,
    }
    if os.name == "nt":
        executor_options["mp_context"] = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(**executor_options)
    task_iterator = iter(tasks)
    in_flight: dict[Any, _CallExtractionTask] = {}
    rows: list[dict[str, Any]] = []
    completed_count = 0

    def submit_next() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        in_flight[executor.submit(_extract_call_rows, task)] = task
        return True

    try:
        for _ in range(min(len(tasks), worker_count * 2)):
            submit_next()
        while in_flight:
            completed, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in completed:
                task = in_flight.pop(future)
                rows.extend(future.result())
                completed_count += 1
                if on_call_done is not None:
                    on_call_done(completed_count, len(tasks), task.call_id)
                submit_next()
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return rows


def extract_candidate_feature_frame(
    manifest: pd.DataFrame,
    *,
    data_root: str | Path,
    split: str,
    on_call_done: Callable[[int, int, str], None] | None = None,
    worker_count: int | None = None,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> pd.DataFrame:
    """Read each call once and extract all frozen blocks for its utterances."""

    required = {
        "split",
        "record_key",
        "call_id",
        "audio_path",
        "crop_start_frame",
        "crop_end_frame",
        "crop_ok",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"candidate extraction manifest is missing columns: {missing}")
    source = manifest.loc[manifest["split"].astype(str) == split].copy()
    if source.empty or not source["record_key"].astype(str).is_unique:
        raise ValueError(f"{split} extraction rows must have unique record_key values")
    if not source["crop_ok"].astype(bool).all():
        raise ValueError(f"{split} contains invalid crops; rows cannot be dropped silently")
    source["source_position"] = np.arange(len(source), dtype=np.int64)
    ids = _validate_block_ids(block_ids)
    feature_names = tuple(
        name for block_id in ids for name in BLOCK_BY_ID[block_id].feature_names
    )
    tasks = _build_call_extraction_tasks(
        source, data_root=data_root, block_ids=ids
    )
    resolved_worker_count = _resolve_worker_count(worker_count)
    if resolved_worker_count == 1 or len(tasks) == 1:
        rows = []
        # A serial run shares the Notebook process, so restore its numeric
        # thread settings before later sklearn/numpy work in that session.
        limit = _threadpool_limits(limits=1) if _threadpool_limits else nullcontext()
        with limit:
            for call_number, task in enumerate(tasks, start=1):
                rows.extend(_extract_call_rows(task))
                if on_call_done is not None:
                    on_call_done(call_number, len(tasks), task.call_id)
    else:
        rows = _extract_tasks_parallel(
            tasks,
            worker_count=min(resolved_worker_count, len(tasks)),
            on_call_done=on_call_done,
        )
    frame = pd.DataFrame(rows).sort_values("source_position", kind="stable")
    frame = frame.drop(columns="source_position").reset_index(drop=True)
    feature_values = frame.loc[:, feature_names].to_numpy(dtype=np.float32)
    for column_index, feature_name in enumerate(feature_names):
        frame[feature_name] = feature_values[:, column_index]
    if len(frame) != len(source) or not frame["record_key"].is_unique:
        raise AssertionError("candidate extraction changed row count or record keys")
    if set(frame["record_key"]) != set(source["record_key"].astype(str)):
        raise AssertionError("candidate extraction changed the canonical record_key set")
    if frame["record_key"].tolist() != source["record_key"].astype(str).tolist():
        raise AssertionError("candidate extraction changed canonical source_position order")
    if frame["call_id"].tolist() != source["call_id"].astype(str).tolist():
        raise AssertionError("candidate extraction changed canonical call_id alignment")
    return frame


def build_candidate_feature_cache(
    *,
    manifest: pd.DataFrame,
    data_root: str | Path,
    cache_root: str | Path,
    split: str,
    allow_full_extraction: bool = False,
    on_call_done: Callable[[int, int, str], None] | None = None,
    worker_count: int | None = None,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> tuple[Path, Path]:
    """Build one immutable all-block cache only after explicit user opt-in."""

    if not allow_full_extraction:
        raise RuntimeError(
            "STOP: full acoustic extraction requires allow_full_extraction=True"
        )
    ids = _validate_block_ids(block_ids)
    metadata = _expected_cache_metadata(manifest, split=split, block_ids=ids)
    parquet_path, metadata_path = candidate_cache_paths(
        cache_root, split=split, block_ids=ids
    )
    if parquet_path.exists() or metadata_path.exists():
        raise FileExistsError("candidate cache already exists; validate and reuse it")
    frame = extract_candidate_feature_frame(
        manifest,
        data_root=data_root,
        split=split,
        on_call_done=on_call_done,
        worker_count=worker_count,
        block_ids=ids,
    )
    if len(frame) != metadata["row_count"]:
        raise AssertionError("candidate cache row count differs from frozen metadata")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary_parquet, index=False)
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_parquet.replace(parquet_path)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_parquet.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return parquet_path, metadata_path


def load_candidate_feature_cache(
    *,
    manifest: pd.DataFrame,
    cache_root: str | Path,
    split: str,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> pd.DataFrame:
    """Strict-load the exact precommit artifact; never extract on cache miss."""

    ids = _validate_block_ids(block_ids)
    feature_names = tuple(
        name for block_id in ids for name in BLOCK_BY_ID[block_id].feature_names
    )
    parquet_path, metadata_path = candidate_cache_paths(
        cache_root, split=split, block_ids=ids
    )
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("candidate cache is missing; automatic extraction is disabled")
    expected = _expected_cache_metadata(manifest, split=split, block_ids=ids)
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError("candidate cache metadata differs from the precommit contract")
    frame = pd.read_parquet(parquet_path)
    required = {"record_key", "call_id", *feature_names}
    if not required <= set(frame.columns) or len(frame) != expected["row_count"]:
        raise ValueError("candidate cache columns or row count are invalid")
    if not frame["record_key"].astype(str).is_unique:
        raise ValueError("candidate cache record_key must be one-to-one")
    source = manifest.loc[manifest["split"].astype(str) == split]
    loaded_keys = frame["record_key"].astype(str).to_numpy()
    canonical_keys = source["record_key"].astype(str).to_numpy()
    if record_key_signature(loaded_keys) != observed["record_key_signature"]:
        raise ValueError("candidate cache parquet record_key_signature mismatch")
    # The signature sorts keys, so order and call alignment need direct checks.
    if not np.array_equal(loaded_keys, canonical_keys):
        raise ValueError("candidate cache record_key canonical order mismatch")
    if not np.array_equal(
        frame["call_id"].astype(str).to_numpy(),
        source["call_id"].astype(str).to_numpy(),
    ):
        raise ValueError("candidate cache call_id alignment mismatch")
    values = frame.loc[:, feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("candidate cache contains NaN or inf")
    for column_index, feature_name in enumerate(feature_names):
        frame[feature_name] = values[:, column_index]
    return frame


def _validate_e05_inputs(
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(X_e05)
    labels = np.asarray(y, dtype=np.int64)
    keys = np.asarray(record_keys).astype(str)
    calls = np.asarray(call_ids).astype(str)
    if matrix.ndim != 2 or matrix.shape[1] != len(E05_FEATURE_NAMES):
        raise ValueError("reference matrix must satisfy exact E05 42D shape")
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("E05 reference matrix must be finite float32")
    if not (len(matrix) == len(labels) == len(keys) == len(calls)):
        raise ValueError("E05 X/y/record_key/call_id row alignment mismatch")
    if len(np.unique(keys)) != len(keys) or np.any(np.char.str_len(keys) == 0):
        raise ValueError("canonical record_key must be non-empty and one-to-one")
    _require_binary_labels(labels)
    return matrix, labels, keys, calls


def _require_binary_labels(labels: np.ndarray) -> None:
    """Use the same speaker-label rule for E05 and residual preparations."""

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("speaker labels must be binary")


def assemble_single_block_matrix(
    *,
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
    block_id: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append one block after E05 while preserving canonical row and column order."""

    matrix, _, keys, calls = _validate_e05_inputs(X_e05, y, record_keys, call_ids)
    block = BLOCK_BY_ID.get(block_id)
    if block is None:
        raise ValueError(f"unknown acoustic feature block: {block_id}")
    required = {"record_key", "call_id", *block.feature_names}
    missing = sorted(required - set(candidate_frame.columns))
    if missing:
        raise ValueError(f"candidate frame is missing columns: {missing}")
    candidate = candidate_frame.loc[:, ["record_key", "call_id", *block.feature_names]].copy()
    candidate["record_key"] = candidate["record_key"].astype(str)
    candidate["call_id"] = candidate["call_id"].astype(str)
    if not candidate["record_key"].is_unique or set(candidate["record_key"]) != set(keys):
        raise ValueError("candidate record_key alignment is not one-to-one and exact")
    canonical = pd.DataFrame(
        {"record_key": keys, "call_id_expected": calls, "_row": np.arange(len(keys))}
    )
    joined = canonical.merge(
        candidate,
        on="record_key",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row", kind="stable")
    if joined[list(block.feature_names)].isna().any().any():
        raise ValueError("candidate alignment produced missing feature values")
    if not np.array_equal(
        joined["call_id_expected"].to_numpy(), joined["call_id"].to_numpy()
    ):
        raise ValueError("candidate call_id alignment mismatch")
    extra = joined.loc[:, block.feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(extra).all():
        raise ValueError("candidate block contains NaN or inf")
    combined = np.empty((len(matrix), matrix.shape[1] + len(block.feature_names)), dtype=np.float32)
    combined[:, : matrix.shape[1]] = matrix
    combined[:, matrix.shape[1] :] = extra
    feature_names = validate_model_feature_names((*E05_FEATURE_NAMES, *block.feature_names))
    if not np.array_equal(combined[:, :42], matrix):
        raise AssertionError("candidate assembly changed the frozen E05 prefix")
    return combined, feature_names


def _prepare_representation(
    *,
    representation_id: str,
    X: np.ndarray,
    y: np.ndarray,
    record_keys: np.ndarray,
    feature_names: Sequence[str],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    split_audit: Mapping[str, Any],
) -> PreparedRepresentation:
    names = validate_model_feature_names(feature_names)
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.shape != (len(y), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("representation matrix shape or finite contract failed")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(matrix[train_idx]).astype(np.float32, copy=False)
    X_validation_scaled = scaler.transform(matrix[validation_idx]).astype(
        np.float32, copy=False
    )
    if scaler.n_samples_seen_ != len(train_idx):
        raise AssertionError("scaler was not fit on internal Training rows only")
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_validation_scaled).all():
        raise ValueError("scaled representation contains NaN or inf")
    return PreparedRepresentation(
        representation_id=representation_id,
        feature_names=names,
        X_train_scaled=X_train_scaled,
        X_evaluation_scaled=X_validation_scaled,
        y_train=y[train_idx],
        y_evaluation=y[validation_idx],
        training_record_keys=record_keys[train_idx],
        evaluation_record_keys=record_keys[validation_idx],
        train_idx=train_idx.copy(),
        validation_idx=validation_idx.copy(),
        scaler=scaler,
        split_audit=dict(split_audit),
    )


def prepare_single_block_screening(
    *,
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
    expected_split_contract: Mapping[str, int] | None,
    block_ids: Sequence[str] = tuple(block.block_id for block in FEATURE_BLOCKS),
) -> dict[str, PreparedRepresentation]:
    """Prepare E05 and every E05+single-block matrix on one fixed GroupSplit."""

    matrix, labels, keys, calls = _validate_e05_inputs(
        X_e05, y, record_keys, call_ids
    )
    train_idx, validation_idx, split_audit = create_fixed_internal_split(
        calls,
        split_function=standalone_group_split_indices,
        expected_contract=expected_split_contract,
    )
    prepared = {
        "E05_REFERENCE": _prepare_representation(
            representation_id="E05_REFERENCE",
            X=matrix,
            y=labels,
            record_keys=keys,
            feature_names=E05_FEATURE_NAMES,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    }
    for block_id in _validate_block_ids(block_ids):
        block = BLOCK_BY_ID[block_id]
        combined, feature_names = assemble_single_block_matrix(
            X_e05=matrix,
            y=labels,
            record_keys=keys,
            call_ids=calls,
            candidate_frame=candidate_frame,
            block_id=block.block_id,
        )
        prepared[block.block_id] = _prepare_representation(
            representation_id=block.block_id,
            X=combined,
            y=labels,
            record_keys=keys,
            feature_names=feature_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    return prepared


def prepare_round2_single_block_screening(
    *,
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
    expected_split_contract: Mapping[str, int] | None,
) -> dict[str, PreparedRepresentation]:
    """Prepare the fixed AF02 47D reference and each Round 2 49D candidate."""

    matrix, labels, keys, calls = _validate_e05_inputs(
        X_e05, y, record_keys, call_ids
    )
    reference_matrix, reference_names = assemble_single_block_matrix(
        X_e05=matrix,
        y=labels,
        record_keys=keys,
        call_ids=calls,
        candidate_frame=candidate_frame,
        block_id="AF02_PITCH_VOICING",
    )
    if reference_names != ROUND2_REFERENCE_FEATURE_NAMES:
        raise AssertionError("Round 2 reference differs from the frozen AF02 47D order")
    train_idx, validation_idx, split_audit = create_fixed_internal_split(
        calls,
        split_function=standalone_group_split_indices,
        expected_contract=expected_split_contract,
    )
    prepared = {
        ROUND2_REFERENCE_ID: _prepare_representation(
            representation_id=ROUND2_REFERENCE_ID,
            X=reference_matrix,
            y=labels,
            record_keys=keys,
            feature_names=reference_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    }
    for block_id in ROUND2_BLOCK_IDS:
        e05_plus_block, names = assemble_single_block_matrix(
            X_e05=matrix,
            y=labels,
            record_keys=keys,
            call_ids=calls,
            candidate_frame=candidate_frame,
            block_id=block_id,
        )
        block = BLOCK_BY_ID[block_id]
        combined = np.concatenate(
            [reference_matrix, e05_plus_block[:, -block.output_dimension :]], axis=1
        ).astype(np.float32, copy=False)
        feature_names = validate_model_feature_names(
            (*ROUND2_REFERENCE_FEATURE_NAMES, *block.feature_names)
        )
        if names[-block.output_dimension :] != block.feature_names:
            raise AssertionError("Round 2 candidate assembly changed block order")
        prepared[block_id] = _prepare_representation(
            representation_id=block_id,
            X=combined,
            y=labels,
            record_keys=keys,
            feature_names=feature_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    return prepared


def _assemble_frozen_49d_matrix(
    *,
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assemble the immutable E05+AF02+AF04 prefix without redefining a block."""

    reference_47d, names_47d = assemble_single_block_matrix(
        X_e05=X_e05,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate_frame,
        block_id="AF02_PITCH_VOICING",
    )
    e05_plus_af04, af04_names = assemble_single_block_matrix(
        X_e05=X_e05,
        y=y,
        record_keys=record_keys,
        call_ids=call_ids,
        candidate_frame=candidate_frame,
        block_id="AF04_CPP",
    )
    if names_47d != ROUND2_REFERENCE_FEATURE_NAMES:
        raise AssertionError("frozen 47D prefix order changed")
    af04 = BLOCK_BY_ID["AF04_CPP"]
    if af04_names[-af04.output_dimension :] != af04.feature_names:
        raise AssertionError("frozen AF04 order changed")
    matrix = np.concatenate(
        [reference_47d, e05_plus_af04[:, -af04.output_dimension :]], axis=1
    ).astype(np.float32, copy=False)
    if matrix.shape[1] != 49 or not np.array_equal(matrix[:, :47], reference_47d):
        raise AssertionError("frozen 49D prefix assembly changed")
    return matrix, FROZEN_49D_FEATURE_NAMES


def prepare_frozen_49d_single_block_screening(
    *,
    X_e05: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
    expected_split_contract: Mapping[str, int] | None,
    block_ids: Sequence[str] = NEXT_CANDIDATE_BLOCK_IDS,
) -> dict[str, PreparedRepresentation]:
    """Prepare frozen 49D and 49D-plus-one candidate representations."""

    matrix, labels, keys, calls = _validate_e05_inputs(
        X_e05, y, record_keys, call_ids
    )
    ids = _validate_block_ids(block_ids)
    if any(block_id not in NEXT_CANDIDATE_BLOCK_IDS for block_id in ids):
        raise ValueError("49D screening accepts only next-candidate blocks")
    reference_matrix, reference_names = _assemble_frozen_49d_matrix(
        X_e05=matrix,
        y=labels,
        record_keys=keys,
        call_ids=calls,
        candidate_frame=candidate_frame,
    )
    train_idx, validation_idx, split_audit = create_fixed_internal_split(
        calls,
        split_function=standalone_group_split_indices,
        expected_contract=expected_split_contract,
    )
    prepared = {
        FROZEN_49D_REFERENCE_ID: _prepare_representation(
            representation_id=FROZEN_49D_REFERENCE_ID,
            X=reference_matrix,
            y=labels,
            record_keys=keys,
            feature_names=reference_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    }
    for block_id in ids:
        e05_plus_block, names = assemble_single_block_matrix(
            X_e05=matrix,
            y=labels,
            record_keys=keys,
            call_ids=calls,
            candidate_frame=candidate_frame,
            block_id=block_id,
        )
        block = BLOCK_BY_ID[block_id]
        if names[-block.output_dimension :] != block.feature_names:
            raise AssertionError("candidate block feature order changed")
        combined = np.concatenate(
            [reference_matrix, e05_plus_block[:, -block.output_dimension :]], axis=1
        ).astype(np.float32, copy=False)
        feature_names = validate_model_feature_names(
            (*FROZEN_49D_FEATURE_NAMES, *block.feature_names)
        )
        if not np.array_equal(combined[:, :49], reference_matrix):
            raise AssertionError("candidate assembly changed the frozen 49D prefix")
        prepared[block_id] = _prepare_representation(
            representation_id=block_id,
            X=combined,
            y=labels,
            record_keys=keys,
            feature_names=feature_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=split_audit,
        )
    return prepared


def _scaler_fingerprint(
    scaler: StandardScaler,
    *,
    array_fingerprint: Callable[[np.ndarray], str] = _array_fingerprint,
) -> str:
    return _json_fingerprint(
        {
            "mean": array_fingerprint(np.asarray(scaler.mean_)),
            "scale": array_fingerprint(np.asarray(scaler.scale_)),
            "var": array_fingerprint(np.asarray(scaler.var_)),
            "n_samples_seen": int(scaler.n_samples_seen_),
        }
    )


def _prepared_array_fingerprints(
    prepared: PreparedRepresentation,
    *,
    array_fingerprint: Callable[[np.ndarray], str] = _array_fingerprint,
) -> dict[str, str]:
    """Bind one array algorithm to the split and scaler provenance together."""

    return {
        "split_fingerprint": _json_fingerprint(
            {
                "train_idx": array_fingerprint(prepared.train_idx),
                "validation_idx": array_fingerprint(prepared.validation_idx),
                "audit": dict(prepared.split_audit),
            }
        ),
        "scaler_fingerprint": _scaler_fingerprint(
            prepared.scaler, array_fingerprint=array_fingerprint
        ),
    }


def validate_canonical_52d_prepared(
    prepared: PreparedRepresentation,
    *,
    canonical_matrix: np.ndarray,
    canonical_labels: Sequence[int],
    canonical_record_keys: Sequence[str],
    canonical_train_idx: Sequence[int],
    canonical_validation_idx: Sequence[int],
    expected_split_contract: Mapping[str, int] = EXPECTED_INTERNAL_SPLIT,
) -> dict[str, Any]:
    """Validate the independent 52D sources before any stale-result refit.

    The caller supplies the canonical matrix, labels, keys, and fixed split used
    to construct ``prepared``.  Recomputing scaler statistics from those sources
    prevents a self-consistent but stale ``PreparedRepresentation`` from being
    accepted merely because its own fields agree with one another.
    """

    matrix = np.asarray(canonical_matrix)
    labels = np.asarray(canonical_labels, dtype=np.int64)
    record_keys = np.asarray(canonical_record_keys).astype(str)
    train_idx = np.asarray(canonical_train_idx, dtype=np.int64)
    validation_idx = np.asarray(canonical_validation_idx, dtype=np.int64)
    expected_total = int(expected_split_contract["total_rows"])
    expected_train = int(expected_split_contract["train_rows"])
    expected_validation = int(expected_split_contract["validation_rows"])
    expected_train_shape = (expected_train, len(CANONICAL_52D_FEATURE_NAMES))
    expected_validation_shape = (
        expected_validation,
        len(CANONICAL_52D_FEATURE_NAMES),
    )
    mismatches: list[str] = []

    if prepared.representation_id != CANONICAL_52D_PREPARED_ID:
        mismatches.append("representation_id")
    if tuple(prepared.feature_names) != CANONICAL_52D_FEATURE_NAMES:
        mismatches.append("feature_names/order")
    if len(prepared.feature_names) != 52:
        mismatches.append("feature_dimension")
    if matrix.shape != (expected_total, 52) or matrix.dtype != np.float32:
        mismatches.append("canonical_matrix_shape/dtype")
    if labels.shape != (expected_total,):
        mismatches.append("canonical_labels_shape")
    if record_keys.shape != (expected_total,) or len(np.unique(record_keys)) != expected_total:
        mismatches.append("canonical_record_keys")
    if train_idx.shape != (expected_train,) or validation_idx.shape != (
        expected_validation,
    ):
        mismatches.append("canonical_split_shape")
    elif (
        np.any(train_idx < 0)
        or np.any(validation_idx < 0)
        or np.any(train_idx >= expected_total)
        or np.any(validation_idx >= expected_total)
        or not np.array_equal(
            np.sort(np.concatenate([train_idx, validation_idx])),
            np.arange(expected_total, dtype=np.int64),
        )
    ):
        mismatches.append("canonical_split_partition")

    if not np.array_equal(prepared.train_idx, train_idx):
        mismatches.append("train_idx")
    if not np.array_equal(prepared.validation_idx, validation_idx):
        mismatches.append("validation_idx")
    if prepared.X_train_scaled.shape != expected_train_shape:
        mismatches.append("train_shape")
    if prepared.X_evaluation_scaled.shape != expected_validation_shape:
        mismatches.append("evaluation_shape")
    if not np.array_equal(prepared.training_record_keys.astype(str), record_keys[train_idx]):
        mismatches.append("training_record_keys")
    if not np.array_equal(
        prepared.evaluation_record_keys.astype(str), record_keys[validation_idx]
    ):
        mismatches.append("evaluation_record_keys")
    if not np.array_equal(prepared.y_train, labels[train_idx]):
        mismatches.append("training_labels")
    if not np.array_equal(prepared.y_evaluation, labels[validation_idx]):
        mismatches.append("evaluation_labels")
    for name, expected in expected_split_contract.items():
        if prepared.split_audit.get(name) != expected:
            mismatches.append(f"split_audit.{name}")

    finite_arrays = (
        matrix,
        prepared.X_train_scaled,
        prepared.X_evaluation_scaled,
        np.asarray(prepared.scaler.mean_),
        np.asarray(prepared.scaler.scale_),
    )
    if any(not np.isfinite(values).all() for values in finite_arrays):
        mismatches.append("finite_values")
    if prepared.X_train_scaled.dtype != np.float32:
        mismatches.append("train_dtype")
    if prepared.X_evaluation_scaled.dtype != np.float32:
        mismatches.append("evaluation_dtype")

    expected_scaler = None
    if matrix.shape == (expected_total, 52) and train_idx.shape == (expected_train,):
        expected_scaler = StandardScaler().fit(matrix[train_idx])
        if int(prepared.scaler.n_samples_seen_) != expected_train:
            mismatches.append("scaler_fit_rows")
        if not np.array_equal(prepared.scaler.mean_, expected_scaler.mean_):
            mismatches.append("scaler.mean_")
        if not np.array_equal(prepared.scaler.scale_, expected_scaler.scale_):
            mismatches.append("scaler.scale_")
    else:
        mismatches.append("scaler_source_unavailable")

    if mismatches:
        raise RuntimeError(
            "STOP: prepared_52d canonical contract failed: "
            + ", ".join(dict.fromkeys(mismatches))
        )
    assert expected_scaler is not None
    return {
        "feature_dimension": 52,
        "train_shape": expected_train_shape,
        "evaluation_shape": expected_validation_shape,
        "scaler_fit_rows": expected_train,
        "scaler_fit_scope": "internal_train_only",
        "array_fingerprint_version": ARRAY_FINGERPRINT_VERSION,
        "feature_fingerprint": _json_fingerprint(list(prepared.feature_names)),
        **_prepared_array_fingerprints(prepared),
        "row_fingerprint": _json_fingerprint(
            {
                "training": prepared.training_record_keys.tolist(),
                "evaluation": prepared.evaluation_record_keys.tolist(),
            }
        ),
        "label_fingerprint": _json_fingerprint(
            {
                "training": _array_fingerprint(prepared.y_train),
                "evaluation": _array_fingerprint(prepared.y_evaluation),
            }
        ),
    }


def prepare_residual_combination_screening(
    *,
    prepared_52d: PreparedRepresentation,
    X_52d: np.ndarray,
    y: Sequence[int],
    record_keys: Sequence[str],
    call_ids: Sequence[str],
    candidate_frame: pd.DataFrame,
    canonical_train_idx: Sequence[int],
    canonical_validation_idx: Sequence[int],
    expected_split_contract: Mapping[str, int] = EXPECTED_INTERNAL_SPLIT,
) -> dict[str, PreparedRepresentation]:
    """Prepare exactly four 52D-plus-residual-block combinations.

    The caller must strict-load the existing combined AF10/AF11/AF12 cache.
    This function performs no waveform or feature extraction and refuses to
    build candidates until the supplied 52D payload passes its canonical
    prepared-object contract.
    """

    matrix = np.asarray(X_52d)
    raw_labels = np.asarray(y)
    _require_binary_labels(raw_labels)
    labels = np.asarray(raw_labels, dtype=np.int64)
    keys = np.asarray(record_keys).astype(str)
    calls = np.asarray(call_ids).astype(str)
    train_idx = np.asarray(canonical_train_idx, dtype=np.int64)
    validation_idx = np.asarray(canonical_validation_idx, dtype=np.int64)
    validate_canonical_52d_prepared(
        prepared_52d,
        canonical_matrix=matrix,
        canonical_labels=labels,
        canonical_record_keys=keys,
        canonical_train_idx=train_idx,
        canonical_validation_idx=validation_idx,
        expected_split_contract=expected_split_contract,
    )
    if calls.shape != keys.shape or np.any(np.char.str_len(calls) == 0):
        raise ValueError("canonical call_id shape/non-empty contract failed")

    cache_feature_names = tuple(
        feature_name
        for block_id in RESIDUAL_TARGET_BLOCK_IDS
        for feature_name in BLOCK_BY_ID[block_id].feature_names
    )
    required_columns = {"record_key", "call_id", *cache_feature_names}
    missing_columns = sorted(required_columns - set(candidate_frame.columns))
    if missing_columns:
        raise ValueError(
            f"combined residual cache is missing columns: {missing_columns}"
        )
    candidate = candidate_frame.loc[
        :, ["record_key", "call_id", *cache_feature_names]
    ].copy()
    candidate["record_key"] = candidate["record_key"].astype(str)
    candidate["call_id"] = candidate["call_id"].astype(str)
    if not candidate["record_key"].is_unique or set(candidate["record_key"]) != set(
        keys
    ):
        raise ValueError("combined residual cache record_key alignment is not exact")

    canonical_order = pd.DataFrame(
        {
            "record_key": keys,
            "call_id_expected": calls,
            "_row": np.arange(len(keys), dtype=np.int64),
        }
    )
    aligned = canonical_order.merge(
        candidate,
        on="record_key",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row", kind="stable")
    if not np.array_equal(
        aligned["call_id_expected"].to_numpy(),
        aligned["call_id"].to_numpy(),
    ):
        raise ValueError("combined residual cache call_id alignment mismatch")
    cache_matrix = aligned.loc[:, list(cache_feature_names)].to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(cache_matrix).all():
        raise ValueError("combined residual cache contains NaN or inf")

    prepared: dict[str, PreparedRepresentation] = {}
    for combination_id, block_ids in RESIDUAL_COMBINATION_SPECS.items():
        combination_feature_names = tuple(
            feature_name
            for block_id in block_ids
            for feature_name in BLOCK_BY_ID[block_id].feature_names
        )
        extra_matrix = aligned.loc[:, list(combination_feature_names)].to_numpy(
            dtype=np.float32
        )
        combination_matrix = np.concatenate(
            [matrix, extra_matrix], axis=1
        ).astype(np.float32, copy=False)
        if not np.array_equal(combination_matrix[:, :52], matrix):
            raise AssertionError(f"{combination_id} changed the canonical 52D prefix")
        payload = _prepare_representation(
            representation_id=combination_id,
            X=combination_matrix,
            y=labels,
            record_keys=keys,
            feature_names=(
                *CANONICAL_52D_FEATURE_NAMES,
                *combination_feature_names,
            ),
            train_idx=train_idx,
            validation_idx=validation_idx,
            split_audit=prepared_52d.split_audit,
        )
        if not np.array_equal(
            payload.X_train_scaled[:, :52], prepared_52d.X_train_scaled
        ) or not np.array_equal(
            payload.X_evaluation_scaled[:, :52],
            prepared_52d.X_evaluation_scaled,
        ):
            raise AssertionError(
                f"{combination_id} changed the scaled canonical 52D prefix"
            )
        prepared[combination_id] = payload
    return prepared


def _normalized_confusion_matrix(value: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    try:
        matrix = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (2, 2):
        return None
    return tuple(tuple(int(item) for item in row) for row in matrix)  # type: ignore[return-value]


def audit_canonical_52d_result(
    result: ModelExperimentResult | None,
    prepared: PreparedRepresentation,
    *,
    prepared_audit: Mapping[str, Any],
    expected_accuracy: float = CANONICAL_52D_ACCURACY,
    expected_confusion_matrix: Sequence[Sequence[int]] = (
        CANONICAL_52D_CONFUSION_MATRIX
    ),
) -> dict[str, Any]:
    """Return every reason an in-memory 52D result is unsafe to reuse."""

    if result is None:
        return {"is_canonical": False, "mismatches": ("missing_result",)}

    mismatches: list[str] = []
    fit_audit = dict(getattr(result, "fit_audit", {}))
    expected_params = MLPClassifier(
        **dict(M04_ITER100_FEATURE_SCREENING_CONFIG)
    ).get_params(deep=False)
    model = getattr(result, "model", None)
    observed_params = (
        model.get_params(deep=False) if hasattr(model, "get_params") else None
    )

    if result.experiment_id != CANONICAL_52D_REFERENCE_ID:
        mismatches.append("experiment_id")
    if tuple(result.feature_names) != CANONICAL_52D_FEATURE_NAMES:
        mismatches.append("feature_names/order")
    if result.train_shape != prepared.X_train_scaled.shape:
        mismatches.append("train_shape")
    if result.evaluation_shape != prepared.X_evaluation_scaled.shape:
        mismatches.append("evaluation_shape")
    for name, expected in (
        ("train_idx", prepared.train_idx),
        ("validation_idx", prepared.validation_idx),
        ("training_record_keys", prepared.training_record_keys),
        ("validation_record_keys", prepared.evaluation_record_keys),
    ):
        observed = fit_audit.get(name)
        if observed is None or not np.array_equal(observed, expected):
            mismatches.append(name)
    if not np.array_equal(result.y_true, prepared.y_evaluation):
        mismatches.append("evaluation_labels")
    try:
        validate_binary_probabilities(
            result.probabilities,
            expected_rows=len(prepared.y_evaluation),
            classes=getattr(model, "classes_", None),
        )
    except (TypeError, ValueError):
        mismatches.append("probabilities")
    if dict(result.model_config) != dict(M04_ITER100_FEATURE_SCREENING_CONFIG):
        mismatches.append("model_config")
    if observed_params != expected_params:
        mismatches.append("model_full_params")
    if fit_audit.get("seed") != int(
        M04_ITER100_FEATURE_SCREENING_CONFIG["random_state"]
    ):
        mismatches.append("seed")
    if fit_audit.get("scaler_fit_scope") != "internal_train_only":
        mismatches.append("scaler_fit_scope")
    if int(getattr(result.scaler, "n_samples_seen_", -1)) != int(
        prepared.scaler.n_samples_seen_
    ):
        mismatches.append("scaler_fit_rows")
    for attribute in ("mean_", "scale_"):
        if not np.array_equal(
            getattr(result.scaler, attribute, None),
            getattr(prepared.scaler, attribute),
        ):
            mismatches.append(f"scaler.{attribute}")
    array_version = fit_audit.get("array_fingerprint_version")
    if array_version is None:
        # Historical in-memory acoustic results used tuple-shaped hashes.
        expected_array_fingerprints = _prepared_array_fingerprints(
            prepared, array_fingerprint=_legacy_acoustic_array_fingerprint
        )
    elif array_version == ARRAY_FINGERPRINT_VERSION:
        expected_array_fingerprints = prepared_audit
    else:
        mismatches.append("array_fingerprint_version")
        expected_array_fingerprints = prepared_audit
    for name in (
        "feature_fingerprint",
        "split_fingerprint",
        "row_fingerprint",
        "scaler_fingerprint",
    ):
        expected = (
            expected_array_fingerprints.get(name)
            if name in {"split_fingerprint", "scaler_fingerprint"}
            else prepared_audit.get(name)
        )
        if fit_audit.get(name) != expected:
            mismatches.append(name)

    observed_accuracy = result.metrics.get("accuracy")
    observed_confusion = _normalized_confusion_matrix(
        result.metrics.get("confusion_matrix")
    )
    expected_confusion = _normalized_confusion_matrix(expected_confusion_matrix)
    if expected_confusion is None:
        mismatches.append("expected_confusion_matrix")
    if observed_confusion != expected_confusion:
        mismatches.append("confusion_matrix")
    expected_rows = len(prepared.y_evaluation)
    expected_correct = None
    if expected_confusion is not None:
        expected_correct = expected_confusion[0][0] + expected_confusion[1][1]
        if sum(sum(row) for row in expected_confusion) != expected_rows:
            mismatches.append("expected_confusion_row_count")
        if not _rate_matches_count(expected_accuracy, expected_correct, expected_rows):
            mismatches.append("expected_accuracy_count")
        if not _rate_matches_count(observed_accuracy, expected_correct, expected_rows):
            mismatches.append("accuracy")
        if result.metrics.get("correct_count") != expected_correct:
            mismatches.append("correct_count")

    try:
        recalculated = calculate_prediction_metrics(result.y_true, result.y_pred)
    except (TypeError, ValueError):
        recalculated = {}
    if not _rate_matches_count(
        observed_accuracy,
        recalculated.get("correct_count", -1),
        recalculated.get("sample_count", -1),
    ):
        mismatches.append("prediction_accuracy")
    if recalculated.get("correct_count") != result.metrics.get("correct_count"):
        mismatches.append("prediction_correct_count")
    if _normalized_confusion_matrix(
        recalculated.get("confusion_matrix")
    ) != observed_confusion:
        mismatches.append("prediction_confusion_matrix")

    historical_accuracy = float(observed_accuracy) == float(
        HISTORICAL_NON_REPRODUCED_52D_ACCURACY
    ) if observed_accuracy is not None else False
    historical_confusion = (
        observed_confusion == HISTORICAL_NON_REPRODUCED_52D_CONFUSION_MATRIX
    )
    if historical_accuracy and historical_confusion:
        mismatches.append("historical_non_reproduced_result")

    unique_mismatches = tuple(dict.fromkeys(mismatches))
    return {
        "is_canonical": not unique_mismatches,
        "mismatches": unique_mismatches,
        "expected_accuracy": float(expected_accuracy),
        "observed_accuracy": observed_accuracy,
        "expected_confusion_matrix": expected_confusion,
        "observed_confusion_matrix": observed_confusion,
        "expected_correct_count": expected_correct,
        "observed_correct_count": recalculated.get("correct_count"),
        "environment_versions": fit_audit.get("environment_versions"),
        "array_fingerprint_version": array_version or "acoustic-array-legacy-v0",
        "source_modules": fit_audit.get("source_modules"),
        "model_config_fingerprint": fit_audit.get("model_config_fingerprint"),
        "row_fingerprint": fit_audit.get("row_fingerprint"),
        "prediction_fingerprint": fit_audit.get("prediction_fingerprint"),
    }


def resolve_canonical_52d_reference(
    prepared: PreparedRepresentation,
    *,
    existing_result: ModelExperimentResult | None,
    canonical_matrix: np.ndarray,
    canonical_labels: Sequence[int],
    canonical_record_keys: Sequence[str],
    canonical_train_idx: Sequence[int],
    canonical_validation_idx: Sequence[int],
    expected_split_contract: Mapping[str, int] = EXPECTED_INTERNAL_SPLIT,
    allow_refit: bool = False,
    expected_accuracy: float = CANONICAL_52D_ACCURACY,
    expected_confusion_matrix: Sequence[Sequence[int]] = (
        CANONICAL_52D_CONFUSION_MATRIX
    ),
) -> Canonical52DResolution:
    """Reuse only a canonical 52D result; otherwise refit validated inputs."""

    prepared_audit = validate_canonical_52d_prepared(
        prepared,
        canonical_matrix=canonical_matrix,
        canonical_labels=canonical_labels,
        canonical_record_keys=canonical_record_keys,
        canonical_train_idx=canonical_train_idx,
        canonical_validation_idx=canonical_validation_idx,
        expected_split_contract=expected_split_contract,
    )
    existing_audit = audit_canonical_52d_result(
        existing_result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=expected_accuracy,
        expected_confusion_matrix=expected_confusion_matrix,
    )
    if existing_audit["is_canonical"]:
        assert existing_result is not None
        return Canonical52DResolution(
            result=existing_result,
            action="reused_canonical",
            stale_mismatches=(),
            prepared_audit=prepared_audit,
        )
    if not allow_refit:
        raise RuntimeError(
            "STOP: canonical 52D is missing or stale and refit is disabled: "
            + ", ".join(existing_audit["mismatches"])
            + f"; reproducibility diagnostic={existing_audit!r}"
        )

    refit_result = _run_prepared_representation(
        prepared,
        experiment_id=CANONICAL_52D_REFERENCE_ID,
        allow_fit=True,
    )
    refit_audit = audit_canonical_52d_result(
        refit_result,
        prepared,
        prepared_audit=prepared_audit,
        expected_accuracy=expected_accuracy,
        expected_confusion_matrix=expected_confusion_matrix,
    )
    if not refit_audit["is_canonical"]:
        raise RuntimeError(
            "STOP: refitted 52D did not reproduce the canonical contract: "
            + ", ".join(refit_audit["mismatches"])
            + f"; reproducibility diagnostic={refit_audit!r}"
        )
    return Canonical52DResolution(
        result=refit_result,
        action="refit_stale_or_missing",
        stale_mismatches=tuple(existing_audit["mismatches"]),
        prepared_audit=prepared_audit,
    )


def build_reference_contract(prepared: PreparedRepresentation) -> dict[str, Any]:
    """Freeze structural provenance before fitting the M04-100 measurement model."""

    valid_reference = (
        prepared.representation_id == "E05_REFERENCE"
        and prepared.feature_names == E05_FEATURE_NAMES
    ) or (
        prepared.representation_id == ROUND2_REFERENCE_ID
        and prepared.feature_names == ROUND2_REFERENCE_FEATURE_NAMES
    ) or (
        prepared.representation_id == FROZEN_49D_REFERENCE_ID
        and prepared.feature_names == FROZEN_49D_FEATURE_NAMES
    )
    if not valid_reference:
        raise ValueError(
            "reference contract requires exact E05 42D, AF02 47D, or frozen 49D data"
        )
    return {
        "array_fingerprint_version": ARRAY_FINGERPRINT_VERSION,
        "row_fingerprint": _json_fingerprint(
            {
                "training": prepared.training_record_keys.tolist(),
                "evaluation": prepared.evaluation_record_keys.tolist(),
            }
        ),
        **_prepared_array_fingerprints(prepared),
        "feature_fingerprint": _json_fingerprint(list(prepared.feature_names)),
        "config_fingerprint": config_fingerprint(M04_ITER100_FEATURE_SCREENING_CONFIG),
        "seed": int(M04_ITER100_FEATURE_SCREENING_CONFIG["random_state"]),
        "scaler_fit_scope": "internal_train_only",
    }


def _run_prepared_representation(
    prepared: PreparedRepresentation,
    *,
    experiment_id: str,
    allow_fit: bool,
) -> ModelExperimentResult:
    if not allow_fit:
        raise RuntimeError("STOP: model fitting requires an explicit True execution switch")
    model = MLPClassifier(**dict(M04_ITER100_FEATURE_SCREENING_CONFIG))
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(prepared.X_train_scaled, prepared.y_train)
    fit_seconds = perf_counter() - started
    prediction_started = perf_counter()
    prediction = model.predict(prepared.X_evaluation_scaled).astype(np.int64)
    probabilities = validate_binary_probabilities(
        model.predict_proba(prepared.X_evaluation_scaled),
        expected_rows=len(prepared.y_evaluation),
        classes=model.classes_,
    )
    prediction_seconds = perf_counter() - prediction_started
    metrics = calculate_prediction_metrics(prepared.y_evaluation, prediction)
    runtime_provenance = _runtime_provenance(model)
    contract = (
        build_reference_contract(prepared)
        if prepared.representation_id in {
            "E05_REFERENCE",
            ROUND2_REFERENCE_ID,
            FROZEN_49D_REFERENCE_ID,
        }
        else None
    )
    canonical_fingerprints = _prepared_array_fingerprints(prepared)
    legacy_fingerprints = _prepared_array_fingerprints(
        prepared, array_fingerprint=_legacy_acoustic_array_fingerprint
    )
    fit_audit = {
        "train_idx": prepared.train_idx.copy(),
        "validation_idx": prepared.validation_idx.copy(),
        "training_record_keys": prepared.training_record_keys.copy(),
        "validation_record_keys": prepared.evaluation_record_keys.copy(),
        "scaler_fit_record_keys": prepared.training_record_keys.copy(),
        "training_scope": "Training/internal_train",
        "evaluation_scope": "Training/internal_validation",
        "scaler_fit_scope": "internal_train_only",
        "array_fingerprint_version": ARRAY_FINGERPRINT_VERSION,
        "model_config_fingerprint": config_fingerprint(
            M04_ITER100_FEATURE_SCREENING_CONFIG
        ),
        "feature_fingerprint": _json_fingerprint(list(prepared.feature_names)),
        **canonical_fingerprints,
        "legacy_split_fingerprint": legacy_fingerprints["split_fingerprint"],
        "legacy_scaler_fingerprint": legacy_fingerprints["scaler_fingerprint"],
        "row_fingerprint": _json_fingerprint(
            {
                "training": prepared.training_record_keys.tolist(),
                "evaluation": prepared.evaluation_record_keys.tolist(),
            }
        ),
        "seed": int(M04_ITER100_FEATURE_SCREENING_CONFIG["random_state"]),
        "prediction_shape": prediction.shape,
        "prediction_fingerprint": _array_fingerprint(prediction),
        "probability_shape": probabilities.shape,
        "probability_fingerprint": _array_fingerprint(probabilities),
        "confusion_matrix": metrics["confusion_matrix"],
        "accuracy": float(metrics["accuracy"]),
        "n_iter": int(model.n_iter_),
        "loss_curve": tuple(float(value) for value in model.loss_curve_),
        **runtime_provenance,
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "runtime_seconds": float(fit_seconds + prediction_seconds),
        "warning_messages": tuple(str(item.message) for item in caught),
        "convergence_warning_raised": any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }
    if contract is not None:
        for name, value in contract.items():
            if name in fit_audit and fit_audit[name] != value:
                raise AssertionError(f"reference fit changed pre-fit fingerprint: {name}")
    return ModelExperimentResult(
        experiment_id=experiment_id,
        model_name="MLPClassifier",
        model_config=dict(M04_ITER100_FEATURE_SCREENING_CONFIG),
        feature_names=prepared.feature_names,
        train_shape=prepared.X_train_scaled.shape,
        evaluation_shape=prepared.X_evaluation_scaled.shape,
        y_true=prepared.y_evaluation.copy(),
        y_pred=prediction,
        probabilities=probabilities,
        metrics=metrics,
        scaler=prepared.scaler,
        model=model,
        fit_audit=fit_audit,
    )


def run_feature_screening_reference(
    prepared: PreparedRepresentation, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Fit exact M04-100 on E05 only after the explicit reference switch."""

    if prepared.representation_id != "E05_REFERENCE" or prepared.feature_names != E05_FEATURE_NAMES:
        raise ValueError("Feature Screening Reference requires exact E05 42D")
    return _run_prepared_representation(
        prepared,
        experiment_id="AF00_E05_M04_ITER100_REFERENCE",
        allow_fit=allow_fit,
    )


def run_feature_screening_candidate(
    prepared: PreparedRepresentation,
    *,
    block_id: str,
    allow_fit: bool = False,
) -> ModelExperimentResult:
    """Fit exactly one E05+block representation; never generate combinations."""

    block = BLOCK_BY_ID.get(block_id)
    if block is None or prepared.representation_id != block_id:
        raise ValueError("candidate prepared payload does not match the requested block")
    if prepared.feature_names != (*E05_FEATURE_NAMES, *block.feature_names):
        raise ValueError("candidate feature order differs from E05 plus one frozen block")
    return _run_prepared_representation(
        prepared,
        experiment_id=block_id,
        allow_fit=allow_fit,
    )


def run_round2_feature_screening_reference(
    prepared: PreparedRepresentation, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Fit M04-100 on the AF02 47D reference only after explicit opt-in."""

    if (
        prepared.representation_id != ROUND2_REFERENCE_ID
        or prepared.feature_names != ROUND2_REFERENCE_FEATURE_NAMES
    ):
        raise ValueError("Round 2 reference requires exact AF02 47D prepared data")
    return _run_prepared_representation(
        prepared,
        experiment_id="AF02_47D_M04_ITER100_REFERENCE",
        allow_fit=allow_fit,
    )


def run_round2_feature_screening_candidate(
    prepared: PreparedRepresentation,
    *,
    block_id: str,
    allow_fit: bool = False,
) -> ModelExperimentResult:
    """Fit AF02 47D plus exactly one Round 2 block; combinations are forbidden."""

    if block_id not in ROUND2_BLOCK_IDS or prepared.representation_id != block_id:
        raise ValueError("Round 2 candidate must be one frozen AF04/AF05 block")
    block = BLOCK_BY_ID[block_id]
    expected_names = (*ROUND2_REFERENCE_FEATURE_NAMES, *block.feature_names)
    if prepared.feature_names != expected_names:
        raise ValueError("Round 2 candidate differs from AF02 47D plus one block")
    return _run_prepared_representation(
        prepared,
        experiment_id=block_id,
        allow_fit=allow_fit,
    )


def run_frozen_49d_screening_reference(
    prepared: PreparedRepresentation, *, allow_fit: bool = False
) -> ModelExperimentResult:
    """Fit the already-frozen 49D reference only on an explicit user call."""

    if (
        prepared.representation_id != FROZEN_49D_REFERENCE_ID
        or prepared.feature_names != FROZEN_49D_FEATURE_NAMES
    ):
        raise ValueError("frozen reference runner requires the exact 49D payload")
    return _run_prepared_representation(
        prepared,
        experiment_id="AF02_AF04_49D_M04_ITER100_REFERENCE",
        allow_fit=allow_fit,
    )


def run_frozen_49d_screening_candidate(
    prepared: PreparedRepresentation,
    *,
    block_id: str,
    allow_fit: bool = False,
) -> ModelExperimentResult:
    """Fit frozen 49D plus exactly one independent next-candidate block."""

    if block_id not in NEXT_CANDIDATE_BLOCK_IDS:
        raise ValueError("unknown frozen-49D candidate block")
    if prepared.representation_id != block_id:
        raise ValueError("prepared payload does not match the requested candidate")
    block = BLOCK_BY_ID[block_id]
    if prepared.feature_names != (*FROZEN_49D_FEATURE_NAMES, *block.feature_names):
        raise ValueError("candidate differs from frozen 49D plus one block")
    return _run_prepared_representation(
        prepared,
        experiment_id=block_id,
        allow_fit=allow_fit,
    )


def audit_reference_reproduction(
    result: ModelExperimentResult,
    prefit_contract: Mapping[str, Any],
    *,
    accuracy_target: float = M04_ITER100_ACCURACY_TARGET,
    accuracy_atol: float = 5e-7,
) -> dict[str, Any]:
    """Fail on provenance mismatch; report Accuracy only as a reproduction target."""

    fingerprint_fields = (
        "row_fingerprint",
        "split_fingerprint",
        "feature_fingerprint",
        "scaler_fingerprint",
        "config_fingerprint",
        "seed",
        "scaler_fit_scope",
    )
    observed = dict(result.fit_audit)
    observed["config_fingerprint"] = observed["model_config_fingerprint"]
    prefit_version = prefit_contract.get("array_fingerprint_version")
    result_version = observed.get("array_fingerprint_version")
    if prefit_version is None:
        if result_version == ARRAY_FINGERPRINT_VERSION:
            # Compare a frozen v0 prefit contract against the v0 hashes also
            # recorded in a newly produced result.
            observed["split_fingerprint"] = observed.get("legacy_split_fingerprint")
            observed["scaler_fingerprint"] = observed.get("legacy_scaler_fingerprint")
        elif result_version is not None:
            raise ValueError(f"unsupported result array fingerprint version: {result_version}")
    elif prefit_version != ARRAY_FINGERPRINT_VERSION or result_version != prefit_version:
        raise ValueError(
            "reference array fingerprint version mismatch: "
            f"prefit={prefit_version!r}, result={result_version!r}"
        )
    mismatches = {
        name: {"expected": prefit_contract.get(name), "observed": observed.get(name)}
        for name in fingerprint_fields
        if prefit_contract.get(name) != observed.get(name)
    }
    if mismatches:
        raise ValueError(f"reference fingerprint contract mismatch: {mismatches}")
    accuracy = float(result.metrics["accuracy"])
    return {
        "fingerprint_contract_passed": True,
        "array_fingerprint_version": prefit_version or "acoustic-array-legacy-v0",
        "accuracy": accuracy,
        "accuracy_target": float(accuracy_target),
        "accuracy_delta_from_target": float(accuracy - accuracy_target),
        "accuracy_target_match": bool(
            np.isclose(accuracy, accuracy_target, rtol=0.0, atol=accuracy_atol)
        ),
        "accuracy_is_reproduction_target_not_gate": True,
        "status": (
            "REPRODUCTION_TARGET_MATCH"
            if np.isclose(accuracy, accuracy_target, rtol=0.0, atol=accuracy_atol)
            else "REVIEW_ACCURACY_DIFFERENCE_AFTER_PROVENANCE_PASS"
        ),
    }


def compare_feature_candidate_to_reference(
    reference: ModelExperimentResult, candidate: ModelExperimentResult
) -> dict[str, Any]:
    """Return same-run paired deltas without selecting or promoting a block."""

    if reference.experiment_id == "AF00_E05_M04_ITER100_REFERENCE":
        reference_names = E05_FEATURE_NAMES
        allowed_blocks = tuple(BLOCK_BY_ID)
    elif reference.experiment_id == "AF02_47D_M04_ITER100_REFERENCE":
        reference_names = ROUND2_REFERENCE_FEATURE_NAMES
        allowed_blocks = ROUND2_BLOCK_IDS
    elif reference.experiment_id == "AF02_AF04_49D_M04_ITER100_REFERENCE":
        reference_names = FROZEN_49D_FEATURE_NAMES
        allowed_blocks = NEXT_CANDIDATE_BLOCK_IDS
    else:
        raise ValueError("paired comparison requires a recognized same-run reference")
    if candidate.experiment_id not in allowed_blocks:
        raise ValueError("paired comparison requires one allowed candidate block")
    if reference.feature_names != reference_names:
        raise ValueError("reference feature contract differs from its frozen definition")
    block = BLOCK_BY_ID[candidate.experiment_id]
    if candidate.feature_names != (*reference_names, *block.feature_names):
        raise ValueError("candidate is not the reference plus exactly one frozen block")
    for name in ("train_idx", "validation_idx", "training_record_keys", "validation_record_keys"):
        if not np.array_equal(reference.fit_audit[name], candidate.fit_audit[name]):
            raise ValueError(f"same-run paired {name} alignment mismatch")
    if not np.array_equal(reference.y_true, candidate.y_true):
        raise ValueError("same-run paired label alignment mismatch")
    if dict(reference.model_config) != dict(candidate.model_config):
        raise ValueError("same-run paired classifier/config mismatch")
    truth = np.asarray(reference.y_true)
    reference_correct = np.asarray(reference.y_pred) == truth
    candidate_correct = np.asarray(candidate.y_pred) == truth
    reference_wrong_to_candidate_correct = int(
        np.sum((~reference_correct) & candidate_correct)
    )
    reference_correct_to_candidate_wrong = int(
        np.sum(reference_correct & (~candidate_correct))
    )
    disagreement = np.asarray(reference.y_pred) != np.asarray(candidate.y_pred)
    return {
        "reference_id": reference.experiment_id,
        "candidate_id": candidate.experiment_id,
        "feature_block": candidate.experiment_id,
        "total_dimension": len(candidate.feature_names),
        "train_rows": candidate.train_shape[0],
        "validation_rows": candidate.evaluation_shape[0],
        "accuracy": float(candidate.metrics["accuracy"]),
        "accuracy_delta": float(
            candidate.metrics["accuracy"] - reference.metrics["accuracy"]
        ),
        "confusion_matrix": candidate.metrics["confusion_matrix"],
        "recall_speaker_0": float(candidate.metrics["recall_speaker_0"]),
        "recall_speaker_1": float(candidate.metrics["recall_speaker_1"]),
        "recall_speaker_0_delta": float(
            candidate.metrics["recall_speaker_0"]
            - reference.metrics["recall_speaker_0"]
        ),
        "recall_speaker_1_delta": float(
            candidate.metrics["recall_speaker_1"]
            - reference.metrics["recall_speaker_1"]
        ),
        "correct_count_delta": int(
            candidate.metrics["correct_count"] - reference.metrics["correct_count"]
        ),
        "reference_wrong_to_candidate_correct": reference_wrong_to_candidate_correct,
        "reference_correct_to_candidate_wrong": reference_correct_to_candidate_wrong,
        "both_correct": int(np.sum(reference_correct & candidate_correct)),
        "both_wrong": int(np.sum((~reference_correct) & (~candidate_correct))),
        "prediction_disagreement_count": int(np.sum(disagreement)),
        "prediction_disagreement_rate": float(np.mean(disagreement)),
        "runtime_seconds": candidate.fit_audit["runtime_seconds"],
        "automatic_promotion": False,
    }


def _diagnostic_slice_row(
    label: str,
    selected: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Calculate the full binary metric contract for one diagnostic slice."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != truth.shape:
        raise ValueError("diagnostic slice mask shape mismatch")
    if not np.any(mask):
        return {
            "slice": label,
            "n": 0,
            "accuracy": np.nan,
            "recall_speaker_0": np.nan,
            "recall_speaker_1": np.nan,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    metrics = calculate_prediction_metrics(truth[mask], prediction[mask])
    return {
        "slice": label,
        "n": int(np.sum(mask)),
        "accuracy": metrics["accuracy"],
        "recall_speaker_0": metrics["recall_speaker_0"],
        "recall_speaker_1": metrics["recall_speaker_1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _error_group_overlap_tables(
    groups: Mapping[str, np.ndarray], *, error_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pair_rows = []
    normalized = {
        name: np.asarray(mask, dtype=bool) for name, mask in groups.items()
    }
    for name, mask in normalized.items():
        count = int(np.sum(mask))
        summary_rows.append(
            {
                "group": name,
                "count": count,
                "fraction_of_49d_errors": (
                    float(count / error_count) if error_count else 0.0
                ),
            }
        )
    names = tuple(normalized)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = normalized[left_name]
            right = normalized[right_name]
            intersection = int(np.sum(left & right))
            union = int(np.sum(left | right))
            left_count = int(np.sum(left))
            pair_rows.append(
                {
                    "group_a": left_name,
                    "group_b": right_name,
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": float(intersection / union) if union else 0.0,
                    "fraction_of_group_a": (
                        float(intersection / left_count) if left_count else 0.0
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def build_frozen_49d_residual_analysis(
    reference_47d: ModelExperimentResult,
    reference_49d: ModelExperimentResult,
    *,
    full_training_tracking: pd.DataFrame,
    high_confidence_threshold: float = 0.90,
) -> dict[str, Any]:
    """Analyze already-observed 49D residuals without extracting or fitting.

    Diagnostic columns are deliberately kept in the returned ledger and never
    passed to a representation-preparation function.
    """

    if tuple(reference_47d.feature_names) != ROUND2_REFERENCE_FEATURE_NAMES:
        raise ValueError("residual analysis requires the exact observed 47D result")
    if tuple(reference_49d.feature_names) != FROZEN_49D_FEATURE_NAMES:
        raise ValueError("residual analysis requires the exact frozen 49D result")
    keys_47d = np.asarray(
        reference_47d.fit_audit["validation_record_keys"]
    ).astype(str)
    keys_49d = np.asarray(
        reference_49d.fit_audit["validation_record_keys"]
    ).astype(str)
    if not np.array_equal(keys_47d, keys_49d):
        raise ValueError("47D/49D record_key alignment mismatch")
    if len(np.unique(keys_49d)) != len(keys_49d):
        raise ValueError("49D evaluation record_key must be one-to-one")
    truth = np.asarray(reference_49d.y_true, dtype=np.int64)
    truth_47d = np.asarray(reference_47d.y_true, dtype=np.int64)
    pred_47d = np.asarray(reference_47d.y_pred, dtype=np.int64)
    pred_49d = np.asarray(reference_49d.y_pred, dtype=np.int64)
    probabilities = np.asarray(reference_49d.probabilities, dtype=np.float64)
    if not np.array_equal(truth, truth_47d):
        raise ValueError("47D/49D label alignment mismatch")
    if pred_47d.shape != truth.shape or pred_49d.shape != truth.shape:
        raise ValueError("47D/49D prediction shape mismatch")
    if probabilities.shape != (len(truth), 2):
        raise ValueError("49D probabilities must have shape (rows, 2)")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
    ):
        raise ValueError("49D probabilities must be finite and sum to one")

    validation_idx = np.asarray(
        reference_49d.fit_audit["validation_idx"], dtype=np.int64
    )
    evaluation_tracking = full_training_tracking.iloc[validation_idx].reset_index(
        drop=True
    )
    required_tracking = {"record_key", "call_id", "startAt_ms", "endAt_ms"}
    missing = sorted(required_tracking - set(evaluation_tracking.columns))
    if missing:
        raise ValueError(f"residual tracking is missing columns: {missing}")
    tracking_keys = evaluation_tracking["record_key"].astype(str).to_numpy()
    if not np.array_equal(tracking_keys, keys_49d):
        raise ValueError("tracking/result record_key alignment mismatch")
    duration_sec = (
        pd.to_numeric(evaluation_tracking["endAt_ms"], errors="raise").to_numpy(
            dtype=np.float64
        )
        - pd.to_numeric(
            evaluation_tracking["startAt_ms"], errors="raise"
        ).to_numpy(dtype=np.float64)
    ) / 1_000.0
    if not np.isfinite(duration_sec).all() or np.any(duration_sec < 0.0):
        raise ValueError("residual duration contains invalid values")
    overlap = _overlap_mask(evaluation_tracking)
    confidence = np.max(probabilities, axis=1)
    probability_speaker_1 = probabilities[:, 1]

    correct_47d = pred_47d == truth
    correct_49d = pred_49d == truth
    persistent = (~correct_47d) & (~correct_49d)
    fixed = (~correct_47d) & correct_49d
    regressed = correct_47d & (~correct_49d)
    both_correct = correct_47d & correct_49d
    transition_type = np.full(len(truth), "both_correct", dtype=object)
    transition_type[persistent] = "persistent"
    transition_type[fixed] = "fixed"
    transition_type[regressed] = "regressed"

    false_positive = (truth == 0) & (pred_49d == 1)
    false_negative = (truth == 1) & (pred_49d == 0)
    high_confidence = confidence >= float(high_confidence_threshold)
    short = duration_sec < 1.0
    groups = {
        "FP": false_positive,
        "FN": false_negative,
        "high_confidence_FP": false_positive & high_confidence,
        "high_confidence_FN": false_negative & high_confidence,
        "short_persistent": short & persistent,
        "overlap_persistent": overlap & persistent,
        "short_overlap_persistent": short & overlap & persistent,
    }
    error_count = int(np.sum(~correct_49d))
    group_summary, group_overlap = _error_group_overlap_tables(
        groups, error_count=error_count
    )

    global_metrics = calculate_prediction_metrics(truth, pred_49d)
    global_metrics.update(
        {
            "false_positive_count": int(np.sum(false_positive)),
            "false_negative_count": int(np.sum(false_negative)),
        }
    )
    duration_masks = (
        ("<0.5s", duration_sec < 0.5),
        ("0.5-1.0s", (duration_sec >= 0.5) & (duration_sec < 1.0)),
        ("1.0-2.0s", (duration_sec >= 1.0) & (duration_sec < 2.0)),
        ("2.0-5.0s", (duration_sec >= 2.0) & (duration_sec < 5.0)),
        (">=5.0s", duration_sec >= 5.0),
    )
    fixed_duration_summary = pd.DataFrame(
        [
            _diagnostic_slice_row(label, mask, truth, pred_49d)
            for label, mask in duration_masks
        ]
    )
    overlap_summary = pd.DataFrame(
        [
            _diagnostic_slice_row("overlap", overlap, truth, pred_49d),
            _diagnostic_slice_row("no_overlap", ~overlap, truth, pred_49d),
        ]
    )
    current_diagnostics = build_result_diagnostics(
        reference_49d, full_training_tracking=full_training_tracking
    )

    error_type = np.full(len(truth), "", dtype=object)
    error_type[false_positive] = "FP"
    error_type[false_negative] = "FN"
    ledger = evaluation_tracking.loc[:, ["record_key", "call_id"]].copy()
    ledger["true_label"] = truth
    ledger["predicted_label"] = pred_49d
    ledger["probability_speaker_1"] = probability_speaker_1
    ledger["confidence"] = confidence
    ledger["duration_sec"] = duration_sec
    ledger["overlap_diagnostic"] = overlap
    ledger["error_type"] = error_type
    ledger["transition_type"] = transition_type
    ledger["persistent"] = persistent
    ledger["fixed"] = fixed
    ledger["regressed"] = regressed
    for name, mask in groups.items():
        ledger[name] = mask
    error_ledger = ledger.loc[~correct_49d].reset_index(drop=True)

    return {
        "global_metrics": global_metrics,
        "fixed_duration_summary": fixed_duration_summary,
        "overlap_summary": overlap_summary,
        "confidence_summary": current_diagnostics["confidence_summary"],
        "high_confidence_error_count": int(
            np.sum((~correct_49d) & high_confidence)
        ),
        "high_confidence_threshold": float(high_confidence_threshold),
        "paired_transitions": {
            "wrong_to_correct": int(np.sum(fixed)),
            "correct_to_wrong": int(np.sum(regressed)),
            "both_wrong": int(np.sum(persistent)),
            "both_correct": int(np.sum(both_correct)),
        },
        "error_group_summary": group_summary,
        "error_group_overlap": group_overlap,
        "ledger": ledger,
        "error_ledger": error_ledger,
    }


def build_result_diagnostics(
    result: ModelExperimentResult,
    *,
    full_training_tracking: pd.DataFrame,
) -> dict[str, Any]:
    """Build duration/overlap/confidence slices on the result's exact split rows."""

    train_idx = np.asarray(result.fit_audit["train_idx"], dtype=np.int64)
    validation_idx = np.asarray(result.fit_audit["validation_idx"], dtype=np.int64)
    training_tracking = full_training_tracking.iloc[train_idx].reset_index(drop=True)
    evaluation_tracking = full_training_tracking.iloc[validation_idx].reset_index(drop=True)
    return build_standalone_diagnostic_report(
        result,
        training_tracking=training_tracking,
        evaluation_tracking=evaluation_tracking,
    )


__all__ = [
    "ALL_CANDIDATE_FEATURE_NAMES",
    "BLOCK_BY_ID",
    "CANONICAL_52D_ACCURACY",
    "CANONICAL_52D_CONFUSION_MATRIX",
    "CANONICAL_52D_FEATURE_NAMES",
    "CANONICAL_52D_PREPARED_ID",
    "CANONICAL_52D_REFERENCE_ID",
    "Canonical52DResolution",
    "FEATURE_BLOCKS",
    "FROZEN_49D_FEATURE_NAMES",
    "FROZEN_49D_REFERENCE_ID",
    "M04_ITER100_ACCURACY_TARGET",
    "M04_ITER100_FEATURE_SCREENING_CONFIG",
    "HISTORICAL_NON_REPRODUCED_52D_ACCURACY",
    "HISTORICAL_NON_REPRODUCED_52D_CONFUSION_MATRIX",
    "PITCH_EXTRACTION_CONFIG",
    "PreparedRepresentation",
    "NEXT_CANDIDATE_BLOCK_IDS",
    "RESIDUAL_COMBINATION_SPECS",
    "RESIDUAL_TARGET_BLOCK_IDS",
    "ROUND1_BLOCK_IDS",
    "ROUND2_BLOCK_IDS",
    "ROUND2_M04_ITER100_ACCURACY_TARGET",
    "ROUND2_REFERENCE_FEATURE_NAMES",
    "ROUND2_REFERENCE_ID",
    "assemble_single_block_matrix",
    "audit_canonical_52d_result",
    "audit_reference_reproduction",
    "benchmark_pitch_extraction",
    "build_candidate_feature_cache",
    "build_frozen_49d_residual_analysis",
    "build_reference_contract",
    "build_result_diagnostics",
    "candidate_cache_paths",
    "candidate_hypothesis_table",
    "candidate_manifest_table",
    "compare_feature_candidate_to_reference",
    "config_fingerprint",
    "extract_acoustic_blocks",
    "implemented_candidate_definition_table",
    "load_candidate_feature_cache",
    "normalized_spectral_flux",
    "prepare_single_block_screening",
    "prepare_round2_single_block_screening",
    "prepare_frozen_49d_single_block_screening",
    "prepare_residual_combination_screening",
    "run_frozen_49d_screening_candidate",
    "run_frozen_49d_screening_reference",
    "run_round2_feature_screening_candidate",
    "run_round2_feature_screening_reference",
    "run_feature_screening_candidate",
    "run_feature_screening_reference",
    "resolve_canonical_52d_reference",
    "validate_canonical_52d_prepared",
    "validate_model_feature_names",
]
