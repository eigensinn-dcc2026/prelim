"""Sequential, Training-only acoustic feature grid for Mission 2.

Importing this module does not read competition data, inspect an Official
Validation artifact, install packages, or train a model. Every execution path
requires the persisted Stage 6 Internal Validation membership and fails closed
if row, label, call, split, or artifact identity differs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import uuid

import numpy as np
import pandas as pd


SEED = 42
THRESHOLD = 0.5
STAGE6_REFERENCE_RELATIVE = Path(
    "data/processed/mission2_complex_tabular_screen/"
    "20261006T031510Z_a9a01a2d"
)
STAGE11_REFERENCE_RELATIVE = Path(
    "data/processed/mission2_time_frequency_screen/20261007T141754Z_7b1e9644"
)
STAGE12_COMPUTE_GATE_RELATIVE = Path(
    "data/processed/mission2_raw_waveform_pretrained_speech/"
    "stage12_wavlm_canonical_reassessment_20261008T043852Z_fac003a8"
)
STAGE13_REFERENCE_RELATIVE = Path(
    "data/processed/mission2_stage13_untested_model_family_first_screen/"
    "20261009T051303Z_7d149a2b"
)
EGEMAPS_88D_RELATIVE = Path(
    "data/processed/mission2_egemaps_colab_v1/training_egemaps_v02_88d.npz"
)
GRID_RELATIVE = Path("data/processed/mission2_acoustic_feature_grid")
COLAB_SOURCE_RUN_ID = "20261010T012333Z_8de41728"
COLAB_OUTPUT_DIR_NAME = "mission2_grid_runs"


def resolve_colab_drive_layout(mydrive_root):
    """Resolve the confirmed MyDrive/prelim shortcut without guessing paths.

    The Training archive folder is returned lexically and is checked separately
    before any ZIP path is resolved, so a broken Drive shortcut stays visible
    as an access failure instead of collapsing to a misleading root path.
    """
    mydrive = Path(mydrive_root).expanduser()
    if mydrive.name != "MyDrive" or not mydrive.is_dir():
        raise FileNotFoundError(f"Expected the mounted Google Drive MyDrive root: {mydrive}")
    mydrive = mydrive.resolve(strict=True)

    project_shortcut = mydrive / "prelim"
    if not project_shortcut.is_dir():
        raise FileNotFoundError(
            f"The expected MyDrive/prelim shared-project shortcut is unavailable: {project_shortcut}"
        )
    repo_root = project_shortcut.resolve(strict=True)
    module_path = repo_root / "src" / "dcc_data" / "mission2_acoustic_feature_grid.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Resolved prelim target is not the Mission 2 repository: {repo_root}")

    source_run = repo_root / GRID_RELATIVE / COLAB_SOURCE_RUN_ID
    if not source_run.is_dir() or not (source_run / "contract.json").is_file() \
            or not (source_run / "run_ledger.json").is_file():
        raise FileNotFoundError(
            f"The configured prior Feature Grid run is incomplete or missing: {source_run}"
        )
    return {
        "mydrive_root": mydrive,
        "project_shortcut": project_shortcut,
        "repo_root": repo_root,
        "source_run_root": source_run.resolve(strict=True),
        "training_archive_root": repo_root / "data" / "raw_original" / "대학부 데이터",
        "output_root": mydrive / COLAB_OUTPUT_DIR_NAME,
    }

F04_NAMES = (
    "slopeV0-500_sma3nz_amean",
    "slopeV0-500_sma3nz_stddevNorm",
    "slopeV500-1500_sma3nz_amean",
    "slopeV500-1500_sma3nz_stddevNorm",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
)
F05_NAMES = (
    "spectralFluxV_sma3nz_amean",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "mfcc2V_sma3nz_amean",
    "mfcc2V_sma3nz_stddevNorm",
    "mfcc3V_sma3nz_amean",
    "mfcc3V_sma3nz_stddevNorm",
    "mfcc4V_sma3nz_amean",
    "mfcc4V_sma3nz_stddevNorm",
    "spectralFluxUV_sma3nz_amean",
)
_F01_NAMES = tuple(
    f"mfcc_{index:02d}_{stat}"
    for index in range(1, 21)
    for stat in ("q10", "median", "q90", "iqr")
)
_F02_NAMES = tuple(
    f"spectral_contrast_band{band}_{stat}"
    for band in range(5)
    for stat in ("mean", "std")
)
_F03_NAMES = tuple(
    f"spectral_power_ratio_{band}_{stat}"
    for band in ("0_500", "500_1000", "1000_2000", "2000_4000")
    for stat in ("mean", "std")
)
_F06_NAMES = ("delta_mfcc_norm_mean", "delta_mfcc_norm_std")
_F07_NAMES = ("early_middle_distance", "middle_late_distance", "early_late_distance")


@dataclass(frozen=True)
class FeatureBlock:
    """Immutable public feature schema; config_json keeps nested config frozen."""

    block_id: str
    feature_names: tuple[str, ...]
    family: str
    rationale: str
    config_json: str

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def config(self) -> dict:
        return json.loads(self.config_json)


def _block(block_id, names, family, rationale, config):
    return FeatureBlock(block_id, tuple(names), family, rationale,
                        json.dumps(config, sort_keys=True, separators=(",", ":")))


FEATURE_BLOCKS = (
    _block("F01", _F01_NAMES, "MFCC distribution", "MFCC20 level and spread quantiles beyond mean/std.",
           {"q": [0.10, 0.50, 0.90], "spread": "q75-q25", "dimension": 80}),
    _block("F02", _F02_NAMES, "spectral contrast", "Five octave bands cover 0-4 kHz, including 2-4 kHz.",
           {"fmin_hz": 250, "n_bands": 4, "quantile": 0.02, "linear": False,
            "bands_hz": [[0, 250], [250, 500], [500, 1000], [1000, 2000], [2000, 4000]],
            "stft": {"n_fft": 512, "win_length": 512, "hop_length": 160,
                     "window": "hann", "center": True, "pad_mode": "constant"},
            "functionals": ["mean", "population_std"], "dimension": 10}),
    _block("F03", _F03_NAMES, "spectral power ratios", "Framewise power proportions in four contiguous Nyquist-complete bands.",
           {"bands_hz": [[0, 500], [500, 1000], [1000, 2000], [2000, 4000]],
            "stft": {"n_fft": 512, "win_length": 512, "hop_length": 160,
                     "window": "hann", "center": True, "pad_mode": "constant"},
            "zero_energy_frame_policy": "all band ratios zero and count recorded",
            "functionals": ["mean", "population_std"], "dimension": 8}),
    _block("F04", F04_NAMES, "eGeMAPSv02 slopes", "Low/mid-band voiced and unvoiced spectral slope functionals.",
           {"source_schema": "Training eGeMAPSv02 88D cache", "dimension": 6}),
    _block("F05", F05_NAMES, "eGeMAPSv02 spectral dynamics", "Voiced MFCC1-4 and spectral-flux level/spread plus unvoiced flux mean.",
           {"source_schema": "Training eGeMAPSv02 88D cache", "dimension": 11}),
    _block("F06", _F06_NAMES, "AF05 delta MFCC dynamics", "Reuses the existing AF05 matrix-level equation and short-frame fallback.",
           {"helper": "_delta_mfcc_features_from_matrix", "fallback_frames_lt": 3, "dimension": 2}),
    _block("F07", _F07_NAMES, "AF12 temporal MFCC contrast", "Reuses existing AF12 matrix-level region distances and short-crop fallback.",
           {"helper": "_temporal_mfcc_contrast_features_from_matrix", "fallback_samples_lt": 480,
            "dimension": 3}),
)
BLOCK_BY_ID = {block.block_id: block for block in FEATURE_BLOCKS}
MODEL_IDS = ("tabm", "ft_transformer", "tabular_resnet", "dcnv2", "gandalf")
COLAB_CANDIDATE_MODEL_IDS = ("tabm", "tabular_resnet", "dcnv2", "gandalf")

EXPECTED_SPLIT = {
    "total_rows": 889_358,
    "train_rows": 711_467,
    "validation_rows": 177_891,
    "train_calls": 23_360,
    "validation_calls": 5_840,
    "call_overlap": 0,
}


def experiment_ids() -> tuple[str, ...]:
    """Five clean baselines first, then seven single additions per model."""
    return tuple([f"baseline_{model}" for model in MODEL_IDS] + [
        f"{model}_{block.block_id}" for model in MODEL_IDS for block in FEATURE_BLOCKS
    ])


def user_scope_excluded_experiment_ids() -> tuple[str, ...]:
    """The user-approved runtime reduction omits only FT feature additions."""
    return tuple(f"ft_transformer_F0{index}" for index in range(1, 8))


def active_experiment_ids() -> tuple[str, ...]:
    """Return the 33 fits in scope while retaining the original 40-ID order."""
    excluded = set(user_scope_excluded_experiment_ids())
    return tuple(name for name in experiment_ids() if name not in excluded)


def candidate_experiment_ids() -> tuple[str, ...]:
    """The 28 new fits; the five frozen baselines are verified and reused."""
    return tuple(f"{model}_{block.block_id}"
                 for model in COLAB_CANDIDATE_MODEL_IDS
                 for block in FEATURE_BLOCKS)


def training_archive_paths(archive_root) -> tuple[Path, ...]:
    """Resolve the one confirmed ZIP folder without following a broken shortcut."""
    root = Path(archive_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            "Training ZIP folder is unavailable in this Colab account: "
            f"{root}. Expected the shared Drive shortcut at "
            "prelim/data/raw_original/대학부 데이터; check its access before staging."
        )
    paths = tuple(
        root / f"대학부 데이터-20260901T090511Z-1-{number:03d}.zip"
        for number in range(1, 13)
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Training ZIP folder is visible but {len(missing)}/12 expected archives are missing: "
            + ", ".join(missing[:3])
        )
    return paths


def _scope_amendment(root: Path, *, required=False):
    """Verify the additive user-scope record against the untouched run contract."""
    path = Path(root) / "scope_amendment.json"
    if not path.is_file():
        if required:
            raise ValueError("Approved scope amendment is missing")
        return None
    amendment = _read_json(path)
    expected = {
        "schema_version": "mission2_acoustic_feature_grid_scope_amendment_v1",
        "contract_sha256": _sha256_file(Path(root) / "contract.json"),
        "excluded_experiments": list(user_scope_excluded_experiment_ids()),
        "scheduled_fit_count": len(active_experiment_ids()),
        "single_feature_candidate_count": len(active_experiment_ids()) - len(MODEL_IDS),
        "baseline_ft_transformer_retained": True,
        "reason": "SKIP FT-Transformer F01-F07 feature additions by user-approved scope; preserve its completed 59D baseline.",
    }
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError("User-scope amendment differs from the approved exclusion or frozen contract")
    return amendment


def apply_user_scope_exclusions(run_root):
    """Record the approved seven-run exclusion without rewriting the frozen contract."""
    root = Path(run_root).resolve()
    contract_path = root / "contract.json"
    ledger_path = root / "run_ledger.json"
    contract = _read_json(contract_path)
    ledger = _read_json(ledger_path)
    if (contract.get("schema_version") != "mission2_acoustic_feature_grid_v1"
            or tuple(contract.get("experiments", ())) != experiment_ids()):
        raise ValueError("Scope change requires the original frozen 40-experiment contract")
    if set(ledger.get("experiments", {})) != set(experiment_ids()):
        raise ValueError("Run ledger experiment IDs differ from the frozen contract")
    if ledger.get("status") == "RUNNING":
        raise RuntimeError("Cannot amend experiment scope while a fit is marked RUNNING")

    excluded = user_scope_excluded_experiment_ids()
    existing_amendment = _scope_amendment(root)
    if existing_amendment is None:
        for experiment_id in excluded:
            if (ledger["experiments"].get(experiment_id) != "PENDING"
                    or (root / experiment_id).exists()):
                raise RuntimeError(
                    f"Cannot exclude {experiment_id}: it has a non-pending status or artifacts"
                )
        amendment = {
            "schema_version": "mission2_acoustic_feature_grid_scope_amendment_v1",
            "contract_sha256": _sha256_file(contract_path),
            "excluded_experiments": list(excluded),
            "scheduled_fit_count": len(active_experiment_ids()),
            "single_feature_candidate_count": len(active_experiment_ids()) - len(MODEL_IDS),
            "baseline_ft_transformer_retained": True,
            "reason": "SKIP FT-Transformer F01-F07 feature additions by user-approved scope; preserve its completed 59D baseline.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_new(root / "scope_amendment.json", amendment)
    else:
        amendment = existing_amendment

    for experiment_id in excluded:
        status = ledger["experiments"].get(experiment_id)
        if status not in ("PENDING", "SKIPPED_BY_USER_SCOPE"):
            raise RuntimeError(f"Approved exclusion has unexpected status: {experiment_id}={status}")
        if (root / experiment_id).exists():
            raise RuntimeError(f"Approved exclusion has unexpected artifacts: {experiment_id}")
        ledger["experiments"][experiment_id] = "SKIPPED_BY_USER_SCOPE"

    complete = [name for name, value in ledger["experiments"].items() if value == "COMPLETE"]
    ledger["completed_count"] = len(complete)
    ledger["skipped_count"] = len(excluded)
    pending = [name for name in active_experiment_ids()
               if ledger["experiments"].get(name) == "PENDING"]
    ledger["next_experiment"] = pending[0] if pending else None
    if (ledger.get("status") == "FAILED_STOP"
            or any(value == "FAILED" for value in ledger["experiments"].values())):
        ledger["status"] = "FAILED_STOP"
    elif not pending and len(complete) == len(active_experiment_ids()):
        ledger["status"] = "ALL_33_COMPLETE_WITH_7_SKIPPED"
    else:
        ledger["status"] = "READY"
    _write_json_replace(ledger_path, ledger)
    return {"status": ledger["status"], "scheduled_fit_count": len(active_experiment_ids()),
            "completed_count": len(complete), "skipped_count": len(excluded),
            "next_experiment": ledger["next_experiment"],
            "scope_amendment_sha256": _sha256_file(root / "scope_amendment.json")}


def stage10_to13_prior_run_roots(repo_root):
    """Return known historical roots; every result is contract-gated on load."""
    root = Path(repo_root).resolve()
    family_root = root / "data/processed/mission2_model_family_screen"
    discovered_stage10 = sorted(
        path for path in family_root.iterdir()
        if path.is_dir() and (path / "contract.json").is_file()
    ) if family_root.is_dir() else []
    return tuple([
        root / STAGE6_REFERENCE_RELATIVE,
        *discovered_stage10,
        root / STAGE11_REFERENCE_RELATIVE,
        root / STAGE12_COMPUTE_GATE_RELATIVE,
        root / STAGE13_REFERENCE_RELATIVE,
    ])


def _array_fingerprint(values: np.ndarray) -> str:
    """Match the Stage 6 dtype/shape/C-order SHA-256 identity contract."""
    original = np.asarray(values)
    if original.dtype.hasobject:
        raise TypeError("array fingerprint does not support object dtype")
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(str(original.dtype).encode("ascii"))
    digest.update(json.dumps(original.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _record_key_signature(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_fixed_internal_split_manifest(
    record_keys,
    call_ids,
    labels,
    persisted_validation_predictions,
    persisted_contract,
    *,
    expected_split=EXPECTED_SPLIT,
):
    """Load Stage 6 Validation keys, derive only their Training complement.

    The prediction snapshots are treated as a persisted membership manifest,
    not as model results. No splitter is called here.
    """
    source_keys = np.asarray(record_keys)
    source_calls = np.asarray(call_ids)
    keys = source_keys.astype(str)
    calls = source_calls.astype(str)
    y = np.asarray(labels)
    if keys.ndim != 1 or calls.shape != keys.shape or y.shape != keys.shape or not len(keys):
        raise ValueError("Training record_key/call_id/label vectors must align")
    if len(np.unique(keys)) != len(keys) or np.any(np.char.str_len(keys) == 0):
        raise ValueError("Training record_key identity is empty or duplicated")
    if np.any(np.char.str_len(calls) == 0) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Training call_id or binary label identity is invalid")
    if not isinstance(persisted_contract, dict):
        raise ValueError("Persisted Stage 6 contract is missing or invalid")
    if (persisted_contract.get("source_split") != "Training"
            or persisted_contract.get("evaluation_set") != "fixed_internal_validation"
            or persisted_contract.get("seed") != SEED
            or persisted_contract.get("threshold") != THRESHOLD
            or persisted_contract.get("official_validation_used") is not False):
        raise ValueError("Persisted split contract source/seed/threshold/Official gate differs")
    contract_split = persisted_contract.get("split")
    contract_identity = persisted_contract.get("identity")
    if not isinstance(contract_split, dict) or not isinstance(contract_identity, dict):
        raise ValueError("Persisted split counts or identity hashes are missing")
    if (contract_split.get("seed") != SEED or contract_split.get("test_size") != 0.2
            or contract_split.get("call_overlap") != 0):
        raise ValueError("Persisted split seed/test_size/call-overlap differs")
    for name, value in dict(expected_split).items():
        if contract_split.get(name) != value:
            raise ValueError(f"Persisted split count differs: {name}")
    required_hashes = ("train_keys", "validation_keys", "train_y", "validation_y",
                       "source_y", "source_keys", "source_calls")
    if any(not isinstance(contract_identity.get(name), str) for name in required_hashes):
        raise ValueError("Persisted split identity hash is missing")
    if not isinstance(persisted_validation_predictions, (tuple, list)) or len(persisted_validation_predictions) != 2:
        raise ValueError("Two independent Stage 6 Validation prediction snapshots are required")
    frames = []
    for frame in persisted_validation_predictions:
        if not isinstance(frame, pd.DataFrame) or not {"record_key", "call_id", "y_true"}.issubset(frame.columns):
            raise ValueError("Persisted Validation membership snapshot schema differs")
        frame = frame.loc[:, ["record_key", "call_id", "y_true"]].copy()
        frame["record_key"] = frame.record_key.astype(str)
        frame["call_id"] = frame.call_id.astype(str)
        raw_labels = pd.to_numeric(frame.y_true, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, (0, 1)).all():
            raise ValueError("Persisted Validation labels are not exact binary values")
        labels_frame = raw_labels.astype(y.dtype)
        frame["y_true"] = labels_frame
        if not frame.record_key.is_unique or frame.record_key.str.len().eq(0).any():
            raise ValueError("Persisted Validation keys are empty or duplicated")
        frames.append(frame)
    first, second = frames
    for name in ("record_key", "call_id", "y_true"):
        if not np.array_equal(first[name].to_numpy(), second[name].to_numpy()):
            raise ValueError(f"Stage 6 Validation snapshots disagree on {name} identity")
    val_keys = first.record_key.to_numpy(dtype=str)
    val_calls = first.call_id.to_numpy(dtype=str)
    val_y = first.y_true.to_numpy(dtype=y.dtype)
    if len(val_keys) != expected_split["validation_rows"] or np.any(np.char.str_len(val_calls) == 0):
        raise ValueError("Persisted Validation row/call membership counts differ")
    key_to_index = {key: index for index, key in enumerate(keys)}
    if any(key not in key_to_index for key in val_keys):
        raise ValueError("Persisted Validation contains a missing/non-Training record_key")
    validation_idx = np.sort(np.fromiter((key_to_index[key] for key in val_keys), dtype=np.int64))
    train_mask = np.ones(len(keys), dtype=bool)
    train_mask[validation_idx] = False
    train_idx = np.flatnonzero(train_mask)
    # Require the saved rows' supplied order to be the exact source-order subset;
    # this prevents silently sorting or repairing a corrupted prediction file.
    source_order_val_keys = keys[validation_idx]
    if not np.array_equal(val_keys, source_order_val_keys):
        raise ValueError("Persisted Validation manifest order differs from Training source order")
    if not np.array_equal(calls[validation_idx], val_calls) or not np.array_equal(y[validation_idx], val_y):
        raise ValueError("Persisted Validation call_id/label differs from Training source")
    train_calls, validation_calls = set(calls[train_idx]), set(calls[validation_idx])
    overlap = train_calls & validation_calls
    audit = {
        "seed": SEED,
        "test_size": 0.2,
        "total_rows": int(len(keys)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "train_calls": int(len(train_calls)),
        "validation_calls": int(len(validation_calls)),
        "call_overlap": int(len(overlap)),
        "manifest_rows": int(len(val_keys)),
        "manifest_record_key_signature": _record_key_signature(val_keys),
        "official_validation_used": False,
    }
    if overlap:
        raise ValueError("Persisted Internal Split has call_id overlap")
    for name in ("total_rows", "train_rows", "validation_rows", "train_calls", "validation_calls", "call_overlap"):
        if audit[name] != expected_split[name]:
            raise ValueError(f"Current Training membership count differs: {name}")
    observed_hashes = {
        # The saved Stage 6 context cast each selected object-key array to a
        # minimally sized Unicode dtype. Preserve that dtype behavior for the
        # subset hashes while source_keys remains the full-source Unicode hash.
        "train_keys": _array_fingerprint(source_keys[train_idx].astype(str)),
        "validation_keys": _array_fingerprint(source_keys[validation_idx].astype(str)),
        "train_y": _array_fingerprint(y[train_idx]),
        "validation_y": _array_fingerprint(y[validation_idx]),
        "source_y": _array_fingerprint(y),
        "source_keys": _array_fingerprint(source_keys.astype(str)),
        "source_calls": _array_fingerprint(calls),
    }
    if observed_hashes != {name: contract_identity[name] for name in required_hashes}:
        differing = [name for name in required_hashes if observed_hashes[name] != contract_identity[name]]
        raise ValueError(f"Persisted Stage 6 row/key/label/call hashes differ: {differing}")
    audit["identity_hashes"] = observed_hashes
    return train_idx, validation_idx, audit


def _mono_crop(waveform, sample_rate):
    if sample_rate != 8_000:
        raise ValueError(f"Mission 2 current-utterance crop must be native 8 kHz, got {sample_rate}")
    signal = np.asarray(waveform)
    if signal.ndim == 2:
        # Existing WAV reader returns (channels, frames); mono averaging is the
        # same channel rule used by the clean-59D MFCC builder.
        if signal.shape[0] <= 8:
            signal = signal.mean(axis=0)
        elif signal.shape[1] <= 8:
            signal = signal.mean(axis=1)
        else:
            raise ValueError("Ambiguous multichannel waveform layout")
    if signal.ndim != 1 or signal.size == 0 or not np.issubdtype(signal.dtype, np.number):
        raise ValueError("Current-utterance crop must be a non-empty numeric mono vector")
    signal = signal.astype(np.float32, copy=False)
    if not np.isfinite(signal).all():
        raise ValueError("Current-utterance waveform contains NaN/Inf")
    return signal


def _clean_mfcc_matrix(signal):
    """Explicit equivalent of the clean 59D librosa MFCC frontend."""
    import librosa
    return librosa.feature.mfcc(
        y=signal, sr=8_000, n_mfcc=20, n_fft=512, win_length=512,
        hop_length=160, window="hann", center=True, pad_mode="constant",
        power=2.0, n_mels=128, fmin=0.0, fmax=4_000.0, htk=False,
        mel_norm="slaney", dct_type=2, norm="ortho", lifter=0,
    )


def _spectrum(signal):
    import librosa
    stft = librosa.stft(signal, n_fft=512, win_length=512, hop_length=160,
                        window="hann", center=True, pad_mode="constant")
    return np.abs(stft), np.square(np.abs(stft), dtype=np.float64)


def _waveform_vectors(signal):
    import librosa
    from . import mission2_acoustic_feature_expansion as acoustic

    mfcc = _clean_mfcc_matrix(signal)
    if mfcc.shape[0] != 20 or mfcc.shape[1] == 0 or not np.isfinite(mfcc).all():
        raise ValueError("Clean frontend MFCC matrix is empty, malformed, or non-finite")
    mfcc64 = np.asarray(mfcc, dtype=np.float64)
    quantiles = np.quantile(mfcc64, [0.10, 0.50, 0.90], axis=1)
    f01 = np.stack((quantiles[0], quantiles[1], quantiles[2],
                    np.quantile(mfcc64, 0.75, axis=1) - np.quantile(mfcc64, 0.25, axis=1)), axis=1).reshape(-1)

    magnitude, power = _spectrum(signal)
    contrast = librosa.feature.spectral_contrast(
        S=magnitude, sr=8_000, n_fft=512, hop_length=160,
        fmin=250.0, n_bands=4, quantile=0.02, linear=False,
    )
    if contrast.shape[0] != 5 or contrast.shape[1] != magnitude.shape[1]:
        raise ValueError("Spectral contrast did not produce five Nyquist-complete bands")
    f02 = np.stack((contrast.mean(axis=1), contrast.std(axis=1, ddof=0)), axis=1).reshape(-1)

    frequencies = np.fft.rfftfreq(512, d=1 / 8_000)
    band_edges = ((0.0, 500.0), (500.0, 1_000.0), (1_000.0, 2_000.0), (2_000.0, 4_000.0))
    total = power.sum(axis=0)
    valid = total > 0.0
    ratios = np.zeros((4, power.shape[1]), dtype=np.float64)
    for band_index, (lower, upper) in enumerate(band_edges):
        selected = (frequencies >= lower) & ((frequencies <= upper) if upper == 4_000 else (frequencies < upper))
        ratios[band_index] = np.divide(power[selected].sum(axis=0), total,
                                       out=np.zeros_like(total), where=valid)
    f03 = np.stack((ratios.mean(axis=1), ratios.std(axis=1, ddof=0)), axis=1).reshape(-1)

    f06_values = acoustic._delta_mfcc_features_from_matrix(mfcc)
    if signal.size < 480:
        f07_values = {name: np.float32(0.0) for name in _F07_NAMES}
    else:
        f07_values = acoustic._temporal_mfcc_contrast_features_from_matrix(mfcc)
    blocks = {
        "F01": np.asarray(f01, dtype=np.float32),
        "F02": np.asarray(f02, dtype=np.float32),
        "F03": np.asarray(f03, dtype=np.float32),
        "F06": np.asarray([f06_values[name] for name in _F06_NAMES], dtype=np.float32),
        "F07": np.asarray([f07_values[name] for name in _F07_NAMES], dtype=np.float32),
    }
    details = {
        "mfcc_frames": int(mfcc.shape[1]),
        "zero_energy_frames": int((~valid).sum()),
        "af05_short_fallback": bool(mfcc.shape[1] < 3),
        "af12_short_fallback": bool(signal.size < 480 or mfcc.shape[1] < 3),
    }
    for block_id, vector in blocks.items():
        if vector.shape != (BLOCK_BY_ID[block_id].dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"{block_id} produced invalid dimensions or values")
    return blocks, details


def extract_waveform_blocks(crop, sample_rate: int) -> dict[str, np.ndarray]:
    """Return only waveform-derived F01/F02/F03/F06/F07 values in frozen order."""
    signal = _mono_crop(crop, sample_rate)
    blocks, _ = _waveform_vectors(signal)
    return blocks


def select_egemaps_blocks(matrix, names, record_keys, expected_keys):
    """Strictly select F04/F05 from the ordered, exact 88D Training schema."""
    values = np.asarray(matrix)
    schema = tuple(str(name) for name in names)
    keys = np.asarray(record_keys).astype(str)
    expected = np.asarray(expected_keys).astype(str)
    if schema != tuple(names) or len(schema) != 88 or len(set(schema)) != 88:
        raise ValueError("Stored eGeMAPSv02 schema must be exactly 88 unique ordered names")
    if values.shape != (len(keys), 88) or not np.isfinite(values).all():
        raise ValueError("Stored eGeMAPSv02 matrix shape/finite contract differs")
    if not np.array_equal(keys, expected):
        raise ValueError("eGeMAPSv02 record_key order differs from clean Training source")
    if set(F04_NAMES) & set(F05_NAMES):
        raise AssertionError("F04/F05 frozen columns overlap")
    selected = {}
    for block_id, block_names in (("F04", F04_NAMES), ("F05", F05_NAMES)):
        if any(schema.count(name) != 1 for name in block_names):
            raise ValueError(f"{block_id} columns do not occur exactly once in eGeMAPSv02 cache")
        positions = [schema.index(name) for name in block_names]
        selected[block_id] = np.asarray(values[:, positions], dtype=np.float32)
        if selected[block_id].shape != (len(keys), BLOCK_BY_ID[block_id].dimension):
            raise ValueError(f"{block_id} selected dimensions differ")
    return selected


def validate_feature_block(block_id, matrix, keys, *, expected_keys=None, fallback_counts=None):
    if block_id not in BLOCK_BY_ID:
        raise KeyError(block_id)
    values = np.asarray(matrix)
    record_keys = np.asarray(keys).astype(str)
    expected = record_keys if expected_keys is None else np.asarray(expected_keys).astype(str)
    if (values.ndim != 2 or values.shape != (len(record_keys), BLOCK_BY_ID[block_id].dimension)
            or values.dtype != np.float32 or not np.array_equal(record_keys, expected)
            or len(np.unique(record_keys)) != len(record_keys)):
        raise ValueError(f"{block_id} row, dtype, dimension, or key alignment contract differs")
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(f"{block_id} contains NaN/Inf; values are never imputed")
    return {
        "block_id": block_id, "rows": int(len(values)), "dimension": int(values.shape[1]),
        "dtype": str(values.dtype), "non_finite_values": int((~finite).sum()),
        "constant_columns": [BLOCK_BY_ID[block_id].feature_names[i]
                             for i, column in enumerate(values.T) if np.ptp(column) == 0],
        "fallback_counts": dict(fallback_counts or {}),
        "record_key_signature": _record_key_signature(record_keys),
    }


@dataclass
class TrainingContext:
    """Aligned Training-only rows and the persisted Internal Split indices."""

    clean_X: np.ndarray
    y: np.ndarray
    record_keys: np.ndarray
    call_ids: np.ndarray
    feature_names_59d: tuple[str, ...]
    train_idx: np.ndarray
    validation_idx: np.ndarray
    split_audit: dict
    clean_fingerprint: str
    egemaps_blocks: dict[str, np.ndarray]
    egemaps_audit: dict
    manifest: pd.DataFrame
    stage6_root: Path
    prior_predictions: dict[str, pd.DataFrame]
    stage6_scaler: dict
    feature_matrices: dict[str, np.ndarray] | None = None

    @property
    def validation_record_keys(self):
        return self.record_keys[self.validation_idx]

    @property
    def validation_call_ids(self):
        return self.call_ids[self.validation_idx]

    @property
    def validation_labels(self):
        return self.y[self.validation_idx]


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON contract: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON contract must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage6_validation_snapshots(stage6_root: Path):
    contract_path = stage6_root / "contract.json"
    reference_a_path = stage6_root / "stage6_reference" / "mlp_reference_predictions.parquet"
    reference_b_path = stage6_root / "stage6_reference" / "stronger_mlp_predictions.parquet"
    scaler_path = stage6_root / "scaler.json"
    raw_contract = _read_json(contract_path)
    contract = _normalized_prior_contract(raw_contract)
    stage6_metadata = raw_contract.get("stage6")
    if not isinstance(stage6_metadata, dict) or not isinstance(stage6_metadata.get("hashes"), dict):
        raise ValueError("Pinned Stage 6 prediction hashes are missing")
    stage6_hashes = stage6_metadata["hashes"]
    reference_hashes = (
        (reference_a_path, "candidates/mlp_reference/validation_predictions.parquet"),
        (reference_b_path, "candidates/stronger_mlp/validation_predictions.parquet"),
    )
    for path in (reference_a_path, reference_b_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(f"Persisted Stage 6 split snapshot is missing: {path}")
    observed_reference_hashes = {}
    for path, relative_path in reference_hashes:
        expected_hash = stage6_hashes.get(relative_path)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Pinned Stage 6 prediction hash is missing: {relative_path}")
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Stage 6 reference prediction hash differs: {relative_path}")
        observed_reference_hashes[relative_path] = observed_hash
    return contract, pd.read_parquet(reference_a_path), pd.read_parquet(reference_b_path), {
        "contract_sha256": _sha256_file(contract_path),
        "stage6_reference_a_sha256": observed_reference_hashes[reference_hashes[0][1]],
        "stage6_reference_b_sha256": observed_reference_hashes[reference_hashes[1][1]],
        "scaler_sha256": _sha256_file(scaler_path),
    }


def load_training_context(
    repo_root,
    raw_root,
    split_manifest_path=None,
    prior_run_roots=(),
    run_root=None,
    *,
    stage6_root=None,
    egemaps_cache_path=None,
    transfer_identity_path=None,
):
    """Load Training-only clean/eGeMAPS data and the persisted Stage 6 split.

    ``split_manifest_path`` is the Training manifest Parquet. The Internal
    Validation membership comes from the saved Stage 6 prediction rows and
    contract; no GroupShuffleSplit or alternate split is generated here.
    """
    from . import mission2_clean_stage4 as clean_stage4
    from . import mission2_model_experiments as models
    from . import mission2_model_family_screen as screen
    from . import mission2_standard_representation as standard

    root = Path(repo_root).resolve()
    raw = Path(raw_root).resolve()
    paths = models.standalone_training_cache_paths(root)
    manifest_path = Path(split_manifest_path) if split_manifest_path else paths.manifest_parquet
    stage6 = Path(stage6_root) if stage6_root else root / STAGE6_REFERENCE_RELATIVE
    egemaps_path = Path(egemaps_cache_path) if egemaps_cache_path else root / EGEMAPS_88D_RELATIVE
    manifest_columns = [
        "split", "record_key", "call_id", "annotation_row", "speaker", "audio_path",
        "json_path", "crop_start_frame", "crop_end_frame", "crop_ok", "startAt_ms", "endAt_ms",
    ]
    # Parquet predicate pushdown reads only organizer Training manifest rows.
    manifest = pd.read_parquet(manifest_path, columns=manifest_columns,
                               filters=[("split", "==", "Training")])
    source = clean_stage4._source_rows(
        manifest, expected_rows=EXPECTED_SPLIT["total_rows"],
        expected_calls=EXPECTED_SPLIT["train_calls"] + EXPECTED_SPLIT["validation_calls"],
    )
    keys = source.record_key.astype(str).to_numpy()
    calls = source.call_id.astype(str).to_numpy()
    labels = source.speaker.to_numpy(dtype=np.int64)
    stage6_contract, tabm_snapshot, resnet_snapshot, stage6_hashes = _stage6_validation_snapshots(stage6)
    train_idx, validation_idx, split_audit = load_fixed_internal_split_manifest(
        keys, calls, labels, (tabm_snapshot, resnet_snapshot), stage6_contract,
    )
    clean = clean_stage4.load_clean_stage4_59d(
        root / screen.CLEAN_CHAIN_RELATIVE, manifest_path, paths.manifest_metadata,
        raw, transfer_identity_path=transfer_identity_path,
        fixed_validation_record_keys=keys[validation_idx],
    )
    if transfer_identity_path is not None:
        import librosa

        cached_versions = clean.raw_source_verification["cache_dependency_versions"]
        current_frontend_versions = {"librosa": librosa.__version__, "numpy": np.__version__}
        if any(cached_versions[name] != version
               for name, version in current_frontend_versions.items()):
            raise ValueError(
                "Runtime librosa/numpy versions differ from the frozen Clean 59D frontend: "
                f"cached={cached_versions}, current={current_frontend_versions}"
            )
    if (not np.array_equal(clean.record_keys.astype(str), keys)
            or not np.array_equal(clean.call_ids.astype(str), calls)
            or not np.array_equal(clean.y, labels)
            or clean.provenance_fingerprint != screen.CLEAN_FINGERPRINT):
        raise ValueError("Training clean-59D data differs from the fixed-split source rows")
    # Freeze object-backed Parquet strings as Unicode before hashing/saving the
    # prepared context; the Stage 6 fingerprints use the same explicit cast.
    keys = keys.astype(str)
    calls = calls.astype(str)
    stage6_scaler = _read_json(stage6 / "scaler.json")
    if (stage6_scaler.get("fit_scope") != "internal_train_only"
            or stage6_scaler.get("feature_names") != list(clean.feature_names)
            or stage6_scaler.get("n_samples_seen") != len(train_idx)):
        raise ValueError("Persisted Stage 6 scaler training-scope/schema differs")
    from sklearn.preprocessing import StandardScaler
    clean_scaler = StandardScaler().fit(clean.X[train_idx])
    for name in ("mean", "scale", "var"):
        if not np.array_equal(getattr(clean_scaler, name + "_"), np.asarray(stage6_scaler[name])):
            raise ValueError(f"Persisted Stage 6 clean-59D scaler differs: {name}")
    egemaps = standard.load_egemaps_cache(
        egemaps_path, keys, calls, source_provenance=clean.provenance_fingerprint,
    )
    runtime_names = standard.get_egemaps_feature_names()
    if tuple(egemaps.feature_names) != tuple(runtime_names):
        raise ValueError("Training 88D cache feature_names differ from runtime eGeMAPSv02")
    egemaps_blocks = select_egemaps_blocks(
        egemaps.X, egemaps.feature_names, egemaps.record_keys, keys,
    )
    egemaps_audit = {
        "cache_sha256": _sha256_file(egemaps_path),
        "feature_names": list(egemaps.feature_names),
        "feature_names_sha256": hashlib.sha256("\n".join(egemaps.feature_names).encode("utf-8")).hexdigest(),
        "source_provenance": egemaps.metadata.get("source_provenance"),
        "metadata": dict(egemaps.metadata),
    }
    prior = validate_prior_run_predictions(
        prior_run_roots, keys[validation_idx], calls[validation_idx], labels[validation_idx],
        expected_split=stage6_contract.get("split"),
    ) if prior_run_roots else {}
    return TrainingContext(
        clean_X=clean.X, y=labels, record_keys=keys, call_ids=calls,
        feature_names_59d=tuple(clean.feature_names), train_idx=train_idx,
        validation_idx=validation_idx, split_audit={
            **split_audit, "stage6_artifact_hashes": stage6_hashes,
            "raw_source_verification": clean.raw_source_verification,
        },
        clean_fingerprint=clean.provenance_fingerprint, egemaps_blocks=egemaps_blocks,
        egemaps_audit=egemaps_audit, manifest=source, stage6_root=stage6,
        prior_predictions=prior, stage6_scaler=stage6_scaler,
    )


def validate_frontend_parity():
    """Synthetic-only parity gate for crop lengths that expose padding differences."""
    from . import mission2_acoustic_feature_expansion as acoustic
    import librosa

    results = []
    for length in (160, 320, 480, 512, 1_600):
        signal = np.linspace(-0.1, 0.1, length, dtype=np.float32)
        clean_matrix = _clean_mfcc_matrix(signal)
        direct = librosa.feature.mfcc(
            y=signal, sr=8_000, n_mfcc=20, n_fft=512, hop_length=160,
        )
        if not np.array_equal(clean_matrix, direct):
            raise ValueError(f"Explicit clean MFCC frontend differs from Stage 4 defaults at n={length}")
        expected_frames = 1 + length // 160
        if clean_matrix.shape != (20, expected_frames):
            raise ValueError(f"Centered MFCC frame count differs at n={length}")
        if length >= 512:
            af05_direct = acoustic._delta_mfcc_features(signal)
            af05_matrix = acoustic._delta_mfcc_features_from_matrix(clean_matrix)
            if af05_direct != af05_matrix:
                raise ValueError(f"AF05 matrix-helper parity differs at n={length}")
        af12_matrix = acoustic._temporal_mfcc_contrast_features_from_matrix(clean_matrix)
        if length < 480:
            # AF12's public waveform wrapper applies its original-crop-length
            # fallback before MFCC framing. The matrix helper can be nonzero
            # at three frames, so do not mistake that for a fallback failure.
            af12_direct = acoustic._temporal_mfcc_contrast_features(signal)
            af12_feature = np.zeros(len(_F07_NAMES), dtype=np.float32)
            if (any(float(value) != 0.0 for value in af12_direct.values())
                    or np.any(af12_feature != 0.0)):
                raise ValueError("AF12 original-crop short fallback differs")
        elif length >= 512:
            af12_direct = acoustic._temporal_mfcc_contrast_features(signal)
            if any(af12_direct[name] != af12_matrix[name] for name in _F07_NAMES):
                raise ValueError(f"AF12 matrix-helper parity differs at n={length}")
        results.append({"samples": length, "frames": int(clean_matrix.shape[1]),
                        "af05_fallback": bool(clean_matrix.shape[1] < 3),
                        "af12_fallback": bool(length < 480),
                        "clean_mfcc_frontend": "EXACT_PASS"})
    return {"status": "PASS", "synthetic_rows_only": True, "checks": results}


def extract_training_feature_blocks(context: TrainingContext, raw_root, run_root):
    """Extract five waveform blocks once and persist seven aligned Training matrices."""
    from .audio import read_wav_frames

    root = Path(run_root).resolve()
    cache_dir = root / "feature_cache"
    cache_dir.mkdir(exist_ok=False)
    rows = len(context.record_keys)
    _write_json_new(cache_dir / "status.json", {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "rows_expected": rows,
    })
    matrices = {
        "F04": context.egemaps_blocks["F04"],
        "F05": context.egemaps_blocks["F05"],
        **{name: np.empty((rows, BLOCK_BY_ID[name].dimension), dtype=np.float32)
           for name in ("F01", "F02", "F03", "F06", "F07")},
    }
    counters = {name: {"zero_energy_frames": 0, "af05_short_fallback": 0,
                       "af12_short_fallback": 0} for name in matrices}
    grouped = context.manifest.groupby(["call_id", "audio_path"], sort=False)
    processed = 0
    raw = Path(raw_root).resolve()
    for (_call_id, audio_path), group in grouped:
        audio = read_wav_frames(raw / str(audio_path))
        waveform = np.asarray(audio.waveform)
        if audio.sample_rate != 8_000:
            raise ValueError(f"Unexpected source sample rate: {audio_path}")
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        for row in group.itertuples():
            start, end = int(row.crop_start_frame), int(row.crop_end_frame)
            if start < 0 or end <= start or end > mono.shape[0]:
                raise ValueError(f"Invalid current-utterance crop: {row.record_key}")
            values, details = _waveform_vectors(_mono_crop(mono[start:end], audio.sample_rate))
            for name, vector in values.items():
                matrices[name][row.Index] = vector
            counters["F01"]["zero_energy_frames"] += details["zero_energy_frames"]
            counters["F02"]["zero_energy_frames"] += details["zero_energy_frames"]
            counters["F03"]["zero_energy_frames"] += details["zero_energy_frames"]
            counters["F06"]["af05_short_fallback"] += int(details["af05_short_fallback"])
            counters["F07"]["af12_short_fallback"] += int(details["af12_short_fallback"])
            processed += 1
        if processed and processed % 10_000 < len(group):
            print(f"WAVEFORM_FEATURE_ROWS={processed}/{rows}", flush=True)
    if processed != rows:
        raise ValueError(f"Waveform extraction covered {processed} rows, expected {rows}")
    audits = {}
    for block in FEATURE_BLOCKS:
        matrix = np.asarray(matrices[block.block_id], dtype=np.float32)
        fallback = counters[block.block_id]
        audits[block.block_id] = validate_feature_block(
            block.block_id, matrix, context.record_keys, expected_keys=context.record_keys,
            fallback_counts=fallback,
        )
        path = cache_dir / f"{block.block_id}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, X=matrix, record_keys=context.record_keys,
                                call_ids=context.call_ids,
                                feature_names=np.asarray(block.feature_names, dtype=str))
        audits[block.block_id]["artifact_sha256"] = _sha256_file(path)
        audits[block.block_id]["feature_names"] = list(block.feature_names)
    manifest = {
        "schema_version": "mission2_acoustic_feature_grid_cache_v1",
        "source_split": "Training", "clean_fingerprint": context.clean_fingerprint,
        "record_key_signature": _record_key_signature(context.record_keys),
        "call_id_hash": _array_fingerprint(context.call_ids),
        "labels_hash": _array_fingerprint(context.y), "feature_blocks": audits,
        "egemaps_cache": context.egemaps_audit,
        "frontend_parity": validate_frontend_parity(),
        "official_validation_used": False,
    }
    _write_json_new(cache_dir / "manifest.json", manifest)
    _write_json_replace(cache_dir / "status.json", {
        "status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "manifest_sha256": _sha256_file(cache_dir / "manifest.json"),
    })
    context.feature_matrices = matrices
    return matrices, manifest


def load_feature_blocks(run_root, context):
    """Read-only verify persisted feature matrices; partial or stale cache stops."""
    root = Path(run_root).resolve() / "feature_cache"
    manifest = _read_json(root / "manifest.json")
    status = _read_json(root / "status.json")
    if status.get("status") != "COMPLETE" or status.get("manifest_sha256") != _sha256_file(root / "manifest.json"):
        raise ValueError("Feature cache extraction is incomplete or its completion marker differs")
    if (manifest.get("schema_version") != "mission2_acoustic_feature_grid_cache_v1"
            or manifest.get("source_split") != "Training"
            or manifest.get("official_validation_used") is not False
            or manifest.get("record_key_signature") != _record_key_signature(context.record_keys)
            or manifest.get("call_id_hash") != _array_fingerprint(context.call_ids)
            or manifest.get("labels_hash") != _array_fingerprint(context.y)):
        raise ValueError("Feature cache manifest source identity differs")
    matrices = {}
    for block in FEATURE_BLOCKS:
        path = root / f"{block.block_id}.npz"
        expected_audit = manifest.get("feature_blocks", {}).get(block.block_id, {})
        if not path.is_file() or _sha256_file(path) != expected_audit.get("artifact_sha256"):
            raise ValueError(f"Feature cache artifact missing or corrupt: {block.block_id}")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"X", "record_keys", "call_ids", "feature_names"}:
                raise ValueError(f"Feature cache schema differs: {block.block_id}")
            matrix, keys, calls = saved["X"], saved["record_keys"], saved["call_ids"]
            names = tuple(saved["feature_names"].tolist())
        if (names != block.feature_names or matrix.dtype != np.float32
                or not np.array_equal(keys.astype(str), context.record_keys)
                or not np.array_equal(calls.astype(str), context.call_ids)):
            raise ValueError(f"Feature cache key/call/schema identity differs: {block.block_id}")
        validate_feature_block(block.block_id, matrix, keys, expected_keys=context.record_keys)
        matrices[block.block_id] = matrix
    context.feature_matrices = matrices
    return matrices, manifest


def _json_fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_new(path: Path, value: dict):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _write_json_replace(path: Path, value: dict):
    target = Path(path)
    temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _model_config(model_id: str, input_dim: int) -> dict:
    """Clone the saved Stage 6/13 recipe, changing only its input width."""
    from . import mission2_complex_tabular_screen as complex_screen
    from . import mission2_model_experiments as models
    from . import mission2_stage13_untested_model_family as stage13

    if model_id in complex_screen.CANDIDATES:
        config = deepcopy(complex_screen.get_config(model_id))
        architecture = config["architecture"]
        dimension_name = {"tabm": "n_num_features", "ft_transformer": "n_cont_features",
                          "tabular_resnet": "d_in"}[model_id]
        architecture[dimension_name] = int(input_dim)
        config.pop("config_id", None)
        config["input_dim"] = int(input_dim)
        config["config_id"] = models.config_fingerprint(config)
        return config
    config = deepcopy(stage13.CONFIGS[model_id])
    config.pop("config_id", None)
    config["input_dim"] = int(input_dim)
    config["config_id"] = models.config_fingerprint(config)
    return config


def _all_configs():
    return {
        experiment_id: _model_config(
            experiment_id.removeprefix("baseline_") if experiment_id.startswith("baseline_")
            else experiment_id.rsplit("_F", 1)[0],
            59 if experiment_id.startswith("baseline_")
            else 59 + BLOCK_BY_ID["F" + experiment_id.rsplit("_F", 1)[1]].dimension,
        )
        for experiment_id in experiment_ids()
    }


def _context_identity(context: TrainingContext) -> dict:
    """Hash source order and persisted split membership for every run contract."""
    return {
        "clean_fingerprint": context.clean_fingerprint,
        "source_keys": _array_fingerprint(context.record_keys),
        "source_labels": _array_fingerprint(context.y),
        "source_calls": _array_fingerprint(context.call_ids),
        "train_keys": _array_fingerprint(context.record_keys[context.train_idx]),
        "validation_keys": _array_fingerprint(context.record_keys[context.validation_idx]),
        "train_labels": _array_fingerprint(context.y[context.train_idx]),
        "validation_labels": _array_fingerprint(context.y[context.validation_idx]),
        "egemaps_cache_sha256": context.egemaps_audit["cache_sha256"],
        "stage6_scaler_sha256": _sha256_file(context.stage6_root / "scaler.json"),
        "prior_prediction_sha256": {
            name: item["sha256"] for name, item
            in context.prior_predictions.get("predictions", {}).items()
        },
        "prior_unavailable": context.prior_predictions.get("unavailable", []),
    }


def get_or_create_run_root(repo_root, context: TrainingContext, *, run_root=None):
    """Reserve one durable run; restarting reuses its exact root, never forks it."""
    base = Path(repo_root).resolve() / GRID_RELATIVE
    base.mkdir(parents=True, exist_ok=True)
    pointer = base / "active_run.json"
    context_identity = _context_identity(context)
    schema = {block.block_id: list(block.feature_names) for block in FEATURE_BLOCKS}
    schema_fingerprint = _json_fingerprint(schema)
    if run_root is not None:
        root = Path(run_root).resolve()
        if not root.is_dir() or root.parent != base:
            raise ValueError("Explicit run_root must be an existing grid child directory")
        contract = _read_json(root / "contract.json")
        if (contract.get("context_identity") != context_identity
                or contract.get("split") != context.split_audit
                or contract.get("feature_schema_fingerprint") != schema_fingerprint):
            raise ValueError("Explicit run root belongs to another source/split/schema")
        return root
    if pointer.exists():
        payload = _read_json(pointer)
        root = Path(payload.get("run_root", "")).resolve()
        if not root.is_dir() or root.parent != base:
            raise ValueError("Active grid pointer is missing or escapes the grid output root")
        if payload.get("contract_sha256") != _sha256_file(root / "contract.json"):
            raise ValueError("Active grid pointer contract hash differs")
        contract = _read_json(root / "contract.json")
        if (contract.get("context_identity") != context_identity
                or contract.get("split") != context.split_audit
                or contract.get("feature_schema_fingerprint") != schema_fingerprint):
            raise ValueError("Active grid run has different Training/split/schema identity")
        return root

    configs = _all_configs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    root = base / run_id
    root.mkdir(exist_ok=False)
    contract = {
        "schema_version": "mission2_acoustic_feature_grid_v1",
        "source_split": "Training", "evaluation_set": "fixed_internal_validation",
        "split_seed": SEED, "group_key": "call_id", "threshold": THRESHOLD,
        "official_validation_used": False, "split": deepcopy(context.split_audit),
        "context_identity": context_identity, "feature_schema": schema,
        "feature_schema_fingerprint": schema_fingerprint,
        "models": list(MODEL_IDS), "experiments": list(experiment_ids()),
        "model_configs": configs,
        "comparison_policy": {
            "primary": "fresh_per_model_baseline_vs_single_feature_candidate_on_exact_shared_rows",
            "secondary": "saved_stage10_to_stage13_internal_predictions_after_exact_key_call_label_alignment",
            "internal_validation_reuse_disclosed": True,
            "35_candidate_selection_bias_disclosed": True,
        },
        "stage6_reference_root": str(context.stage6_root),
        "prior_run_inventory": {
            "verified_models": {name: {key: value for key, value in item.items() if key != "predictions"}
                                for name, item in context.prior_predictions.get("predictions", {}).items()},
            "unavailable": context.prior_predictions.get("unavailable", []),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(root / "contract.json", contract)
    ledger = {
        "schema_version": "mission2_acoustic_feature_grid_ledger_v1",
        "run_id": run_id, "status": "READY", "next_experiment": experiment_ids()[0],
        "experiments": {name: "PENDING" for name in experiment_ids()},
        "completed_count": 0,
    }
    _write_json_new(root / "run_ledger.json", ledger)
    _write_json_new(root / "environment.json", {
        "status": "NOT_CHECKED", "runtime": "Colab must record actual device and packages",
    })
    _write_json_new(pointer, {"run_id": run_id, "run_root": str(root),
                              "contract_sha256": _sha256_file(root / "contract.json")})
    return root


def _baseline_source_inventory(context: TrainingContext, source_root: Path) -> dict:
    """Verify and pin the five completed baseline artifacts without fitting them."""
    source = Path(source_root).resolve()
    contract_path = source / "contract.json"
    ledger_path = source / "run_ledger.json"
    if not contract_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("Existing Feature Grid baseline run is missing its contract or ledger")
    source_contract = _read_json(contract_path)
    source_ledger = _read_json(ledger_path)
    if (source_contract.get("schema_version") != "mission2_acoustic_feature_grid_v1"
            or tuple(source_contract.get("experiments", ())) != experiment_ids()
            or source_contract.get("context_identity") != _context_identity(context)
            or source_contract.get("split") != context.split_audit
            or source_contract.get("official_validation_used") is not False):
        raise ValueError("Existing baseline run differs from the loaded Training/fixed-split contract")
    if not (source / "environment.json").is_file():
        raise FileNotFoundError("Existing baseline run environment record is missing")
    baseline_sources = {}
    for experiment_id in (f"baseline_{model}" for model in MODEL_IDS):
        if source_ledger.get("experiments", {}).get(experiment_id) != "COMPLETE":
            raise RuntimeError(f"Required saved baseline is not COMPLETE: {experiment_id}")
        verify_completed_experiment(context, source, experiment_id)
        directory = source / experiment_id
        artifact_manifest_path = directory / "artifact_manifest.json"
        artifact_manifest = _verify_artifact_manifest(directory)
        checkpoint_files = _checkpoint_files_from_manifest(artifact_manifest["files"])
        if not checkpoint_files:
            raise ValueError(f"Saved baseline weights are missing: {experiment_id}")
        baseline_sources[experiment_id] = {
            "source_directory": str(directory),
            "source_artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
            "source_contract_sha256": _sha256_file(contract_path),
            "source_environment_sha256": _sha256_file(source / "environment.json"),
            "artifact_files": artifact_manifest["files"],
            "checkpoint_files": checkpoint_files,
        }
    return {
        "source_run_root": str(source),
        "source_run_id": source.name,
        "source_contract_sha256": _sha256_file(contract_path),
        "source_ledger_sha256": _sha256_file(ledger_path),
        "source_environment_sha256": _sha256_file(source / "environment.json"),
        "baselines": baseline_sources,
    }


def _checkpoint_files_from_manifest(artifact_files: dict) -> list[str]:
    """Find persisted model weights across the recipes' checkpoint formats."""
    weight_suffixes = {".pt", ".pth", ".ckpt"}
    return sorted(
        relative for relative in artifact_files
        if Path(relative).suffix.lower() in weight_suffixes
    )


def _verify_reused_baseline(context: TrainingContext, run_root, experiment_id):
    """Verify a read-only copied checkpoint/prediction baseline in a continuation run."""
    root = Path(run_root)
    contract = _read_json(root / "contract.json")
    inventory = _read_json(root / "baseline_reuse_manifest.json")
    source_environment = root / "baseline_reuse" / "source_environment.json"
    if (not source_environment.is_file()
            or _sha256_file(source_environment) != inventory.get("source_environment_sha256")
            or _sha256_file(source_environment) != inventory.get("copied_environment_sha256")):
        raise ValueError("Reused baseline environment snapshot hash differs")
    reference = inventory.get("baselines", {}).get(experiment_id)
    if reference is None or experiment_id not in (f"baseline_{model}" for model in MODEL_IDS):
        raise ValueError(f"Missing or unexpected reused baseline reference: {experiment_id}")
    if reference.get("source_contract_sha256") != inventory.get("source_contract_sha256"):
        raise ValueError("Reused baseline source contract hash differs")
    directory = root / "baseline_reuse" / experiment_id
    manifest = _verify_artifact_manifest(directory)
    if (_sha256_file(directory / "artifact_manifest.json")
            != reference.get("source_artifact_manifest_sha256")
            or manifest.get("experiment_id") != experiment_id
            or manifest.get("config_id") != contract["model_configs"][experiment_id]["config_id"]
            or manifest.get("split_identity") != contract.get("context_identity")
            or manifest.get("feature_schema_fingerprint") != contract.get("feature_schema_fingerprint")
            or manifest.get("files") != reference.get("artifact_files")):
        raise ValueError(f"Reused baseline artifact manifest differs: {experiment_id}")
    metrics = _read_json(directory / "metrics.json")
    if (metrics.get("status") != "COMPLETE"
            or metrics.get("checkpoint_reload") not in ("EXACT_PASS", "ATOL_1E-7_PASS")
            or metrics.get("split_identity") != contract.get("context_identity")
            or metrics.get("environment_sha256") != inventory.get("source_environment_sha256")):
        raise ValueError(f"Reused baseline metrics/provenance differ: {experiment_id}")
    config = _read_json(directory / "config.json")
    if config != contract["model_configs"][experiment_id]:
        raise ValueError(f"Reused baseline model config differs: {experiment_id}")
    for checkpoint in reference.get("checkpoint_files", []):
        if checkpoint not in manifest.get("files", {}):
            raise ValueError(f"Reused baseline checkpoint is not hash-registered: {experiment_id}")
    scaler = _read_json(directory / "scaler.json")
    if (scaler.get("fit_scope") != "fixed_internal_train_only"
            or scaler.get("training_rows") != len(context.train_idx)
            or len(scaler.get("mean", [])) != 59
            or len(scaler.get("scale", [])) != 59
            or len(scaler.get("var", [])) != 59):
        raise ValueError(f"Reused baseline scaler scope/schema differs: {experiment_id}")
    frame = pd.read_parquet(directory / "predictions.parquet")
    _prediction_identity(frame, context.validation_record_keys,
                         context.validation_call_ids, context.validation_labels)
    observed, reconstructed = _metric_bundle(
        context.validation_labels, frame.probability_speaker1.to_numpy(dtype=np.float64),
        context.validation_record_keys, context.validation_call_ids,
    )
    if not frame.equals(reconstructed):
        raise ValueError(f"Reused baseline predictions do not reconstruct: {experiment_id}")
    for name, value in observed.items():
        if metrics.get(name) != value:
            raise ValueError(f"Reused baseline metric does not reconstruct: {experiment_id}/{name}")
    return {"status": "VERIFIED_REUSED_BASELINE", "experiment_id": experiment_id,
            "validation_accuracy": observed["validation_accuracy"],
            "checkpoint_files": reference["checkpoint_files"]}


def _resume_ledger_valid(ledger: dict) -> bool:
    expected = {name: "REUSED_VERIFIED" for name in (f"baseline_{model}" for model in MODEL_IDS)}
    expected.update({name: "SKIPPED_BY_USER_SCOPE" for name in user_scope_excluded_experiment_ids()})
    for name in candidate_experiment_ids():
        expected[name] = ledger.get("experiments", {}).get(name)
    return ledger.get("experiments") == expected


def validate_colab_resume_run(context: TrainingContext, run_root):
    """Preflight an existing continuation run; never repair partial files."""
    root = Path(run_root).resolve()
    contract_path = root / "contract.json"
    ledger_path = root / "run_ledger.json"
    if not contract_path.is_file() or not ledger_path.is_file():
        raise RuntimeError(f"Continuation run is partial (contract/ledger missing): {root}")
    contract = _read_json(contract_path)
    ledger = _read_json(ledger_path)
    if (contract.get("schema_version") != "mission2_acoustic_feature_grid_resume_v1"
            or contract.get("context_identity") != _context_identity(context)
            or contract.get("split") != context.split_audit
            or contract.get("official_validation_used") is not False
            or tuple(contract.get("candidate_experiments", ())) != candidate_experiment_ids()
            or contract.get("output_root") != str(root.parent)):
        raise ValueError("Continuation run path/Training/split/feature contract differs")
    if not _resume_ledger_valid(ledger):
        raise ValueError("Continuation ledger differs from 5 reused / 7 skipped / 28 candidate contract")
    reuse_manifest_path = root / "baseline_reuse_manifest.json"
    if _sha256_file(reuse_manifest_path) != contract.get("baseline_reuse_manifest_sha256"):
        raise ValueError("Reused baseline manifest hash differs from the run contract")
    reuse_manifest = _read_json(reuse_manifest_path)
    if (reuse_manifest.get("source_run_root") != contract.get("baseline_source_run_root")
            or reuse_manifest.get("source_contract_sha256") != contract.get("baseline_source_contract_sha256")
            or reuse_manifest.get("source_environment_sha256") != contract.get("baseline_source_environment_sha256")):
        raise ValueError("Reused baseline source identity differs from the run contract")
    for experiment_id in (f"baseline_{model}" for model in MODEL_IDS):
        _verify_reused_baseline(context, root, experiment_id)
    feature_reference = _read_json(root / "feature_cache" / "source_reference.json")
    if (feature_reference.get("source_run_root") != contract.get("feature_cache_source_run_root")
            or feature_reference.get("manifest_sha256") != contract.get("feature_cache_manifest_sha256")
            or _sha256_file(root / "feature_cache" / "manifest.json") != contract.get("feature_cache_manifest_sha256")):
        raise ValueError("Feature-cache reference differs from the validated source cache")
    for experiment_id, status in ledger["experiments"].items():
        directory = root / experiment_id
        if status == "COMPLETE":
            verify_completed_experiment(context, root, experiment_id)
        elif experiment_id in candidate_experiment_ids() and (status != "PENDING" or directory.exists()):
            raise RuntimeError(f"Candidate is incomplete/failed and will not be retried: {experiment_id}={status}")
    if ledger.get("status") in ("FAILED_STOP", "RUNNING"):
        raise RuntimeError(f"Continuation run cannot resume from state {ledger['status']}")
    return {"status": ledger.get("status"), "run_root": str(root),
            "completed_count": ledger.get("completed_count"),
            "candidate_count": len(candidate_experiment_ids()),
            "next_experiment": ledger.get("next_experiment")}


def prepare_colab_resume_run(context: TrainingContext, source_run_root, output_root):
    """Reuse five verified baselines/cache, then create a 28-fit Drive continuation."""
    source = Path(source_run_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Existing Feature Grid run is unavailable: {source}")
    baseline_inventory = _baseline_source_inventory(context, source)
    feature_matrices, feature_audit = load_feature_blocks(source, context)
    context.feature_matrices = feature_matrices
    source_cache = source / "feature_cache"
    cache_manifest_path = source_cache / "manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError("Existing verified seven-block feature cache is missing")
    cache_manifest_sha = _sha256_file(cache_manifest_path)

    base = Path(output_root).expanduser().resolve()
    if base.name != COLAB_OUTPUT_DIR_NAME or base.parent.name != "MyDrive":
        raise ValueError(
            "Continuation output must be a direct child of the mounted MyDrive/mission2_grid_runs"
        )
    if base == source or base.is_relative_to(source):
        raise ValueError("New Drive output root must not be inside the read-only source run")
    run_id = f"{source.name}_gpu_resume"
    root = base / run_id
    if root.exists():
        result = validate_colab_resume_run(context, root)
        contract = _read_json(root / "contract.json")
        if (contract.get("baseline_source_run_root") != str(source)
                or contract.get("feature_cache_source_run_root") != str(source)
                or contract.get("feature_cache_manifest_sha256") != cache_manifest_sha):
            raise ValueError("Existing continuation run points to a different source/cache")
        return root, {**result, "baseline_reuse": "PASS", "feature_cache": feature_audit}

    # Do not create a Drive output tree until source baseline and cache checks pass.
    base.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    baseline_copy_root = root / "baseline_reuse"
    baseline_copy_root.mkdir(exist_ok=False)
    copied_baselines = {}
    try:
        for experiment_id, reference in baseline_inventory["baselines"].items():
            destination = baseline_copy_root / experiment_id
            shutil.copytree(reference["source_directory"], destination)
            copied_manifest = _verify_artifact_manifest(destination)
            if (copied_manifest.get("files") != reference["artifact_files"]
                    or _sha256_file(destination / "artifact_manifest.json")
                    != reference["source_artifact_manifest_sha256"]):
                raise ValueError(f"Copied baseline artifact changed during transfer: {experiment_id}")
            copied_baselines[experiment_id] = reference

        shutil.copy2(source / "environment.json", baseline_copy_root / "source_environment.json")
        cache_directory = root / "feature_cache"
        cache_directory.mkdir(exist_ok=False)
        shutil.copy2(cache_manifest_path, cache_directory / "manifest.json")
        _write_json_new(cache_directory / "source_reference.json", {
            "source_run_root": str(source), "manifest_sha256": cache_manifest_sha,
            "feature_cache_audit": feature_audit,
            "copied_matrices": False,
        })
        reuse_manifest = {
            **{key: value for key, value in baseline_inventory.items() if key != "baselines"},
            "baselines": copied_baselines,
            "copied_environment_sha256": _sha256_file(baseline_copy_root / "source_environment.json"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_new(root / "baseline_reuse_manifest.json", reuse_manifest)
        schema = {block.block_id: list(block.feature_names) for block in FEATURE_BLOCKS}
        configs = _all_configs()
        contract = {
            "schema_version": "mission2_acoustic_feature_grid_resume_v1",
            "run_id": run_id, "source_run_id": source.name,
            "source_split": "Training", "evaluation_set": "fixed_internal_validation",
            "split_seed": SEED, "group_key": "call_id", "threshold": THRESHOLD,
            "official_validation_used": False, "split": deepcopy(context.split_audit),
            "context_identity": _context_identity(context),
            "feature_schema": schema, "feature_schema_fingerprint": _json_fingerprint(schema),
            "models": list(MODEL_IDS), "experiments": list(experiment_ids()),
            "baseline_experiments": [f"baseline_{model}" for model in MODEL_IDS],
            "baseline_source_run_root": str(source),
            "baseline_source_contract_sha256": baseline_inventory["source_contract_sha256"],
            "baseline_source_environment_sha256": baseline_inventory["source_environment_sha256"],
            "baseline_reuse_manifest_sha256": _sha256_file(root / "baseline_reuse_manifest.json"),
            "candidate_experiments": list(candidate_experiment_ids()),
            "excluded_experiments": list(user_scope_excluded_experiment_ids()),
            "scheduled_fit_count": len(candidate_experiment_ids()),
            "model_configs": configs,
            "feature_cache_source_run_root": str(source),
            "feature_cache_manifest_sha256": cache_manifest_sha,
            "feature_cache_audit": feature_audit,
            "output_root": str(base),
            "attempt_policy": {
                "tabm_F01": "new attempt in this run; preserve any source-run attempt unchanged",
                "cpu_fallback": False,
                "automatic_retry_or_overwrite": False,
            },
            "comparison_policy": {
                "primary": "reused_stage13_59d_baseline_vs_single_feature_candidate_on_exact_shared_rows",
                "secondary": "saved_stage10_to_stage13_internal_predictions_after_exact_key_call_label_alignment",
                "internal_validation_reuse_disclosed": True,
                "28_candidate_selection_bias_disclosed": True,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_new(root / "contract.json", contract)
        ledger = {
            "schema_version": "mission2_acoustic_feature_grid_resume_ledger_v1",
            "run_id": run_id, "status": "READY",
            "next_experiment": candidate_experiment_ids()[0],
            "experiments": {
                **{f"baseline_{model}": "REUSED_VERIFIED" for model in MODEL_IDS},
                **{name: "SKIPPED_BY_USER_SCOPE" for name in user_scope_excluded_experiment_ids()},
                **{name: "PENDING" for name in candidate_experiment_ids()},
            },
            "completed_count": 0, "scheduled_fit_count": len(candidate_experiment_ids()),
            "reused_baseline_count": len(MODEL_IDS),
            "skipped_count": len(user_scope_excluded_experiment_ids()),
        }
        _write_json_new(root / "run_ledger.json", ledger)
        _write_json_new(root / "environment.json", {
            "status": "NOT_CHECKED", "runtime": "Colab GPU preflight required",
            "device_required": "cuda", "cpu_fallback": False,
        })
    except Exception:
        # The visible directory is intentionally retained as a partial attempt;
        # the next setup must report it instead of silently replacing files.
        raise
    result = validate_colab_resume_run(context, root)
    return root, {**result, "baseline_reuse": "PASS", "feature_cache": feature_audit}


def _probe_candidate_model_apis(torch, complex_screen, device):
    """Exercise each in-scope model API on small synthetic GPU tensors only."""
    dimensions = sorted({59 + block.dimension for block in FEATURE_BLOCKS})
    probes = {model_id: [] for model_id in COLAB_CANDIDATE_MODEL_IDS}
    for input_dim in dimensions:
        sample = torch.zeros((4, input_dim), dtype=torch.float32, device=device)
        for model_id in ("tabm", "tabular_resnet"):
            config = _model_config(model_id, input_dim)
            model = _make_complex_model(model_id, config, device)
            model.eval()
            with torch.inference_mode():
                logits = complex_screen._logits(model_id, model, sample)
            if logits.shape[0] != len(sample) or not torch.isfinite(logits).all():
                raise RuntimeError(f"Synthetic GPU API probe failed: {model_id}/{input_dim}D")
            probes[model_id].append(input_dim)
            del model, logits

        dcn_model = _dcn_model_class(torch, input_dim)().to(device)
        dcn_model.eval()
        with torch.inference_mode():
            dcn_logits = dcn_model(sample)
        if dcn_logits.shape != (len(sample),) or not torch.isfinite(dcn_logits).all():
            raise RuntimeError(f"Synthetic GPU API probe failed: dcnv2/{input_dim}D")
        probes["dcnv2"].append(input_dim)
        del dcn_model, dcn_logits, sample
        torch.cuda.empty_cache()

    # Stage 13's GANDALF helper is high-level and does not expose a stable
    # standalone forward method. Construct the exact pinned config objects
    # before any real fit so dependency/API mismatches fail at preflight.
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig

    names = [f"probe_feature_{index:03d}" for index in range(max(dimensions))]
    data_config = DataConfig(target=["target"], continuous_cols=names, categorical_cols=[])
    model_config = GANDALFConfig(
        task="classification", gflu_stages=6, gflu_dropout=.1,
        gflu_feature_init_sparsity=.3, learning_rate=.001, metrics=["accuracy"],
    )
    trainer_config = TrainerConfig(
        max_epochs=20, batch_size=4096, accelerator="auto", devices=1,
        early_stopping="valid_accuracy", early_stopping_mode="max",
        early_stopping_patience=5, checkpoints="valid_accuracy",
        checkpoints_mode="max", checkpoints_save_top_k=1,
        load_best=True, deterministic=True,
    )
    gandalf = TabularModel(
        data_config=data_config, model_config=model_config,
        optimizer_config=OptimizerConfig(), trainer_config=trainer_config,
    )
    del gandalf, data_config, model_config, trainer_config
    probes["gandalf"] = {"config_constructor": "PASS", "input_dimension": max(dimensions)}
    return {"status": "PASS", "synthetic_only": True, "input_dimensions": dimensions,
            "models": probes}


def record_environment(run_root, *, device="cuda"):
    """Record real versions and require the frozen GPU backend before fitting."""
    import torch
    from . import mission2_complex_tabular_screen as complex_screen

    required = {"tabm": "0.0.3", "rtdl-revisiting-models": "0.0.2",
                "rtdl-num-embeddings": "0.0.12", "pytorch-tabular": "1.2.0"}
    versions = {name: importlib.metadata.version(name) for name in required}
    for name, expected in required.items():
        if versions[name] != expected:
            raise RuntimeError(f"Frozen package mismatch: {name}={versions[name]} expected={expected}")
    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if not (2, 6) <= torch_version < (3, 0):
        raise RuntimeError("Stage 6/13 recipes require PyTorch >= 2.6,<3")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Feature Grid requires the selected Colab CUDA runtime; CPU fallback is disabled")
    # Synthetic API probes only. No competition rows are read by this check.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED)
        candidate_api_probe = _probe_candidate_model_apis(torch, complex_screen, device)
    environment = {
        "status": "READY", "python": platform.python_version(),
        "platform": platform.platform(), "packages": {**versions,
            **{name: importlib.metadata.version(name) for name in
               ("torch", "numpy", "pandas", "scikit-learn", "librosa", "opensmile", "pyarrow")}},
        "device": device, "gpu": torch.cuda.get_device_name(),
        "cuda_build": torch.version.cuda, "official_api_probe": "PASS",
        "candidate_api_probe": candidate_api_probe,
        "cpu_fallback": False,
    }
    environment_path = Path(run_root) / "environment.json"
    previous = _read_json(environment_path)
    if previous.get("status") == "READY":
        if previous != environment:
            raise RuntimeError("Colab runtime differs from the frozen run environment; do not resume fits")
    else:
        _write_json_replace(environment_path, environment)
    return environment


def _make_complex_model(model_id, config, device):
    import torch
    if model_id == "tabm":
        from tabm import TabM
        model = TabM.make(**config["architecture"])
    else:
        from rtdl_revisiting_models import FTTransformer, ResNet
        factory = FTTransformer if model_id == "ft_transformer" else ResNet
        model = factory(**config["architecture"])
    return model.to(device=device, dtype=torch.float32)


def _fit_complex_model(model_id, config, X_train, y_train, X_validation, y_validation, *, device):
    """Stage 6/13 tabular recipe with only its numerical input width generalized."""
    import torch
    from . import mission2_complex_tabular_screen as screen

    input_dim = config["input_dim"]
    for X, y in ((X_train, y_train), (X_validation, y_validation)):
        if (X.dtype != np.float32 or X.shape != (len(y), input_dim) or len(y) < 2
                or not np.isfinite(X).all() or not set(np.unique(y)).issubset({0, 1})):
            raise ValueError("Model input rows/dimension/finite/binary contract differs")
    started = __import__("time").perf_counter()
    with screen._deterministic_training(device, config["seed"]):
        model = _make_complex_model(model_id, config, device)
        parameters = model.make_parameter_groups() if model_id == "ft_transformer" else model.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"],
                                      betas=tuple(config["adam_betas"]), eps=config["adam_eps"])
        rng = np.random.default_rng(config["seed"])
        history, best_state, best_correct, best_epoch, stale = [], None, -1, 0, 0
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            order = rng.permutation(len(y_train))
            batches = [order[left:left + config["batch_size"]]
                       for left in range(0, len(order), config["batch_size"])]
            if len(batches[-1]) == 1 and len(batches) > 1:
                tail = batches.pop()
                batches[-1] = np.concatenate((batches[-1], tail))
            loss_total = 0.0
            for indices in batches:
                xb = torch.from_numpy(X_train[indices].copy()).to(device)
                yb = torch.from_numpy(y_train[indices].astype(np.float32, copy=False)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = screen._logits(model_id, model, xb)
                if model_id == "tabm":
                    yb = yb[:, None].expand_as(logits)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise ValueError("Non-finite model gradient")
                optimizer.step()
                loss_total += float(loss.detach().cpu()) * len(indices)
            val_p = screen.predict_probability(model_id, model, X_validation,
                                               device=device, batch_size=config["batch_size"])
            correct = int(((val_p >= THRESHOLD) == y_validation).sum())
            if correct > best_correct:
                best_correct, best_epoch, stale = correct, epoch, 0
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": loss_total / len(y_train),
                            "validation_accuracy": correct / len(y_validation),
                            "validation_correct": correct, "best_epoch": best_epoch})
            print(f"{model_id} epoch={epoch} validation_accuracy={correct / len(y_validation):.6f} "
                  f"best_epoch={best_epoch}", flush=True)
            if stale >= config["patience"]:
                break
        if best_state is None:
            raise RuntimeError("No Best Checkpoint was produced")
        model.load_state_dict(best_state, strict=True)
    return model, history, {"best_epoch": int(best_epoch), "epochs_run": len(history),
                            "best_validation_correct": int(best_correct),
                            "fit_seconds": __import__("time").perf_counter() - started}


def _dcn_model_class(torch, input_dim):
    nn = torch.nn

    class DCNv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(2)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
            for weight in self.cross_w:
                nn.init.xavier_uniform_(weight)
            self.deep = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(.1),
                                      nn.Linear(128, 64), nn.ReLU())
            self.output = nn.Linear(input_dim + 64, 1)

        def forward(self, values):
            original, crossed = values, values
            for weight, bias in zip(self.cross_w, self.cross_b):
                crossed = original * (crossed @ weight + bias) + crossed
            return self.output(torch.cat((crossed, self.deep(values)), dim=1)).squeeze(1)

    return DCNv2


def _torch_probability(model, X, device):
    import torch
    model.eval()
    probability = np.empty(len(X), dtype=np.float64)
    with torch.inference_mode():
        for left in range(0, len(X), 32_768):
            batch = torch.from_numpy(X[left:left + 32_768].copy()).to(device)
            probability[left:left + len(batch)] = torch.sigmoid(model(batch)).cpu().numpy()
    return probability


def _fit_dcnv2(config, X_train, y_train, X_validation, y_validation, folder, device):
    import torch
    input_dim = config["input_dim"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model_class = _dcn_model_class(torch, input_dim)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    best_state, best_correct, best_epoch, stale, history = None, -1, 0, 0, []
    started = __import__("time").perf_counter()
    for epoch in range(1, 21):
        model.train()
        indices = rng.permutation(len(y_train))
        loss_total = 0.0
        for left in range(0, len(indices), 4096):
            batch_indices = indices[left:left + 4096]
            xb = torch.from_numpy(X_train[batch_indices].copy()).to(device)
            yb = torch.from_numpy(y_train[batch_indices].astype(np.float32, copy=False)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite DCNv2 loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(batch_indices)
        val_p = _torch_probability(model, X_validation, device)
        correct = int(((val_p >= THRESHOLD) == y_validation).sum())
        history.append({"epoch": epoch, "train_loss": loss_total / len(y_train),
                        "validation_accuracy": correct / len(y_validation),
                        "validation_correct": correct})
        if correct > best_correct:
            best_correct, best_epoch, stale = correct, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
        print(f"dcnv2 epoch={epoch} validation_accuracy={correct / len(y_validation):.6f} "
              f"best_epoch={best_epoch}", flush=True)
        if stale >= 5:
            break
    if best_state is None:
        raise RuntimeError("DCNv2 Best Checkpoint was not produced")
    model.load_state_dict(best_state, strict=True)
    before = _torch_probability(model, X_validation, device)
    path = folder / "checkpoint.pt"
    with path.open("xb") as stream:
        torch.save({"state_dict": best_state, "config": config}, stream)
    model.cpu()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("DCNv2 checkpoint recipe/input width differs")
    restored = model_class().to(device)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    after = _torch_probability(restored, X_validation, device)
    if not np.array_equal(before, after):
        raise ValueError("DCNv2 checkpoint reload predictions differ")
    return restored, history, {"best_epoch": int(best_epoch), "epochs_run": len(history),
                               "best_validation_correct": int(best_correct),
                               "fit_seconds": __import__("time").perf_counter() - started,
                               "checkpoint_reload": "EXACT_PASS"}


def _metric_bundle(y_true, probability, record_keys, call_ids):
    p1 = np.asarray(probability, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if (p1.shape != y.shape or not np.isfinite(p1).all()
            or np.any((p1 < 0) | (p1 > 1)) or not set(np.unique(y)).issubset({0, 1})):
        raise ValueError("Saved prediction probabilities or labels are invalid")
    prediction = (p1 >= THRESHOLD).astype(np.int64)
    correct = prediction == y
    tn = int(np.count_nonzero((y == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((y == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((y == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((y == 1) & (prediction == 1)))
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    confidence = np.maximum(1.0 - p1, p1)
    error_type = np.where(y == 0, np.where(prediction == 0, "TN", "FP"),
                          np.where(prediction == 0, "FN", "TP"))
    frame = pd.DataFrame({
        "record_key": np.asarray(record_keys).astype(str),
        "call_id": np.asarray(call_ids).astype(str), "y_true": y,
        "probability_speaker1": p1, "prediction": prediction,
        "confidence": confidence, "correct": correct, "error_type": error_type,
    }, columns=["record_key", "call_id", "y_true", "probability_speaker1", "prediction",
                "confidence", "correct", "error_type"])
    metrics = {
        "status": "COMPLETE", "threshold": THRESHOLD,
        "validation_accuracy": float(correct.mean()), "balanced_accuracy": (recall_0 + recall_1) / 2,
        "recall_0": recall_0, "recall_1": recall_1,
        "confusion_matrix": [[tn, fp], [fn, tp]], "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "errors": int((~correct).sum()), "hce": int((~correct & (confidence >= .90)).sum()),
        "mean_confidence": float(confidence.mean()),
    }
    return metrics, frame


def _prediction_identity(frame, keys, calls, labels):
    for name, expected in (("record_key", keys), ("call_id", calls), ("y_true", labels)):
        if name not in frame or not np.array_equal(frame[name].to_numpy(), expected):
            raise ValueError(f"Prediction row/call/label identity differs: {name}")


def _experiment_model_and_block(experiment_id):
    if experiment_id not in experiment_ids():
        raise KeyError(f"Unknown Feature Grid experiment: {experiment_id}")
    if experiment_id.startswith("baseline_"):
        return experiment_id.removeprefix("baseline_"), None
    model_id, feature_id = experiment_id.rsplit("_F", 1)
    return model_id, "F" + feature_id


def _experiment_input(context, experiment_id):
    if context.feature_matrices is None:
        raise RuntimeError("Feature matrices must be extracted and verified before any fit")
    model_id, block_id = _experiment_model_and_block(experiment_id)
    if block_id is None:
        X = context.clean_X
        names = context.feature_names_59d
    else:
        if block_id not in context.feature_matrices:
            raise ValueError(f"Missing frozen feature block matrix: {block_id}")
        X = np.concatenate((context.clean_X, context.feature_matrices[block_id]), axis=1).astype(np.float32, copy=False)
        names = (*context.feature_names_59d, *BLOCK_BY_ID[block_id].feature_names)
    expected_dim = 59 if block_id is None else 59 + BLOCK_BY_ID[block_id].dimension
    if (X.dtype != np.float32 or X.shape != (len(context.y), expected_dim)
            or not np.isfinite(X).all() or len(set(names)) != len(names)):
        raise ValueError(f"{experiment_id} frozen model-input matrix contract differs")
    return model_id, block_id, X, tuple(names)


def _normalized_prior_contract(value):
    for key in ("source_contract", "preprocessing", "contract"):
        nested = value.get(key)
        if isinstance(nested, dict) and "split" in nested:
            return nested
    return value


def _prior_prediction_paths(root: Path):
    """Inspect only known Internal prediction layouts; never recurse the data tree."""
    candidates = []
    for parent in (root, root / "stage6_reference", root / "frozen_references",
                   root / "candidates"):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                for name in ("predictions.parquet", "validation_predictions.parquet"):
                    path = child / name
                    if path.is_file():
                        candidates.append(path)
            elif child.is_file() and child.suffix == ".parquet" and "prediction" in child.name:
                candidates.append(child)
    return tuple(sorted(set(candidates)))


def validate_prior_run_predictions(
    prior_run_roots,
    validation_record_keys,
    validation_call_ids,
    validation_labels,
    *,
    expected_split=EXPECTED_SPLIT,
):
    """Load only explicitly named, non-Official historical Internal predictions."""
    keys = np.asarray(validation_record_keys).astype(str)
    calls = np.asarray(validation_call_ids).astype(str)
    labels = np.asarray(validation_labels, dtype=np.int64)
    results, unavailable = {}, []
    for supplied_root in prior_run_roots:
        root = Path(supplied_root).resolve()
        if not root.exists():
            unavailable.append({"root": str(root), "reason": "path_missing"})
            continue
        contract_path = root / "contract.json" if root.is_dir() else root.parent / "contract.json"
        if not contract_path.is_file():
            unavailable.append({"root": str(root), "reason": "contract_missing"})
            continue
        contract = _normalized_prior_contract(_read_json(contract_path))
        if (contract.get("source_split") != "Training"
                or contract.get("evaluation_set") != "fixed_internal_validation"
                or contract.get("seed") != SEED
                or contract.get("threshold") != THRESHOLD
                or contract.get("official_validation_used") is not False):
            unavailable.append({"root": str(root), "reason": "not_proven_training_internal_only"})
            continue
        split = contract.get("split")
        if not isinstance(split, dict) or any(split.get(name) != value for name, value in expected_split.items()):
            unavailable.append({"root": str(root), "reason": "fixed_split_counts_differ"})
            continue
        for prediction_path in _prior_prediction_paths(root):
            name = f"{root.name}/{prediction_path.parent.name or prediction_path.stem}"
            if name in results:
                name = f"{name}:{prediction_path.stem}"
            metrics_path = prediction_path.parent / "metrics.json"
            if metrics_path.is_file():
                metrics = _read_json(metrics_path)
                if metrics.get("status") not in ("SUCCESS", "COMPLETE"):
                    unavailable.append({"root": str(prediction_path), "reason": "historical_model_not_complete"})
                    continue
            frame = pd.read_parquet(prediction_path)
            _prediction_identity(frame, keys, calls, labels)
            probability_name = next((column for column in
                ("probability_speaker1", "p_speaker1", "probability_1") if column in frame), None)
            if probability_name is None:
                unavailable.append({"root": str(prediction_path), "reason": "probability_column_missing"})
                continue
            observed, reconstructed = _metric_bundle(
                labels, frame[probability_name].to_numpy(dtype=np.float64), keys, calls,
            )
            if "correct" in frame and not np.array_equal(frame.correct.to_numpy(dtype=bool), reconstructed.correct.to_numpy()):
                raise ValueError(f"Historical correctness does not reconstruct: {prediction_path}")
            results[name] = {
                "path": str(prediction_path), "sha256": _sha256_file(prediction_path),
                "model_id": str(frame.get("candidate", pd.Series([prediction_path.parent.name])).iloc[0]),
                "metrics": observed, "predictions": reconstructed,
            }
    return {"predictions": results, "unavailable": unavailable,
            "contract": "Training/fixed_internal_validation/seed42/official_false"}


def _scaler_for_experiment(X, context):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[context.train_idx]).astype(np.float32, copy=False)
    X_validation = scaler.transform(X[context.validation_idx]).astype(np.float32, copy=False)
    if (int(scaler.n_samples_seen_) != len(context.train_idx)
            or not np.isfinite(X_train).all() or not np.isfinite(X_validation).all()):
        raise ValueError("Scaler must fit finite Internal Train rows only")
    for name in ("mean", "scale", "var"):
        if not np.array_equal(getattr(scaler, name + "_")[:59],
                              np.asarray(context.stage6_scaler[name])):
            raise ValueError(f"Candidate scaler changed the matched clean 59D {name}")
    return scaler, X_train, X_validation


def _config_for_experiment(root, experiment_id):
    configs = _read_json(root / "contract.json").get("model_configs", {})
    if experiment_id not in configs:
        raise ValueError(f"Frozen config is missing for {experiment_id}")
    return configs[experiment_id]


def _run_artifact_files(directory: Path):
    excluded = {"status.json", "artifact_manifest.json"}
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.name not in excluded)


def _verify_artifact_manifest(directory: Path):
    manifest_path = directory / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Completed run artifact hash manifest is missing")
    actual_paths = {path.relative_to(directory).as_posix(): path for path in _run_artifact_files(directory)}
    if set(actual_paths) != set(expected):
        raise ValueError("Completed run contains missing or unregistered artifacts")
    for relative, path in actual_paths.items():
        if _sha256_file(path) != expected[relative]:
            raise ValueError(f"Completed run artifact hash mismatch: {relative}")
    return manifest


def verify_completed_experiment(context: TrainingContext, run_root, experiment_id):
    """Verify a previous COMPLETE run before skipping it during sequential resume."""
    root = Path(run_root).resolve()
    directory = root / experiment_id
    status = _read_json(directory / "status.json")
    if status.get("status") != "COMPLETE":
        raise RuntimeError(f"Experiment {experiment_id} is {status.get('status')}; do not retry/overwrite")
    artifact_manifest = _verify_artifact_manifest(directory)
    root_contract = _read_json(root / "contract.json")
    if (artifact_manifest.get("experiment_id") != experiment_id
            or artifact_manifest.get("config_id") != _config_for_experiment(root, experiment_id)["config_id"]
            or artifact_manifest.get("split_identity") != root_contract.get("context_identity")
            or artifact_manifest.get("feature_schema_fingerprint") != root_contract.get("feature_schema_fingerprint")):
        raise ValueError("Completed artifact manifest config/split/schema identity differs")
    config = _config_for_experiment(root, experiment_id)
    if _read_json(directory / "config.json") != config:
        raise ValueError("Completed experiment config differs from frozen run contract")
    metrics = _read_json(directory / "metrics.json")
    if (metrics.get("status") != "COMPLETE" or metrics.get("checkpoint_reload") not in
            ("EXACT_PASS", "ATOL_1E-7_PASS")):
        raise ValueError("Completed experiment lacks successful fit/reload evidence")
    if (metrics.get("split_identity") != root_contract.get("context_identity")
            or metrics.get("environment_sha256") != _sha256_file(root / "environment.json")):
        raise ValueError("Completed experiment split/environment identity differs")
    expected_feature_hash = (
        context.clean_fingerprint if metrics.get("block_id") is None else
        _read_json(root / "feature_cache" / "manifest.json")
        ["feature_blocks"][metrics["block_id"]]["artifact_sha256"]
    )
    if metrics.get("feature_cache_sha256") != expected_feature_hash:
        raise ValueError("Completed experiment feature-cache hash differs")
    scaler = _read_json(directory / "scaler.json")
    if (scaler.get("fit_scope") != "fixed_internal_train_only"
            or scaler.get("training_rows") != len(context.train_idx)
            or scaler.get("feature_names") != metrics.get("feature_names")
            or len(scaler.get("mean", [])) != metrics.get("input_dimension")
            or len(scaler.get("scale", [])) != metrics.get("input_dimension")
            or len(scaler.get("var", [])) != metrics.get("input_dimension")):
        raise ValueError("Completed experiment scaler scope/schema differs")
    frame = pd.read_parquet(directory / "predictions.parquet")
    _prediction_identity(frame, context.validation_record_keys,
                         context.validation_call_ids, context.validation_labels)
    observed, reconstructed = _metric_bundle(
        context.validation_labels, frame.probability_speaker1.to_numpy(dtype=np.float64),
        context.validation_record_keys, context.validation_call_ids,
    )
    if not frame.equals(reconstructed):
        raise ValueError("Completed predictions do not reconstruct exactly")
    for name, value in observed.items():
        if metrics.get(name) != value:
            raise ValueError(f"Completed metrics do not reconstruct: {name}")
    return {"status": "VERIFIED_COMPLETE", "experiment_id": experiment_id,
            "validation_accuracy": observed["validation_accuracy"]}


def _write_experiment_status(path: Path, status: str, **details):
    _write_json_replace(path, {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
                               **details})


def _mark_ledger(root: Path, experiment_id: str, status: str):
    ledger_path = root / "run_ledger.json"
    ledger = _read_json(ledger_path)
    contract = _read_json(root / "contract.json")
    if contract.get("schema_version") == "mission2_acoustic_feature_grid_resume_v1":
        if experiment_id not in candidate_experiment_ids():
            raise ValueError(f"Continuation ledger only accepts one of the 28 candidates: {experiment_id}")
        expected_fixed = {
            **{f"baseline_{model}": "REUSED_VERIFIED" for model in MODEL_IDS},
            **{name: "SKIPPED_BY_USER_SCOPE" for name in user_scope_excluded_experiment_ids()},
        }
        if any(ledger["experiments"].get(name) != value
               for name, value in expected_fixed.items()):
            raise ValueError("Continuation baseline/skip ledger entries were modified")
        ledger["experiments"][experiment_id] = status
        completed = [name for name in candidate_experiment_ids()
                     if ledger["experiments"].get(name) == "COMPLETE"]
        pending = [name for name in candidate_experiment_ids()
                   if ledger["experiments"].get(name) == "PENDING"]
        ledger["completed_count"] = len(completed)
        ledger["reused_baseline_count"] = len(MODEL_IDS)
        ledger["skipped_count"] = len(user_scope_excluded_experiment_ids())
        ledger["next_experiment"] = pending[0] if pending else None
        if status == "FAILED":
            ledger["status"] = "FAILED_STOP"
        elif len(completed) == len(candidate_experiment_ids()) and not pending:
            ledger["status"] = "ALL_28_COMPLETE_WITH_5_BASELINES_REUSED_7_SKIPPED"
        else:
            ledger["status"] = "RUNNING" if status == "RUNNING" else "READY"
        _write_json_replace(ledger_path, ledger)
        return

    ledger["experiments"][experiment_id] = status
    complete = [name for name, value in ledger["experiments"].items() if value == "COMPLETE"]
    ledger["completed_count"] = len(complete)
    amendment = _scope_amendment(root)
    excluded = set(amendment["excluded_experiments"]) if amendment else set()
    observed_skips = {name for name, value in ledger["experiments"].items()
                      if value == "SKIPPED_BY_USER_SCOPE"}
    if observed_skips != excluded:
        raise ValueError("Ledger skip statuses differ from the approved scope amendment")
    pending = [name for name in experiment_ids()
               if name not in excluded and ledger["experiments"].get(name) == "PENDING"]
    ledger["skipped_count"] = len(observed_skips)
    ledger["next_experiment"] = pending[0] if pending else None
    if status == "FAILED":
        ledger["status"] = "FAILED_STOP"
    elif (amendment is not None and not pending
          and len(complete) == len(active_experiment_ids())):
        ledger["status"] = "ALL_33_COMPLETE_WITH_7_SKIPPED"
    elif amendment is None and not pending and len(complete) == len(experiment_ids()):
        ledger["status"] = "ALL_40_COMPLETE"
    else:
        ledger["status"] = "RUNNING" if status == "RUNNING" else "READY"
    _write_json_replace(ledger_path, ledger)


def _save_scaler(scaler, directory, feature_names):
    _write_json_new(directory / "scaler.json", {
        "fit_scope": "fixed_internal_train_only", "training_rows": int(scaler.n_samples_seen_),
        "feature_names": list(feature_names), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
    })


def _fit_gandalf(config, feature_names, X_train, y_train, X_validation, y_validation, folder):
    from types import SimpleNamespace
    from . import mission2_stage13_untested_model_family as stage13

    payload = SimpleNamespace(
        feature_names=feature_names, X_train_scaled=X_train, y_train=y_train,
        X_evaluation_scaled=X_validation, y_evaluation=y_validation,
    )
    model, train_probability, val_probability, fit_seconds, _, reload_status, history = stage13._fit_gandalf(
        payload, folder, config=config,
    )
    if reload_status not in ("EXACT_PASS", "ATOL_1E-7_PASS"):
        raise ValueError("Stage 13 GANDALF checkpoint reload failed")
    return model, train_probability, val_probability, history, {
        "fit_seconds": float(fit_seconds), "best_epoch": stage13._best_epoch(history),
        "epochs_run": stage13._epochs_run(history), "checkpoint_reload": reload_status,
        "history_kind": history.attrs.get("history_kind", "epoch_history"),
    }


def _best_epoch_from_history(history):
    if "epoch" in history and len(history):
        return int(history.iloc[-1]["epoch"])
    return None


def _parameter_count(model):
    candidate = model if hasattr(model, "parameters") else getattr(model, "model", None)
    if candidate is None or not hasattr(candidate, "parameters"):
        return None
    return int(sum(parameter.numel() for parameter in candidate.parameters()))


def _fit_one_model(model_id, config, feature_names, X_train, y_train,
                   X_validation, y_validation, folder, device):
    if model_id in ("tabm", "ft_transformer", "tabular_resnet"):
        from . import mission2_complex_tabular_screen as screen
        model, history, fit = _fit_complex_model(
            model_id, config, X_train, y_train, X_validation, y_validation, device=device,
        )
        train_probability = screen.predict_probability(
            model_id, model, X_train, device=device, batch_size=config["batch_size"],
        )
        val_probability = screen.predict_probability(
            model_id, model, X_validation, device=device, batch_size=config["batch_size"],
        )
        checkpoint = folder / "checkpoint.pt"
        with checkpoint.open("xb") as stream:
            import torch
            torch.save({"candidate": model_id, "config": config,
                        "feature_names": list(feature_names),
                        "state_dict": {name: value.detach().cpu()
                                       for name, value in model.state_dict().items()}}, stream)
        model.cpu()
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["feature_names"] != list(feature_names):
            raise ValueError("Checkpoint config or feature schema differs")
        reloaded = _make_complex_model(model_id, config, device)
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        check = screen.predict_probability(model_id, reloaded, X_validation,
                                           device=device, batch_size=config["batch_size"])
        if not np.array_equal(val_probability, check):
            raise ValueError("Checkpoint reload probabilities differ exactly")
        fit["checkpoint_reload"] = "EXACT_PASS"
        return reloaded, train_probability, check, pd.DataFrame(history), fit
    if model_id == "dcnv2":
        model, history, fit = _fit_dcnv2(
            config, X_train, y_train, X_validation, y_validation, folder, device,
        )
        return model, _torch_probability(model, X_train, device), _torch_probability(model, X_validation, device), pd.DataFrame(history), fit
    return _fit_gandalf(config, feature_names, X_train, y_train,
                        X_validation, y_validation, folder)


def run_one(experiment_id, context: TrainingContext, run_root, *, device="cuda",
            _batch_preflight_validated=False):
    """Fit exactly one next experiment; incomplete or failed runs stop permanently."""
    root = Path(run_root).resolve()
    contract = _read_json(root / "contract.json")
    continuation = contract.get("schema_version") == "mission2_acoustic_feature_grid_resume_v1"
    allowed_ids = candidate_experiment_ids() if continuation else active_experiment_ids()
    if experiment_id not in allowed_ids:
        raise KeyError(f"run_one accepts one in-scope feature experiment only: {experiment_id}")
    if continuation:
        if (root.name != contract.get("run_id")
                or root.parent != Path(contract.get("output_root", "")).resolve()):
            raise ValueError("Continuation writes must remain under the configured MyDrive output folder")
    else:
        grid_base = root.parent
        for _ in GRID_RELATIVE.parts:
            grid_base = grid_base.parent
        if root.parent != (grid_base / GRID_RELATIVE).resolve():
            raise ValueError("Refuse training writes outside mission2_acoustic_feature_grid")
    if (contract.get("context_identity") != _context_identity(context)
            or contract.get("split") != context.split_audit
            or contract.get("official_validation_used") is not False):
        raise ValueError("Run contract Training/split/eGeMAPS/Official gate differs")
    if context.feature_matrices is None:
        raise RuntimeError("All seven verified feature matrices must be loaded before fitting")
    ledger_path = root / "run_ledger.json"
    ledger = _read_json(ledger_path)
    if continuation:
        if not _resume_ledger_valid(ledger):
            raise ValueError("Continuation ledger differs from 5 reused / 7 skipped / 28 fit scope")
        if _sha256_file(root / "baseline_reuse_manifest.json") != contract.get("baseline_reuse_manifest_sha256"):
            raise ValueError("Reused baseline manifest changed before candidate fit")
        if not _batch_preflight_validated:
            validate_colab_resume_run(context, root)
        order = candidate_experiment_ids()
    else:
        order = experiment_ids()
        amendment = _scope_amendment(root)
        excluded = set(amendment["excluded_experiments"]) if amendment else set()
        observed_skips = {name for name, value in ledger["experiments"].items()
                          if value == "SKIPPED_BY_USER_SCOPE"}
        if observed_skips != excluded:
            raise ValueError("Ledger skip statuses differ from the approved scope amendment")
    position = order.index(experiment_id)
    current_status = ledger["experiments"].get(experiment_id)
    if ledger.get("status") == "FAILED_STOP" and current_status != "COMPLETE":
        raise RuntimeError("Sequential stop: this run is FAILED_STOP and cannot start another fit")
    for prior_id in order[:position]:
        prior_status = ledger["experiments"].get(prior_id)
        if not continuation and prior_status == "SKIPPED_BY_USER_SCOPE" and prior_id in excluded:
            continue
        if prior_status != "COMPLETE":
            raise RuntimeError(f"Sequential stop: earlier experiment {prior_id} is {prior_status}")
        if not _batch_preflight_validated:
            verify_completed_experiment(context, root, prior_id)
    directory = root / experiment_id
    if current_status == "COMPLETE":
        return verify_completed_experiment(context, root, experiment_id)
    if current_status != "PENDING" or directory.exists():
        raise RuntimeError(f"Sequential stop: {experiment_id} is {current_status} or has partial artifacts")
    environment = _read_json(root / "environment.json")
    if environment.get("status") != "READY" or environment.get("device") != device:
        raise RuntimeError("Actual Colab environment/device must pass and be recorded before fitting")
    if device != "cuda":
        raise RuntimeError("Feature Grid is GPU-only; CPU fallback is disabled")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable before fit; no CPU retry is permitted")
    model_id, block_id, X, names = _experiment_input(context, experiment_id)
    config = _config_for_experiment(root, experiment_id)
    if config.get("input_dim") != X.shape[1]:
        raise ValueError("Frozen model input dimension differs from feature matrix")
    scaler, X_train, X_validation = _scaler_for_experiment(X, context)
    y_train = context.y[context.train_idx]
    y_validation = context.y[context.validation_idx]
    directory.mkdir(exist_ok=False)
    _write_json_new(directory / "config.json", config)
    _write_json_new(directory / "status.json", {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id, "run_id": root.name,
    })
    _mark_ledger(root, experiment_id, "RUNNING")
    _save_scaler(scaler, directory, names)
    model = None
    try:
        model, train_probability, val_probability, history, fit = _fit_one_model(
            model_id, config, names, X_train, y_train, X_validation, y_validation,
            directory, device,
        )
        metrics, predictions = _metric_bundle(
            y_validation, val_probability, context.validation_record_keys,
            context.validation_call_ids,
        )
        if not np.array_equal(predictions.y_true.to_numpy(), y_validation):
            raise ValueError("Prediction labels differ from persisted Internal Validation")
        train_accuracy = float(((train_probability >= THRESHOLD) == y_train).mean())
        metrics.update({
            "experiment_id": experiment_id, "model_id": model_id, "block_id": block_id,
            "input_dimension": int(X.shape[1]), "config_id": config["config_id"],
            "seed": SEED, "train_accuracy": train_accuracy,
            "train_validation_gap": train_accuracy - metrics["validation_accuracy"],
            "parameter_count": _parameter_count(model),
            "best_epoch": fit.get("best_epoch"), "epochs_run": fit.get("epochs_run"),
            "history_kind": fit.get("history_kind", "epoch_history"),
            "fit_seconds": float(fit["fit_seconds"]),
            "checkpoint_reload": fit["checkpoint_reload"],
            "feature_names": list(names), "scaler_fit_scope": "fixed_internal_train_only",
            "feature_cache_sha256": (
                context.clean_fingerprint if block_id is None else
                _read_json(root / "feature_cache" / "manifest.json")
                ["feature_blocks"][block_id]["artifact_sha256"]
            ),
            "split_identity": deepcopy(contract["context_identity"]),
            "environment_sha256": _sha256_file(root / "environment.json"),
        })
        history.to_csv(directory / "history.csv", index=False, mode="x")
        with (directory / "predictions.parquet").open("xb") as stream:
            predictions.to_parquet(stream, index=False)
        _write_json_new(directory / "metrics.json", metrics)
        artifact_hashes = {
            path.relative_to(directory).as_posix(): _sha256_file(path)
            for path in _run_artifact_files(directory)
        }
        _write_json_new(directory / "artifact_manifest.json", {
            "schema_version": "mission2_grid_run_artifacts_v1",
            "experiment_id": experiment_id, "config_id": config["config_id"],
            "split_identity": contract["context_identity"], "feature_schema_fingerprint": contract["feature_schema_fingerprint"],
            "files": artifact_hashes,
        })
        _write_experiment_status(directory / "status.json", "COMPLETE",
                                 experiment_id=experiment_id,
                                 completed_utc=datetime.now(timezone.utc).isoformat(),
                                 checkpoint_reload=fit["checkpoint_reload"])
        _mark_ledger(root, experiment_id, "COMPLETE")
        return metrics
    except Exception as error:
        _write_experiment_status(directory / "status.json", "FAILED",
                                 experiment_id=experiment_id,
                                 reason=f"{type(error).__name__}: {error}")
        _mark_ledger(root, experiment_id, "FAILED")
        raise
    finally:
        del model
        import gc
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()


def validate_candidate_preflight(context: TrainingContext, run_root):
    """Check all seven matrix/scaler/config pairs before spending GPU time."""
    root = Path(run_root).resolve()
    contract = _read_json(root / "contract.json")
    if (contract.get("schema_version") != "mission2_acoustic_feature_grid_resume_v1"
            or tuple(contract.get("candidate_experiments", ())) != candidate_experiment_ids()
            or contract.get("official_validation_used") is not False):
        raise ValueError("Candidate preflight requires the frozen 28-fit Training-only contract")
    if context.feature_matrices is None:
        raise RuntimeError("Verified cached feature matrices must be loaded before preflight")
    checks = []
    for block in FEATURE_BLOCKS:
        matrix = context.feature_matrices[block.block_id]
        if (matrix.dtype != np.float32
                or matrix.shape != (len(context.y), block.dimension)
                or not np.isfinite(matrix).all()):
            raise ValueError(f"Feature preflight failed for {block.block_id}: shape/dtype/NaN/Inf")
        experiment_id = f"{COLAB_CANDIDATE_MODEL_IDS[0]}_{block.block_id}"
        _, _, X, names = _experiment_input(context, experiment_id)
        if (X.shape[1] != 59 + block.dimension
                or len(names) != len(set(names))
                or len(names) != X.shape[1]):
            raise ValueError(f"Candidate schema preflight failed for {block.block_id}")
        _scaler_for_experiment(X, context)
        for model_id in COLAB_CANDIDATE_MODEL_IDS:
            config = contract.get("model_configs", {}).get(f"{model_id}_{block.block_id}")
            if (config is None or config.get("input_dim") != X.shape[1]
                    or config.get("seed") != SEED):
                raise ValueError(f"Frozen recipe preflight failed: {model_id}_{block.block_id}")
        checks.append({"feature_id": block.block_id, "rows": int(len(matrix)),
                       "feature_dimension": block.dimension, "model_input_dimension": int(X.shape[1]),
                       "finite": True, "train_only_scaler": True})
    return {"status": "PASS", "candidate_count": len(candidate_experiment_ids()),
            "feature_blocks": checks}


def run_remaining_candidate_fits(context: TrainingContext, run_root, *, device="cuda"):
    """Train remaining candidates serially; verify and persist after each fit."""
    root = Path(run_root).resolve()
    if device != "cuda":
        raise RuntimeError("Feature Grid continuation is GPU-only; CPU fallback is disabled")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab CUDA is unavailable; no candidate fit was started")
    validate_colab_resume_run(context, root)
    environment = _read_json(root / "environment.json")
    if environment.get("status") != "READY" or environment.get("device") != "cuda":
        raise RuntimeError("The live Colab GPU and package environment must pass before candidate fits")
    preflight = validate_candidate_preflight(context, root)
    results = []
    ledger = _read_json(root / "run_ledger.json")
    for index, experiment_id in enumerate(candidate_experiment_ids(), start=1):
        status = ledger["experiments"].get(experiment_id)
        if status == "COMPLETE":
            result = verify_completed_experiment(context, root, experiment_id)
            results.append(result)
            print(f"[{index}/{len(candidate_experiment_ids())}] VERIFIED_EXISTING {experiment_id}", flush=True)
            continue
        if status != "PENDING" or (root / experiment_id).exists():
            raise RuntimeError(
                f"Stopped without retry/overwrite: {experiment_id} is {status} or has partial artifacts"
            )
        result = run_one(
            experiment_id, context, root, device="cuda",
            _batch_preflight_validated=True,
        )
        verified = verify_completed_experiment(context, root, experiment_id)
        results.append(verified)
        ledger = _read_json(root / "run_ledger.json")
        print(
            f"[{index}/{len(candidate_experiment_ids())}] COMPLETE {experiment_id} "
            f"accuracy={result['validation_accuracy']:.8f} "
            f"fit_seconds={result['fit_seconds']:.1f} artifact={verified['status']}",
            flush=True,
        )
    final_ledger = _read_json(root / "run_ledger.json")
    if final_ledger.get("status") != "ALL_28_COMPLETE_WITH_5_BASELINES_REUSED_7_SKIPPED":
        raise RuntimeError("Candidate batch ended before all 28 artifacts were complete")
    return {"status": final_ledger["status"], "completed_count": final_ledger["completed_count"],
            "candidate_count": len(candidate_experiment_ids()), "preflight": preflight,
            "results": results}


def _paired_counts(reference, candidate):
    _prediction_identity(candidate, reference.record_key.to_numpy(),
                         reference.call_id.to_numpy(), reference.y_true.to_numpy())
    ref_correct = reference.correct.to_numpy(dtype=bool)
    cand_correct = candidate.correct.to_numpy(dtype=bool)
    regressed = ref_correct & ~cand_correct
    corrected = ~ref_correct & cand_correct
    return {
        "both_correct": int((ref_correct & cand_correct).sum()),
        "both_wrong": int((~ref_correct & ~cand_correct).sum()),
        "regressions": int(regressed.sum()), "corrections": int(corrected.sum()),
        "net_corrections": int(corrected.sum() - regressed.sum()),
        "changed_rows": int((regressed | corrected).sum()),
    }, regressed, corrected


def _analysis_input_fingerprint(context, root):
    """Bind derived comparison files to the exact fit artifacts and prior rows."""
    root = Path(root)
    contract = _read_json(root / "contract.json")
    continuation = contract.get("schema_version") == "mission2_acoustic_feature_grid_resume_v1"
    if continuation:
        experiment_ids_to_bind = tuple(
            f"baseline_{model}" for model in MODEL_IDS
        ) + candidate_experiment_ids()
        baseline_manifest_sha = _sha256_file(root / "baseline_reuse_manifest.json")
    else:
        amendment = _scope_amendment(root)
        experiment_ids_to_bind = active_experiment_ids() if amendment else experiment_ids()
        baseline_manifest_sha = None
    artifact_hashes = {}
    for experiment_id in experiment_ids_to_bind:
        directory = (root / "baseline_reuse" / experiment_id
                     if continuation and experiment_id.startswith("baseline_")
                     else root / experiment_id)
        artifact_hashes[experiment_id] = _sha256_file(directory / "artifact_manifest.json")
    prior_hashes = {
        name: item.get("sha256")
        for name, item in context.prior_predictions.get("predictions", {}).items()
    }
    return _json_fingerprint({
        "contract_sha256": _sha256_file(root / "contract.json"),
        "ledger_sha256": _sha256_file(root / "run_ledger.json"),
        "baseline_reuse_manifest_sha256": baseline_manifest_sha,
        "fit_artifact_manifest_sha256": artifact_hashes,
        "prior_prediction_sha256": prior_hashes,
    })


def _load_existing_analysis(context, root, input_fingerprint):
    """Return only a fully written and hash-verified prior analysis snapshot."""
    root = Path(root)
    output_names = (
        "comparison.csv", "historical_comparison.csv", "slice_comparison.csv",
        "paired_error_transitions.parquet", "analysis_metadata.json",
        "analysis_artifact_manifest.json",
    )
    existing = [name for name in output_names if (root / name).exists()]
    if not existing:
        return None
    if len(existing) != len(output_names):
        raise RuntimeError(
            "Comparison output set is incomplete; refusing to overwrite partial analysis artifacts"
        )
    metadata = _read_json(root / "analysis_metadata.json")
    manifest = _read_json(root / "analysis_artifact_manifest.json")
    expected_files = set(output_names) - {"analysis_artifact_manifest.json"}
    if (metadata.get("status") != "COMPLETE"
            or metadata.get("input_fingerprint") != input_fingerprint
            or manifest.get("status") != "COMPLETE"
            or manifest.get("input_fingerprint") != input_fingerprint
            or set(manifest.get("files", {})) != expected_files
            or metadata.get("official_validation_used") is not False):
        raise ValueError("Existing comparison output provenance differs from verified current inputs")
    for name, expected_hash in manifest["files"].items():
        if _sha256_file(root / name) != expected_hash:
            raise ValueError(f"Existing comparison output hash differs: {name}")
    return (
        pd.read_csv(root / "comparison.csv"),
        pd.read_csv(root / "historical_comparison.csv"),
        pd.read_csv(root / "slice_comparison.csv"),
        pd.read_parquet(root / "paired_error_transitions.parquet"),
    )


def _record_transitions(output_rows, reference_name, candidate_name, reference, candidate):
    counts, regressed, corrected = _paired_counts(reference, candidate)
    changed = regressed | corrected
    if changed.any():
        output_rows.append(pd.DataFrame({
            "reference": reference_name, "candidate": candidate_name,
            "record_key": reference.loc[changed, "record_key"].to_numpy(),
            "call_id": reference.loc[changed, "call_id"].to_numpy(),
            "y_true": reference.loc[changed, "y_true"].to_numpy(),
            "reference_prediction": reference.loc[changed, "prediction"].to_numpy(),
            "candidate_prediction": candidate.loc[changed, "prediction"].to_numpy(),
            "reference_error_type": reference.loc[changed, "error_type"].to_numpy(),
            "candidate_error_type": candidate.loc[changed, "error_type"].to_numpy(),
            "transition": np.where(regressed[changed], "CORRECT_TO_WRONG", "WRONG_TO_CORRECT"),
        }))
    return counts


def _load_grid_predictions(context, root):
    root = Path(root)
    ledger = _read_json(root / "run_ledger.json")
    contract = _read_json(root / "contract.json")
    if contract.get("schema_version") == "mission2_acoustic_feature_grid_resume_v1":
        result = validate_colab_resume_run(context, root)
        if result["status"] != "ALL_28_COMPLETE_WITH_5_BASELINES_REUSED_7_SKIPPED":
            raise RuntimeError("Comparison requires all 28 candidate fits to be complete")
        frames, metrics = {}, {}
        for experiment_id in (f"baseline_{model}" for model in MODEL_IDS):
            _verify_reused_baseline(context, root, experiment_id)
            folder = root / "baseline_reuse" / experiment_id
            frames[experiment_id] = pd.read_parquet(folder / "predictions.parquet")
            metrics[experiment_id] = _read_json(folder / "metrics.json")
        for experiment_id in candidate_experiment_ids():
            verify_completed_experiment(context, root, experiment_id)
            folder = root / experiment_id
            frames[experiment_id] = pd.read_parquet(folder / "predictions.parquet")
            metrics[experiment_id] = _read_json(folder / "metrics.json")
        return frames, metrics

    amendment = _scope_amendment(root)
    expected_ids = active_experiment_ids() if amendment else experiment_ids()
    expected_status = ("ALL_33_COMPLETE_WITH_7_SKIPPED" if amendment
                       else "ALL_40_COMPLETE")
    expected_skips = set(amendment["excluded_experiments"]) if amendment else set()
    observed_skips = {name for name, value in ledger.get("experiments", {}).items()
                      if value == "SKIPPED_BY_USER_SCOPE"}
    expected_count = len(expected_ids)
    if (set(ledger.get("experiments", {})) != set(experiment_ids())
            or ledger.get("status") != expected_status
            or ledger.get("completed_count") != expected_count
            or ledger.get("skipped_count", 0) != len(expected_skips)
            or observed_skips != expected_skips):
        raise RuntimeError(
            f"Comparison requires {expected_count} verified fits and "
            f"{len(expected_skips)} approved scope skips"
        )
    frames, metrics = {}, {}
    for experiment_id in expected_ids:
        if ledger["experiments"].get(experiment_id) != "COMPLETE":
            raise RuntimeError(f"Incomplete experiment prevents comparison: {experiment_id}")
        verify_completed_experiment(context, root, experiment_id)
        folder = Path(root) / experiment_id
        frames[experiment_id] = pd.read_parquet(folder / "predictions.parquet")
        metrics[experiment_id] = _read_json(folder / "metrics.json")
    return frames, metrics


def compare_grid_results(context: TrainingContext, run_root):
    """Produce matched-baseline, historical-model, slice, and changed-row evidence."""
    from . import mission2_model_experiments as models
    from . import mission2_standard_representation_comparison as diagnostics

    root = Path(run_root).resolve()
    frames, metrics = _load_grid_predictions(context, root)
    input_fingerprint = _analysis_input_fingerprint(context, root)
    existing_analysis = _load_existing_analysis(context, root, input_fingerprint)
    if existing_analysis is not None:
        return existing_analysis
    comparison_rows, historical_rows, slices, transitions = [], [], [], []
    validation_manifest = context.manifest.iloc[context.validation_idx].reset_index(drop=True).copy()
    duration = (pd.to_numeric(validation_manifest.endAt_ms, errors="raise").to_numpy(dtype=np.float64)
                - pd.to_numeric(validation_manifest.startAt_ms, errors="raise").to_numpy(dtype=np.float64)) / 1000.0
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("Internal Validation duration diagnostics are invalid")
    validation_manifest["duration_sec"] = duration
    validation_manifest["overlap"] = models._overlap_mask(validation_manifest)
    for name, expected in (("record_key", context.validation_record_keys),
                           ("call_id", context.validation_call_ids),
                           ("speaker", context.validation_labels)):
        if not np.array_equal(validation_manifest[name].astype(str).to_numpy(), expected.astype(str)):
            raise ValueError(f"Validation analysis-only metadata alignment differs: {name}")
    slice_masks = diagnostics.slice_masks(pd.DataFrame({
        "duration_sec": validation_manifest.duration_sec,
        "overlap": validation_manifest.overlap,
        "y_true": context.validation_labels,
    }))
    saved_prior = context.prior_predictions.get("predictions", {})

    for model_id in MODEL_IDS:
        baseline_id = f"baseline_{model_id}"
        reference = frames[baseline_id]
        baseline_metrics = metrics[baseline_id]
        for block in FEATURE_BLOCKS:
            candidate_id = f"{model_id}_{block.block_id}"
            if candidate_id not in frames:
                continue
            candidate = frames[candidate_id]
            candidate_metrics = metrics[candidate_id]
            paired, _, _ = _paired_counts(reference, candidate)
            comparison_rows.append({
                "comparison_type": (
                    "reused_stage13_baseline"
                    if _read_json(root / "contract.json").get("schema_version")
                    == "mission2_acoustic_feature_grid_resume_v1"
                    else "fresh_matched_baseline"
                ),
                "model_id": model_id, "feature_id": block.block_id,
                "feature_dimension": block.dimension,
                "baseline_accuracy": baseline_metrics["validation_accuracy"],
                "candidate_accuracy": candidate_metrics["validation_accuracy"],
                "accuracy_delta": candidate_metrics["validation_accuracy"] - baseline_metrics["validation_accuracy"],
                "baseline_recall_0": baseline_metrics["recall_0"],
                "candidate_recall_0": candidate_metrics["recall_0"],
                "baseline_recall_1": baseline_metrics["recall_1"],
                "candidate_recall_1": candidate_metrics["recall_1"],
                "baseline_errors": baseline_metrics["errors"],
                "candidate_errors": candidate_metrics["errors"],
                "baseline_tn": baseline_metrics["tn"], "baseline_fp": baseline_metrics["fp"],
                "baseline_fn": baseline_metrics["fn"], "baseline_tp": baseline_metrics["tp"],
                "candidate_tn": candidate_metrics["tn"], "candidate_fp": candidate_metrics["fp"],
                "candidate_fn": candidate_metrics["fn"], "candidate_tp": candidate_metrics["tp"],
                "baseline_hce": baseline_metrics["hce"], "candidate_hce": candidate_metrics["hce"],
                **paired,
            })
            _record_transitions(transitions, baseline_id, candidate_id, reference, candidate)
            for slice_name, mask in slice_masks.items():
                n = int(mask.sum())
                if n == 0:
                    continue
                ref_correct = reference.correct.to_numpy(dtype=bool)
                cand_correct = candidate.correct.to_numpy(dtype=bool)
                slices.append({
                    "model_id": model_id, "feature_id": block.block_id,
                    "slice": slice_name, "rows": n,
                    "baseline_accuracy": float(ref_correct[mask].mean()),
                    "candidate_accuracy": float(cand_correct[mask].mean()),
                    "baseline_hce": int((mask & ~ref_correct & (reference.confidence.to_numpy() >= .9)).sum()),
                    "candidate_hce": int((mask & ~cand_correct & (candidate.confidence.to_numpy() >= .9)).sum()),
                    "corrections": int((mask & ~ref_correct & cand_correct).sum()),
                    "regressions": int((mask & ref_correct & ~cand_correct).sum()),
                })
            confidence_bins = pd.cut(candidate.confidence, bins=[.5, .6, .7, .8, .9, 1.0000001],
                                     labels=["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
                                     include_lowest=True)
            confidence_labels = np.asarray(confidence_bins.astype(str))
            for label in pd.unique(confidence_labels):
                mask = confidence_labels == label
                if mask.any():
                    slices.append({
                        "model_id": model_id, "feature_id": block.block_id,
                        "slice": f"confidence_{label}", "rows": int(mask.sum()),
                        "baseline_accuracy": float(reference.correct.to_numpy()[mask].mean()),
                        "candidate_accuracy": float(candidate.correct.to_numpy()[mask].mean()),
                        "baseline_hce": int((mask & ~reference.correct.to_numpy(dtype=bool)
                                              & (reference.confidence.to_numpy() >= .9)).sum()),
                        "candidate_hce": int((mask & ~candidate.correct.to_numpy(dtype=bool)
                                              & (candidate.confidence.to_numpy() >= .9)).sum()),
                        "corrections": None, "regressions": None,
                    })
            for historical_name, historical in saved_prior.items():
                old_frame = historical["predictions"]
                paired_old, _, _ = _paired_counts(old_frame, candidate)
                historical_rows.append({
                    "candidate_id": candidate_id, "historical_model": historical_name,
                    "historical_model_id": historical["model_id"],
                    "historical_accuracy": historical["metrics"]["validation_accuracy"],
                    "candidate_accuracy": candidate_metrics["validation_accuracy"],
                    "accuracy_delta": candidate_metrics["validation_accuracy"] - historical["metrics"]["validation_accuracy"],
                    "historical_errors": historical["metrics"]["errors"],
                    "candidate_errors": candidate_metrics["errors"],
                    **paired_old,
                })
                _record_transitions(transitions, historical_name, candidate_id, old_frame, candidate)

    comparison = pd.DataFrame(comparison_rows)
    historical_comparison = pd.DataFrame(historical_rows, columns=[
        "candidate_id", "historical_model", "historical_model_id",
        "historical_accuracy", "candidate_accuracy", "accuracy_delta",
        "historical_errors", "candidate_errors", "both_correct", "both_wrong",
        "regressions", "corrections", "net_corrections", "changed_rows",
    ])
    slice_comparison = pd.DataFrame(slices)
    comparison.to_csv(root / "comparison.csv", index=False, mode="x", encoding="utf-8")
    historical_comparison.to_csv(root / "historical_comparison.csv", index=False,
                                 mode="x", encoding="utf-8")
    slice_comparison.to_csv(root / "slice_comparison.csv", index=False, mode="x", encoding="utf-8")
    transitions_frame = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame(
        columns=["reference", "candidate", "record_key", "call_id", "y_true",
                 "reference_prediction", "candidate_prediction", "reference_error_type",
                 "candidate_error_type", "transition"]
    )
    with (root / "paired_error_transitions.parquet").open("xb") as stream:
        transitions_frame.to_parquet(stream, index=False)
    _write_json_new(root / "analysis_metadata.json", {
        "status": "COMPLETE", "input_fingerprint": input_fingerprint,
        "source_split": "Training", "evaluation_set": "fixed_internal_validation",
        "official_validation_used": False, "split_seed": SEED, "group_key": "call_id",
        "historical_internal_validation_reused": True,
        "candidate_selection_bias": (
            "28 single-feature candidates compared on one repeatedly used Internal Validation"
            if _read_json(root / "contract.json").get("schema_version")
            == "mission2_acoustic_feature_grid_resume_v1" else
            f"{len(active_experiment_ids()) - len(MODEL_IDS)} single-feature candidates compared on one repeatedly used Internal Validation"
            if _scope_amendment(root) else
            "35 single-feature candidates compared on one repeatedly used Internal Validation"
        ),
        "feature_selection": "none; no automated winning model or block selected",
        "analysis_metadata_inputs": "duration/overlap only; never model features",
        "prior_models_verified": list(saved_prior),
        "prior_models_unavailable": context.prior_predictions.get("unavailable", []),
        "transition_rows_only": True,
    })
    output_names = (
        "comparison.csv", "historical_comparison.csv", "slice_comparison.csv",
        "paired_error_transitions.parquet", "analysis_metadata.json",
    )
    _write_json_new(root / "analysis_artifact_manifest.json", {
        "status": "COMPLETE", "input_fingerprint": input_fingerprint,
        "files": {name: _sha256_file(root / name) for name in output_names},
    })
    return comparison, historical_comparison, slice_comparison, transitions_frame


def write_final_report_ko(context: TrainingContext, run_root):
    """Write the Stage 14 evidence handoff without selecting a final model."""
    root = Path(run_root).resolve()
    ledger = _read_json(root / "run_ledger.json")
    contract = _read_json(root / "contract.json")
    continuation = contract.get("schema_version") == "mission2_acoustic_feature_grid_resume_v1"
    if continuation:
        if validate_colab_resume_run(context, root)["status"] != "ALL_28_COMPLETE_WITH_5_BASELINES_REUSED_7_SKIPPED":
            raise RuntimeError("Final report requires all 28 candidate fits and five reused baselines")
        expected_status = "ALL_28_COMPLETE_WITH_5_BASELINES_REUSED_7_SKIPPED"
        expected_count = len(candidate_experiment_ids())
    else:
        expected_status = "ALL_33_COMPLETE_WITH_7_SKIPPED" if _scope_amendment(root) else "ALL_40_COMPLETE"
        expected_count = len(active_experiment_ids()) if _scope_amendment(root) else len(experiment_ids())
    amendment = _scope_amendment(root)
    if (ledger.get("status") != expected_status
            or ledger.get("completed_count") != expected_count
            or ledger.get("skipped_count", 0) != (len(user_scope_excluded_experiment_ids())
                                                     if (amendment or continuation) else 0)
            or ({name for name, value in ledger.get("experiments", {}).items()
                 if value == "SKIPPED_BY_USER_SCOPE"}
                != (set(user_scope_excluded_experiment_ids()) if (amendment or continuation) else set()))):
        raise RuntimeError(f"Final report requires all {expected_count} in-scope verified fits")
    comparison = pd.read_csv(root / "comparison.csv")
    historical = pd.read_csv(root / "historical_comparison.csv")
    environment = _read_json(root / "environment.json")
    retained_ft_baseline = None
    if continuation:
        _verify_reused_baseline(context, root, "baseline_ft_transformer")
        retained_ft_baseline = _read_json(
            root / "baseline_reuse" / "baseline_ft_transformer" / "metrics.json"
        )
    elif amendment:
        verify_completed_experiment(context, root, "baseline_ft_transformer")
        retained_ft_baseline = _read_json(root / "baseline_ft_transformer" / "metrics.json")
    lines = [
        "# DCC Mission 2 Acoustic Feature Grid 결과 보고서",
        "",
        "## 범위와 실행 상태",
        "",
        (f"- 실행 상태: {ledger['status']} ({ledger['completed_count']}/{expected_count} 신규 학습 완료, "
         f"기존 Baseline {ledger.get('reused_baseline_count', 0)}개 재사용, Feature fit {len(candidate_experiment_ids())}개 대상)."
         if continuation else
         f"- 실행 상태: {ledger['status']} ({ledger['completed_count']}/{expected_count} 학습 완료)"),
        (f"- 사용자 범위 제외: {ledger.get('skipped_count', 0)}개 FT-Transformer Feature 실험; 기존 FT-Transformer Baseline은 검증 후 재사용함."
         if continuation else
         f"- 사용자 범위 제외: {ledger.get('skipped_count', 0)}개 SKIPPED_BY_USER_SCOPE (FT-Transformer F01–F07); 완료된 FT-Transformer 59D Baseline은 유지함."
         if amendment else ""),
        (f"- 보존된 FT-Transformer Baseline: Accuracy {retained_ft_baseline['validation_accuracy']:.6f}, "
         f"Recall 0/1 {retained_ft_baseline['recall_0']:.6f}/{retained_ft_baseline['recall_1']:.6f}, "
         f"HCE {retained_ft_baseline['hce']}, 학습 {retained_ft_baseline['fit_seconds']:.1f}초."
         if amendment else ""),
        f"- 런타임: {environment.get('device')} / {environment.get('gpu')}",
        "- 데이터: organizer Training만 사용. Official Validation은 열거나 평가하지 않음.",
        "- Split: 기존 Stage 6 고정 Internal Split, seed 42, 그룹 키 call_id.",
        f"- 행 수: Train {contract['split']['train_rows']:,} / Validation {contract['split']['validation_rows']:,}; "
        f"통화 수: Train {contract['split']['train_calls']:,} / Validation {contract['split']['validation_calls']:,}.",
        ("- 비교: 기존 실행의 동일 모델 59D Baseline을 검증 후 재사용해 Feature 증분을 비교하고, 저장된 Stage 10–13 예측 비교는 별도로 제시함."
         if continuation else
         "- 비교: 새 모델별 59D Baseline 대비 Feature 효과와 저장된 Stage 10–13 Internal 예측 대비 변화를 분리함."),
        "",
        ("## 기존 Baseline 재사용 및 28개 Feature 비교 결과" if continuation else
         "## 사전 고정 33회 학습 및 28개 Feature 비교 결과" if amendment
         else "## 사전 고정 40회 실험 결과"),
        "",
        ("모든 값은 동일한 기존 Internal Validation에서 얻은 Development 결과다. 다섯 59D Baseline은 기존 Run의 완료 artifact를 검증해 재사용했다. 28개 신규 Feature fit은 Baseline과 고정 Split·Seed·recipe 기준을 맞췄고, 우승을 자동 선정하지 않는다. FT-Transformer Feature 7개는 제외했다."
         if continuation else
         "모든 값은 동일한 기존 Internal Validation에서 얻은 Development 결과다. 표는 사전 고정 순서이며, 우승 모델을 자동 선정하지 않는다. FT-Transformer의 일곱 Feature 추가는 사용자 범위 변경으로 제외했다."
         if amendment else
         "모든 값은 동일한 기존 Internal Validation에서 얻은 Development 결과다. 표는 사전 고정 모델·Feature 순서이며, 우승 모델을 자동 선정하지 않는다."),
        "",
        "| 모델 | Feature | 추가 차원 | Baseline Acc | 후보 Acc | Δ Acc | Baseline Recall 0/1 | 후보 Recall 0/1 | 교정/회귀 | HCE baseline→후보 | 혼동행렬 TN/FP/FN/TP (Baseline→후보) |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.feature_id} | {row.feature_dimension} | "
            f"{row.baseline_accuracy:.6f} | {row.candidate_accuracy:.6f} | {row.accuracy_delta:+.6f} | "
            f"{row.baseline_recall_0:.6f}/{row.baseline_recall_1:.6f} | "
            f"{row.candidate_recall_0:.6f}/{row.candidate_recall_1:.6f} | "
            f"{row.corrections}/{row.regressions} | {row.baseline_hce}→{row.candidate_hce} | "
            f"{row.baseline_tn}/{row.baseline_fp}/{row.baseline_fn}/{row.baseline_tp}→"
            f"{row.candidate_tn}/{row.candidate_fp}/{row.candidate_fn}/{row.candidate_tp} |"
        )
    lines.extend([
        "",
        "## 기존 저장 모델 대비",
        "",
        f"정확한 record_key, call_id, label 정합성을 통과한 역사적 예측 비교 행: {len(historical)}.",
        "성능 변화와 실제 교정·회귀 수는 historical_comparison.csv에서 확인한다. 행 단위 차이는 paired_error_transitions.parquet에 교정 또는 회귀가 발생한 행만 저장했다.",
        f"현재 전달된 prior-run 경로에서 확인한 저장 모델 수: {len(context.prior_predictions.get('predictions', {}))}.",
        "경로 누락 또는 내부 Validation 근거 부족으로 제외된 artifact는 contract.json 및 analysis_metadata.json에 사유를 남겼다.",
        "",
        "## 해석 제한",
        "",
        "- 동일 Internal Validation은 과거 Stage 10–13 실험에도 재사용됐다. 누적 사용에 따른 적응 위험이 있다.",
         ("- 28개 단일 Feature 후보를 같은 Validation에서 비교했다. 기존 Baseline은 과거 모델 선택에 사용된 결과를 재사용하므로, 과거 누적 사용과 이번 후보 비교 모두 선택 편향·점수 낙관 편향에 기여한다."
          if continuation else
          "- 28개 단일 Feature 후보를 같은 Validation에서 비교했다. 최고 점수를 골라내면 선택 편향과 점수 낙관 편향이 생긴다."
          if amendment else
         "- 35개 단일 Feature 후보를 같은 Validation에서 비교했다. 최고 점수를 골라내면 선택 편향과 점수 낙관 편향이 생긴다."),
        "- Accuracy, 클래스별 Recall, 혼동행렬, HCE, paired 교정/회귀와 duration/overlap/confidence slice는 Development 분석이다.",
        "- duration과 overlap은 오류 분석 metadata로만 사용했고 모델 입력에는 포함하지 않았다. threshold는 0.5다.",
        "- 이 결과는 최종 모델 확정이나 Official Validation 결과가 아니다. 후보 확정과 이후 Official Validation은 기존 로드맵을 따른다.",
        "",
        "## Stage 14 인계",
        "",
        ("Stage 14 검토를 위해 재사용한 5개 Baseline과 28개 Feature checkpoint, config, scaler, history, Internal Validation prediction/metric, row-level 교정·회귀를 보존했다. baseline_reuse/에는 기존 checkpoint 복사본과 원본 SHA 근거가 있고, 이전 Run은 수정하지 않았다. Ensemble은 실행하지 않았고 최종 후보도 지정하지 않았다."
         if continuation else
         "Stage 14에서 검토할 수 있도록 각 모델의 59D Baseline과 단일 Feature checkpoint, frozen config, scaler, history, Validation prediction/metric, row-level 교정·회귀를 보존했다. 이 단계에서는 Ensemble을 학습하지 않았고 최종 후보를 지정하지 않았다."),
        "",
        "## 산출물",
        "",
        "- contract.json, run_ledger.json, environment.json",
        ("- baseline_reuse/: 재사용 Baseline 5개 checkpoint/prediction/metric/config/scaler와 원본 artifact hash"
         if continuation else ""),
        ("- feature_cache/: 원본 Feature cache manifest/hash 참조; 행렬은 기존 Run에서 재사용"
         if continuation else "- feature_cache/: Training-only matrix, schema, hash, short/silent fallback count"),
        "- 각 experiment_id 디렉터리: config, scaler, history, checkpoint, predictions, metrics, artifact hashes",
        "- comparison.csv, historical_comparison.csv, slice_comparison.csv, paired_error_transitions.parquet",
        "- analysis_metadata.json",
    ])
    path = root / "final_report_ko.md"
    report_text = "\n".join(lines) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != report_text:
            raise ValueError("Existing final report differs from the verified run; refusing overwrite")
        return path
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(report_text)
    return path
