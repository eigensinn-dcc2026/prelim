"""Mission 2 module: 신고자(speaker=1) / 119대원(speaker=0) per utterance (owner: DCC M2).

Wraps the frozen ``scripts/mission2_inference.py`` (do not edit it: frozen-zip
reruns depend on it) and returns the official columns
``audio file name,startAt,endAt,speaker`` instead of its ``audio_file_name``.
Pairing (``pair_by_stem_strict``), the expected-row count, the completeness
check, the L1/L2 predictors, the ensemble soft vote, the L2 config checks and
the per-utterance majority fallback are all the script's own functions; the
per-utterance loop below is the script's ``main()`` loop, row for row, minus
the CSV writing (mission modules must not write files).

Rules (#41): each prediction uses only that utterance's own audio slice
(``startAt``..``endAt``). No ``text``, no utterance order/index, no call-level
information is read; labels pass through unchanged.

Checkpoint selection is by ``ckpt_path`` (the official ``--ckpt_path``):

- ``.pt`` / ``.pth`` / ``.ckpt`` -> loaded with ``torch.load(weights_only=True)``.
  If it is an L1 checkpoint converted by ``scripts/convert_mission2_l1_pkl.py``
  (``format == L1_CKPT_FORMAT``) the embedded L1 pickle is used; otherwise it
  is an L2 CNN checkpoint (single model or packed seed ensemble) and goes to
  the script's ``_load_l2_predictor``.
- anything else (``.pkl``) -> the script's L1 loader (``load_model`` pickle).

The official rules only accept ``.pt``/``.pth``/``.ckpt`` weight files, hence
the L1 converter. ``predict(..., ckpt_path=None)`` falls back to
``DEFAULT_CKPT``. Checkpoints are not committed (see ``CKPT_FILENAMES``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd

from dcc_data.mission2_checkpoint import CheckpointConfigError
from dcc_data.mission2_features import (
    FEATURE_COLUMNS,
    N_FFT,
    read_call_waveform,
    slice_and_pad,
    utterance_feature_vector,
    utterance_frame_range,
)

COLUMNS = ("audio file name", "startAt", "endAt", "speaker")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "mission2_inference.py"

# Expected file names under checkpoints/mission2/ (not committed; Drive + SHA-256 in reports).
CKPT_FILENAMES = {
    # L2: 3-seed SpeakerCNNLarge soft-vote ensemble, scripts/pack_mission2_l2_ensemble.py
    "l2_ensemble": "m2_cnn_large_raw_ensemble.pt",
    # L1 9k insurance: LightGBM pickle from train_mission2_baseline.py, then
    # scripts/convert_mission2_l1_pkl.py -> same model inside a .ckpt
    "l1_9k_pkl": "m2_baseline_lgbm_9k.pkl",
    "l1_9k": "m2_baseline_lgbm_9k.ckpt",
}
# Final M2 submission is not decided yet (L2 ensemble vs L1 9k); change this one line when it is.
DEFAULT_CKPT = REPOSITORY_ROOT / "checkpoints" / "mission2" / CKPT_FILENAMES["l2_ensemble"]

TORCH_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

# torch file that carries an L1 (GBM) pickle verbatim; see save_l1_torch_ckpt.
L1_CKPT_FORMAT = "dcc2026-mission2-l1-pickle-v1"

PredictFn = Callable[[np.ndarray, int, float], int]


def _script() -> ModuleType:
    """The frozen scripts/mission2_inference.py, imported once (same module object the tests use)."""

    module = sys.modules.get("mission2_inference")
    if module is not None and Path(getattr(module, "__file__", "")).resolve() == SCRIPT_PATH.resolve():
        return module
    spec = importlib.util.spec_from_file_location("mission2_inference", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mission2_inference"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("mission2_inference", None)
        raise
    return module


def _use_all_cpu_threads() -> None:
    """Safety net: root inference.py sets OMP/MKL threads, but mission2_features' setdefault(..., '1')
    wins if this module is imported on its own."""

    import torch

    torch.set_num_threads(os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# L1 (.pkl) inside a torch .ckpt
# ---------------------------------------------------------------------------


def save_l1_torch_ckpt(pkl_path: str | Path, ckpt_path: str | Path) -> dict:
    """Write the L1 pickle's exact bytes into a torch file; returns the stored header."""

    import torch

    raw = Path(pkl_path).read_bytes()
    pickle.loads(raw)  # fail here, not at inference, if it is not a readable pickle
    checkpoint = {
        "format": L1_CKPT_FORMAT,
        "l1_pickle": raw,
        "l1_pickle_sha256": hashlib.sha256(raw).hexdigest(),
        "source_name": Path(pkl_path).name,
    }
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    return {key: value for key, value in checkpoint.items() if key != "l1_pickle"}


def _l1_from_torch_checkpoint(checkpoint: dict) -> tuple[Any, dict]:
    raw = checkpoint["l1_pickle"]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != checkpoint.get("l1_pickle_sha256"):
        raise CheckpointConfigError(f"L1 pickle sha256 {digest} != recorded {checkpoint.get('l1_pickle_sha256')}")
    payload = pickle.loads(raw)  # same as dcc_data.mission2_train.load_model
    return payload["model"], payload["meta"]


def _l1_predictor(model: Any, meta: dict) -> tuple[PredictFn, int]:
    """Same as the script's _load_l1_predictor, but from an already-loaded (model, meta)."""

    if tuple(meta["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match the current feature extractor "
            "(dcc_data.mission2_features.FEATURE_COLUMNS) -- retrain or update the checkpoint"
        )
    fallback_speaker = int(meta.get("majority_class", 1))

    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        vector = utterance_feature_vector(segment, sample_rate, duration_sec)
        return int(model.predict(vector.reshape(1, -1).astype(np.float64))[0])

    return predict, fallback_speaker


def load_predictor(ckpt_path: str | Path) -> tuple[PredictFn, int, str]:
    """(predict_fn, fallback_speaker, kind) with kind in {"l1", "l1-ckpt", "l2"}."""

    ckpt_path = Path(ckpt_path)
    script = _script()
    if ckpt_path.suffix.lower() not in TORCH_CHECKPOINT_SUFFIXES:
        predict_fn, fallback = script._load_l1_predictor(ckpt_path)
        return predict_fn, fallback, "l1"

    import torch

    _use_all_cpu_threads()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == L1_CKPT_FORMAT:
        predict_fn, fallback = _l1_predictor(*_l1_from_torch_checkpoint(checkpoint))
        return predict_fn, fallback, "l1-ckpt"
    del checkpoint
    predict_fn, fallback = script._load_l2_predictor(ckpt_path)
    return predict_fn, fallback, "l2"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path | None = None) -> pd.DataFrame:
    """Official Mission 2 rows, one per labelled utterance, in each label file's own order."""

    ckpt_path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    script = _script()
    predict_fn, fallback_speaker, kind = load_predictor(ckpt_path)

    pairs = script.pair_by_stem_strict(Path(audio_dir), Path(label_dir))
    expected_rows = script.count_label_utterances(json_path for _, _, json_path in pairs)

    rows: list[tuple] = []
    failure_count = 0
    calls_seen = 0
    for stem, wav_path, json_path in pairs:
        calls_seen += 1
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        utterances = document.get("utterances", [])

        waveform = None
        sample_rate = None
        try:
            waveform, sample_rate = read_call_waveform(wav_path)
        except Exception as exc:  # noqa: BLE001 - one bad call must not stop the whole run
            print(f"[mission2] failed to read {wav_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

        for utt_idx, utterance in enumerate(utterances):  # utt_idx is only used in log messages
            start_at = utterance.get("startAt")
            end_at = utterance.get("endAt")
            speaker = fallback_speaker
            if waveform is not None:
                try:
                    start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
                    segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
                    duration_sec = (float(end_at) - float(start_at)) / 1000.0
                    speaker = predict_fn(segment, sample_rate, duration_sec)
                except CheckpointConfigError:
                    raise  # setup error, never a per-utterance fallback
                except Exception as exc:  # noqa: BLE001 - majority fallback, keep the row
                    failure_count += 1
                    print(
                        f"[mission2] feature/predict failed for stem={stem} utt_idx={utt_idx}: "
                        f"{type(exc).__name__}: {exc} -- falling back to majority class",
                        file=sys.stderr,
                    )
            else:
                failure_count += 1
            rows.append((wav_path.name, start_at, end_at, int(speaker)))

    script.verify_output_complete(
        expected_calls=len(pairs), calls_seen=calls_seen, expected_rows=expected_rows, rows=rows
    )
    print(
        f"[mission2] calls={len(pairs)} rows={len(rows)} fallback={failure_count} ckpt={kind}:{ckpt_path.name}",
        file=sys.stderr,
    )
    # object dtype keeps startAt/endAt exactly as the JSON had them (no int -> float upcast),
    # so the CSV text matches the script's csv.DictWriter output.
    frame = pd.DataFrame(rows, columns=list(COLUMNS), dtype=object)
    frame["speaker"] = frame["speaker"].astype("int64")
    return frame
