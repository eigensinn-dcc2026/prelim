"""Mission 2 baseline (L1) feature pipeline: acoustic statistics only.

Builds on the repository's existing manifest/guards/audio primitives instead
of re-implementing wav<->json pairing or split isolation:

- ``dcc_data.manifest`` already pairs wav/json files by filename stem (the
  ``audioPath`` field inside each JSON is known to be unreliable -- see
  ``manifest.build_manifest``). We reuse its output (a loaded manifest) here.
- ``dcc_data.guards.ensure_training_rows`` already enforces that Validation
  data cannot reach a training path. We call it rather than re-checking
  ``row.split`` by hand.
- ``dcc_data.audio`` already does dependency-light PCM WAV reading. We read
  each call's WAV exactly once (``read_wav_frames(path, 0, frame_count)``)
  and slice utterances from the in-memory array, instead of reopening the
  file per utterance -- opening per utterance was measured (see the M2
  baseline planning notes) to be the dominant cost at ~30 utterances/call.

Only ``startAt``/``endAt``/``speaker`` are ever read from a labeling JSON's
utterances. ``text`` is never touched. ``utt_idx`` (the utterance's position
in the JSON) is carried only so downstream code can restore submission row
order -- it must never be used as a model feature, since the contest
explicitly forbids order-derived signals ("발화의 홀짝 등 순서 사용불가").
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

# Must be set before numpy/librosa are imported (BLAS thread count is fixed
# at library load time). build_seoul_training_dataset()'s max_workers > 1
# path already parallelizes across OS processes; without this, each
# process's own BLAS backend ALSO spawns one thread per CPU core on top of
# that -- N worker processes x C BLAS threads each is pure oversubscription.
# On a 12-core/16GB machine this reliably crashed every worker ("OpenBLAS
# error: Memory allocation still failed after 10 retries" / Windows "paging
# file too small") with as few as 4 workers. Every caller of this module
# (the CLI script, tests, ad-hoc notebooks) benefits from pinning this here
# rather than each having to remember to set it themselves.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import librosa

from .audio import read_wav_frames, read_wav_info
from .guards import TRAINING, ensure_training_rows
from .manifest import ManifestRow
from .mission2 import milliseconds_to_frame


# ---------------------------------------------------------------------------
# Region filtering (Training must be Seoul-only; see the contest rules and
# the team EDA). ManifestRow does not carry a region field, so we recover it
# from the path using the same "TS_<region>_구급"-style folder convention
# scripts/audit_dcc_dataset.py already relies on for the same purpose.
# ---------------------------------------------------------------------------

_REGION_DIR_RE = re.compile(r"^(TS|TL|VS|VL)_([^_]+)_구급$")
SEOUL_REGION = unicodedata.normalize("NFC", "서울")


def path_region(relative_path: str) -> str | None:
    """Return the region token encoded in a manifest-relative path, if any.

    Paths that started life as Korean-named folders inside a zip archive (as
    this dataset's do) can arrive as NFD (decomposed jamo) instead of NFC
    (precomposed) -- the two render identically but compare unequal, and NFD
    also breaks a literal match against "_구급$" in the regex below. Every
    path part is normalized to NFC before matching so this is independent of
    whatever form the filesystem/zip handed us.
    """

    for part in PurePosixPath(relative_path).parts:
        normalized_part = unicodedata.normalize("NFC", part)
        match = _REGION_DIR_RE.fullmatch(normalized_part)
        if match:
            return match.group(2)
    return None


def is_seoul_path(relative_path: str) -> bool:
    return path_region(relative_path) == SEOUL_REGION


def select_seoul_training_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Training-only, Seoul-only rows -- the only rows this pipeline may use.

    ``ensure_training_rows`` is a fail-fast *assertion* that a batch is
    already pure Training (it raises on a mixed batch, it does not filter)
    -- see ``dcc_data.guards``. A manifest loaded via ``load_manifest``
    normally contains both splits together, so we filter by ``row.split``
    first and then run the guard as defense-in-depth against a filtering bug.
    """

    training_only = [row for row in rows if row.split == TRAINING]
    training_rows = ensure_training_rows(training_only)
    return [
        row
        for row in training_rows
        if is_seoul_path(row.wav_path) and is_seoul_path(row.json_path)
    ]


def sample_calls(rows: list[ManifestRow], n_calls: int, seed: int) -> list[ManifestRow]:
    """Reproducibly sample ``n_calls`` calls (not utterances) by call/row.

    Sampling at the call level (rather than the utterance level) is required
    for GroupShuffleSplit-by-call_id to make sense downstream.
    """

    if n_calls <= 0 or n_calls >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n_calls)


# ---------------------------------------------------------------------------
# Module A: utterance-level index (metadata only, no audio touched here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtteranceRecord:
    call_id: str
    wav_stem: str
    wav_path: str
    json_path: str
    utt_idx: int
    start_at: float
    end_at: float
    duration_sec: float
    speaker: int


def _load_call_utterances(json_path: Path) -> tuple[str, list[dict]]:
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    utterances = document.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"utterances must be a list: {json_path}")
    call_id = document.get("_id") or json_path.stem
    return str(call_id), utterances


def build_utterance_records(row: ManifestRow, data_root: Path) -> list[UtteranceRecord]:
    """Expand one call's manifest row into its utterance-level index rows.

    Only ``startAt``/``endAt``/``speaker`` are read from each utterance --
    never ``text`` (Mission 2 forbids using transcript content as a feature).
    """

    json_path = data_root / row.json_path
    call_id, utterances = _load_call_utterances(json_path)
    records: list[UtteranceRecord] = []
    for idx, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            raise ValueError(f"utterance must be an object: {json_path}[{idx}]")
        start_at = utterance["startAt"]
        end_at = utterance["endAt"]
        speaker = utterance["speaker"]
        if not isinstance(start_at, Real) or isinstance(start_at, bool):
            raise TypeError(f"startAt must be numeric: {json_path}[{idx}]")
        if not isinstance(end_at, Real) or isinstance(end_at, bool):
            raise TypeError(f"endAt must be numeric: {json_path}[{idx}]")
        if start_at < 0:
            raise ValueError(f"startAt must be non-negative: {json_path}[{idx}] got {start_at!r}")
        if end_at <= start_at:
            raise ValueError(
                f"endAt must be greater than startAt: {json_path}[{idx}] "
                f"got startAt={start_at!r} endAt={end_at!r}"
            )
        if speaker not in (0, 1):
            raise ValueError(f"speaker must be 0 or 1: {json_path}[{idx}] got {speaker!r}")
        records.append(
            UtteranceRecord(
                call_id=call_id,
                wav_stem=row.stem,
                wav_path=row.wav_path,
                json_path=row.json_path,
                utt_idx=idx,
                start_at=float(start_at),
                end_at=float(end_at),
                duration_sec=(float(end_at) - float(start_at)) / 1000.0,
                speaker=int(speaker),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Audio access: one WAV open per call, in-memory slicing per utterance
# ---------------------------------------------------------------------------


def read_call_waveform(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read an entire call's WAV once. Team EDA confirmed all WAVs are mono."""

    info = read_wav_info(wav_path)
    if info.channels != 1:
        raise ValueError(f"expected mono WAV, got {info.channels} channels: {wav_path}")
    segment = read_wav_frames(wav_path, 0, info.frame_count)
    return segment.waveform[0], segment.sample_rate


def utterance_frame_range(
    start_at_ms: Real, end_at_ms: Real, sample_rate: int, total_frames: int
) -> tuple[int, int]:
    start_frame = max(0, min(milliseconds_to_frame(start_at_ms, sample_rate), total_frames))
    end_frame = max(0, min(milliseconds_to_frame(end_at_ms, sample_rate), total_frames))
    if end_frame <= start_frame:
        raise ValueError(
            f"non-positive frame range after clamping: start={start_frame} end={end_frame} "
            f"(raw ms {start_at_ms}-{end_at_ms}, total_frames={total_frames})"
        )
    return start_frame, end_frame


def slice_and_pad(waveform: np.ndarray, start_frame: int, end_frame: int, min_length: int) -> np.ndarray:
    """Slice one utterance out of a call waveform, zero-padding short segments.

    Short utterances must not be discarded (the contest's own 100ms-and-under
    utterances are real data, not noise) -- they are zero-padded up to
    ``min_length`` so downstream STFT-based features never crash on them.
    """

    segment = waveform[start_frame:end_frame]
    if len(segment) < min_length:
        segment = np.pad(segment, (0, min_length - len(segment)))
    return segment.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Module B: 53-dim acoustic feature vector (duration + spectral statistics)
# ---------------------------------------------------------------------------

N_FFT = 512
HOP_LENGTH = 160
N_MFCC = 20
N_ACOUSTIC_FEATURES = 52
N_FEATURES = 53  # duration_sec + the 52 acoustic statistics below

FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_sec",
    *(f"mfcc_mean_{i}" for i in range(N_MFCC)),
    *(f"mfcc_std_{i}" for i in range(N_MFCC)),
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
)
assert len(FEATURE_COLUMNS) == N_FEATURES


def acoustic_feature_vector(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """52-dim MFCC/RMS/ZCR/spectral statistics. Caller prepends duration_sec."""

    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=segment, n_fft=N_FFT, hop_length=HOP_LENGTH)

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [rms.mean(), rms.std()],
        [zcr.mean(), zcr.std()],
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [bandwidth.mean(), bandwidth.std()],
        [flatness.mean(), flatness.std()],
    ]
    vector = np.concatenate([np.asarray(part, dtype=np.float64).ravel() for part in parts])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.shape[0] != N_ACOUSTIC_FEATURES:
        raise ValueError(f"acoustic feature vector has {vector.shape[0]} dims, expected {N_ACOUSTIC_FEATURES}")
    return vector.astype(np.float32)


def utterance_feature_vector(segment: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    acoustic = acoustic_feature_vector(segment, sample_rate)
    vector = np.concatenate([np.array([duration_sec], dtype=np.float32), acoustic])
    if vector.shape[0] != N_FEATURES:
        raise ValueError(f"feature vector has {vector.shape[0]} dims, expected {N_FEATURES}")
    return vector


# ---------------------------------------------------------------------------
# Module C + orchestration: sample calls, build the index, extract features
# ---------------------------------------------------------------------------

OnCallDone = Callable[[int, int], None]

_CallResult = tuple[list[UtteranceRecord], list[dict], list[dict]]


def _process_one_call(row: ManifestRow, data_root: Path) -> _CallResult:
    """Process exactly one call end-to-end: index + feature extraction.

    Pure function of ``(row, data_root)`` with no shared state -- this is the
    unit of work ``build_seoul_training_dataset`` uses for both its
    sequential loop and its process-pool path, and it must stay a top-level
    function (not a closure) so it can be pickled and sent to worker
    processes on Windows (spawn start method).
    """

    index_records: list[UtteranceRecord] = []
    feature_rows: list[dict] = []
    failures: list[dict] = []

    wav_path = data_root / row.wav_path
    try:
        records = build_utterance_records(row, data_root)
    except Exception as exc:  # noqa: BLE001 - see module docstring: logged, not swallowed
        failures.append(
            {
                "stage": "index",
                "call_id": row.stem,
                "utt_idx": None,
                "wav_path": row.wav_path,
                "json_path": row.json_path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return index_records, feature_rows, failures

    index_records.extend(records)

    try:
        waveform, sample_rate = read_call_waveform(wav_path)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        for record in records:
            failures.append(
                {
                    "stage": "audio_open",
                    "call_id": record.call_id,
                    "utt_idx": record.utt_idx,
                    "wav_path": row.wav_path,
                    "json_path": row.json_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        return index_records, feature_rows, failures

    total_frames = waveform.shape[0]
    for record in records:
        try:
            start_frame, end_frame = utterance_frame_range(
                record.start_at, record.end_at, sample_rate, total_frames
            )
            segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
            vector = utterance_feature_vector(segment, sample_rate, record.duration_sec)
        except Exception as exc:  # noqa: BLE001 - see module docstring
            failures.append(
                {
                    "stage": "feature",
                    "call_id": record.call_id,
                    "utt_idx": record.utt_idx,
                    "wav_path": row.wav_path,
                    "json_path": row.json_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        feature_rows.append(
            {
                "call_id": record.call_id,
                "utt_idx": record.utt_idx,
                "speaker": record.speaker,
                **dict(zip(FEATURE_COLUMNS, vector.tolist())),
            }
        )

    return index_records, feature_rows, failures


def build_seoul_training_dataset(
    rows: Iterable[ManifestRow],
    data_root: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[UtteranceRecord] | None = None,
    feature_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
) -> tuple[list[UtteranceRecord], list[dict], list[dict]]:
    """Sample Seoul/Training calls and build the utterance index + feature cache.

    ``index_sink``/``feature_sink``/``failure_sink`` may be passed in so a
    caller (e.g. the CLI script) can flush partial progress mid-run; they are
    mutated in place and also returned for convenience.

    A failure extracting one utterance's features never drops that utterance
    from the index (module A) and never aborts the rest of the call -- it is
    recorded in ``failure_sink`` with the call_id/utt_idx/reason instead, per
    the "원인 로깅 없는 광범위한 except 금지" rule: broad catches are used
    here because audio/librosa failure modes are numerous, but every one is
    logged with enough detail to act on.

    ``max_workers`` > 1 processes calls in parallel via a process pool -- each
    call only touches its own wav file and has no shared state, so this is a
    pure speed optimization: the aggregate contents of the three sinks are
    identical to a ``max_workers=1`` run (just interleaved in a different,
    completion-order-dependent sequence, which nothing downstream relies on --
    utt_idx order *within* a call is unaffected either way).
    """

    index_sink = index_sink if index_sink is not None else []
    feature_sink = feature_sink if feature_sink is not None else []
    failure_sink = failure_sink if failure_sink is not None else []

    seoul_rows = select_seoul_training_rows(rows)
    sampled_rows = sample_calls(seoul_rows, n_calls, seed)
    total = len(sampled_rows)

    if max_workers <= 1:
        for call_number, row in enumerate(sampled_rows, start=1):
            records, features, failures = _process_one_call(row, data_root)
            index_sink.extend(records)
            feature_sink.extend(features)
            failure_sink.extend(failures)
            if on_call_done:
                on_call_done(call_number, total)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_one_call, row, data_root) for row in sampled_rows]
            for call_number, future in enumerate(as_completed(futures), start=1):
                records, features, failures = future.result()
                index_sink.extend(records)
                feature_sink.extend(features)
                failure_sink.extend(failures)
                if on_call_done:
                    on_call_done(call_number, total)

    return index_sink, feature_sink, failure_sink


# ---------------------------------------------------------------------------
# Module E support: flat stem pairing for inference (no Training/Validation
# path component is guaranteed in the folders the grader hands us, so
# dcc_data.manifest.build_manifest -- which requires exactly that -- does not
# apply here; this mirrors its pairing logic without the split requirement).
# ---------------------------------------------------------------------------


def pair_by_stem(audio_dir: Path, label_dir: Path) -> list[tuple[str, Path, Path]]:
    wav_files = {p.stem: p for p in Path(audio_dir).rglob("*.wav")}
    json_files = {p.stem: p for p in Path(label_dir).rglob("*.json")}
    common = sorted(set(wav_files) & set(json_files))
    return [(stem, wav_files[stem], json_files[stem]) for stem in common]
