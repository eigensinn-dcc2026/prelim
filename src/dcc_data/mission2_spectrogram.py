"""Mission 2 L2 input representation: log-mel spectrograms for a CNN.

L1 (dcc_data.mission2_features) collapses each utterance to 53-64 hand-
picked summary statistics. Three rounds of adding more such statistics
(pitch/energy, call-relative z-scores, an autocorrelation-based harmonicity/
overlap signal) each showed a real effect in isolation but failed to move
the two error modes that matter most -- the duration-bucket 대원/신고자 bias
and the overlap-driven accuracy gap (see reports/mission2/). This module
keeps the raw time-frequency structure instead, on the theory that exactly
the information those two error modes depend on (fine-grained pitch/energy
*contours*, or the interference pattern of two overlapping voices) is what
collapsing to a mean/std/percentile throws away.

Reuses dcc_data.mission2_features' framing constants (N_FFT, HOP_LENGTH) and
its manifest/audio-slicing pipeline (read_call_waveform, slice_and_pad,
build_utterance_records) rather than re-deriving any of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import librosa

from .guards import TRAINING, SplitContaminationError
from .manifest import ManifestRow
from .mission2_features import (
    HOP_LENGTH,
    N_FFT,
    build_utterance_records,
    read_call_waveform,
    sample_calls,
    select_seoul_training_rows,
    select_seoul_validation_rows,
    slice_and_pad,
    utterance_frame_range,
)

if TYPE_CHECKING:
    import pandas as pd

N_MELS = 40
# librosa.power_to_db's own default clip range (top_db=80, relative to the
# spectrogram's own max) -- pad frames get set to exactly this floor so a
# padded "silent" frame reads as silence to the CNN, not as loud signal
# (naively padding with 0.0 would do the opposite: 0 dB is this utterance's
# *loudest* frame under the ref=np.max normalization below).
SPEC_FLOOR_DB = -80.0

# Every index row records where its spectrogram came from (the manifest split of its call) and the
# settings it was built with, so a training run can prove its input is Training-only and the checkpoint
# can record what it was trained on. See assert_training_spectrogram_index().
INDEX_PROVENANCE_COLUMNS = ("split", "n_mels", "hop_length", "n_fft", "sample_rate")


def assert_training_spectrogram_index(index_df: "pd.DataFrame") -> dict:
    """Fail unless every row of a spectrogram index is stamped as built from Training.

    Returns the single ``{split, n_mels, hop_length, n_fft, sample_rate}`` the index was built with.
    An index without the stamp (built before the stamp existed) cannot be shown to be Training-only, so
    it is refused rather than trusted -- the official Validation folder must never reach a training loop,
    and a Validation index built into the same paths as the Training one would look identical otherwise.
    """

    missing = [column for column in INDEX_PROVENANCE_COLUMNS if column not in index_df.columns]
    if missing:
        raise SplitContaminationError(
            f"spectrogram index has no build provenance (missing column(s): {missing}), so it cannot be shown "
            "to come from Training only. Rebuild it with scripts/build_mission2_spectrograms.py, or, if you "
            "know it is a Training cache built with the default settings, stamp it once: "
            f"df.assign(split={TRAINING!r}, n_mels={N_MELS}, hop_length={HOP_LENGTH}, n_fft={N_FFT}, "
            "sample_rate=8000).to_parquet(path, index=False)"
        )
    if len(index_df) == 0:
        raise ValueError("spectrogram index is empty")

    foreign = index_df.loc[index_df["split"] != TRAINING, "split"]
    if len(foreign):
        counts = foreign.astype(str).value_counts().to_dict()
        raise SplitContaminationError(
            f"Training run received {len(foreign)} non-Training spectrogram index row(s): {counts}"
        )

    provenance = {}
    for column in INDEX_PROVENANCE_COLUMNS:
        values = index_df[column].unique()
        if len(values) != 1:
            raise ValueError(f"spectrogram index mixes several {column} values: {sorted(map(str, values))[:5]}")
        value = values[0]
        provenance[column] = value.item() if hasattr(value, "item") else value
    return provenance


def mel_spectrogram(
    segment: np.ndarray, sample_rate: int, *, n_mels: int = N_MELS, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, n_frames), normalized to this
    segment's own peak (0 dB ceiling, SPEC_FLOOR_DB floor) -- per-utterance
    relative loudness, the same reason CALL_RELATIVE_FEATURE_COLUMNS z-score
    against each call's own mean/std rather than an absolute scale: absolute
    recording gain varies per call and isn't a speaker-identity signal.

    ``hop_length`` defaults to the same HOP_LENGTH L1's hand-crafted features
    use, so every existing caller is unaffected. A smaller value gives more
    frames for the same audio -- SpeakerCNN's 3 rounds of MaxPool(stride 2)
    can otherwise leave a short (backchannel-length) utterance with only 1-2
    surviving timesteps before AdaptiveAvgPool2d; see
    scripts/build_mission2_spectrograms.py's --hop-length.
    """

    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sample_rate, n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=-SPEC_FLOOR_DB)
    return np.nan_to_num(log_mel, nan=SPEC_FLOOR_DB, neginf=SPEC_FLOOR_DB).astype(np.float32)


def linear_spectrogram(
    segment: np.ndarray, sample_rate: int, *, n_bins: int = N_MELS, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """Log spectrogram on a uniform (linear) Hz scale, shape (n_bins, n_frames).

    A mel filterbank spends most of its bins on low frequencies and gives
    high frequencies fewer, wider bins. muju0629's forensic analysis (see
    reports/미션2 화자 분류 해결 방식 조사.md) found a single 2.2-3.6kHz vs
    0.3-2kHz energy-ratio feature alone reaches 70.1% speaker-role accuracy --
    exactly the band a 40-band mel scale resolves most coarsely. This keeps
    the same band count but spaces bins uniformly in Hz instead, so that band
    gets proportionally more resolution. Same per-utterance peak
    normalization and floor as mel_spectrogram, so the two are drop-in
    interchangeable for the rest of the pipeline (same shape, same value
    range).
    """

    if len(segment) < N_FFT:
        segment = np.pad(segment, (0, N_FFT - len(segment)))
    segment = segment.astype(np.float32, copy=False)

    stft_power = np.abs(librosa.stft(segment, n_fft=N_FFT, hop_length=hop_length)) ** 2
    binned = np.stack([group.mean(axis=0) for group in np.array_split(stft_power, n_bins, axis=0)])
    log_spec = librosa.power_to_db(binned, ref=np.max, top_db=-SPEC_FLOOR_DB)
    return np.nan_to_num(log_spec, nan=SPEC_FLOOR_DB, neginf=SPEC_FLOOR_DB).astype(np.float32)


FREQ_SCALES = {"mel": mel_spectrogram, "linear": linear_spectrogram}


def compute_spectrogram(
    segment: np.ndarray, sample_rate: int, *, freq_scale: str, n_mels: int, hop_length: int
) -> np.ndarray:
    """Dispatch to mel_spectrogram or linear_spectrogram by name.

    Both take the band count as their second keyword (``n_mels``/``n_bins``
    respectively) -- unified here under the ``n_mels`` CLI/call-site name so
    build_mission2_spectrograms.py doesn't need a scale-specific flag.
    """

    try:
        spectrogram_fn = FREQ_SCALES[freq_scale]
    except KeyError:
        raise ValueError(f"unknown freq_scale {freq_scale!r}, expected one of {sorted(FREQ_SCALES)}") from None
    if spectrogram_fn is mel_spectrogram:
        return spectrogram_fn(segment, sample_rate, n_mels=n_mels, hop_length=hop_length)
    return spectrogram_fn(segment, sample_rate, n_bins=n_mels, hop_length=hop_length)


# ---------------------------------------------------------------------------
# Orchestration: sample calls, slice each utterance, save one .npy per
# utterance + an index of what was saved (mirrors
# mission2_features.build_seoul_training_dataset, output format only differs
# because a spectrogram doesn't fit in a flat tabular row like the 61-dim
# vector does).
# ---------------------------------------------------------------------------

OnCallDone = Callable[[int, int], None]


def _spectrogram_path(output_dir: Path, call_id: str, utt_idx: int) -> Path:
    return output_dir / call_id / f"{utt_idx}.npy"


def _process_one_call_spectrograms(
    row: ManifestRow,
    data_root: Path,
    output_dir: Path,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    freq_scale: str = "mel",
) -> tuple[list[dict], list[dict]]:
    """Process exactly one call: slice every utterance, save its spectrogram.

    Same "log, never silently drop" failure policy as
    mission2_features._process_one_call, and the same reason it must stay a
    top-level function: picklable for a process-pool worker on Windows.
    """

    index_rows: list[dict] = []
    failures: list[dict] = []

    wav_path = data_root / row.wav_path
    try:
        records = build_utterance_records(row, data_root)
    except Exception as exc:  # noqa: BLE001 - logged, not swallowed
        failures.append(
            {"stage": "index", "call_id": row.stem, "utt_idx": None, "error_type": type(exc).__name__, "error_message": str(exc)}
        )
        return index_rows, failures

    try:
        waveform, sample_rate = read_call_waveform(wav_path)
    except Exception as exc:  # noqa: BLE001
        for record in records:
            failures.append(
                {"stage": "audio_open", "call_id": record.call_id, "utt_idx": record.utt_idx, "error_type": type(exc).__name__, "error_message": str(exc)}
            )
        return index_rows, failures

    total_frames = waveform.shape[0]
    for record in records:
        try:
            start_frame, end_frame = utterance_frame_range(record.start_at, record.end_at, sample_rate, total_frames)
            segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
            spec = compute_spectrogram(segment, sample_rate, freq_scale=freq_scale, n_mels=n_mels, hop_length=hop_length)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"stage": "spectrogram", "call_id": record.call_id, "utt_idx": record.utt_idx, "error_type": type(exc).__name__, "error_message": str(exc)}
            )
            continue

        path = _spectrogram_path(output_dir, record.call_id, record.utt_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, spec)
        index_rows.append(
            {
                "call_id": record.call_id,
                "utt_idx": record.utt_idx,
                "speaker": record.speaker,
                "duration_sec": record.duration_sec,
                "n_frames": spec.shape[1],
                "npy_path": str(path.relative_to(output_dir)),
                "split": row.split,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "n_fft": N_FFT,
                "sample_rate": int(sample_rate),
                "freq_scale": freq_scale,
            }
        )

    return index_rows, failures


def _build_seoul_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    select_rows: Callable[[Iterable[ManifestRow]], list[ManifestRow]],
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    freq_scale: str = "mel",
) -> tuple[list[dict], list[dict]]:
    """Shared implementation behind build_seoul_training_spectrograms and
    build_seoul_validation_spectrograms -- only which rows are eligible
    (``select_rows``) differs, mirroring mission2_features._build_seoul_dataset.
    """

    index_sink = index_sink if index_sink is not None else []
    failure_sink = failure_sink if failure_sink is not None else []

    seoul_rows = select_rows(rows)
    sampled_rows = sample_calls(seoul_rows, n_calls, seed)
    total = len(sampled_rows)
    output_dir = Path(output_dir)

    if max_workers <= 1:
        for call_number, row in enumerate(sampled_rows, start=1):
            index_rows, failures = _process_one_call_spectrograms(
                row, data_root, output_dir, hop_length, n_mels, freq_scale
            )
            index_sink.extend(index_rows)
            failure_sink.extend(failures)
            if on_call_done:
                on_call_done(call_number, total)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_process_one_call_spectrograms, row, data_root, output_dir, hop_length, n_mels, freq_scale)
                for row in sampled_rows
            ]
            for call_number, future in enumerate(as_completed(futures), start=1):
                index_rows, failures = future.result()
                index_sink.extend(index_rows)
                failure_sink.extend(failures)
                if on_call_done:
                    on_call_done(call_number, total)

    return index_sink, failure_sink


def build_seoul_training_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    freq_scale: str = "mel",
) -> tuple[list[dict], list[dict]]:
    """Sample Seoul/Training calls and save one spectrogram .npy per utterance.

    Mirrors mission2_features.build_seoul_training_dataset's sampling/
    parallelism/failure-logging exactly; only the per-utterance output
    differs (a saved .npy + an index row, instead of a 61-dim feature dict).

    ``hop_length`` defaults to HOP_LENGTH (unchanged behavior); pass a
    smaller value to build a separate, higher time-resolution spectrogram
    set into a different ``output_dir`` -- see mel_spectrogram()'s docstring.
    ``n_mels`` defaults to N_MELS (unchanged behavior); pass a larger value
    for a higher frequency-resolution set, same ``output_dir`` caveat.
    ``freq_scale`` is ``"mel"`` (unchanged behavior) or ``"linear"`` -- see
    linear_spectrogram()'s docstring; also use a different ``output_dir``.
    """

    return _build_seoul_spectrograms(
        rows,
        data_root,
        output_dir,
        select_rows=select_seoul_training_rows,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
        hop_length=hop_length,
        n_mels=n_mels,
        freq_scale=freq_scale,
    )


def build_seoul_validation_spectrograms(
    rows: Iterable[ManifestRow],
    data_root: Path,
    output_dir: Path,
    *,
    n_calls: int,
    seed: int,
    index_sink: list[dict] | None = None,
    failure_sink: list[dict] | None = None,
    on_call_done: OnCallDone | None = None,
    max_workers: int = 1,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    freq_scale: str = "mel",
) -> tuple[list[dict], list[dict]]:
    """Same as build_seoul_training_spectrograms, but over the official
    Validation split (dcc_data.mission2_features.select_seoul_validation_rows)
    -- eval-only, never pass this to a training loop's ``.fit()``/backprop.
    """

    return _build_seoul_spectrograms(
        rows,
        data_root,
        output_dir,
        select_rows=select_seoul_validation_rows,
        n_calls=n_calls,
        seed=seed,
        index_sink=index_sink,
        failure_sink=failure_sink,
        on_call_done=on_call_done,
        max_workers=max_workers,
        hop_length=hop_length,
        n_mels=n_mels,
        freq_scale=freq_scale,
    )
