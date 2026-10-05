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

# How an index's ``speaker`` column relates to the provided label files. The organizers ruled (2026-10-01)
# that Mission 2 training must use the provided speaker labels as they are: no correction, and no data
# selection, on the basis of ``text``. So an index has to say its labels are the provided ones.
LABEL_SOURCE_PROVIDED = "provided_json"  # written by the builder straight from the provided label JSON
LABEL_SOURCE_VERIFIED = "verified_against_provided_json"  # re-checked row by row by scripts/verify_mission2_raw_labels.py
ALLOWED_LABEL_SOURCES = (LABEL_SOURCE_PROVIDED, LABEL_SOURCE_VERIFIED)

# Every index row records where its spectrogram came from (the manifest split of its call), the settings it
# was built with and where its labels came from, so a training run can prove its input is Training-only with
# the provided labels and the checkpoint can record what it was trained on. See
# assert_training_spectrogram_index().
INDEX_PROVENANCE_COLUMNS = ("split", "n_mels", "hop_length", "n_fft", "sample_rate", "label_source")


class LabelProvenanceError(ValueError):
    """A spectrogram index does not say its speaker labels are the provided ones."""


def assert_training_spectrogram_index(index_df: "pd.DataFrame") -> dict:
    """Fail unless every row of a spectrogram index is stamped as built from Training, with provided labels.

    Returns the single ``{split, n_mels, hop_length, n_fft, sample_rate, label_source}`` the index was built
    with. An index without the stamp (built before the stamp existed) cannot be shown to be Training-only or
    to carry the provided speaker labels, so it is refused rather than trusted -- the official Validation
    folder must never reach a training loop, a Validation index built into the same paths as the Training
    one would look identical otherwise, and a cache whose labels were edited would train on them silently.
    """

    missing = [column for column in INDEX_PROVENANCE_COLUMNS if column not in index_df.columns]
    if missing:
        raise SplitContaminationError(
            f"spectrogram index has no build provenance (missing column(s): {missing}), so it cannot be shown "
            "to come from Training only with the provided speaker labels. Rebuild it with "
            "scripts/build_mission2_spectrograms.py, or, for a cache you know is a Training-only, "
            "default-settings one, run scripts/verify_mission2_raw_labels.py --index <index> --label-root "
            "<Training label folder> --output <new index> --stamp-training-defaults -- it checks every speaker "
            "label against the provided label files, restores any that differ, and writes the stamp."
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
    if provenance["label_source"] not in ALLOWED_LABEL_SOURCES:
        raise LabelProvenanceError(
            f"spectrogram index label_source is {provenance['label_source']!r}; training needs the provided "
            f"speaker labels as they are ({', '.join(ALLOWED_LABEL_SOURCES)}). Rebuild the index, or check and "
            "restore its labels with scripts/verify_mission2_raw_labels.py."
        )
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
    row: ManifestRow, data_root: Path, output_dir: Path, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS
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
            spec = mel_spectrogram(segment, sample_rate, hop_length=hop_length, n_mels=n_mels)
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
                "label_source": LABEL_SOURCE_PROVIDED,
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
            index_rows, failures = _process_one_call_spectrograms(row, data_root, output_dir, hop_length, n_mels)
            index_sink.extend(index_rows)
            failure_sink.extend(failures)
            if on_call_done:
                on_call_done(call_number, total)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_process_one_call_spectrograms, row, data_root, output_dir, hop_length, n_mels)
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
    )
