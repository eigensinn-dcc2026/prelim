"""Check a cache's speaker labels against the provided label files, and restore them where they differ.

The organizers ruled (2026-10-01) that Mission 2 training must use the provided ``speaker`` labels as they
are: reading ``text`` to look at Training errors is fine, but correcting labels (or selecting training data)
on the strength of it and then training on the result is not. Earlier L1/L2 caches were built with six
Training calls' labels corrected (109 utterances), so a cache from before that ruling cannot be assumed to
carry the provided labels. This module compares any cache that has ``call_id``/``utt_idx``/``speaker``
columns -- an L2 spectrogram index or an L1 utterance/feature table -- with the label JSON files, row by
row, and can write a copy whose labels are the provided ones.

Only ``_id`` and each utterance's ``speaker`` are read from the label files -- never ``text``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .guards import TRAINING
from .mission2_features import HOP_LENGTH, N_FFT, _load_call_utterances
from .mission2_spectrogram import (
    LABEL_SOURCE_VERIFIED,
    N_MELS,
    assert_training_spectrogram_index,
)

if TYPE_CHECKING:
    import pandas as pd

REQUIRED_INDEX_COLUMNS = ("call_id", "utt_idx", "speaker")
PROGRESS_EVERY = 2000

# What a spectrogram index built before provenance existed was built with (the project's defaults at the time).
# Only used when the caller says so explicitly (--stamp-training-defaults), and only for columns the index lacks.
LEGACY_DEFAULT_SAMPLE_RATE = 8000


class LabelVerificationError(ValueError):
    """The cache cannot be compared with the provided labels (wrong folder, split or call ids)."""


def load_provided_speakers(
    label_root: str | Path, on_progress: Callable[[int, int], None] | None = None
) -> dict[str, list[int]]:
    """``{call_id: [speaker of utterance 0, 1, ...]}`` for every label JSON under ``label_root``.

    ``call_id`` is read the way dcc_data.mission2_features.build_utterance_records reads it (the JSON's
    ``_id``, else the file stem), and the list position is the same ``utt_idx``. The full Training folder is
    29,200 files and takes several minutes to read; ``on_progress(done, total)`` is called every
    PROGRESS_EVERY files so a caller can show that it is still going.
    """

    provided: dict[str, list[int]] = {}
    where: dict[str, Path] = {}
    json_paths = sorted(Path(label_root).rglob("*.json"))
    for done, json_path in enumerate(json_paths, start=1):
        if on_progress is not None and done % PROGRESS_EVERY == 0:
            on_progress(done, len(json_paths))
        call_id, utterances = _load_call_utterances(json_path)
        if call_id in provided:
            raise LabelVerificationError(f"call_id {call_id!r} appears in two label files: {where[call_id]} and {json_path}")
        speakers = []
        for idx, utterance in enumerate(utterances):
            speaker = utterance.get("speaker") if isinstance(utterance, dict) else None
            if speaker not in (0, 1):
                raise LabelVerificationError(f"speaker must be 0 or 1: {json_path}[{idx}] got {speaker!r}")
            speakers.append(int(speaker))
        provided[call_id] = speakers
        where[call_id] = json_path
    if not provided:
        raise LabelVerificationError(f"no label .json files found under {label_root}")
    return provided


@dataclass(frozen=True)
class LabelComparison:
    n_rows: int
    n_calls: int
    unverifiable_by_call: dict[str, int]  # call not in the label files, or utt_idx past its last utterance
    mismatched_by_call: dict[str, int]  # the cache's speaker differs from the provided one

    @property
    def unverifiable_rows(self) -> int:
        return sum(self.unverifiable_by_call.values())

    @property
    def mismatched_rows(self) -> int:
        return sum(self.mismatched_by_call.values())

    @property
    def matches(self) -> bool:
        return self.unverifiable_rows == 0 and self.mismatched_rows == 0


def _require_columns(index_df: "pd.DataFrame") -> None:
    missing = [column for column in REQUIRED_INDEX_COLUMNS if column not in index_df.columns]
    if missing:
        raise LabelVerificationError(f"index is missing column(s) {missing}; need {list(REQUIRED_INDEX_COLUMNS)}")


def compare_index_labels(index_df: "pd.DataFrame", provided: dict[str, list[int]]) -> LabelComparison:
    _require_columns(index_df)
    mismatched: Counter[str] = Counter()
    unverifiable: Counter[str] = Counter()
    for call_id, utt_idx, speaker in zip(
        index_df["call_id"].astype(str), index_df["utt_idx"].astype(int), index_df["speaker"].astype(int)
    ):
        labels = provided.get(call_id)
        if labels is None or not 0 <= utt_idx < len(labels):
            unverifiable[call_id] += 1
        elif labels[utt_idx] != speaker:
            mismatched[call_id] += 1
    return LabelComparison(
        n_rows=len(index_df),
        n_calls=int(index_df["call_id"].nunique()),
        unverifiable_by_call=dict(unverifiable),
        mismatched_by_call=dict(mismatched),
    )


def restore_provided_labels(
    index_df: "pd.DataFrame", provided: dict[str, list[int]], *, stamp_training_defaults: bool = False
) -> "pd.DataFrame":
    """A copy of ``index_df`` whose ``speaker`` column is the provided labels, or LabelVerificationError.

    Refuses if any row cannot be found in the label files (it could not be checked, let alone restored).
    For a spectrogram index (it has ``npy_path``) the result is also stamped ``label_source`` =
    verified_against_provided_json and has to pass assert_training_spectrogram_index. A spectrogram index
    built before provenance existed has no ``split``/``n_mels``/... columns: ``stamp_training_defaults``
    fills the missing ones with the Training defaults, which is only right if that cache really is a
    Training-only, default-settings one -- the caller has to say so.
    """

    comparison = compare_index_labels(index_df, provided)
    if comparison.unverifiable_rows:
        raise LabelVerificationError(
            f"{comparison.unverifiable_rows} row(s) in {len(comparison.unverifiable_by_call)} call(s) are not in the "
            f"label files (first: {sorted(comparison.unverifiable_by_call)[:5]}) -- wrong --label-root, or an index "
            "from another split"
        )

    restored = index_df.copy()
    restored["speaker"] = [
        provided[call_id][utt_idx]
        for call_id, utt_idx in zip(restored["call_id"].astype(str), restored["utt_idx"].astype(int))
    ]
    restored["speaker"] = restored["speaker"].astype(index_df["speaker"].dtype)

    if "npy_path" in restored.columns:  # an L2 spectrogram index
        if stamp_training_defaults:
            defaults = {
                "split": TRAINING,
                "n_mels": N_MELS,
                "hop_length": HOP_LENGTH,
                "n_fft": N_FFT,
                "sample_rate": LEGACY_DEFAULT_SAMPLE_RATE,
            }
            restored = restored.assign(**{name: value for name, value in defaults.items() if name not in restored.columns})
        restored["label_source"] = LABEL_SOURCE_VERIFIED
        assert_training_spectrogram_index(restored)  # raises if still not a Training index with its provenance

    if not compare_index_labels(restored, provided).matches:  # cannot happen; guards a future edit
        raise LabelVerificationError("restored labels still differ from the provided ones")
    return restored
