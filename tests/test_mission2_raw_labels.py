"""Tests for dcc_data.mission2_raw_labels and scripts/verify_mission2_raw_labels.py.

A cache built before the organizers' 2026-10-01 ruling (training must use the provided speaker labels as they
are) may carry corrected labels. These tests make a pair of small label folders and caches -- one whose
labels equal the provided ones, one with a whole call and the tail of another flipped, the way the old
correction did it -- and check that the tool finds exactly the flipped rows, restores them, and stamps a
spectrogram index so that the trainer's guard accepts it.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from dcc_data.guards import DataGuardError, SplitContaminationError  # noqa: E402
from dcc_data.mission2_features import HOP_LENGTH, N_FFT  # noqa: E402
from dcc_data.mission2_raw_labels import (  # noqa: E402
    LabelVerificationError,
    compare_index_labels,
    load_provided_speakers,
    restore_provided_labels,
)
from dcc_data.mission2_spectrogram import (  # noqa: E402
    LABEL_SOURCE_VERIFIED,
    N_MELS,
    assert_training_spectrogram_index,
)

verify_script = importlib.import_module("verify_mission2_raw_labels")

PROVIDED = {
    "call_a": [0, 1, 0, 1, 0, 1],
    "call_b": [1, 0, 1, 0],
    "call_c": [0, 0, 1],
}


def _write_labels(root: Path, provided: dict[str, list[int]]) -> Path:
    label_root = root / "Training" / "2.라벨링데이터" / "TL_서울_구급"
    label_root.mkdir(parents=True, exist_ok=True)
    for call_id, speakers in provided.items():
        document = {
            "_id": call_id,
            # no ``text`` key at all: the tool must never need it
            "utterances": [
                {"id": f"u{i}", "startAt": i * 1000, "endAt": i * 1000 + 500, "speaker": speaker}
                for i, speaker in enumerate(speakers)
            ],
        }
        (label_root / f"{call_id}_20230101.json").write_text(json.dumps(document), encoding="utf-8")
    return root / "Training" / "2.라벨링데이터"


def _frame(speakers_by_call: dict[str, list[int]], **extra) -> pd.DataFrame:
    rows = [
        {"call_id": call_id, "utt_idx": idx, "speaker": speaker, "duration_sec": 0.5, "n_frames": 4,
         "npy_path": f"{call_id}/{idx}.npy"}
        for call_id, speakers in speakers_by_call.items()
        for idx, speaker in enumerate(speakers)
    ]
    return pd.DataFrame(rows).assign(**extra)


def _corrected(provided: dict[str, list[int]]) -> dict[str, list[int]]:
    """What the old correction did: flip all of call_a, and call_b from utterance 2 on."""

    corrected = {call_id: list(speakers) for call_id, speakers in provided.items()}
    corrected["call_a"] = [1 - speaker for speaker in provided["call_a"]]
    corrected["call_b"] = provided["call_b"][:2] + [1 - speaker for speaker in provided["call_b"][2:]]
    return corrected


FULL_STAMP = {
    "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT, "sample_rate": 8000,
    "label_source": "provided_json",
}


# --------------------------------------------------------------------------- reading the label files


def test_load_provided_speakers_returns_each_calls_speakers_in_utterance_order(tmp_path: Path) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)

    assert load_provided_speakers(label_root) == PROVIDED


def test_load_reports_progress_while_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dcc_data.mission2_raw_labels as raw_labels

    monkeypatch.setattr(raw_labels, "PROGRESS_EVERY", 2)
    label_root = _write_labels(tmp_path, PROVIDED)  # 3 files
    seen = []

    load_provided_speakers(label_root, on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(2, 3)]


def test_load_refuses_two_label_files_with_the_same_call_id(tmp_path: Path) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    duplicate = label_root / "TL_서울_구급" / "call_a_copy.json"
    duplicate.write_text(json.dumps({"_id": "call_a", "utterances": []}), encoding="utf-8")

    with pytest.raises(LabelVerificationError, match="appears in two label files"):
        load_provided_speakers(label_root)


def test_load_refuses_a_speaker_that_is_not_0_or_1(tmp_path: Path) -> None:
    label_root = _write_labels(tmp_path, {"call_x": [0, 1]})
    path = next(label_root.rglob("*.json"))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["utterances"][1]["speaker"] = 2
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LabelVerificationError, match="speaker must be 0 or 1"):
        load_provided_speakers(label_root)


def test_load_refuses_a_folder_without_label_files(tmp_path: Path) -> None:
    with pytest.raises(LabelVerificationError, match="no label .json files"):
        load_provided_speakers(tmp_path)


# --------------------------------------------------------------------------- comparing


def test_a_cache_with_the_provided_labels_matches(tmp_path: Path) -> None:
    comparison = compare_index_labels(_frame(PROVIDED), PROVIDED)

    assert comparison.matches and comparison.n_rows == 13 and comparison.n_calls == 3


def test_the_comparison_names_exactly_the_calls_and_rows_that_were_changed() -> None:
    comparison = compare_index_labels(_frame(_corrected(PROVIDED)), PROVIDED)

    assert comparison.mismatched_by_call == {"call_a": 6, "call_b": 2}  # call_c is untouched
    assert comparison.unverifiable_rows == 0 and not comparison.matches


def test_rows_that_are_not_in_the_label_files_cannot_be_verified() -> None:
    frame = _frame({**PROVIDED, "call_missing": [0, 1], "call_c": [0, 0, 1, 1]})  # an unknown call; call_c has an extra utterance

    comparison = compare_index_labels(frame, PROVIDED)

    assert comparison.unverifiable_by_call == {"call_missing": 2, "call_c": 1}
    assert not comparison.matches


def test_comparison_needs_the_three_columns() -> None:
    with pytest.raises(LabelVerificationError, match="missing column"):
        compare_index_labels(_frame(PROVIDED).drop(columns=["utt_idx"]), PROVIDED)


# --------------------------------------------------------------------------- restoring


def test_restore_puts_the_provided_labels_back_and_leaves_everything_else_alone() -> None:
    edited = _frame(_corrected(PROVIDED), **FULL_STAMP)
    before = edited.copy()

    restored = restore_provided_labels(edited, PROVIDED)

    assert compare_index_labels(restored, PROVIDED).matches
    assert restored["speaker"].dtype == edited["speaker"].dtype
    pd.testing.assert_frame_equal(restored.drop(columns=["speaker", "label_source"]), edited.drop(columns=["speaker", "label_source"]))
    pd.testing.assert_frame_equal(edited, before)  # the input is not modified


def test_a_restored_spectrogram_index_is_stamped_verified_and_passes_the_trainer_guard() -> None:
    restored = restore_provided_labels(_frame(_corrected(PROVIDED), **FULL_STAMP), PROVIDED)

    provenance = assert_training_spectrogram_index(restored)

    assert provenance["label_source"] == LABEL_SOURCE_VERIFIED


def test_a_table_that_is_not_a_spectrogram_index_gets_no_extra_columns() -> None:
    l1_like = _frame(_corrected(PROVIDED)).drop(columns=["npy_path"])

    restored = restore_provided_labels(l1_like, PROVIDED)

    assert list(restored.columns) == list(l1_like.columns)
    assert compare_index_labels(restored, PROVIDED).matches


def test_a_legacy_index_without_provenance_is_only_stamped_when_the_caller_says_so() -> None:
    legacy = _frame(_corrected(PROVIDED))  # no split/n_mels/... columns, like the Drive index built before provenance

    with pytest.raises(DataGuardError, match="no build provenance"):
        restore_provided_labels(legacy, PROVIDED)

    restored = restore_provided_labels(legacy, PROVIDED, stamp_training_defaults=True)
    assert assert_training_spectrogram_index(restored) == {**FULL_STAMP, "label_source": LABEL_SOURCE_VERIFIED}


def test_stamping_defaults_never_overwrites_provenance_the_index_already_has() -> None:
    edited = _frame(_corrected(PROVIDED), **{**FULL_STAMP, "hop_length": HOP_LENGTH // 2})

    restored = restore_provided_labels(edited, PROVIDED, stamp_training_defaults=True)

    assert set(restored["hop_length"]) == {HOP_LENGTH // 2}


def test_restore_refuses_a_cache_that_has_rows_the_label_files_do_not_have() -> None:
    with pytest.raises(LabelVerificationError, match="not in the label files"):
        restore_provided_labels(_frame({**PROVIDED, "call_missing": [0, 1]}, **FULL_STAMP), PROVIDED)


def test_restore_still_refuses_an_index_that_is_not_Training() -> None:
    validation_index = _frame(PROVIDED, **{**FULL_STAMP, "split": "Validation"})

    with pytest.raises(SplitContaminationError, match="non-Training"):
        restore_provided_labels(validation_index, PROVIDED)


# --------------------------------------------------------------------------- the command line


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["verify_mission2_raw_labels.py", *argv])
    return verify_script.main()


def _write_index(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_parquet(path, index=False)
    return path


def test_cli_exits_0_when_every_label_is_the_provided_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "index.parquet", _frame(PROVIDED, **FULL_STAMP))

    code = _run(monkeypatch, "--index", str(index), "--label-root", str(label_root))

    assert code == 0 and "OK: every speaker label equals the provided one" in capsys.readouterr().out


def test_cli_exits_1_and_names_the_changed_calls_when_labels_differ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "index.parquet", _frame(_corrected(PROVIDED), **FULL_STAMP))

    code = _run(monkeypatch, "--index", str(index), "--label-root", str(label_root))

    output = capsys.readouterr().out
    assert code == 1
    assert "call_a: 6 row(s)" in output and "call_b: 2 row(s)" in output and "call_c" not in output
    assert "DIFFERS" in output


def test_cli_exits_2_when_the_rows_cannot_be_found_in_the_label_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    label_root = _write_labels(tmp_path, {"call_c": PROVIDED["call_c"]})
    index = _write_index(tmp_path / "index.parquet", _frame(PROVIDED, **FULL_STAMP))

    assert _run(monkeypatch, "--index", str(index), "--label-root", str(label_root)) == 2


def test_cli_writes_a_restored_and_stamped_copy_that_the_trainer_guard_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "legacy.parquet", _frame(_corrected(PROVIDED)))
    output = tmp_path / "out" / "restored.parquet"

    code = _run(
        monkeypatch, "--index", str(index), "--label-root", str(label_root),
        "--output", str(output), "--stamp-training-defaults",
    )

    assert code == 0
    written = pd.read_parquet(output)
    assert compare_index_labels(written, PROVIDED).matches
    assert assert_training_spectrogram_index(written)["label_source"] == LABEL_SOURCE_VERIFIED
    # the input file is untouched
    assert not compare_index_labels(pd.read_parquet(index), PROVIDED).matches


def test_cli_without_the_stamp_flag_does_not_write_an_unstamped_spectrogram_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "legacy.parquet", _frame(_corrected(PROVIDED)))
    output = tmp_path / "restored.parquet"

    code = _run(monkeypatch, "--index", str(index), "--label-root", str(label_root), "--output", str(output))

    assert code == 2 and not output.exists()
    assert "CANNOT WRITE" in capsys.readouterr().out


def test_cli_refuses_to_overwrite_its_own_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "index.parquet", _frame(_corrected(PROVIDED), **FULL_STAMP))

    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, "--index", str(index), "--label-root", str(label_root), "--output", str(index))

    assert excinfo.value.code == 2
    assert not compare_index_labels(pd.read_parquet(index), PROVIDED).matches  # still as it was


def test_cli_stamp_flag_needs_an_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    label_root = _write_labels(tmp_path, PROVIDED)
    index = _write_index(tmp_path / "index.parquet", _frame(PROVIDED))

    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, "--index", str(index), "--label-root", str(label_root), "--stamp-training-defaults")

    assert excinfo.value.code == 2
