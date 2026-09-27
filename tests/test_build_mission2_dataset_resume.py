from __future__ import annotations

import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from dcc_data.manifest import ManifestRow, write_manifest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

build_mission2_dataset = importlib.import_module("build_mission2_dataset")

SAMPLE_RATE = 8000


def _write_wav(path: Path, duration_s: float, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(n) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _write_call(root: Path, stem: str, *, seed: int) -> ManifestRow:
    wav_relative = f"Training/1.원천데이터/TS_서울_구급/{stem}.wav"
    json_relative = f"Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
    _write_wav(root / wav_relative, 3.0, seed=seed)
    document = {
        "_id": stem,
        "utterances": [
            {"id": "u0", "startAt": 0, "endAt": 1000, "speaker": 0, "text": "x"},
            {"id": "u1", "startAt": 1000, "endAt": 2500, "speaker": 1, "text": "y"},
        ],
    }
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return ManifestRow("Training", stem, wav_relative, json_relative)


def _run_main(argv: list[str]) -> None:
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["build_mission2_dataset.py", *argv]
    try:
        build_mission2_dataset.main()
    finally:
        _sys.argv = old_argv


def test_resume_skips_already_processed_calls_and_appends_the_rest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stems = ["call_a", "call_b", "call_c"]
    rows = [_write_call(root, stem, seed=i) for i, stem in enumerate(stems)]

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    # "Interrupted first run": only call_a and call_b were ever available/reachable.
    partial_manifest = tmp_path / "manifest_partial.csv"
    write_manifest(rows[:2], partial_manifest, overwrite=True)
    _run_main(
        [
            "--manifest",
            str(partial_manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
        ]
    )
    first_pass_index = pd.read_parquet(index_output)
    assert set(first_pass_index["wav_stem"]) == {"call_a", "call_b"}

    # "Resumed run": the full manifest (all 3 calls) is now available; --resume
    # must skip call_a/call_b (already in index_output) and process only call_c.
    full_manifest = tmp_path / "manifest_full.csv"
    write_manifest(rows, full_manifest, overwrite=True)
    _run_main(
        [
            "--manifest",
            str(full_manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",
        ]
    )

    final_index = pd.read_parquet(index_output)
    final_features = pd.read_parquet(features_output)
    assert set(final_index["wav_stem"]) == {"call_a", "call_b", "call_c"}
    # Original call_a/call_b rows must be untouched, not just re-derived identically.
    assert len(final_index) == 3 * 2  # 2 utterances/call * 3 calls
    assert len(final_features) == len(final_index)


def test_resume_reprocesses_a_call_that_was_indexed_but_never_resolved(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stems = ["call_a", "call_b", "call_c"]
    rows = [_write_call(root, stem, seed=i) for i, stem in enumerate(stems)]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    _run_main(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
        ]
    )

    full_index = pd.read_parquet(index_output)
    full_features = pd.read_parquet(features_output)
    assert set(full_index["call_id"]) == {"call_a", "call_b", "call_c"}
    assert len(full_features) == len(full_index)

    # Simulate a crash that happened right after call_c's utterances were
    # indexed but before its audio was opened / features extracted: its
    # index rows survive on disk, but it ends up with zero feature rows and
    # zero failure entries -- the exact state that silently and permanently
    # lost 2,170/29,200 real calls under the old --resume logic.
    crashed_features = full_features[full_features["call_id"] != "call_c"]
    crashed_features.to_parquet(features_output, index=False)
    assert not failures_log.exists() or failures_log.stat().st_size == 0

    _run_main(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",
        ]
    )

    resumed_index = pd.read_parquet(index_output)
    resumed_features = pd.read_parquet(features_output)
    assert set(resumed_index["call_id"]) == {"call_a", "call_b", "call_c"}
    assert len(resumed_index) == len(full_index)  # no duplicate rows introduced
    # call_c must be fully recovered, not skipped forever because it was
    # already (incompletely) present in the index.
    assert set(resumed_features.loc[resumed_features["call_id"] == "call_c", "utt_idx"]) == {0, 1}
    assert len(resumed_features) == len(resumed_index)


def test_resume_preserves_survivor_file_when_only_one_output_exists(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    # Simulate a crash that landed between the two files being written
    # inside _write_outputs: only the index file survives on disk.
    stray_index = pd.DataFrame(
        [
            {
                "call_id": "stray",
                "wav_stem": "stray",
                "wav_path": "stray.wav",
                "json_path": "stray.json",
                "utt_idx": 0,
                "startAt": 0,
                "endAt": 1000,
                "duration_sec": 1.0,
                "speaker": 0,
            }
        ]
    )
    stray_index.to_parquet(index_output, index=False)

    _run_main(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",
        ]
    )

    # The stray survivor must be preserved under a different name, not
    # silently overwritten by the fresh run's unrelated output.
    backups = list(tmp_path.glob("index.stale-*.parquet"))
    assert len(backups) == 1
    preserved = pd.read_parquet(backups[0])
    assert list(preserved["call_id"]) == ["stray"]

    # And the fresh run itself must have proceeded normally, unaffected.
    fresh_index = pd.read_parquet(index_output)
    assert set(fresh_index["wav_stem"]) == {"call_only"}


def _write_broken_call(root: Path, stem: str) -> ManifestRow:
    """A call whose JSON never parses -- fails at the "index" stage on every
    attempt, forever, regardless of --resume."""

    wav_relative = f"Training/1.원천데이터/TS_서울_구급/{stem}.wav"
    json_relative = f"Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
    _write_wav(root / wav_relative, 3.0, seed=99)
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text("{not valid json", encoding="utf-8")
    return ManifestRow("Training", stem, wav_relative, json_relative)


def test_resume_does_not_duplicate_failure_entries_for_a_call_that_keeps_failing(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    good_row = _write_call(root, "call_a", seed=0)
    bad_row = _write_broken_call(root, "call_bad")
    manifest = tmp_path / "manifest.csv"
    write_manifest([good_row, bad_row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    args = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--n-calls",
        "0",
        "--index-output",
        str(index_output),
        "--features-output",
        str(features_output),
        "--failures-log",
        str(failures_log),
        "--flush-every",
        "0",
    ]
    _run_main(args)

    failures_after_first_run = [json.loads(line) for line in failures_log.read_text(encoding="utf-8").splitlines()]
    assert [f for f in failures_after_first_run if f["call_id"] == "call_bad"].__len__() == 1

    # Resume twice more -- call_bad has no way to ever succeed, so it must
    # keep failing, but its failures.jsonl entry must be replaced each time,
    # never duplicated.
    _run_main([*args, "--resume"])
    _run_main([*args, "--resume"])

    final_failures = [json.loads(line) for line in failures_log.read_text(encoding="utf-8").splitlines()]
    bad_failures = [f for f in final_failures if f["call_id"] == "call_bad"]
    assert len(bad_failures) == 1, f"expected exactly 1 failure entry for call_bad, got {len(bad_failures)}"


def test_resume_refuses_ambiguous_call_id_shared_by_two_wav_paths(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_a", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    # Two index rows share call_id "dup" but point at different wav files,
    # each with its OWN (non-colliding) utt_idx -- a scenario the pipeline
    # doesn't otherwise guarantee can't happen, since call_id is just the
    # label JSON's `_id` field with no enforced global uniqueness. utt_idx
    # is deliberately distinct per row (0 vs 1) so this exercises the
    # call_id -> wav_path collision check specifically, not the separate
    # (call_id, utt_idx) duplicate-row check in _validate_loaded_state.
    ambiguous_index = pd.DataFrame(
        [
            {
                "call_id": "dup",
                "wav_stem": "one",
                "wav_path": "one.wav",
                "json_path": "one.json",
                "utt_idx": 0,
                "startAt": 0,
                "endAt": 1000,
                "duration_sec": 1.0,
                "speaker": 0,
            },
            {
                "call_id": "dup",
                "wav_stem": "two",
                "wav_path": "two.wav",
                "json_path": "two.json",
                "utt_idx": 1,
                "startAt": 0,
                "endAt": 1000,
                "duration_sec": 1.0,
                "speaker": 0,
            },
        ]
    )
    ambiguous_index.to_parquet(index_output, index=False)
    pd.DataFrame(
        [
            {"call_id": "dup", "utt_idx": 0, "speaker": 0},
            {"call_id": "dup", "utt_idx": 1, "speaker": 0},
        ]
    ).to_parquet(features_output, index=False)

    import pytest

    from dcc_data.guards import DuplicateIdentifierError

    # DuplicateIdentifierError specifically (not just any ValueError): this
    # should join the same exception taxonomy every other dcc_data/*.py
    # module raises for "identifier isn't unique" (guards.py), so callers
    # elsewhere that catch DuplicateIdentifierError/DataGuardError to
    # distinguish "a known integrity guard tripped" from "an unexpected bug"
    # catch this too.
    with pytest.raises(DuplicateIdentifierError, match="ambiguous call_id"):
        _run_main(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(root),
                "--n-calls",
                "0",
                "--index-output",
                str(index_output),
                "--features-output",
                str(features_output),
                "--failures-log",
                str(failures_log),
                "--flush-every",
                "0",
                "--resume",
            ]
        )


def test_resume_rejects_a_positive_n_calls(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    import pytest

    # A positive --n-calls (the default is 3000, not 0) combined with
    # --resume used to silently delete incomplete calls the sample didn't
    # happen to reselect, without ever reprocessing them (reviewer-
    # reproduced). Now enforced at the argument-parsing level.
    with pytest.raises(SystemExit):
        _run_main(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(root),
                "--n-calls",
                "1",
                "--index-output",
                str(tmp_path / "index.parquet"),
                "--features-output",
                str(tmp_path / "features.parquet"),
                "--failures-log",
                str(tmp_path / "failures.jsonl"),
                "--resume",
            ]
        )


def test_resume_recovers_from_a_corrupted_features_file_instead_of_crashing(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    args = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--n-calls",
        "0",
        "--index-output",
        str(index_output),
        "--features-output",
        str(features_output),
        "--failures-log",
        str(failures_log),
        "--flush-every",
        "0",
    ]
    _run_main(args)

    # Corrupt the features parquet in place -- simulates a crash mid-write
    # (predating the atomic-write fix) or damage from outside this script.
    features_output.write_bytes(b"not a real parquet file")

    # Must NOT crash uncaught: falls back to preserve-and-restart, same as
    # the "only one file exists" case.
    _run_main([*args, "--resume"])

    backups = list(tmp_path.glob("features.stale-*.parquet"))
    assert len(backups) == 1, "the corrupted file must be preserved, not silently discarded"
    assert set(pd.read_parquet(index_output)["wav_stem"]) == {"call_only"}
    assert set(pd.read_parquet(features_output)["call_id"]) == {"call_only"}


def _build_single_call_index_and_empty_features(tmp_path: Path):
    """Shared setup for the three _validate_loaded_state DataGuardError
    branches below: one real call indexed (2 utterances), features/failures
    left for the caller to construct by hand."""

    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    args = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--n-calls",
        "0",
        "--index-output",
        str(index_output),
        "--features-output",
        str(features_output),
        "--failures-log",
        str(failures_log),
        "--flush-every",
        "0",
    ]
    _run_main(args)
    return root, args, index_output, features_output, failures_log


def test_resume_raises_on_an_utterance_with_both_a_feature_row_and_a_failure_entry(tmp_path: Path) -> None:
    _, args, index_output, features_output, failures_log = _build_single_call_index_and_empty_features(tmp_path)

    # utt_idx 0 already has a feature row (from the successful first run);
    # add a failure entry for that same (call_id, utt_idx) -- a state that
    # should never occur, but --resume must refuse it loudly rather than
    # pick one arbitrarily.
    failures_log.write_text(
        json.dumps(
            {
                "stage": "feature",
                "call_id": "call_only",
                "utt_idx": 0,
                "wav_path": "Training/1.원천데이터/TS_서울_구급/call_only.wav",
                "json_path": "Training/2.라벨링데이터/TL_서울_구급/call_only.json",
                "error_type": "RuntimeError",
                "error_message": "simulated",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    from dcc_data.guards import DataGuardError

    with pytest.raises(DataGuardError, match="both a feature row and a failure entry"):
        _run_main([*args, "--resume"])


def test_resume_raises_on_a_failure_entry_for_an_utterance_never_indexed(tmp_path: Path) -> None:
    _, args, index_output, features_output, failures_log = _build_single_call_index_and_empty_features(tmp_path)

    # utt_idx 99 was never in this call's index at all.
    failures_log.write_text(
        json.dumps(
            {
                "stage": "feature",
                "call_id": "call_only",
                "utt_idx": 99,
                "wav_path": "Training/1.원천데이터/TS_서울_구급/call_only.wav",
                "json_path": "Training/2.라벨링데이터/TL_서울_구급/call_only.json",
                "error_type": "RuntimeError",
                "error_message": "simulated",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    from dcc_data.guards import DataGuardError

    with pytest.raises(DataGuardError, match="never indexed"):
        _run_main([*args, "--resume"])


def test_resume_raises_when_a_failure_wav_path_disagrees_with_the_index(tmp_path: Path) -> None:
    _, args, index_output, features_output, failures_log = _build_single_call_index_and_empty_features(tmp_path)

    # Delete utt_idx 1's feature row so it's unresolved, then log a failure
    # for it whose wav_path doesn't match what the index has on file for
    # the same (call_id, utt_idx) -- internally inconsistent provenance.
    features = pd.read_parquet(features_output)
    features = features[features["utt_idx"] != 1]
    features.to_parquet(features_output, index=False)

    failures_log.write_text(
        json.dumps(
            {
                "stage": "feature",
                "call_id": "call_only",
                "utt_idx": 1,
                "wav_path": "Training/1.원천데이터/TS_서울_구급/some_other_call.wav",
                "json_path": "Training/2.라벨링데이터/TL_서울_구급/some_other_call.json",
                "error_type": "RuntimeError",
                "error_message": "simulated",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import pytest

    from dcc_data.guards import DataGuardError

    with pytest.raises(DataGuardError, match="inconsistent data"):
        _run_main([*args, "--resume"])


def test_resume_raises_loudly_on_a_duplicate_feature_row_instead_of_silently_resuming(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    args = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--n-calls",
        "0",
        "--index-output",
        str(index_output),
        "--features-output",
        str(features_output),
        "--failures-log",
        str(failures_log),
        "--flush-every",
        "0",
    ]
    _run_main(args)

    # Duplicate one feature row in place -- a logically-inconsistent but
    # perfectly parseable file (reviewer-reproduced: the old subset-based
    # completeness check judged the call "done" and left the duplicate in
    # the final output forever). Unlike a corrupted/unreadable file, this
    # must be a loud, blocking error -- not silently auto-recovered by
    # discarding the file, since that could throw away real prior work over
    # what might be a one-row fixable mistake.
    features = pd.read_parquet(features_output)
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    duplicated.to_parquet(features_output, index=False)

    import pytest

    from dcc_data.guards import DuplicateIdentifierError

    with pytest.raises(DuplicateIdentifierError, match="duplicate"):
        _run_main([*args, "--resume"])


def test_resume_recovers_a_call_that_crashed_between_a_periodic_flush_and_the_failures_write(
    tmp_path: Path, monkeypatch
) -> None:
    """Faithful reproduction of the mechanism PR review identified as the
    real one (the PR's original description -- "crash between index write
    and audio open" -- doesn't hold up: _process_one_call returns
    index/feature/failure rows for a call together in one tuple, so a
    mid-call worker crash drops all three from the parent sinks, not just
    features).

    The real path: every utterance in a call fails at the *feature* stage
    (not audio_open, e.g. a corrupted WAV that opens but won't decode),
    which still returns (index_records, [], failures) normally and gets
    added to the parent sinks -- but if the process is interrupted exactly
    between that cycle's index/features flush and its failures flush (the
    two used to happen at different times entirely; now they're back to
    back in the same on_call_done call), the on-disk state is
    index-present/features-absent/failures-not-yet-written. Reproduced here
    via a real _write_failures call site, not post-hoc parquet surgery.
    """

    root = tmp_path / "raw"
    bad_row = _write_call(root, "call_bad", seed=0)
    # A WAV file that opens (has a valid-looking header) but has no actual
    # audio frames -- read_call_waveform succeeds at the file-open level
    # (this call gets indexed) but every utterance's slice/feature-extract
    # step fails: a "feature" stage failure, not "audio_open".
    wav_path = root / bad_row.wav_path
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"")  # zero frames

    manifest = tmp_path / "manifest.csv"
    write_manifest([bad_row], manifest, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    real_write_failures = build_mission2_dataset._write_failures

    def crashing_write_failures(failures, path):
        raise RuntimeError("simulated crash: interrupted between this cycle's flush and the failures write")

    monkeypatch.setattr(build_mission2_dataset, "_write_failures", crashing_write_failures)

    import pytest

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_main(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(root),
                "--n-calls",
                "0",
                "--index-output",
                str(index_output),
                "--features-output",
                str(features_output),
                "--failures-log",
                str(failures_log),
                "--flush-every",
                "1",
            ]
        )

    # Reproduces the real on-disk signature: indexed, but not in features,
    # and failures.jsonl never got written for it.
    assert index_output.exists()
    assert set(pd.read_parquet(index_output)["call_id"]) == {"call_bad"}
    assert not features_output.exists() or "call_bad" not in set(pd.read_parquet(features_output)["call_id"])
    assert not failures_log.exists() or failures_log.stat().st_size == 0

    # Resuming for real must recover it (not skip it forever) and end up
    # with a properly logged failure this time.
    monkeypatch.setattr(build_mission2_dataset, "_write_failures", real_write_failures)
    _run_main(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",
        ]
    )

    final_failures = [json.loads(line) for line in failures_log.read_text(encoding="utf-8").splitlines()]
    assert any(f["call_id"] == "call_bad" for f in final_failures)


def test_atomic_write_helpers_clean_up_their_temp_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    """The repo's other atomic-write sites (dcc_data.manifest.write_manifest,
    scripts/mission2_inference.py, scripts/audit_dcc_dataset.py) all delete
    their temp file if the write/replace fails, instead of orphaning it.
    _atomic_write_text/_atomic_write_parquet must match that, not just
    achieve atomicity on the happy path.
    """

    import os

    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("simulated disk-full/permission failure")

    monkeypatch.setattr(build_mission2_dataset.os, "replace", failing_replace)

    text_path = tmp_path / "failures.jsonl"
    try:
        build_mission2_dataset._atomic_write_text(text_path, "irrelevant")
    except OSError:
        pass
    assert list(tmp_path.glob("*.tmp-*")) == []
    assert not text_path.exists()

    parquet_path = tmp_path / "index.parquet"
    try:
        build_mission2_dataset._atomic_write_parquet(pd.DataFrame({"a": [1]}), parquet_path)
    except OSError:
        pass
    assert list(tmp_path.glob("*.tmp-*")) == []
    assert not parquet_path.exists()

    monkeypatch.setattr(build_mission2_dataset.os, "replace", real_replace)


def test_resume_without_existing_outputs_falls_back_to_a_fresh_run(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest([row], manifest_path, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",  # nothing to resume from yet -- must not crash
        ]
    )

    assert set(pd.read_parquet(index_output)["wav_stem"]) == {"call_only"}
