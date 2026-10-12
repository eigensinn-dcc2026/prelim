"""Small synthetic contracts; no competition audio, openSMILE extraction, or Colab."""

from __future__ import annotations

import json
import importlib
import zipfile
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dcc_data import mission2_egemaps_colab as colab
from dcc_data import mission2_egemaps_resume as resume
from dcc_data import mission2_standard_representation as standard


NAMES = tuple(f"f{index}" for index in range(88))
FEATURE_NAMES = tuple(colab.get_egemaps_feature_names())
KEYS = np.asarray(["k0", "k1", "k2"])
CALLS = np.asarray(["a", "a", "b"])


def _manifest():
    return pd.DataFrame({
        "split": ["Training"] * 3, "record_key": KEYS, "call_id": CALLS,
        "audio_path": ["a.wav", "a.wav", "b.wav"],
        "crop_start_frame": [0, 10, 0], "crop_end_frame": [10, 20, 10],
    })


def _partial(tmp_path):
    root = tmp_path / "partial"
    matrix, mask, _ = resume._create_partial(
        root, KEYS, CALLS, NAMES, colab.EXPECTED_FINGERPRINT,
    )
    matrix[:2] = 1
    mask[:2] = True
    resume._checkpoint(root, matrix, mask, KEYS, NAMES, colab.EXPECTED_FINGERPRINT, {})
    return root


def test_valid_partial_read_only_and_no_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "get_egemaps_feature_names", lambda: NAMES)
    root = _partial(tmp_path)
    summary = colab.validate_existing_partial(
        _manifest(), root, KEYS, CALLS, NAMES,
        fingerprint=colab.EXPECTED_FINGERPRINT, expected_rows=3,
    )
    assert summary["status"] == "PASS" and summary["completed"] == 2
    matrix, mask, _ = resume._validate_partial(
        root, KEYS, CALLS, NAMES, colab.EXPECTED_FINGERPRINT, read_only=True,
    )
    assert not matrix.flags.writeable and not mask.flags.writeable
    with pytest.raises(ValueError, match="incomplete"):
        colab.validate_existing_partial(
            _manifest(), tmp_path / "missing", KEYS, CALLS, NAMES,
            fingerprint=colab.EXPECTED_FINGERPRINT, expected_rows=3,
        )


@pytest.mark.parametrize("delta,status", [
    (0, "CROSS_PLATFORM_EXACT_MATCH"),
    (1e-9, "CROSS_PLATFORM_NUMERICALLY_CLOSE"),
    (0.1, "CROSS_PLATFORM_FEATURE_MISMATCH"),
])
def test_cross_platform_status(delta, status):
    local = np.ones((100, 88), dtype=np.float32)
    rerun = local.astype(np.float64)
    rerun[0, 0] += delta
    result = colab.compare_overlap_values(local, rerun, NAMES)
    assert result["status"] == status
    assert result["differing_value_count"] == (0 if delta == 0 else 1)


def _approval_inputs(status):
    return ({"status": "PASS", "fingerprint": colab.EXPECTED_FINGERPRINT},
            {"status": status, "sampled_rows": 100, "padded_available": True,
             "padded_sampled": True},
            {"status": "PASS", "clean_reference_fingerprint": colab.EXPECTED_FINGERPRINT,
             "raw_content_sha256": "raw"})


def test_mismatch_and_unapproved_numerical_close_block_resume():
    with pytest.raises(ValueError, match="mismatch"):
        colab.approve_resume(*_approval_inputs("CROSS_PLATFORM_FEATURE_MISMATCH"))
    inputs = _approval_inputs("CROSS_PLATFORM_NUMERICALLY_CLOSE")
    with pytest.raises(ValueError, match="explicit"):
        colab.approve_resume(*inputs)
    approved = colab.approve_resume(*inputs, explicit_approval=colab.APPROVAL_PHRASE)
    assert approved["numerical_user_approval"] is True
    exact = colab.approve_resume(*_approval_inputs("CROSS_PLATFORM_EXACT_MATCH"))
    assert exact["numerical_user_approval"] is False


def _approved_mismatch_inputs():
    partial = {"status": "PASS", "fingerprint": colab.EXPECTED_FINGERPRINT,
               "rows": 889358, "completed": 470255, "remaining": 419103,
               "feature_dim": 88, "feature_names": list(FEATURE_NAMES), "resume_history": []}
    overlap = {"status": "CROSS_PLATFORM_FEATURE_MISMATCH", "sampled_rows": 100,
               "sampled_calls": 100, "padded_available": True, "padded_sampled": True,
               "sampled_record_keys": [f"k{index}" for index in range(100)],
               "exact_row_count": 48, "numerically_close_row_count": 77,
               "differing_row_count": 52, "differing_value_count": 114,
               "mean_abs_diff": 1.5450959009203043e-08,
               "max_abs_diff": 2.6702880859375e-05}
    identity = {"status": "PASS", "source_split": "Training", "raw_file_count": 58400,
                "rows": 889358, "clean_reference_fingerprint": colab.EXPECTED_FINGERPRINT,
                "raw_content_sha256": "raw"}
    environment = {"platform": "Linux", "packages": {"opensmile": "2.6.0"}}
    return partial, overlap, identity, environment


def test_mismatch_requires_and_records_constrained_manual_approval():
    partial, overlap, identity, environment = _approved_mismatch_inputs()
    with pytest.raises(ValueError, match="explicit"):
        colab.approve_resume(partial, overlap, identity, environment=environment,
                             explicit_approval=None)
    approval = colab.approve_resume(
        partial, overlap, identity, environment=environment, explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
    )
    assert approval["approval_type"] == "APPROVE_CROSS_PLATFORM_MISMATCH"
    assert approval["overlap_status"] == "CROSS_PLATFORM_FEATURE_MISMATCH"
    assert approval["sampled_rows"] == 100 and approval["sampled_calls"] == 100
    assert approval["padded_sampled"] is True
    assert approval["differing_value_count"] == 114
    assert approval["runtime_provenance"] == environment
    assert approval["opensmile_version"] == "2.6.0"
    assert approval["raw_file_count"] == 58400
    assert approval["source_split"] == "Training"
    assert approval["starting_completed_rows"] == 470255
    assert approval["starting_remaining_rows"] == 419103


@pytest.mark.parametrize("field,value", [
    ("mean_abs_diff", 2.0000001e-8),
    ("max_abs_diff", 3.0000001e-5),
    ("differing_value_count", 121),
    ("sampled_calls", 99),
    ("padded_sampled", False),
])
def test_mismatch_override_rejects_overlap_outside_allowlist(field, value):
    partial, overlap, identity, environment = _approved_mismatch_inputs()
    overlap[field] = value
    with pytest.raises(ValueError, match="mismatch override"):
        colab.approve_resume(
            partial, overlap, identity, environment=environment,
            explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
        )


@pytest.mark.parametrize("target,field,value", [
    ("identity", "clean_reference_fingerprint", "wrong"),
    ("identity", "raw_file_count", 58401),
    ("identity", "source_split", "Validation"),
    ("identity", "rows", 1),
    ("partial", "completed", 470254),
    ("partial", "remaining", 419104),
])
def test_mismatch_override_rejects_source_or_partial_contract(target, field, value):
    partial, overlap, identity, environment = _approved_mismatch_inputs()
    (identity if target == "identity" else partial)[field] = value
    with pytest.raises(ValueError, match="mismatch override"):
        colab.approve_resume(
            partial, overlap, identity, environment=environment,
            explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
        )


@pytest.mark.parametrize("target,field,value", [
    ("environment", "opensmile", "2.6.1"),
    ("partial", "feature_dim", 87),
    ("partial", "feature_names", list(reversed(FEATURE_NAMES))),
])
def test_mismatch_override_rejects_runtime_or_feature_contract(target, field, value):
    partial, overlap, identity, environment = _approved_mismatch_inputs()
    if target == "environment":
        environment["packages"][field] = value
    else:
        partial[field] = value
    with pytest.raises(ValueError, match="mismatch override"):
        colab.approve_resume(
            partial, overlap, identity, environment=environment,
            explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
        )


def test_mismatch_resume_records_approval_for_disconnect_recovery(tmp_path, monkeypatch, capsys):
    partial, overlap, identity, environment = _approved_mismatch_inputs()
    approval = colab.approve_resume(
        partial, overlap, identity, environment=environment,
        explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
    )
    root = _partial(tmp_path)
    metadata_path = root / "recovery_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(rows=889358, completed_rows=470255, remaining_rows=419103,
                    dim=88, feature_names=list(FEATURE_NAMES))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(colab, "validate_existing_partial", lambda *args, **kwargs: {
        **partial, "feature_dim": 88, "feature_names": list(FEATURE_NAMES),
    })
    monkeypatch.setattr(colab, "_start_crash_journal", lambda _root: None)
    monkeypatch.setattr(colab, "fill_existing_partial", lambda *args: {"completed": 470255})

    result = colab.resume_approved_partial(
        _manifest(), tmp_path, root, KEYS, CALLS, FEATURE_NAMES,
        approval, overlap, identity, environment,
    )
    assert result["completed"] == 470255
    assert "START_COMPLETED = 470255" in capsys.readouterr().out
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    event = saved["resume_history"][-1]
    assert event["resume_approval"] == approval
    assert event["starting_remaining_rows"] == 419103

    continued = {**partial, "completed": 470300, "remaining": 419058,
                 "resume_history": saved["resume_history"]}
    repeated_approval = colab.approve_resume(
        continued, overlap, identity, environment=environment,
        explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
    )
    assert repeated_approval["starting_completed_rows"] == 470255
    assert repeated_approval["starting_remaining_rows"] == 419103


def _final_mismatch_inputs():
    _, overlap, identity, environment = _approved_mismatch_inputs()
    initial_partial, _, _, _ = _approved_mismatch_inputs()
    approval = colab.approve_resume(
        initial_partial, overlap, identity, environment=environment,
        explicit_approval="APPROVE_CROSS_PLATFORM_MISMATCH",
    )
    history = [{"starting_completed_rows": 470255,
                "starting_remaining_rows": 419103,
                "overlap_status": overlap["status"],
                "raw_content_sha256": identity["raw_content_sha256"],
                "resume_approval": approval}]
    partial = {"status": "PASS", "rows": 889358, "completed": 889358,
               "remaining": 0, "fingerprint": colab.EXPECTED_FINGERPRINT,
               "feature_dim": 88, "feature_names": list(FEATURE_NAMES),
               "resume_history": history}
    return partial, approval, overlap, identity, environment, history


def test_finalization_approval_accepts_only_recorded_valid_manual_mismatch():
    partial, approval, overlap, identity, environment, history = _final_mismatch_inputs()
    assert colab._validate_mismatch_finalization_approval(
        partial, approval, overlap, identity, environment, FEATURE_NAMES, history,
    ) == "PASS"
    with pytest.raises(ValueError, match="approval"):
        colab._validate_mismatch_finalization_approval(
            partial, {}, overlap, identity, environment, FEATURE_NAMES, history,
        )


def test_publish_guard_blocks_mismatch_without_manual_approval(tmp_path):
    _, _, overlap, identity, environment, _ = _final_mismatch_inputs()
    with pytest.raises(ValueError, match="cross-platform approval"):
        colab.publish_approved_cache(
            _manifest(), tmp_path / "partial", tmp_path / "final.npz", KEYS, CALLS,
            FEATURE_NAMES, {}, overlap, identity, environment,
            snapshot_root=tmp_path / "snapshot",
        )
    assert not (tmp_path / "final.npz").exists()


def test_publish_guard_accepts_validated_mismatch_without_publishing(
    tmp_path, monkeypatch,
):
    partial, approval, overlap, identity, environment, history = _final_mismatch_inputs()
    recovery_root = tmp_path / "working"
    snapshot_root = tmp_path / "snapshot"
    recovery_root.mkdir()
    snapshot_root.mkdir()
    (recovery_root / "recovery_metadata.json").write_text(
        json.dumps({"resume_history": history}), encoding="utf-8",
    )
    working_matrix = np.zeros((3, 88), dtype=np.float32)
    snapshot_matrix = working_matrix.copy()
    snapshot_mask = np.asarray([True, True, False])
    working_mask = np.asarray([True, True, True])
    np.save(recovery_root / "completed_mask.npy", working_mask)
    monkeypatch.setattr(colab, "validate_existing_partial", lambda *args, **kwargs: partial)

    def validate_partial(root, *_args, **_kwargs):
        return ((snapshot_matrix, snapshot_mask, []) if Path(root) == snapshot_root
                else (working_matrix, working_mask, []))

    monkeypatch.setattr(colab, "_validate_partial", validate_partial)
    count_nonzero = np.count_nonzero
    monkeypatch.setattr(colab.np, "count_nonzero",
                        lambda values: 470255 if values is snapshot_mask else count_nonzero(values))

    class CacheBuildWouldRun(Exception):
        pass

    monkeypatch.setattr(colab, "build_egemaps_cache",
                        lambda *args, **kwargs: (_ for _ in ()).throw(CacheBuildWouldRun()))
    with pytest.raises(CacheBuildWouldRun):
        colab.publish_approved_cache(
            _manifest(), recovery_root, tmp_path / "final.npz", KEYS, CALLS,
            FEATURE_NAMES, approval, overlap, identity, environment,
            snapshot_root=snapshot_root,
        )
    assert not (tmp_path / "final.npz").exists()


@pytest.mark.parametrize("target,field,value", [
    ("partial", "fingerprint", "wrong"),
    ("identity", "clean_reference_fingerprint", "wrong"),
    ("overlap", "mean_abs_diff", 2.0000001e-8),
    ("overlap", "max_abs_diff", 3.0000001e-5),
    ("overlap", "differing_value_count", 121),
    ("partial", "completed", 889357),
    ("partial", "remaining", 1),
    ("environment", "opensmile", "2.6.1"),
])
def test_finalization_approval_rejects_mismatch_contract_changes(target, field, value):
    partial, approval, overlap, identity, environment, history = _final_mismatch_inputs()
    if target == "partial":
        partial[field] = value
    elif target == "identity":
        identity[field] = value
    elif target == "overlap":
        overlap[field] = value
    else:
        environment["packages"][field] = value
    with pytest.raises(ValueError, match="STOP"):
        colab._validate_mismatch_finalization_approval(
            partial, approval, overlap, identity, environment, FEATURE_NAMES, history,
        )


def test_finalization_approval_requires_history_and_preserved_snapshot_rows():
    partial, approval, overlap, identity, environment, history = _final_mismatch_inputs()
    with pytest.raises(ValueError, match="history"):
        colab._validate_mismatch_finalization_approval(
            partial, approval, overlap, identity, environment, FEATURE_NAMES, [],
        )
    snapshot_matrix = np.asarray([[1, 2], [3, 4], [np.nan, np.nan]], dtype=np.float32)
    snapshot_mask = np.asarray([True, True, False])
    working_matrix = np.asarray([[1, 2], [3, 9], [5, 6]], dtype=np.float32)
    working_mask = np.asarray([True, True, True])
    with pytest.raises(ValueError, match="snapshot completed"):
        colab._assert_snapshot_completed_rows(
            snapshot_matrix, snapshot_mask, working_matrix, working_mask,
        )


def test_overlap_selection_spans_lengths_and_includes_padded_row():
    manifest = pd.DataFrame({
        "record_key": [f"k{index}" for index in range(200)],
        "crop_start_frame": np.zeros(200, dtype=np.int64),
        "crop_end_frame": np.arange(1, 201),
    })
    selected = colab.select_overlap_rows(
        manifest, np.ones(200, dtype=np.bool_), ["k50"], count=100,
    )
    assert len(selected) == len(set(selected)) == 100
    assert 0 in selected and 50 in selected


def test_completed_call_skipped_and_checkpoint_history_preserved(tmp_path, monkeypatch):
    root = _partial(tmp_path)
    original_completed = np.load(root / "matrix_88_partial.npy", mmap_mode="r")[:2].copy()
    reads = []

    def read(path):
        reads.append(path.name)
        return SimpleNamespace(waveform=np.ones((1, 10), dtype=np.float32), sample_rate=8000)

    monkeypatch.setattr(resume, "read_wav_frames", read)
    monkeypatch.setattr(resume, "extract_with_short_audio_fallback",
                        lambda crop, rate: (np.ones(88, dtype=np.float32), NAMES, False))
    metadata_path = root / "recovery_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resume_history"] = [{"starting_completed": 2}]
    metadata["windows_overlap_record_keys"] = ["k0"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = resume.fill_existing_partial(
        _manifest(), tmp_path, root, KEYS, CALLS, NAMES, colab.EXPECTED_FINGERPRINT,
    )
    assert result["completed"] == 3 and reads == ["b.wav"]
    np.testing.assert_array_equal(
        np.load(root / "matrix_88_partial.npy", mmap_mode="r")[:2], original_completed,
    )
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["resume_history"] == metadata["resume_history"]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["windows_overlap_record_keys"] == ["k0"]


def test_short_audio_policy_is_shared():
    assert colab.extract_with_short_audio_fallback is resume.extract_with_short_audio_fallback
    assert colab.SHORT_AUDIO_FALLBACK == "zero_pad_to_800_on_nonfinite_short_segment"
    assert colab.PADDING_TARGET_SAMPLES == 800


def test_interrupted_checkpoint_rolls_back_only_uncommitted_rows(tmp_path, monkeypatch):
    root = _partial(tmp_path)
    monkeypatch.setattr(colab, "get_egemaps_feature_names", lambda: NAMES)
    colab._start_crash_journal(root)
    matrix = np.load(root / "matrix_88_partial.npy", mmap_mode="r+")
    mask = np.load(root / "completed_mask.npy", mmap_mode="r+")
    matrix[2] = 3
    mask[2] = True
    matrix.flush()
    mask.flush()
    assert "ROLLED_BACK" in colab.recover_interrupted_checkpoint(root)
    checked, checked_mask, _ = resume._validate_partial(
        root, KEYS, CALLS, NAMES, colab.EXPECTED_FINGERPRINT, read_only=True,
    )
    assert checked_mask.tolist() == [True, True, False]
    assert np.isnan(checked[2]).all() and np.all(checked[:2] == 1)


def test_publish_guard_requires_complete_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "get_egemaps_feature_names", lambda: NAMES)
    # The synthetic publish fixture has 3 rows; keep production's 889358-row guard.
    monkeypatch.setattr(colab, "validate_existing_partial", lambda *args, **kwargs:
                        resume.validate_existing_partial(*args, **kwargs, expected_rows=3))
    root = _partial(tmp_path)
    final = tmp_path / "final.npz"
    partial, overlap, identity = _approval_inputs("CROSS_PLATFORM_EXACT_MATCH")
    approval = colab.approve_resume(partial, overlap, identity)
    with pytest.raises(ValueError, match="requires all"):
        colab.publish_approved_cache(
            _manifest(), root, final, KEYS, CALLS, NAMES,
            approval, overlap, identity, {"packages": {"opensmile": "2.6.0"}},
        )
    assert not final.exists()


def test_final_publish_keeps_colab_provenance_and_partial(tmp_path, monkeypatch):
    root = _partial(tmp_path)
    monkeypatch.setattr(resume, "get_egemaps_feature_names", lambda: NAMES)
    monkeypatch.setattr(standard, "get_egemaps_feature_names", lambda: NAMES)
    monkeypatch.setattr(colab, "validate_existing_partial", lambda *args, **kwargs:
                        resume.validate_existing_partial(*args, **kwargs, expected_rows=3))
    matrix = np.load(root / "matrix_88_partial.npy", mmap_mode="r+")
    mask = np.load(root / "completed_mask.npy", mmap_mode="r+")
    matrix[2] = 1
    mask[2] = True
    resume._checkpoint(root, matrix, mask, KEYS, NAMES, colab.EXPECTED_FINGERPRINT, {})
    metadata_path = root / "recovery_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resume_history"] = [{"starting_completed": 2}]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    partial, overlap, identity = _approval_inputs("CROSS_PLATFORM_EXACT_MATCH")
    approval = colab.approve_resume(partial, overlap, identity)
    final = tmp_path / "new" / "final.npz"
    result = colab.publish_approved_cache(
        _manifest(), root, final, KEYS, CALLS, NAMES,
        approval, overlap, identity, {"packages": {"opensmile": "2.6.0"}},
    )
    assert result["rows"] == 3 and root.is_dir()
    loaded = standard.load_egemaps_cache(
        final, KEYS, CALLS, source_provenance=colab.EXPECTED_FINGERPRINT,
    )
    assert loaded.metadata["colab_resume"]["resume_approval"] == approval
    with pytest.raises(FileExistsError):
        colab.publish_approved_cache(
            _manifest(), root, final, KEYS, CALLS, NAMES,
            approval, overlap, identity, {"packages": {"opensmile": "2.6.0"}},
        )


def test_import_has_no_extraction_side_effect(monkeypatch):
    called = []
    original = resume.extract_with_short_audio_fallback
    monkeypatch.setattr(resume, "extract_with_short_audio_fallback",
                        lambda *args: called.append(args))
    importlib.reload(colab)
    assert called == []
    monkeypatch.setattr(resume, "extract_with_short_audio_fallback", original)
    importlib.reload(colab)


def _archive_manifest():
    return pd.DataFrame({
        "split": ["Training"], "record_key": ["a:0"], "call_id": ["a"],
        "audio_path": ["Training/1.원천데이터/TS_서울_구급/a.wav"],
        "json_path": ["Training/2.라벨링데이터/TL_서울_구급/a.json"],
        "crop_start_frame": [0], "crop_end_frame": [8000],
    })


def _shared_zip(tmp_path, manifest, *, omitted=None, extra=None):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.zeros(8000, dtype="<i2").tobytes())
    path = tmp_path / "001.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for column, payload in (("audio_path", stream.getvalue()), ("json_path", b"{}")):
            if column != omitted:
                archive.writestr("대학부 데이터/" + manifest.iloc[0][column], payload)
        archive.writestr("대학부 데이터/Validation/ignored.wav", b"not training")
        if extra:
            archive.writestr(extra, b"unexpected")
    return path


def test_shared_archives_stage_manifest_paths_and_existing_loader(tmp_path, monkeypatch):
    manifest = _archive_manifest()
    archive = _shared_zip(tmp_path, manifest)
    root = tmp_path / "raw"
    plan = colab.inspect_shared_training_archives(manifest, [archive])
    assert plan["files"] == 2 and plan["wav_files"] == 1
    assert not root.exists()  # Indexing is read-only.
    result = colab.prepare_shared_training_raw(manifest, [archive], root, reserve_bytes=0)
    assert result["files"] == 2
    from dcc_data.audio import read_wav_frames
    audio = read_wav_frames(root / manifest.iloc[0].audio_path)
    assert audio.sample_rate == 8000 and audio.waveform.shape == (1, 8000)
    assert not (root / "Validation").exists()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    colab.prepare_shared_training_raw(manifest, [archive], root, reserve_bytes=0)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert colab._raw_digest(manifest, root)[1] == 2
    # Overlap/benchmark use this same existing extraction path after staging.
    monkeypatch.setattr(colab, "extract_with_short_audio_fallback", lambda crop, rate:
                        (np.ones(88, dtype=np.float32), NAMES, False))
    vectors, calls_used = colab._extract_rows(manifest, root, np.asarray([0]), NAMES)
    assert calls_used == 1 and vectors[0].shape == (88,)


@pytest.mark.parametrize("case", ["missing", "duplicate", "unsafe", "wrong_path", "wrong_split"])
def test_shared_archive_errors_stop_before_staging(tmp_path, case):
    manifest = _archive_manifest()
    extra = {"unsafe": "../Training/a.wav",
             "wrong_path": "Training/other/a.wav"}.get(case)
    archive = _shared_zip(tmp_path, manifest, omitted="audio_path" if case == "missing" else None,
                          extra=extra)
    archives = [archive]
    if case == "duplicate":
        second = tmp_path / "002.zip"
        second.write_bytes(archive.read_bytes())
        archives.append(second)
    if case == "wrong_split":
        manifest.loc[0, "split"] = "Validation"
    root = tmp_path / "raw"
    with pytest.raises(ValueError, match="STOP"):
        colab.prepare_shared_training_raw(manifest, archives, root, reserve_bytes=0)
    assert not root.exists()


def test_shared_archive_existing_raw_mismatch_is_never_overwritten(tmp_path):
    manifest = _archive_manifest()
    archive = _shared_zip(tmp_path, manifest)
    root = tmp_path / "raw"
    target = root / manifest.iloc[0].audio_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing bytes")
    with pytest.raises(ValueError, match="STOP.*existing"):
        colab.prepare_shared_training_raw(manifest, [archive], root, reserve_bytes=0)
    assert target.read_bytes() == b"existing bytes"


def test_shared_archive_insufficient_disk_stops_before_writes(tmp_path, monkeypatch):
    manifest = _archive_manifest()
    archive = _shared_zip(tmp_path, manifest)
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    root = tmp_path / "raw"
    with pytest.raises(ValueError, match="STOP.*space"):
        colab.prepare_shared_training_raw(manifest, [archive], root, reserve_bytes=0)
    assert not root.exists()
