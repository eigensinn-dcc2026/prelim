"""Offline contracts for Stage 12; never download models or load real data."""
import importlib
import io
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest


def stage12():
    return importlib.import_module("dcc_data.mission2_raw_waveform_pretrained_speech")


def test_single_pinned_frozen_candidate_and_probe_contract():
    s = stage12()
    assert s.CONFIG["encoder"]["repository"] == "microsoft/wavlm-base-plus"
    assert len(s.CONFIG["encoder"]["revision"]) == 40
    assert s.CONFIG["encoder"]["required_sample_rate_hz"] == 16000
    assert s.CONFIG["representation"]["pooling"] == "last_hidden_valid_frame_masked_mean"
    assert s.CONFIG["classifier"]["name"] == "SGDClassifier"
    assert s.CONTRACT["split"] == {
        "seed": 42, "test_size": 0.2,
        "total_rows": 889358, "train_rows": 711467, "validation_rows": 177891,
        "train_calls": 23360, "validation_calls": 5840, "call_overlap": 0,
    }
    assert s.CONTRACT["threshold"] == 0.5
    assert s.CONTRACT["validation_gradient_update"] is False
    assert s.CONTRACT["official_validation_used"] is False
    assert "opensmile" in s.environment_info()["packages"]


def test_fixed_bucket_padding_and_hidden_mask_pooling():
    s = stage12()
    padded, valid_lengths, padded_rows = s.pad_for_encoder(
        [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)],
        sample_rate_hz=16000,
    )
    assert padded.shape == (2, 400)
    assert valid_lengths.tolist() == [2, 1]
    assert padded_rows == 2
    np.testing.assert_array_equal(padded[:, :2], [[1.0, 2.0], [3.0, 0.0]])
    hidden = np.array([[[1.0], [3.0], [100.0]], [[4.0], [100.0], [100.0]]])
    frame_mask = np.array([[True, True, False], [True, False, False]])
    np.testing.assert_array_equal(s.masked_mean_pool(hidden, frame_mask), [[2.0], [4.0]])
    with pytest.raises(ValueError):
        s.pad_for_encoder([np.array([], dtype=np.float32)], sample_rate_hz=16000)


def test_existing_archive_reader_matches_half_open_pcm_crop_without_extraction():
    s = stage12()
    from dcc_data.audio import read_wav_frames

    samples = np.array([-32768, -1234, 0, 8192, 32767, 40, -8, 700], dtype="<i2")
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())
    wav_bytes = wav_buffer.getvalue()

    with tempfile.TemporaryDirectory(prefix=".stage12_zip_test_", dir=Path.cwd()) as temp:
        root = Path(temp)
        archive_path = root / "organizer_training.zip"
        reference_path = root / "reference.wav"
        reference_path.write_bytes(wav_bytes)
        member_name = "Training/audio.wav"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member_name, wav_bytes)
        member = zipfile.ZipFile(archive_path).getinfo(member_name)
        reader = s._ArchiveTrainingAudioReader({
            member_name: (archive_path, member_name, member.file_size, member.CRC)
        })
        try:
            expected = read_wav_frames(reference_path, start_frame=2, end_frame=7)
            actual = reader.read_wav_frames(member_name, start_frame=2, end_frame=7)
            assert actual.sample_rate == expected.sample_rate == 8000
            np.testing.assert_array_equal(actual.waveform, expected.waveform)
            assert not (root / member_name).exists()
            with pytest.raises(ValueError, match="Invalid WAV frame interval"):
                reader.read_wav_frames(member_name, start_frame=7, end_frame=9)
        finally:
            reader.close()


def test_saved_transfer_proof_matches_existing_stage11_verifier_result_exactly():
    s = stage12()
    transfer_file = {
        "schema": "mission2_egemaps_transfer_identity_v1",
        "raw_content_sha256": "a" * 64,
        "raw_file_count": 58400,
    }
    stage11_identity_proof = {**transfer_file, "status": "PASS"}

    assert s._saved_transfer_proof_matches(stage11_identity_proof, transfer_file)
    assert not s._saved_transfer_proof_matches(transfer_file, transfer_file)
    assert not s._saved_transfer_proof_matches(
        {**stage11_identity_proof, "raw_file_count": 58399}, transfer_file
    )
    assert not s._saved_transfer_proof_matches(
        {**stage11_identity_proof, "extra": "unexpected"}, transfer_file
    )


def test_projection_uses_full_train_and_validation_rows_and_runtime_components():
    s = stage12()
    estimate = s.project_runtime(
        benchmark_rows=2000,
        benchmark_seconds=600.0,
        train_rows=711467,
        validation_rows=177891,
        classifier_seconds=300.0,
        artifact_seconds=120.0,
        duration_bin_rates={"<0.5": 3.0, "0.5-1": 2.0, "1-2": 1.5, "2-5": 0.8, ">=5": 0.2},
        duration_bin_shares={"<0.5": .1, "0.5-1": .2, "1-2": .3, "2-5": .3, ">=5": .1},
    )
    assert estimate["total_rows"] == 889358
    assert estimate["mixed_projection_seconds"] > 0
    assert estimate["duration_weighted_projection_seconds"] > 0
    assert estimate["projected_full_hours"] == max(
        estimate["mixed_projection_seconds"], estimate["duration_weighted_projection_seconds"]
    ) / 3600
    assert s.compute_gate(estimate, limit_hours=20)["status"] in {"PASS", "FAIL"}
    assert s.compute_gate({**estimate, "projected_full_hours": 20.01}, limit_hours=20)["status"] == "FAIL"


def test_shard_identity_is_exact_and_never_overwrites():
    s = stage12()
    rows = {"record_key": ["r1", "r2"], "call_id": ["c1", "c2"], "y_true": [0, 1]}
    embeddings = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=".stage12_test_", dir=Path.cwd()) as temp:
        path, meta = s.save_shard(temp, "train", 0, rows, embeddings,
                                  config_hash="cfg", data_fingerprint="data", row_positions=[0, 1])
        loaded = s.load_shard(path, expected_meta=meta, expected_config_hash="cfg",
                              expected_data_fingerprint="data")
        assert loaded["record_key"].tolist() == rows["record_key"]
        assert loaded["embedding"].dtype == np.float32
        with pytest.raises(FileExistsError):
            s.save_shard(temp, "train", 0, rows, embeddings,
                         config_hash="cfg", data_fingerprint="data")
        with pytest.raises(ValueError):
            s.load_shard(path, expected_meta=meta, expected_config_hash="other",
                         expected_data_fingerprint="data")
        path.with_suffix(".json").unlink()
        recovered = s.recover_shard_without_sidecar(
            path, expected_rows=rows, expected_positions=[0, 1],
            expected_config_hash="cfg", expected_data_fingerprint="data")
        assert recovered["sha256"] == s.file_sha256(path)
        assert s.load_shard(path, expected_meta=recovered, expected_config_hash="cfg",
                            expected_data_fingerprint="data")["row_position"].tolist() == [0, 1]


def test_prediction_metrics_and_identity_alignment():
    s = stage12()
    expected = [{"record_key": "a", "call_id": "x", "y_true": 0},
                {"record_key": "b", "call_id": "y", "y_true": 1}]
    metrics, predictions = s.evaluate_predictions(expected, np.array([.9, .8]))
    assert metrics["accuracy"] == .5
    assert metrics["confusion_matrix"] == [[0, 1], [0, 1]]
    assert metrics["recall_0"] == 0.0 and metrics["recall_1"] == 1.0
    assert metrics["errors"] == 1 and metrics["hce_confidence_ge_0_90"] == 1
    shuffled = {key: list(reversed(value)) for key, value in predictions.items()}
    assert [row["record_key"] for row in s.align_predictions(shuffled, expected)] == ["a", "b"]
    slices = s.duration_slice_report(expected, predictions, [.25, 1.5])
    assert slices["status"] == "PASS"
    assert slices["slices"]["duration_<0.5"]["sample_count"] == 1
    assert slices["slices"]["duration_1-2"]["sample_count"] == 1
    with pytest.raises(ValueError):
        s.evaluate_predictions(expected, np.array([np.nan, .8]))


def test_gate_failure_never_authorizes_full_extraction():
    s = stage12()
    with pytest.raises(RuntimeError, match="full extraction is forbidden"):
        s.require_gate_pass({"status": "FAIL", "decision": "RESOURCE/TIME-BUDGET INFEASIBLE"})
    s.require_gate_pass({"status": "PASS"})
