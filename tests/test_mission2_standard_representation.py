from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
import shutil
from types import SimpleNamespace
import wave

import numpy as np
import pandas as pd
import pytest

from dcc_data import mission2_acoustic_feature_expansion as acoustic
from dcc_data import mission2_model_experiments as models


standard = importlib.import_module("dcc_data.mission2_standard_representation")
clean = importlib.import_module("dcc_data.mission2_clean_stage4")


def _sine(samples: int = 8_000) -> np.ndarray:
    time = np.arange(samples) / 8_000
    return (0.1 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)


def _small_comparison():
    keys = np.asarray([f"Training|call_{i}|0" for i in range(10)])
    calls = np.asarray([f"call_{i}" for i in range(10)])
    labels = np.asarray([i % 2 for i in range(10)], dtype=np.int64)
    train_idx, val_idx, audit = models.create_fixed_internal_split(
        calls, split_function=models.standalone_group_split_indices
    )
    names_59 = (
        *acoustic.CANONICAL_52D_FEATURE_NAMES,
        *(name for block in acoustic.RESIDUAL_COMBINATION_SPECS["AF10_AF11_COMBINATION"]
          for name in acoustic.BLOCK_BY_ID[block].feature_names),
    )
    matrix_59 = np.arange(10 * 59, dtype=np.float32).reshape(10, 59) / 1000
    prepared = acoustic._prepare_representation(
        representation_id="AF10_AF11_COMBINATION", X=matrix_59, y=labels,
        record_keys=keys, feature_names=names_59, train_idx=train_idx,
        validation_idx=val_idx, split_audit=audit,
    )
    matrix_88 = np.tile(np.arange(88, dtype=np.float32), (10, 1)) + np.arange(10, dtype=np.float32)[:, None]
    return prepared, matrix_88, labels, keys, calls


def test_extract_exact_88d_is_stable_float32_and_waveform_only():
    first, names = standard.extract_egemaps(_sine(), 8_000)
    second, repeated_names = standard.extract_egemaps(_sine(), 8_000)
    assert first.shape == (88,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert names == repeated_names == standard.get_egemaps_feature_names()
    assert len(names) == len(set(names)) == 88


def test_short_signal_fails_without_imputation_and_silence_is_finite():
    with pytest.warns(UserWarning, match="Segment too short"):
        with pytest.raises(ValueError, match="NaN|finite|short"):
            standard.extract_egemaps(_sine(80), 8_000)
    silence, _ = standard.extract_egemaps(np.zeros(8_000, dtype=np.float32), 8_000)
    assert silence.shape == (88,) and np.isfinite(silence).all()
    with pytest.raises(ValueError, match="8 kHz|8000"):
        standard.extract_egemaps(_sine(), 16_000)
    with pytest.raises(ValueError, match="mono|one-dimensional"):
        standard.extract_egemaps(_sine().reshape(1, -1), 8_000)


def test_cache_rejects_record_key_order_and_config_mismatch(tmp_path):
    path = tmp_path / "egemaps.npz"
    names = standard.get_egemaps_feature_names()
    matrix = np.ones((2, 88), dtype=np.float32)
    keys = ("Training|a|0", "Training|b|0")
    calls = ("a", "b")
    standard.build_egemaps_cache(path, matrix, keys, calls, names, source_provenance="synthetic")
    loaded = standard.load_egemaps_cache(path, keys, calls, source_provenance="synthetic")
    assert np.array_equal(loaded.X, matrix)
    with pytest.raises(ValueError, match="record_key"):
        standard.load_egemaps_cache(path, keys[::-1], calls, source_provenance="synthetic")
    with pytest.raises(ValueError, match="config|provenance"):
        standard.load_egemaps_cache(path, keys, calls, source_provenance="changed")
    with pytest.raises(ValueError, match="feature"):
        standard.build_egemaps_cache(tmp_path / "wrong.npz", matrix, keys, calls, names[::-1], source_provenance="synthetic")
    with np.load(path, allow_pickle=False) as original:
        fields = {name: original[name] for name in original.files}
    metadata = json.loads(fields["metadata"].item())
    metadata["opensmile_version"] = "changed"
    fields["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(tmp_path / "stale.npz", **fields)
    with pytest.raises(ValueError, match="config|provenance"):
        standard.load_egemaps_cache(tmp_path / "stale.npz", keys, calls, source_provenance="synthetic")


def _colab_mismatch_provenance(names):
    fingerprint = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
    overlap = {
        "status": "CROSS_PLATFORM_FEATURE_MISMATCH", "sampled_rows": 100,
        "sampled_calls": 100, "padded_sampled": True,
        "exact_row_count": 48, "numerically_close_row_count": 77,
        "differing_row_count": 52, "differing_value_count": 114,
        "mean_abs_diff": 1.5450959009203043e-08,
        "max_abs_diff": 2.6702880859375e-05,
    }
    identity = {
        "status": "PASS", "source_split": "Training", "raw_file_count": 58400,
        "rows": 889358, "clean_reference_fingerprint": fingerprint,
        "raw_content_sha256": "a" * 64,
    }
    environment = {"platform": "Linux", "packages": {"opensmile": "2.6.0"}}
    approval = {
        "status": "APPROVED", "approval_type": "APPROVE_CROSS_PLATFORM_MISMATCH",
        "overlap_status": overlap["status"], **{
            key: overlap[key] for key in (
                "sampled_rows", "sampled_calls", "padded_sampled", "exact_row_count",
                "numerically_close_row_count", "differing_row_count", "differing_value_count",
                "mean_abs_diff", "max_abs_diff",
            )
        },
        "runtime_provenance": environment, "opensmile_version": "2.6.0",
        "feature_dim": 88, "feature_names": list(names),
        "raw_file_count": 58400, "source_split": "Training", "source_rows": 889358,
        "clean_reference_fingerprint": fingerprint,
        "raw_content_sha256": identity["raw_content_sha256"],
        "starting_completed_rows": 470255, "starting_remaining_rows": 419103,
        "fingerprint": fingerprint,
    }
    history = [{
        "overlap_status": overlap["status"],
        "raw_content_sha256": identity["raw_content_sha256"],
        "starting_completed_rows": 470255, "starting_remaining_rows": 419103,
        "resume_approval": approval,
    }]
    return {"environment": environment, "source_identity": identity,
            "overlap_audit": overlap, "resume_approval": approval,
            "resume_history": history}


def test_make_egemaps_data_rejects_mismatch_without_manual_approval():
    names = standard.get_egemaps_feature_names()
    colab_resume = _colab_mismatch_provenance(names)
    wrong_phrase = {**colab_resume["resume_approval"],
                    "approval_type": "APPROVE_NUMERICALLY_CLOSE"}
    for approval in (None, wrong_phrase):
        metadata = {**colab_resume, "resume_approval": approval}
        with pytest.raises(ValueError, match="mismatch|approval"):
            standard.make_egemaps_data(
                np.ones((2, 88), dtype=np.float32), ("k0", "k1"), ("c0", "c1"), names,
                source_provenance=colab_resume["source_identity"]["clean_reference_fingerprint"],
                colab_resume=metadata,
            )


def test_make_egemaps_data_accepts_valid_recorded_manual_mismatch_approval():
    names = standard.get_egemaps_feature_names()
    colab_resume = _colab_mismatch_provenance(names)
    data = standard.make_egemaps_data(
        np.ones((2, 88), dtype=np.float32), ("k0", "k1"), ("c0", "c1"), names,
        source_provenance=colab_resume["source_identity"]["clean_reference_fingerprint"],
        colab_resume=colab_resume,
    )
    assert data.metadata["colab_resume"]["resume_approval"]["approval_type"] == (
        "APPROVE_CROSS_PLATFORM_MISMATCH"
    )


@pytest.mark.parametrize("field,value", [
    ("differing_value_count", 121),
    ("mean_abs_diff", 2.0000001e-8),
    ("max_abs_diff", 3.0000001e-5),
])
def test_make_egemaps_data_rejects_mismatch_approval_over_threshold(field, value):
    names = standard.get_egemaps_feature_names()
    colab_resume = _colab_mismatch_provenance(names)
    colab_resume["overlap_audit"][field] = value
    with pytest.raises(ValueError, match="mismatch|approval"):
        standard.make_egemaps_data(
            np.ones((2, 88), dtype=np.float32), ("k0", "k1"), ("c0", "c1"), names,
            source_provenance=colab_resume["source_identity"]["clean_reference_fingerprint"],
            colab_resume=colab_resume,
        )


@pytest.mark.parametrize("status", [
    "CROSS_PLATFORM_EXACT_MATCH", "CROSS_PLATFORM_NUMERICALLY_CLOSE",
])
def test_make_egemaps_data_preserves_existing_approved_overlap_statuses(status):
    names = standard.get_egemaps_feature_names()
    colab_resume = {
        "environment": {}, "source_identity": {}, "resume_history": [],
        "overlap_audit": {"status": status},
    }
    data = standard.make_egemaps_data(
        np.ones((2, 88), dtype=np.float32), ("k0", "k1"), ("c0", "c1"), names,
        source_provenance="synthetic", colab_resume=colab_resume,
    )
    assert data.metadata["colab_resume"]["overlap_audit"]["status"] == status


@pytest.mark.parametrize("change", ["split", "label", "model"])
def test_comparison_rejects_changed_contract_before_fitting(monkeypatch, change):
    prepared, matrix, labels, keys, calls = _small_comparison()
    if change == "split":
        prepared = replace(prepared, train_idx=prepared.train_idx[::-1])
    elif change == "label":
        labels = labels.copy()
        labels[prepared.train_idx[0]] ^= 1
    else:
        monkeypatch.setattr(acoustic, "M04_ITER100_FEATURE_SCREENING_CONFIG", {"max_iter": 99})
    monkeypatch.setattr(acoustic, "_run_prepared_representation", lambda *a, **k: pytest.fail("fitted before validation"))
    data = standard.make_egemaps_data(
        matrix, keys, calls, standard.get_egemaps_feature_names(), source_provenance="synthetic"
    )
    with pytest.raises(ValueError, match="split|label|M04"):
        standard.run_standard_representation_comparison(
            prepared, data, labels, keys, calls, expected_split_contract=None
        )


def test_import_does_not_extract_or_fit(monkeypatch):
    import opensmile
    from sklearn.neural_network import MLPClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("import started extraction or fit")

    monkeypatch.setattr(opensmile.Smile, "process_signal", forbidden)
    monkeypatch.setattr(MLPClassifier, "fit", forbidden)
    importlib.reload(standard)


def test_small_synthetic_comparison_reports_paired_contract():
    prepared, matrix, labels, keys, calls = _small_comparison()
    data = standard.make_egemaps_data(
        matrix, keys, calls, standard.get_egemaps_feature_names(), source_provenance="synthetic"
    )
    result = standard.run_standard_representation_comparison(
        prepared, data, labels, keys, calls, expected_split_contract=None
    )
    assert result["reports"]["59D"]["dimension"] == 59
    assert result["reports"]["eGeMAPSv02"]["dimension"] == 88
    for report in result["reports"].values():
        assert 0 <= report["train_accuracy"] <= 1
        assert 0 <= report["internal_validation_accuracy"] <= 1
        assert sum(map(sum, report["confusion_matrix"])) == len(prepared.validation_idx)
    assert sum(result[key] for key in (
        "both_correct", "both_wrong", "59d_wrong_egemaps_correct",
        "59d_correct_egemaps_wrong",
    )) == len(prepared.validation_idx)
    assert result["net_correct_gain"] == (
        result["59d_wrong_egemaps_correct"] - result["59d_correct_egemaps_wrong"]
    )


def _provenance_fixture():
    manifest = pd.DataFrame({
        "record_key": ["Training|a|0", "Training|b|0"],
        "call_id": ["a", "b"], "speaker": [0, 1],
    })
    cache = pd.DataFrame({
        "record_key": manifest["record_key"], "call_id": manifest["call_id"],
        "zcr": [0.1, 0.2], "spectral_rolloff": [10.0, 20.0],
    })
    expected = {"cache_schema_version": "candidate_v1", "row_count": 2,
                "source_split": "Training", "feature_names": ["zcr", "spectral_rolloff"]}
    metadata = {**expected, "source_signature": "inventory-a",
                "source_content_sha256": "a" * 64,
                "source_content_signature_kind": standard.RAW_CONTENT_SIGNATURE_KIND,
                "record_key_signature": models.record_key_signature(manifest["record_key"])}
    manifest_metadata = {"source_signature": "inventory-a",
                         "source_content_sha256": "a" * 64,
                         "source_content_signature_kind": standard.RAW_CONTENT_SIGNATURE_KIND}
    return manifest, cache, expected, metadata, manifest_metadata


def _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata,
                     current_digest="a" * 64):
    return standard.audit_cache_provenance(
        cache_name="E05 candidate", manifest_rows=manifest, cache_rows=cache,
        metadata=metadata, manifest_metadata=manifest_metadata,
        expected_metadata=expected, feature_names=("zcr", "spectral_rolloff"),
        verified_current_content_sha256=current_digest,
    )


def test_provenance_exact_and_filesystem_signature_only_mismatch():
    manifest, cache, expected, metadata, manifest_metadata = _provenance_fixture()
    assert _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata)["status"] == standard.PROVENANCE_EXACT
    manifest_metadata["source_signature"] = "inventory-b"
    result = _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata)
    assert result["status"] == standard.PROVENANCE_FILESYSTEM_ONLY
    assert metadata["source_signature"] == "inventory-a"
    assert manifest_metadata["source_signature"] == "inventory-b"


@pytest.mark.parametrize("change", ["row", "key_order", "key_set", "call", "feature_order", "schema", "split", "content", "key_signature"])
def test_provenance_stale_contracts(change):
    manifest, cache, expected, metadata, manifest_metadata = _provenance_fixture()
    if change == "row":
        cache = cache.iloc[:1].copy()
    elif change == "key_order":
        cache = cache.iloc[::-1].reset_index(drop=True)
    elif change == "key_set":
        cache.loc[1, "record_key"] = "Training|c|0"
    elif change == "call":
        cache.loc[1, "call_id"] = "c"
    elif change == "feature_order":
        cache = cache[["record_key", "call_id", "spectral_rolloff", "zcr"]]
    elif change == "schema":
        metadata["cache_schema_version"] = "candidate_v0"
    elif change == "split":
        metadata["source_split"] = "Validation"
    elif change == "key_signature":
        metadata["record_key_signature"] = "wrong"
    else:
        manifest_metadata["source_content_sha256"] = "b" * 64
    assert _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata)["status"] == standard.PROVENANCE_STALE


def test_provenance_unverifiable_without_raw_content_and_no_metadata_rewrite(tmp_path):
    manifest, cache, expected, metadata, manifest_metadata = _provenance_fixture()
    metadata.pop("source_content_sha256")
    manifest_metadata["source_signature"] = "inventory-b"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    before = path.read_bytes()
    result = _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata)
    assert result["status"] == standard.PROVENANCE_UNVERIFIABLE
    assert path.read_bytes() == before
    assert result["reason"] == "SOURCE_SIGNATURE_MISMATCH"


def test_provenance_missing_source_signature_has_distinct_reason():
    manifest, cache, expected, metadata, manifest_metadata = _provenance_fixture()
    metadata.pop("source_signature")
    metadata.pop("source_content_sha256")
    result = _audit_synthetic(manifest, cache, expected, metadata, manifest_metadata)
    assert result["status"] == standard.PROVENANCE_UNVERIFIABLE
    assert result["reason"] == "RAW_SOURCE_IDENTITY_NOT_RECORDED"


def test_provenance_rejects_label_mismatch():
    manifest, cache, expected, metadata, manifest_metadata = _provenance_fixture()
    cache["y"] = [1, 0]
    result = standard.audit_cache_provenance(
        cache_name="MFCC40", manifest_rows=manifest, cache_rows=cache,
        metadata=metadata, manifest_metadata=manifest_metadata,
        expected_metadata=expected, feature_names=("zcr", "spectral_rolloff"),
        label_column="y", verified_current_content_sha256="a" * 64,
    )
    assert result["status"] == standard.PROVENANCE_STALE
    assert result["label"] == "mismatch"


def test_historical_divergence_requires_artifacts_and_inventory_hashes():
    assert standard.audit_historical_af09_divergence({}, {})["status"] == "UNRESOLVED_DUE_TO_MISSING_ARTIFACT"
    assert standard.compare_manifest_inventories({}, {})["status"] == standard.PROVENANCE_UNVERIFIABLE
    old = {"source_inventory": [{"path": "a.wav", "size_bytes": 4,
                                  "mtime_ns": 1, "sha256": "same"}]}
    new = {"source_inventory": [{"path": "a.wav", "size_bytes": 4,
                                  "mtime_ns": 2, "sha256": "same"}]}
    comparison = standard.compare_manifest_inventories(old, new)
    assert comparison["status"] == "INVENTORY_CONTENT_MATCH"
    assert len(comparison["differences"]) == 1


def test_raw_content_signature_ignores_mtime_but_detects_changed_bytes(tmp_path):
    audio = tmp_path / "a.wav"
    annotation = tmp_path / "a.json"
    audio.write_bytes(b"waveform-a")
    annotation.write_bytes(b'{"speaker": 0}')
    manifest = pd.DataFrame({"audio_path": ["a.wav"], "json_path": ["a.json"]})
    before = standard.raw_content_signature(manifest, tmp_path)
    audio.touch()
    assert standard.raw_content_signature(manifest, tmp_path) == before
    audio.write_bytes(b"waveform-b")
    assert standard.raw_content_signature(manifest, tmp_path) != before


def _small_clean_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index in range(4):
        call = f"call_{index}"
        with wave.open(str(raw / f"{call}.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8_000)
            stream.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
        (raw / f"{call}.json").write_text("{}", encoding="utf-8")
        rows.append({
            "split": "Training", "record_key": f"Training|{call}|0",
            "call_id": call, "annotation_row": 0, "speaker": index % 2,
            "audio_path": f"{call}.wav", "json_path": f"{call}.json",
            "crop_start_frame": 0, "crop_end_frame": 8_000, "crop_ok": True,
            "startAt_ms": 0.0, "endAt_ms": 1000.0,
        })
    manifest = pd.DataFrame(rows)
    path = tmp_path / "manifest.parquet"
    metadata_path = tmp_path / "manifest.json"
    manifest.to_parquet(path, index=False)
    metadata_path.write_text(json.dumps({"schema_version": "mission2_raw_eda_v1"}), encoding="utf-8")
    return manifest, path, metadata_path, raw


def _fake_clean_extractors(monkeypatch):
    def mfcc(source, _raw):
        start = source["startAt_ms"].to_numpy(dtype=np.float64)
        end = source["endAt_ms"].to_numpy(dtype=np.float64)
        return {"X": np.ones((len(source), 40), dtype=np.float32),
                "y": source["speaker"].to_numpy(dtype=np.int64),
                "stem": np.asarray(source["call_id"].astype(str).tolist(), dtype=str),
                "start_at": start, "end_at": end, "duration": end - start}
    def candidate_frame(source, *, feature_names, **_kwargs):
        frame = source[["record_key", "call_id"]].copy()
        for name in feature_names:
            frame[name] = np.float32(0.25)
        return frame
    def acoustic_frame(source, *, block_ids, **_kwargs):
        names = tuple(name for block in block_ids for name in acoustic.BLOCK_BY_ID[block].feature_names)
        return candidate_frame(source, feature_names=names)
    monkeypatch.setattr(clean, "_extract_mfcc40", mfcc)
    monkeypatch.setattr(clean.candidate, "extract_candidate_feature_frame", candidate_frame)
    monkeypatch.setattr(clean.acoustic, "extract_candidate_feature_frame", acoustic_frame)


def test_clean_chain_is_versioned_current_schema_and_never_overwrites(tmp_path, monkeypatch):
    manifest, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    historical = tmp_path / "historical_v1.npz"
    historical.write_bytes(b"preserve-historical")
    root = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "clean",
        expected_rows=4, expected_calls=4,
    )
    assert root.name.startswith("stage4_clean_v1_")
    assert historical.read_bytes() == b"preserve-historical"
    candidate_meta = json.loads((root / "e05_candidate.json").read_text(encoding="utf-8"))
    assert candidate_meta["extractor_contract"]["cache_schema_version"] == "mission2_candidate_audio_features_v2"
    mfcc_meta = json.loads((root / "mfcc40.json").read_text(encoding="utf-8"))
    chain_meta = json.loads((root / "chain.json").read_text(encoding="utf-8"))
    assert chain_meta["derived_nodes"]["e05_42d"]["parent_hashes"]["mfcc40"] == mfcc_meta["artifact_sha256"]
    assert chain_meta["derived_nodes"]["e05_42d"]["parent_hashes"]["e05_candidate"] == candidate_meta["artifact_sha256"]
    assert candidate_meta["record_key_signature"] == models.record_key_signature(manifest["record_key"])
    assert candidate_meta["source_inventory_signature"] != candidate_meta["source_row_signature"]
    for name in clean.AF_BLOCKS:
        metadata = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        assert metadata["source_inventory_signature"]
        assert metadata["config_fingerprint"]
        assert metadata["parent_provenance"]["source_row_signature"] == mfcc_meta["source_row_signature"]
    assembled = clean.load_clean_stage4_59d(
        root, manifest_path, meta_path, raw, expected_rows=4, expected_calls=4,
    )
    assert assembled.X.shape == (4, 59)
    assert assembled.feature_names == clean.clean_59d_feature_names()
    assert assembled.feature_names[:52] == acoustic.CANONICAL_52D_FEATURE_NAMES
    assert assembled.X[:, :52].shape == (4, 52)
    assert np.array_equal(assembled.y, manifest["speaker"].to_numpy())
    assert np.array_equal(assembled.call_ids, manifest["call_id"].to_numpy())
    with pytest.raises(FileExistsError, match="already exists"):
        clean.build_clean_stage4_chain(
            manifest_path, meta_path, raw, tmp_path / "clean",
            expected_rows=4, expected_calls=4,
        )
    assert historical.read_bytes() == b"preserve-historical"
    with pytest.raises(FileNotFoundError):
        clean.load_clean_stage4_59d(
            historical.parent, manifest_path, meta_path, raw,
            expected_rows=4, expected_calls=4,
        )


def test_clean_loader_rejects_historical_schema_and_changed_source(tmp_path, monkeypatch):
    manifest, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    root = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "clean",
        expected_rows=4, expected_calls=4,
    )
    candidate_meta_path = root / "e05_candidate.json"
    metadata = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    metadata["extractor_contract"]["cache_schema_version"] = "mission2_candidate_audio_features_v1"
    candidate_meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact metadata"):
        clean.load_clean_stage4_59d(root, manifest_path, meta_path, raw,
                                     expected_rows=4, expected_calls=4)
    metadata["extractor_contract"]["cache_schema_version"] = "mission2_candidate_audio_features_v2"
    candidate_meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    changed = manifest.copy()
    changed.loc[0, "speaker"] = 1
    changed.to_parquet(manifest_path, index=False)
    with pytest.raises(ValueError, match="source contract"):
        clean.load_clean_stage4_59d(root, manifest_path, meta_path, raw,
                                     expected_rows=4, expected_calls=4)


def test_clean_build_does_not_publish_misaligned_candidate(tmp_path, monkeypatch):
    _, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)

    def shuffled_candidate(source, *, feature_names, **_kwargs):
        frame = source[["record_key", "call_id"]].iloc[::-1].reset_index(drop=True)
        for name in feature_names:
            frame[name] = np.float32(0.25)
        return frame

    monkeypatch.setattr(clean.candidate, "extract_candidate_feature_frame", shuffled_candidate)
    output = tmp_path / "clean"
    with pytest.raises(ValueError, match="key/call/feature order"):
        clean.build_clean_stage4_chain(
            manifest_path, meta_path, raw, output,
            expected_rows=4, expected_calls=4,
        )
    assert not any(output.glob("stage4_clean_v1_*"))


def test_clean_chain_optional_raw_content_proof_detects_changed_bytes(tmp_path, monkeypatch):
    manifest, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    content_sha256 = standard.raw_content_signature(manifest, raw)
    root = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "clean",
        expected_rows=4, expected_calls=4, source_content_sha256=content_sha256,
    )
    clean.load_clean_stage4_59d(root, manifest_path, meta_path, raw,
                                 expected_rows=4, expected_calls=4)
    annotation = raw / "call_0.json"
    original_stat = annotation.stat()
    annotation.write_text("[]", encoding="utf-8")
    os.utime(annotation, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="raw content proof"):
        clean.load_clean_stage4_59d(root, manifest_path, meta_path, raw,
                                     expected_rows=4, expected_calls=4)


def test_clean_mfcc40_matches_baseline_frontend_and_order(tmp_path):
    manifest, _, _, raw = _small_clean_source(tmp_path)
    source = clean._source_rows(manifest, expected_rows=4, expected_calls=4)
    bundle = clean._extract_mfcc40(source.iloc[:1], raw)
    audio = np.zeros(8_000, dtype=np.float32)
    reference = __import__("librosa").feature.mfcc(
        y=audio, sr=8_000, n_mfcc=20, n_fft=512, hop_length=160,
    )
    expected = np.concatenate((reference.mean(axis=1), reference.std(axis=1))).astype(np.float32)
    np.testing.assert_array_equal(bundle["X"][0], expected)
    assert bundle["X"].dtype == np.float32
    assert bundle["y"].tolist() == [0]
    assert bundle["stem"].tolist() == ["call_0"]


@pytest.fixture
def transferred_clean(tmp_path, monkeypatch):
    import shutil
    from dcc_data import mission2_egemaps_colab as colab

    manifest, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    root = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "clean",
        expected_rows=4, expected_calls=4,
    )
    chain = json.loads((root / "chain.json").read_text())
    monkeypatch.setattr(colab, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(colab, "EXPECTED_FINGERPRINT", chain["provenance_fingerprint"])
    identity_path = tmp_path / "transfer.json"
    colab.write_local_transfer_identity(manifest, manifest_path, meta_path, raw,
                                        root / "chain.json", identity_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    for path in raw.iterdir():
        target = staged / path.name
        shutil.copyfile(path, target)
        info = path.stat()
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))
    return root, manifest_path, meta_path, staged, identity_path


def test_shared_clean_load_requires_bytes_and_preserves_frozen_cache(transferred_clean):
    root, manifest_path, meta_path, staged, identity_path = transferred_clean
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(ValueError, match="source_inventory_signature"):
        clean.load_clean_stage4_59d(root, manifest_path, meta_path, staged,
                                   expected_rows=4, expected_calls=4)
    loaded = clean.load_clean_stage4_59d(
        root, manifest_path, meta_path, staged, expected_rows=4, expected_calls=4,
        transfer_identity_path=identity_path,
    )
    assert loaded.X.shape == (4, 59)
    assert loaded.X.dtype == np.float32
    assert loaded.raw_source_verification["mode"] == "shared_training_transfer_sha256"
    assert loaded.raw_source_verification["build_time_content_proof"] == "not_collected"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("damage", ["raw", "proof", "manifest", "artifact", "metadata", "missing_proof"])
def test_shared_clean_rejects_tampering(transferred_clean, damage):
    root, manifest_path, meta_path, staged, identity_path = transferred_clean
    if damage == "raw":
        (staged / "call_0.json").write_text("[]")  # Same size, different bytes.
    elif damage == "proof":
        proof = json.loads(identity_path.read_text())
        proof["clean_reference_fingerprint"] = "wrong"
        identity_path.write_text(json.dumps(proof))
    elif damage == "manifest":
        manifest = pd.read_parquet(manifest_path)
        manifest.loc[0, "speaker"] = 1
        manifest.to_parquet(manifest_path)
    elif damage == "artifact":
        with (root / "mfcc40.npz").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "metadata":
        path = root / "af10.json"
        metadata = json.loads(path.read_text())
        metadata["feature_names"] = metadata["feature_names"][::-1]
        path.write_text(json.dumps(metadata))
    else:
        identity_path = identity_path.with_name("missing.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        clean.load_clean_stage4_59d(
            root, manifest_path, meta_path, staged, expected_rows=4, expected_calls=4,
            transfer_identity_path=identity_path,
        )


def test_shared_clean_keeps_builder_versions_separate_from_reader(transferred_clean):
    root, manifest_path, meta_path, staged, identity_path = transferred_clean
    versions = {"librosa": "builder-librosa", "numpy": "builder-numpy", "pandas": "builder-pandas"}
    for name in clean.ARTIFACT_SUFFIX:
        path = root / f"{name}.json"
        metadata = json.loads(path.read_text())
        metadata["dependency_versions"] = versions
        path.write_text(json.dumps(metadata))
    loaded = clean.load_clean_stage4_59d(
        root, manifest_path, meta_path, staged, expected_rows=4, expected_calls=4,
        transfer_identity_path=identity_path,
    )
    assert loaded.raw_source_verification["cache_dependency_versions"] == versions
    metadata_path = root / "af11.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dependency_versions"]["numpy"] = "inconsistent"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="artifact metadata"):
        clean.load_clean_stage4_59d(
            root, manifest_path, meta_path, staged, expected_rows=4, expected_calls=4,
            transfer_identity_path=identity_path,
        )


def test_clean_module_import_does_not_extract(monkeypatch):
    from dcc_data import audio
    def forbidden(*_args, **_kwargs):
        raise AssertionError("import started extraction")
    with monkeypatch.context() as patch:
        patch.setattr(audio, "read_wav_frames", forbidden)
        importlib.reload(clean)
    importlib.reload(clean)


def test_historical_value_audit_keeps_exact_and_close_separate():
    old = np.asarray([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    exact = standard._feature_value_differences(old, old, ("a", "b"), chunk_rows=1)
    assert exact["exact_feature_matrix_equality"]
    assert exact["numerical_close"]
    assert exact["differing_row_count"] == exact["differing_value_count"] == 0

    changed = old.copy()
    changed[0, 0] = np.float32(1e-9)
    close = standard._feature_value_differences(old, changed, ("a", "b"), chunk_rows=1)
    assert not close["exact_feature_matrix_equality"]
    assert close["numerical_close"]
    assert close["differing_row_count"] == 1
    assert close["differing_value_count"] == 1
    assert close["top_changed_features"][0]["feature"] == "a"
    assert close["max_abs_diff"] > 0
    two_values = old.copy()
    two_values[0] = (1.0, 3.0)
    counts = standard._feature_value_differences(old, two_values, ("a", "b"))
    assert counts["differing_row_count"] == 1
    assert counts["differing_value_count"] == 2
    assert counts["mean_abs_diff"] == 0.5
    assert counts["max_abs_diff"] == 1.0


def test_historical_row_and_assembly_order_mismatch():
    keys = np.asarray(["Training|a|0", "Training|b|0"])
    calls = np.asarray(["a", "b"])
    labels = np.asarray([0, 1])
    aligned = standard._historical_row_alignment(
        keys[::-1], keys, calls, calls, labels, labels,
    )
    assert aligned["record_key_exact"]
    assert not aligned["record_key_order_equal"]
    assert aligned["call_id_exact"] and aligned["label_exact"]
    wrong_key = standard._historical_row_alignment(
        np.asarray(["Training|a|0", "Training|c|0"]), keys,
        calls, calls, labels, labels,
    )
    assert not wrong_key["record_key_exact"]
    stats = standard._feature_value_differences(
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32), ("a", "b"),
    )
    assembly = standard._assembled_feature_differences(
        [("block", stats)], rows=2,
        feature_names=("b", "a"), clean_feature_names=("a", "b"),
    )
    assert assembly["status"] == "INCOMPLETE_HISTORICAL_EVIDENCE"
    assert not assembly["feature_order_equal"]


def _small_historical_audit_chain(tmp_path, monkeypatch):
    manifest, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    chain = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "data/processed/mission2_stage4_clean",
        expected_rows=4, expected_calls=4,
    )
    paths = models.standalone_training_cache_paths(tmp_path)
    for path in (paths.manifest_parquet, paths.manifest_metadata,
                 paths.mfcc_npz, paths.mfcc_metadata,
                 paths.candidate_parquet, paths.candidate_metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, paths.manifest_parquet)
    shutil.copy2(meta_path, paths.manifest_metadata)
    raw_copy = tmp_path / "data/raw_original"
    raw_copy.mkdir(parents=True)
    for source in raw.iterdir():
        shutil.copy2(source, raw_copy / source.name)
    shutil.copy2(chain / "mfcc40.npz", paths.mfcc_npz)
    paths.mfcc_metadata.write_text("{}", encoding="utf-8")
    loaded = clean.load_clean_stage4_59d(
        chain, manifest_path, meta_path, raw, expected_rows=4, expected_calls=4,
    )
    keys, calls = loaded.record_keys, loaded.call_ids
    values_by_name = dict(zip(loaded.feature_names, loaded.X.T))

    def save_frame(path, metadata_path, feature_names):
        frame = pd.DataFrame({"record_key": keys, "call_id": calls})
        for name in feature_names:
            frame[name] = values_by_name.get(name, np.zeros(4, dtype=np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        metadata_path.write_text(json.dumps({"feature_names": list(feature_names)}),
                                 encoding="utf-8")

    save_frame(
        paths.candidate_parquet, paths.candidate_metadata,
        ("zcr", "spectral_centroid", "spectral_rolloff", "spectral_flatness",
         "silence_ratio"),
    )
    old_acoustic_root = tmp_path / "data/processed/mission2_acoustic_feature_expansion"
    for block_ids in (acoustic.ROUND1_BLOCK_IDS, acoustic.ROUND2_BLOCK_IDS,
                      acoustic.NEXT_CANDIDATE_BLOCK_IDS,
                      acoustic.RESIDUAL_TARGET_BLOCK_IDS):
        path, metadata_path = acoustic.candidate_cache_paths(
            old_acoustic_root, split="Training", block_ids=block_ids,
        )
        names = tuple(name for block_id in block_ids
                      for name in acoustic.BLOCK_BY_ID[block_id].feature_names)
        save_frame(path, metadata_path, names)
    return chain, paths, loaded.provenance_fingerprint


def test_historical_bounded_audit_is_read_only_and_no_fit(tmp_path, monkeypatch):
    chain, paths, fingerprint = _small_historical_audit_chain(tmp_path, monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bounded audit started extraction or model fit")

    monkeypatch.setattr(acoustic, "_run_prepared_representation", forbidden)
    monkeypatch.setattr(clean, "build_clean_stage4_chain", forbidden)
    historical_before = paths.candidate_parquet.read_bytes()
    clean_before = (chain / "chain.json").read_bytes()
    result = standard.audit_historical_vs_clean_feature_caches(
        tmp_path, chain, expected_fingerprint=fingerprint,
        expected_rows=4, expected_calls=4,
    )
    assert result["conclusion"]["status"] == "FEATURE_CACHE_DIFFERENCE_EXCLUDED"
    assert result["assembled_52d"]["exact_feature_matrix_equality"]
    assert result["assembled_59d"]["exact_feature_matrix_equality"]
    assert paths.candidate_parquet.read_bytes() == historical_before
    assert (chain / "chain.json").read_bytes() == clean_before
    candidate = result["block_audit"].set_index("block").loc["E05 candidate"]
    assert candidate["historical_stored_feature_dimension"] == 5
    assert candidate["clean_stored_feature_dimension"] == 2
    assert candidate["historical_feature_dimension"] == 2
    assert candidate["model_feature_order_equal"]
    assert not candidate["feature_names_order_equal"]
    assert not candidate["historical_storage_order_equal"]
    frame = pd.read_parquet(paths.candidate_parquet)
    frame.loc[:, "record_key"] = frame["record_key"].iloc[::-1].to_numpy()
    frame.to_parquet(paths.candidate_parquet, index=False)
    incomplete = standard.audit_historical_vs_clean_feature_caches(
        tmp_path, chain, expected_fingerprint=fingerprint,
        expected_rows=4, expected_calls=4,
    )
    assert incomplete["conclusion"]["status"] == "INCOMPLETE_HISTORICAL_EVIDENCE"
    assert incomplete["assembled_52d"]["status"] == "INCOMPLETE_HISTORICAL_EVIDENCE"


def test_historical_audit_module_import_has_no_extraction_or_fit(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("import started extraction or fit")

    with monkeypatch.context() as patch:
        patch.setattr(clean, "build_clean_stage4_chain", forbidden)
        patch.setattr(acoustic, "_run_prepared_representation", forbidden)
        importlib.reload(standard)
    importlib.reload(standard)


def test_b2_reuses_completed_chain_without_builder_or_extractor(tmp_path, monkeypatch):
    _, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    output = tmp_path / "clean"
    original = clean.build_clean_stage4_chain(
        manifest_path, meta_path, raw, output, expected_rows=4, expected_calls=4,
    )
    before = (original / "chain.json").read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reuse started builder or extraction")

    monkeypatch.setattr(clean, "build_clean_stage4_chain", forbidden)
    monkeypatch.setattr(clean, "_extract_mfcc40", forbidden)
    monkeypatch.setattr(clean.candidate, "extract_candidate_feature_frame", forbidden)
    monkeypatch.setattr(clean.acoustic, "extract_candidate_feature_frame", forbidden)
    loaded, status = clean.load_or_build_clean_stage4_chain(
        manifest_path, meta_path, raw, output, expected_rows=4, expected_calls=4,
    )
    assert status == "REUSING_EXISTING_CLEAN_CHAIN"
    assert loaded.chain_root == original
    assert loaded.X.shape == (4, 59)
    assert loaded.feature_names == clean.clean_59d_feature_names()
    assert (original / "chain.json").read_bytes() == before


def test_b2_builds_only_when_no_chain_exists(tmp_path, monkeypatch):
    _, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    _fake_clean_extractors(monkeypatch)
    loaded, status = clean.load_or_build_clean_stage4_chain(
        manifest_path, meta_path, raw, tmp_path / "clean",
        expected_rows=4, expected_calls=4,
    )
    assert status == "BUILT_NEW_CLEAN_CHAIN"
    assert loaded.X.shape == (4, 59)


def test_b2_incomplete_or_mismatched_chain_fails_without_rebuild(tmp_path, monkeypatch):
    _, manifest_path, meta_path, raw = _small_clean_source(tmp_path)
    output = tmp_path / "clean"
    incomplete = output / "stage4_clean_v1_incomplete"
    incomplete.mkdir(parents=True)
    original_builder = clean.build_clean_stage4_chain

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid chain triggered rebuild")

    monkeypatch.setattr(clean, "build_clean_stage4_chain", forbidden)
    with pytest.raises(FileNotFoundError):
        clean.load_or_build_clean_stage4_chain(
            manifest_path, meta_path, raw, output, expected_rows=4, expected_calls=4,
        )
    assert incomplete.is_dir()
    incomplete.rename(output / "unrelated")
    _fake_clean_extractors(monkeypatch)
    # Build a valid fixture with the original builder, then corrupt provenance.
    valid = original_builder(
        manifest_path, meta_path, raw, output, expected_rows=4, expected_calls=4,
    )
    chain_meta_path = valid / "chain.json"
    chain_meta = json.loads(chain_meta_path.read_text(encoding="utf-8"))
    chain_meta["provenance_fingerprint"] = "invalid"
    chain_meta_path.write_text(json.dumps(chain_meta), encoding="utf-8")
    before = chain_meta_path.read_bytes()
    monkeypatch.setattr(clean, "build_clean_stage4_chain", forbidden)
    with pytest.raises(ValueError, match="fingerprint"):
        clean.load_or_build_clean_stage4_chain(
            manifest_path, meta_path, raw, output, expected_rows=4, expected_calls=4,
        )
    assert chain_meta_path.read_bytes() == before


def test_b6_three_e05_rows_have_fixed_split_membership():
    changed = pd.DataFrame({"row_index": [0, 2, 3], "feature_name": ["zcr"] * 3})
    rows = standard.annotate_e05_fixed_split(
        changed, np.asarray([0, 1]), np.asarray([2, 3]), total_rows=4,
    )
    assert rows["fixed_split_membership"].tolist() == [
        "internal-train", "internal-validation", "internal-validation",
    ]
    with pytest.raises(ValueError, match="partition"):
        standard.annotate_e05_fixed_split(
            changed, np.asarray([0, 1]), np.asarray([1, 3]), total_rows=4,
        )


def test_b6_historical_assembly_is_read_only_and_never_extracts(tmp_path, monkeypatch):
    chain, paths, fingerprint = _small_historical_audit_chain(tmp_path, monkeypatch)
    frame = pd.read_parquet(paths.candidate_parquet)
    frame.loc[[0, 1, 2], "zcr"] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    frame.to_parquet(paths.candidate_parquet, index=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("historical assembly started extraction or fit")

    monkeypatch.setattr(clean, "_extract_mfcc40", forbidden)
    monkeypatch.setattr(clean.candidate, "extract_candidate_feature_frame", forbidden)
    monkeypatch.setattr(clean.acoustic, "extract_candidate_feature_frame", forbidden)
    monkeypatch.setattr(acoustic, "_run_prepared_representation", forbidden)
    before = paths.candidate_parquet.read_bytes()
    audit = standard.audit_historical_vs_clean_feature_caches(
        tmp_path, chain, expected_fingerprint=fingerprint,
        expected_rows=4, expected_calls=4, include_reproduction_inputs=True,
    )
    inputs = audit["reproduction_inputs"]
    assert inputs["X_52d"].shape == (4, 52)
    assert inputs["X_59d"].shape == (4, 59)
    assert len(inputs["e05_differences"]) == 3
    assert np.array_equal(inputs["X_52d"], inputs["X_59d"][:, :52])
    assert paths.candidate_parquet.read_bytes() == before


def test_b6_reproduction_enforces_frozen_split_and_config_before_two_fits(monkeypatch):
    prepared, _, labels, keys, calls = _small_comparison()
    matrix = np.arange(10 * 59, dtype=np.float32).reshape(10, 59) / 1000
    seen = []

    def fake_fit(item, *, experiment_id, allow_fit):
        assert allow_fit is True
        assert item.split_audit["seed"] == 42
        assert item.split_audit["call_overlap"] == 0
        assert int(item.scaler.n_samples_seen_) == len(item.train_idx)
        assert np.array_equal(item.validation_idx, prepared.validation_idx)
        seen.append(experiment_id)
        return SimpleNamespace(
            model_config=dict(acoustic.M04_ITER100_FEATURE_SCREENING_CONFIG),
            fit_audit={"scaler_fit_scope": "internal_train_only", "seed": 42,
                       "train_idx": item.train_idx, "validation_idx": item.validation_idx},
            y_true=item.y_evaluation, feature_names=item.feature_names,
        )

    monkeypatch.setattr(acoustic, "_run_prepared_representation", fake_fit)
    inputs = {"X_52d": matrix[:, :52], "X_59d": matrix, "y": labels,
              "record_keys": keys, "call_ids": calls,
              "feature_names": prepared.feature_names,
              "e05_differences": pd.DataFrame({"row_index": [0]}),
              "clean_provenance_fingerprint": "synthetic"}
    result = standard.reproduce_historical_stage4_inputs(inputs)
    assert len(result["results"]) == 2
    assert seen == ["AF09_52D_HISTORICAL_CACHE", "AF10_AF11_59D_HISTORICAL_CACHE"]
    seen.clear()
    monkeypatch.setattr(acoustic, "M04_ITER100_FEATURE_SCREENING_CONFIG", {"max_iter": 99})
    with pytest.raises(ValueError, match="M04-100 config"):
        standard.reproduce_historical_stage4_inputs(inputs)
    assert seen == []
