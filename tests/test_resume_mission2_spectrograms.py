"""Resume a crashed Mission 2 spectrogram build without recomputing finished calls.

The synthetic calls reuse the same WAV/JSON writer as tests/test_mission2_spectrogram.py.
Nothing here reads competition audio.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile

from dcc_data.audio import read_wav_info
from dcc_data.guards import SplitContaminationError
from dcc_data.manifest import ManifestRow, write_manifest
from dcc_data.mission2_features import (
    HOP_LENGTH,
    N_FFT,
    build_utterance_records,
    read_call_waveform,
    sample_calls,
    select_seoul_training_rows,
    select_seoul_validation_rows,
)
from dcc_data.mission2_spectrogram import (
    LABEL_SOURCE_PROVIDED,
    N_MELS,
    assert_training_spectrogram_index,
)

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_mission2_spectrograms as build_script  # noqa: E402
import pack_mission2_spectrograms as pack_script  # noqa: E402
import resume_mission2_spectrograms as resume  # noqa: E402

BUILDER_COLUMNS = [
    "call_id",
    "utt_idx",
    "speaker",
    "duration_sec",
    "n_frames",
    "npy_path",
    "split",
    "n_mels",
    "hop_length",
    "n_fft",
    "sample_rate",
    "label_source",
]


def _spectrogram_helpers():
    path = Path(__file__).with_name("test_mission2_spectrogram.py")
    spec = importlib.util.spec_from_file_location("mission2_spectrogram_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_HELPERS = _spectrogram_helpers()
_write_call = _HELPERS._write_call


def _run(module, argv: list[str]) -> None:
    old_argv = sys.argv
    sys.argv = [f"{module.__name__}.py", *argv]
    try:
        assert module.main() == 0
    finally:
        sys.argv = old_argv


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _by_call(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["call_id", "utt_idx"], kind="mergesort").reset_index(drop=True)


def _npy_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*.npy"))}


def _seoul_training(stem: str) -> tuple[str, str]:
    return (
        f"chunk/Training/1.원천데이터/TS_서울_구급/{stem}.wav",
        f"chunk/Training/2.라벨링데이터/TL_서울_구급/{stem}.json",
    )


def _write_training_call(root: Path, stem: str, utterances, *, seed: int) -> ManifestRow:
    wav_relative, json_relative = _seoul_training(stem)
    _write_call(root, wav_relative, json_relative, utterances, seed=seed)
    return ManifestRow("Training", stem, wav_relative, json_relative)


def _cache_args(
    root: Path,
    manifest: Path,
    output_dir: Path,
    index_output: Path,
    failures_log: Path,
    *,
    split: str = "training",
    n_calls: int = -1,
    seed: int = 42,
    workers: int = 1,
    hop_length: int | None = None,
    n_mels: int | None = None,
    expected_utterances: int | None = None,
    progress_log: Path | None = None,
    progress_every: int | None = None,
    partial_index_every: int | None = None,
) -> list[str]:
    args = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--split",
        split,
        "--output-dir",
        str(output_dir),
        "--n-calls",
        str(n_calls),
        "--seed",
        str(seed),
        "--index-output",
        str(index_output),
        "--failures-log",
        str(failures_log),
        "--workers",
        str(workers),
    ]
    if hop_length is not None:
        args.extend(["--hop-length", str(hop_length)])
    if n_mels is not None:
        args.extend(["--n-mels", str(n_mels)])
    if expected_utterances is not None:
        args.extend(["--expected-utterances", str(expected_utterances)])
    if progress_log is not None:
        args.extend(["--progress-log", str(progress_log)])
    if progress_every is not None:
        args.extend(["--progress-every", str(progress_every)])
    if partial_index_every is not None:
        args.extend(["--partial-index-every", str(partial_index_every)])
    return args


def _utterance_total(rows: list[ManifestRow], root: Path, split: str, n_calls: int, seed: int) -> int:
    sampled = resume.select_and_sample(rows, split, n_calls, seed)
    return sum(len(build_utterance_records(row, root)) for row in sampled)


def test_read_call_waveform_does_not_resample_and_matches_the_header(tmp_path: Path) -> None:
    # 16 kHz, not the dataset's 8 kHz. A resampler aimed at 8 kHz would halve the length.
    path = tmp_path / "odd_rate.wav"
    n_frames = 8000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x01" * n_frames)

    header_rate = read_wav_info(path).sample_rate
    waveform, sample_rate = read_call_waveform(path)
    soundfile_rate = soundfile.info(path).samplerate

    assert header_rate == 16000
    assert sample_rate == header_rate
    assert soundfile_rate == header_rate
    assert len(waveform) == n_frames


def test_n_calls_minus_one_keeps_every_eligible_call_in_manifest_order() -> None:
    """``sample_calls`` treats n_calls <= 0, including -1, as the full filtered list.

    The crashed build passed ``--n-calls -1`` on a 32,840-row manifest. That is every
    Seoul Training call in manifest order, not a draw and not the unfiltered manifest.
    """

    def training(stem: str) -> ManifestRow:
        wav, label = _seoul_training(stem)
        return ManifestRow("Training", stem, wav, label)

    mixed = [
        training("z_call"),
        ManifestRow(
            "Training",
            "busan",
            "chunk/Training/1.원천데이터/TS_부산_구급/busan.wav",
            "chunk/Training/2.라벨링데이터/TL_부산_구급/busan.json",
        ),
        training("a_call"),
        ManifestRow(
            "Validation",
            "val_call",
            "chunk/Validation/1.원천데이터/VS_서울_구급/val_call.wav",
            "chunk/Validation/2.라벨링데이터/VL_서울_구급/val_call.json",
        ),
        training("m_call"),
    ]
    eligible = select_seoul_training_rows(mixed)
    assert [row.stem for row in eligible] == ["z_call", "a_call", "m_call"]
    assert len(eligible) < len(mixed)

    # -1 does not consult the seed and does not shuffle.
    assert sample_calls(eligible, -1, 1) == list(eligible)
    assert sample_calls(eligible, -1, 999) == sample_calls(eligible, 0, 1)
    assert resume.select_and_sample(mixed, "training", -1, 42) == list(eligible)
    assert resume.select_and_sample(mixed, "training", -1, 42) == sample_calls(
        select_seoul_training_rows(mixed), -1, 42
    )
    assert [row.stem for row in resume.select_and_sample(mixed, "validation", -1, 7)] == ["val_call"]


def test_sampling_is_deterministic_and_uses_the_builder_selectors() -> None:
    def training(stem: str) -> ManifestRow:
        wav, label = _seoul_training(stem)
        return ManifestRow("Training", stem, wav, label)

    rows = [
        training("z_call"),
        ManifestRow(
            "Training",
            "busan",
            "chunk/Training/1.원천데이터/TS_부산_구급/busan.wav",
            "chunk/Training/2.라벨링데이터/TL_부산_구급/busan.json",
        ),
        training("a_call"),
        ManifestRow(
            "Validation",
            "val_call",
            "chunk/Validation/1.원천데이터/VS_서울_구급/val_call.wav",
            "chunk/Validation/2.라벨링데이터/VL_서울_구급/val_call.json",
        ),
        training("m_call"),
        training("b_call"),
    ]
    seoul = select_seoul_training_rows(rows)
    assert [row.stem for row in seoul] == ["z_call", "a_call", "m_call", "b_call"]

    # n_calls <= 0 (the Colab build passes -1) keeps manifest order and ignores the seed.
    assert resume.select_and_sample(rows, "training", -1, 1) == seoul
    assert resume.select_and_sample(rows, "training", 0, 99) == seoul
    assert resume.select_and_sample(rows, "training", len(seoul), 5) == list(seoul)
    assert resume.select_and_sample(rows, "training", -1, 1) == sample_calls(seoul, -1, 1)

    subset_a = resume.select_and_sample(rows, "training", 3, 42)
    subset_b = resume.select_and_sample(rows, "training", 3, 42)
    assert subset_a == subset_b
    assert subset_a == sample_calls(seoul, 3, 42)
    assert len(subset_a) == 3
    assert {row.stem for row in subset_a}.isdisjoint({"busan", "val_call"})

    validation = resume.select_and_sample(rows, "validation", -1, 42)
    assert validation == select_seoul_validation_rows(rows)
    assert [row.stem for row in validation] == ["val_call"]


def test_resume_cli_defaults_match_the_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_mission2_spectrograms.py", "--manifest", "m.csv", "--data-root", "raw"])
    training_build = build_script.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_mission2_spectrograms.py", "--manifest", "m.csv", "--data-root", "raw", "--split", "validation"],
    )
    validation_build = build_script.parse_args()

    training = resume.parse_args(["--manifest", "m.csv", "--data-root", "raw"])
    validation = resume.parse_args(["--manifest", "m.csv", "--data-root", "raw", "--split", "validation"])

    for name in ("output_dir", "index_output", "failures_log", "n_calls", "seed", "workers", "hop_length", "n_mels"):
        assert getattr(training, name) == getattr(training_build, name)
        assert getattr(validation, name) == getattr(validation_build, name)
    assert training.expected_utterances is None
    assert validation.expected_utterances is None
    assert training.progress_log is None
    assert training.progress_every == 500
    assert training.partial_index_every is None
    assert training.split == "training"
    assert validation.split == "validation"
    assert resume.parse_args(["--manifest", "m.csv", "--data-root", "raw", "--n-calls", "-1"]).n_calls == -1
    with pytest.raises(SystemExit):
        resume.parse_args(["--manifest", "m.csv", "--data-root", "raw", "--progress-every", "0"])
    with pytest.raises(SystemExit):
        resume.parse_args(["--manifest", "m.csv", "--data-root", "raw", "--partial-index-every", "0"])


def test_npy_reuse_predicate_rejects_bad_files(tmp_path: Path) -> None:
    path = tmp_path / "0.npy"
    np.save(path, np.zeros((N_MELS, 5), dtype=np.float32))
    assert resume.npy_n_frames_if_reusable(path, N_MELS) == 5

    np.save(path, np.zeros((N_MELS, 5), dtype=np.float64))
    assert resume.npy_n_frames_if_reusable(path, N_MELS) is None

    np.save(path, np.zeros((N_MELS, 0), dtype=np.float32))
    assert resume.npy_n_frames_if_reusable(path, N_MELS) is None

    np.save(path, np.zeros((N_MELS + 1, 4), dtype=np.float32))
    assert resume.npy_n_frames_if_reusable(path, N_MELS) is None

    np.save(path, np.zeros((N_MELS,), dtype=np.float32))
    assert resume.npy_n_frames_if_reusable(path, N_MELS) is None

    path.write_bytes(path.read_bytes()[:12])
    assert resume.npy_n_frames_if_reusable(path, N_MELS) is None
    assert resume.npy_n_frames_if_reusable(tmp_path / "missing.npy", N_MELS) is None


def test_expected_utterances_is_checked_before_any_spectrogram_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "raw"
    row = _write_training_call(root, "only", [(0, 500, 0), (600, 1500, 1)], seed=0)
    manifest = tmp_path / "manifest.csv"
    write_manifest([row], manifest, overwrite=True)
    output_dir = tmp_path / "specs"
    index_output = tmp_path / "index.parquet"

    def refuse(*_args, **_kwargs):
        raise AssertionError("spectrogram work started before the utterance check")

    monkeypatch.setattr(resume, "_resume_one_call", refuse)
    monkeypatch.setattr(resume, "read_wav_info", refuse)
    monkeypatch.setattr(resume, "npy_n_frames_if_reusable", refuse)

    real_total = _utterance_total([row], root, "training", -1, 42)
    argv = _cache_args(
        root,
        manifest,
        output_dir,
        index_output,
        tmp_path / "failures.jsonl",
        expected_utterances=real_total + 1,
    )
    old_argv = sys.argv
    sys.argv = ["resume_mission2_spectrograms.py", *argv]
    try:
        with pytest.raises(AssertionError, match="expected-utterances"):
            resume.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr().out
    assert "Sampled calls: 1" in captured
    assert f"Utterances in sample: {real_total}" in captured
    assert not index_output.exists()
    assert list(output_dir.rglob("*.npy")) == []


def test_partial_cache_resume_matches_full_build_index_npy_and_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "raw"
    # Manifest order is not call_id order. Resume must follow sample_calls (here, this order),
    # while pack re-sorts by (call_id, utt_idx) before assigning frame_offset.
    specs = [
        ("call_keep_b", [(0, 500, 1), (600, 2200, 0)], 1),
        ("call_truncated", [(0, 90, 0), (200, 1600, 1), (1700, 3200, 0)], 2),
        ("call_keep_a", [(0, 400, 0), (500, 1800, 1), (1900, 3000, 0)], 3),
        ("call_deleted", [(0, 800, 0), (900, 2100, 1)], 4),
        ("call_missing", [(100, 700, 1), (800, 2400, 0)], 5),
        ("call_empty", [(0, 650, 0), (700, 2000, 1)], 6),
        ("call_badshape", [(0, 550, 1), (600, 1900, 0)], 7),
        ("call_keep_c", [(0, 300, 0), (400, 1200, 1)], 8),
    ]
    rows = [_write_training_call(root, stem, utterances, seed=seed) for stem, utterances, seed in specs]
    rows.append(
        ManifestRow(
            "Validation",
            "call_val",
            "chunk/Validation/1.원천데이터/VS_서울_구급/call_val.wav",
            "chunk/Validation/2.라벨링데이터/VL_서울_구급/call_val.json",
        )
    )
    _write_call(
        root,
        rows[-1].wav_path,
        rows[-1].json_path,
        [(0, 500, 0), (600, 1500, 1)],
        seed=9,
    )
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)
    expected_utterances = _utterance_total(rows, root, "training", -1, 42)
    assert expected_utterances > 0

    reference_dir = tmp_path / "reference"
    reference_index = tmp_path / "reference_index.parquet"
    reference_failures = tmp_path / "reference_failures.jsonl"
    _run(
        build_script,
        _cache_args(root, manifest, reference_dir, reference_index, reference_failures, workers=1),
    )
    reference_df = pd.read_parquet(reference_index)
    assert list(reference_df.columns) == BUILDER_COLUMNS
    assert "frame_offset" not in reference_df.columns
    assert set(reference_df["call_id"]) == {stem for stem, _, _ in specs}

    sampled = resume.select_and_sample(rows, "training", -1, 42)
    reference_rows = reference_df.to_dict(orient="records")
    for stem in ("call_keep_a", "call_keep_b", "call_keep_c"):
        row = next(item for item in sampled if item.stem == stem)
        reused = resume.reuse_index_rows(
            row, build_utterance_records(row, root), root, reference_dir, HOP_LENGTH, N_MELS
        )
        expected = [item for item in reference_rows if _plain(item["call_id"]) == stem]
        assert reused is not None and len(reused) == len(expected) >= 2
        for got, exp in zip(reused, expected, strict=True):
            assert list(got.keys()) == BUILDER_COLUMNS
            for column in BUILDER_COLUMNS:
                assert _plain(got[column]) == _plain(exp[column])
            assert got["split"] == "Training"
            assert got["n_mels"] == N_MELS
            assert got["hop_length"] == HOP_LENGTH
            assert got["n_fft"] == N_FFT
            assert got["sample_rate"] == 8000
            assert got["label_source"] == LABEL_SOURCE_PROVIDED
            assert got["n_frames"] == int(np.load(reference_dir / got["npy_path"]).shape[1])

    resume_dir = tmp_path / "resume"
    shutil.copytree(reference_dir, resume_dir)
    resume_df_preview = reference_df

    def call_paths(stem: str) -> list[Path]:
        relatives = resume_df_preview.loc[resume_df_preview["call_id"] == stem, "npy_path"]
        return [resume_dir / relative for relative in relatives]

    keep_mtimes = {
        path: path.stat().st_mtime_ns
        for stem in ("call_keep_a", "call_keep_b", "call_keep_c")
        for path in call_paths(stem)
    }
    shutil.rmtree(call_paths("call_deleted")[0].parent)
    truncated = call_paths("call_truncated")[1]
    truncated.write_bytes(truncated.read_bytes()[:12])
    call_paths("call_missing")[0].unlink()
    np.save(call_paths("call_empty")[1], np.zeros((N_MELS, 0), dtype=np.float32))
    np.save(call_paths("call_badshape")[0], np.zeros((N_MELS + 1, 4), dtype=np.float32))

    resume_index = tmp_path / "resume_index.parquet"
    resume_failures = tmp_path / "resume_failures.jsonl"
    _run(
        resume,
        _cache_args(
            root,
            manifest,
            resume_dir,
            resume_index,
            resume_failures,
            workers=2,
            expected_utterances=expected_utterances,
        ),
    )
    captured = capsys.readouterr().out
    assert f"Sampled calls: {len(specs)}" in captured
    assert "Calls reused: 3" in captured
    assert "Calls recomputed: 5" in captured

    resume_df = pd.read_parquet(resume_index)
    pd.testing.assert_frame_equal(reference_df, resume_df, check_exact=True)
    pd.testing.assert_frame_equal(_by_call(reference_df), _by_call(resume_df), check_exact=True)
    assert _npy_bytes(reference_dir) == _npy_bytes(resume_dir)
    for path, mtime_ns in keep_mtimes.items():
        assert path.stat().st_mtime_ns == mtime_ns
    assert reference_failures.read_text(encoding="utf-8") == resume_failures.read_text(encoding="utf-8") == ""
    provenance = assert_training_spectrogram_index(resume_df)
    assert provenance["label_source"] == LABEL_SOURCE_PROVIDED
    assert provenance["n_mels"] == N_MELS
    assert provenance["hop_length"] == HOP_LENGTH
    assert provenance["sample_rate"] == 8000

    # Pack sorts by (call_id, utt_idx) itself. Shuffling the resume index must not change
    # frame_offset or the packed array; the resume script still emits sample_calls order.
    seen: list[str] = []
    for call_id in resume_df["call_id"]:
        call_id = _plain(call_id)
        if not seen or seen[-1] != call_id:
            seen.append(call_id)
    assert seen == [stem for stem, _, _ in specs]
    for _call_id, group in resume_df.groupby("call_id", sort=False):
        assert list(group["utt_idx"]) == sorted(int(value) for value in group["utt_idx"])

    full_packed = tmp_path / "full_packed.npy"
    full_packed_index = tmp_path / "full_packed_index.parquet"
    resume_packed = tmp_path / "resume_packed.npy"
    resume_packed_index = tmp_path / "resume_packed_index.parquet"
    shuffled_packed = tmp_path / "shuffled_packed.npy"
    full_packed_df = pack_script.pack_spectrograms(reference_df, reference_dir, full_packed)
    resume_packed_df = pack_script.pack_spectrograms(resume_df, resume_dir, resume_packed)
    shuffled_df = pack_script.pack_spectrograms(
        resume_df.sample(frac=1, random_state=0), resume_dir, shuffled_packed
    )
    pd.testing.assert_frame_equal(full_packed_df, resume_packed_df, check_exact=True)
    pd.testing.assert_frame_equal(full_packed_df, shuffled_df, check_exact=True)
    full_packed_df.to_parquet(full_packed_index, index=False)
    resume_packed_df.to_parquet(resume_packed_index, index=False)
    full_packed_df = pd.read_parquet(full_packed_index)
    resume_packed_df = pd.read_parquet(resume_packed_index)
    pd.testing.assert_frame_equal(full_packed_df, resume_packed_df, check_exact=True)
    assert list(full_packed_df.columns) == BUILDER_COLUMNS + ["frame_offset"]
    packed_keys = list(zip(full_packed_df["call_id"], full_packed_df["utt_idx"]))
    assert packed_keys == sorted(packed_keys)
    n_frames = full_packed_df["n_frames"].to_numpy(dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_frames)[:-1]])
    np.testing.assert_array_equal(full_packed_df["frame_offset"].to_numpy(), offsets)
    assert full_packed.read_bytes() == resume_packed.read_bytes() == shuffled_packed.read_bytes()
    assert_training_spectrogram_index(full_packed_df)

    summary = json.loads(resume.resume_summary_path(resume_index).read_text(encoding="utf-8"))
    assert summary["calls_reused"] == 3
    assert summary["calls_recomputed"] == 5
    assert summary["utterances_indexed"] == len(resume_df)
    assert summary["failures"] == 0
    assert summary["sampled_calls"] == len(specs)
    assert summary["utterances_in_sample"] == expected_utterances
    assert summary["args"]["workers"] == 2
    assert summary["args"]["expected_utterances"] == expected_utterances
    assert summary["args"]["n_calls"] == -1

    rerun_index = tmp_path / "rerun_index.parquet"
    _run(
        resume,
        _cache_args(
            root,
            manifest,
            resume_dir,
            rerun_index,
            tmp_path / "rerun_failures.jsonl",
            workers=2,
            expected_utterances=expected_utterances,
        ),
    )
    rerun_summary = json.loads(resume.resume_summary_path(rerun_index).read_text(encoding="utf-8"))
    assert rerun_summary["calls_reused"] == len(specs)
    assert rerun_summary["calls_recomputed"] == 0
    pd.testing.assert_frame_equal(pd.read_parquet(rerun_index), resume_df, check_exact=True)
    assert _npy_bytes(reference_dir) == _npy_bytes(resume_dir)


def test_complete_cache_resumes_with_zero_recomputes(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rows = [
        _write_training_call(root, "call_a", [(0, 200, 0), (300, 1400, 1)], seed=1),
        _write_training_call(root, "call_b", [(0, 90, 1), (200, 1800, 0), (1900, 2800, 1)], seed=2),
        _write_training_call(root, "call_c", [(0, 1000, 0)], seed=3),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)
    expected_utterances = _utterance_total(rows, root, "training", -1, 42)
    output_dir = tmp_path / "specs"
    build_index = tmp_path / "build.parquet"
    _run(
        build_script,
        _cache_args(root, manifest, output_dir, build_index, tmp_path / "build_failures.jsonl"),
    )
    mtimes = {path: path.stat().st_mtime_ns for path in output_dir.rglob("*.npy")}
    resume_index = tmp_path / "resume.parquet"
    _run(
        resume,
        _cache_args(
            root,
            manifest,
            output_dir,
            resume_index,
            tmp_path / "resume_failures.jsonl",
            workers=1,
            expected_utterances=expected_utterances,
        ),
    )
    summary = json.loads(resume.resume_summary_path(resume_index).read_text(encoding="utf-8"))
    assert summary["calls_reused"] == 3
    assert summary["calls_recomputed"] == 0
    assert summary["failures"] == 0
    assert summary["utterances_indexed"] == expected_utterances
    resume_df = pd.read_parquet(resume_index)
    pd.testing.assert_frame_equal(_by_call(pd.read_parquet(build_index)), _by_call(resume_df), check_exact=True)
    assert_training_spectrogram_index(resume_df)
    assert {path: path.stat().st_mtime_ns for path in output_dir.rglob("*.npy")} == mtimes


def test_subset_sample_and_nondefault_resolution_match_the_builder(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    rows = [
        _write_training_call(root, f"call_{index}", [(0, 400 + 100 * index, 0), (800, 2000, 1)], seed=index)
        for index in range(4)
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)
    hop_length = HOP_LENGTH // 2
    n_mels = N_MELS * 2
    n_calls = 2
    seed = 7
    expected_utterances = _utterance_total(rows, root, "training", n_calls, seed)
    build_dir = tmp_path / "build"
    resume_dir = tmp_path / "resume"
    build_index = tmp_path / "build.parquet"
    resume_index = tmp_path / "resume.parquet"
    shared = dict(n_calls=n_calls, seed=seed, hop_length=hop_length, n_mels=n_mels, workers=2)
    _run(build_script, _cache_args(root, manifest, build_dir, build_index, tmp_path / "build.jsonl", **shared))
    shutil.copytree(build_dir, resume_dir)
    _run(
        resume,
        _cache_args(
            root,
            manifest,
            resume_dir,
            resume_index,
            tmp_path / "resume.jsonl",
            expected_utterances=expected_utterances,
            **shared,
        ),
    )
    summary = json.loads(resume.resume_summary_path(resume_index).read_text(encoding="utf-8"))
    assert summary["calls_recomputed"] == 0
    assert summary["calls_reused"] == n_calls
    assert summary["utterances_indexed"] == expected_utterances
    resume_df = pd.read_parquet(resume_index)
    # The builder's workers>1 index follows completion order. Resume writes sample_calls
    # order. Pack sorts either way; compare the canonical (call_id, utt_idx) order.
    pd.testing.assert_frame_equal(
        _by_call(pd.read_parquet(build_index)), _by_call(resume_df), check_exact=True
    )
    assert set(resume_df["n_mels"]) == {n_mels}
    assert set(resume_df["hop_length"]) == {hop_length}
    assert _npy_bytes(build_dir) == _npy_bytes(resume_dir)


def test_validation_resume_is_not_a_training_index(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    wav = "chunk/Validation/1.원천데이터/VS_서울_구급/call_val.wav"
    label = "chunk/Validation/2.라벨링데이터/VL_서울_구급/call_val.json"
    _write_call(root, wav, label, [(0, 400, 0), (500, 1600, 1)], seed=0)
    train_wav, train_label = _seoul_training("call_train")
    _write_call(root, train_wav, train_label, [(0, 400, 0)], seed=1)
    rows = [
        ManifestRow("Training", "call_train", train_wav, train_label),
        ManifestRow("Validation", "call_val", wav, label),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)
    expected_utterances = _utterance_total(rows, root, "validation", -1, 42)
    output_dir = tmp_path / "specs"
    index_output = tmp_path / "index.parquet"
    _run(
        build_script,
        _cache_args(
            root, manifest, output_dir, tmp_path / "build.parquet", tmp_path / "build.jsonl", split="validation"
        ),
    )
    _run(
        resume,
        _cache_args(
            root,
            manifest,
            output_dir,
            index_output,
            tmp_path / "resume.jsonl",
            split="validation",
            workers=1,
            expected_utterances=expected_utterances,
        ),
    )
    frame = pd.read_parquet(index_output)
    assert set(frame["call_id"]) == {"call_val"}
    assert set(frame["split"]) == {"Validation"}
    summary = json.loads(resume.resume_summary_path(index_output).read_text(encoding="utf-8"))
    assert summary["calls_recomputed"] == 0
    assert summary["calls_reused"] == 1
    with pytest.raises(SplitContaminationError, match="non-Training"):
        assert_training_spectrogram_index(frame)


_PROGRESS_LINE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00) "
    r"done=(?P<done>\d+) total=(?P<total>\d+) reused=(?P<reused>\d+) "
    r"recomputed=(?P<recomputed>\d+) utterances_indexed=(?P<utterances>\d+) "
    r"failures=(?P<failures>\d+)$"
)


def test_partial_index_write_is_atomic_and_leaves_the_official_index_alone(tmp_path: Path) -> None:
    index_output = tmp_path / "index.parquet"
    resume._write_partial_index([{"call_id": "c", "utt_idx": 0}], index_output)
    partial = resume.partial_index_path(index_output)
    assert partial == Path(str(index_output) + ".partial.parquet")
    assert partial.exists()
    assert not index_output.exists()
    assert not partial.with_name(partial.name + ".tmp").exists()
    assert pd.read_parquet(partial)["call_id"].tolist() == ["c"]

    resume._write_partial_index([{"call_id": "d", "utt_idx": 1}], index_output)
    assert pd.read_parquet(partial)["call_id"].tolist() == ["d"]
    assert not index_output.exists()


def test_progress_log_appends_flushed_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "progress.log"
    resume._append_progress(path, "line-one")
    assert path.read_text(encoding="utf-8") == "line-one\n"
    resume._append_progress(path, "line-two")
    assert path.read_text(encoding="utf-8") == "line-one\nline-two\n"


def test_progress_log_and_partial_index_and_n_calls_minus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-1 keeps every eligible call. Progress lines append. The partial index is not an input."""

    root = tmp_path / "raw"
    rows = [
        _write_training_call(root, stem, [(0, 500, 0), (600, 1600, 1)], seed=seed)
        for seed, stem in enumerate(("z_call", "a_call", "m_call", "b_call"))
    ]
    busan_wav = "chunk/Training/1.원천데이터/TS_부산_구급/busan.wav"
    busan_json = "chunk/Training/2.라벨링데이터/TL_부산_구급/busan.json"
    _write_call(root, busan_wav, busan_json, [(0, 500, 0), (600, 1600, 1)], seed=9)
    rows.append(ManifestRow("Training", "busan", busan_wav, busan_json))
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest, overwrite=True)
    assert len(rows) == 5
    expected_utterances = _utterance_total(rows, root, "training", -1, 42)
    assert expected_utterances == 8

    build_dir = tmp_path / "build"
    build_index = tmp_path / "build.parquet"
    _run(
        build_script,
        _cache_args(root, manifest, build_dir, build_index, tmp_path / "build.jsonl", n_calls=-1, workers=2),
    )

    resume_dir = tmp_path / "resume"
    resume_index = tmp_path / "resume.parquet"
    progress_log = tmp_path / "local" / "progress.log"
    poison = resume.partial_index_path(resume_index)
    poison.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"call_id": "not-a-real-call", "utt_idx": 0}]).to_parquet(poison, index=False)

    partial_sizes: list[int] = []
    real_write = resume._write_partial_index

    def spy(index_rows, index_output):
        partial_sizes.append(len(index_rows))
        real_write(index_rows, index_output)

    monkeypatch.setattr(resume, "_write_partial_index", spy)
    shared = dict(
        n_calls=-1,
        workers=2,
        progress_log=progress_log,
        progress_every=2,
        partial_index_every=2,
        expected_utterances=expected_utterances,
    )
    _run(resume, _cache_args(root, manifest, resume_dir, resume_index, tmp_path / "resume.jsonl", **shared))

    resume_df = pd.read_parquet(resume_index)
    assert partial_sizes == [4, 8]
    assert "not-a-real-call" not in set(resume_df["call_id"])
    assert set(resume_df["call_id"]) == {"z_call", "a_call", "m_call", "b_call"}
    seen: list[str] = []
    for call_id in resume_df["call_id"]:
        call_id = _plain(call_id)
        if not seen or seen[-1] != call_id:
            seen.append(call_id)
    assert seen == ["z_call", "a_call", "m_call", "b_call"]
    pd.testing.assert_frame_equal(_by_call(pd.read_parquet(build_index)), _by_call(resume_df), check_exact=True)
    partial = pd.read_parquet(resume.partial_index_path(resume_index))
    pd.testing.assert_frame_equal(partial, resume_df, check_exact=True)
    assert not resume.partial_index_path(resume_index).with_name(
        resume.partial_index_path(resume_index).name + ".tmp"
    ).exists()
    assert _npy_bytes(build_dir) == _npy_bytes(resume_dir)

    lines = progress_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [_PROGRESS_LINE.fullmatch(line) for line in lines]
    assert all(parsed)
    assert (int(parsed[0]["done"]), int(parsed[1]["done"])) == (2, 4)
    assert all(int(match["total"]) == 4 and int(match["failures"]) == 0 for match in parsed)
    assert int(parsed[0]["reused"]) + int(parsed[0]["recomputed"]) == 2
    assert int(parsed[1]["reused"]) + int(parsed[1]["recomputed"]) == 4
    assert int(parsed[0]["utterances"]) == 4
    assert int(parsed[1]["utterances"]) == 8
    assert int(parsed[1]["recomputed"]) == 4

    summary = json.loads(resume.resume_summary_path(resume_index).read_text(encoding="utf-8"))
    assert summary["calls_recomputed"] == 4
    assert summary["sampled_calls"] == 4
    assert summary["args"]["n_calls"] == -1
    assert summary["args"]["progress_every"] == 2
    assert summary["args"]["partial_index_every"] == 2

    # A second run appends to the progress log and still ignores any partial checkpoint.
    _run(resume, _cache_args(root, manifest, resume_dir, resume_index, tmp_path / "resume.jsonl", **shared))
    rerun_lines = progress_log.read_text(encoding="utf-8").splitlines()
    assert len(rerun_lines) == 4
    rerun_tail = _PROGRESS_LINE.fullmatch(rerun_lines[-1])
    assert rerun_tail is not None
    assert int(rerun_tail["done"]) == 4
    assert int(rerun_tail["reused"]) == 4
    assert int(rerun_tail["recomputed"]) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(resume_index), resume_df, check_exact=True)
