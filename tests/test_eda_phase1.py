from __future__ import annotations

import importlib.util
import struct
import sys
import types
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _ensure_dcc_data_modules() -> None:
    try:
        from dcc_data.manifest import ManifestRow  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    @dataclass(frozen=True)
    class ManifestRow:
        split: str
        stem: str
        wav_path: str
        json_path: str

    def read_wav_info(path: str | Path):
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(f"Only uncompressed PCM WAV is supported: {path}")
            info = types.SimpleNamespace(
                channels=wav_file.getnchannels(),
                sample_rate=wav_file.getframerate(),
                sample_width=wav_file.getsampwidth(),
                frame_count=wav_file.getnframes(),
            )
        if min(info.channels, info.sample_rate, info.sample_width, info.frame_count) <= 0:
            raise ValueError(f"Invalid WAV parameters: {info}")
        return info

    def load_manifest(*_args, **_kwargs):
        return []

    package = types.ModuleType("dcc_data")
    package.__path__ = []
    manifest = types.ModuleType("dcc_data.manifest")
    manifest.ManifestRow = ManifestRow
    manifest.load_manifest = load_manifest
    audio = types.ModuleType("dcc_data.audio")
    audio.read_wav_info = read_wav_info
    mission3 = types.ModuleType("dcc_data.mission3")
    mission3.SYMPTOM_CLASSES = (
        "고열",
        "구토",
        "두통",
        "복통",
        "어지러움",
        "열상",
        "오심",
        "전신쇠약",
        "호흡곤란",
    )
    sys.modules["dcc_data"] = package
    sys.modules["dcc_data.manifest"] = manifest
    sys.modules["dcc_data.audio"] = audio
    sys.modules["dcc_data.mission3"] = mission3


_ensure_dcc_data_modules()
from dcc_data.manifest import ManifestRow

SCRIPT_PATH = REPO_ROOT / "scripts" / "eda_dcc_dataset.py"


def load_eda_module():
    spec = importlib.util.spec_from_file_location("eda_dcc_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eda = load_eda_module()


def write_wav(path: Path, *, n_frames: int = 20, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(struct.pack("<h", index) for index in range(n_frames))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def test_summarize_values_includes_unit_and_percentiles() -> None:
    empty = eda.summarize_values([], "seconds")
    assert empty["n"] == 0
    assert empty["unit"] == "seconds"
    assert empty["p99"] is None

    summary = eda.summarize_values([1, 2, 3, 4, 5], "count")
    assert summary["n"] == 5
    assert summary["min"] == 1
    assert summary["max"] == 5
    assert summary["p50"] == 3
    assert summary["unit"] == "count"
    assert set(summary) == {
        "n",
        "min",
        "mean",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "unit",
    }


def test_empty_split_block_has_mission_keys() -> None:
    block = eda.empty_split_block()
    assert set(block) == {"common", "mission1", "mission2", "mission3"}
    assert block["common"] == {}
    assert block["mission2"] == {}


def test_assert_not_legacy_output_blocks_preserved_file() -> None:
    with pytest.raises(ValueError, match="legacy EDA"):
        eda.assert_not_legacy_output(eda.LEGACY_EDA_PATH)


def test_main_refuses_legacy_eda_summary_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eda_dcc_dataset.py", "--output", str(eda.LEGACY_EDA_PATH)],
    )
    with pytest.raises(ValueError, match="legacy EDA"):
        eda.main()


def test_wav_header_cache_reuses_then_refreshes_on_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "raw"
    wav_rel = "Training/audio/a.wav"
    wav_path = data_root / wav_rel
    write_wav(wav_path)
    row = ManifestRow("Training", "a", wav_rel, "Training/label/a.json")
    cache_path = tmp_path / "wav_header_cache.csv"

    calls = {"n": 0}
    real = eda.read_wav_info

    def counting_read_wav_info(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(eda, "read_wav_info", counting_read_wav_info)

    first, stats_first = eda.read_or_build_wav_header_cache(
        [row], data_root, cache_path, rebuild=False
    )
    assert stats_first["n_computed"] == 1
    assert stats_first["n_reused"] == 0
    assert stats_first["n_error"] == 0
    assert calls["n"] == 1
    assert first[("Training", "a")]["sample_rate"] == "8000"
    assert first[("Training", "a")]["channels"] == "1"
    assert first[("Training", "a")]["sample_width"] == "2"
    assert first[("Training", "a")]["error"] == ""

    second, stats_second = eda.read_or_build_wav_header_cache(
        [row], data_root, cache_path, rebuild=False
    )
    assert stats_second["n_computed"] == 0
    assert stats_second["n_reused"] == 1
    assert calls["n"] == 1
    assert second[("Training", "a")]["frame_count"] == first[("Training", "a")]["frame_count"]

    write_wav(wav_path, n_frames=40)
    third, stats_third = eda.read_or_build_wav_header_cache(
        [row], data_root, cache_path, rebuild=False
    )
    assert stats_third["n_computed"] == 1
    assert stats_third["n_reused"] == 0
    assert calls["n"] == 2
    assert third[("Training", "a")]["frame_count"] == "40"

    eda.read_or_build_wav_header_cache([row], data_root, cache_path, rebuild=True)
    assert calls["n"] == 3


def test_wav_header_cache_records_error_without_raising(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    row = ManifestRow("Validation", "missing", "Validation/audio/x.wav", "Validation/x.json")
    cache_path = tmp_path / "wav_header_cache.csv"
    merged, stats = eda.read_or_build_wav_header_cache(
        [row], data_root, cache_path, rebuild=False
    )
    assert stats["n_error"] == 1
    assert "FileNotFoundError" in merged[("Validation", "missing")]["error"]
    assert merged[("Validation", "missing")]["sample_rate"] == ""


def test_phase2_skeletons_do_not_compute_stats() -> None:
    assert eda.accumulate_common() == {}
    assert eda.accumulate_mission1() == {}
    assert eda.accumulate_mission2() == {}
    assert eda.accumulate_mission3() == {}
