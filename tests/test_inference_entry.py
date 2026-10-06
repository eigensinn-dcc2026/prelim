"""Root inference.py dispatch and CSV contract (no data needed)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcc_inference import registry  # noqa: E402


def _load_entry():
    spec = importlib.util.spec_from_file_location("dcc_root_inference", ROOT / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,expected",
    [("mission1.csv", 1), ("mission2.csv", 2), ("Mission3.csv", 3), ("out_mission_2.csv", 2)],
)
def test_resolve_mission_from_output_name(name, expected):
    assert registry.resolve_mission(Path("outputs") / name) == expected


def test_resolve_mission_explicit_and_conflict():
    assert registry.resolve_mission(Path("out.csv"), 3) == 3
    with pytest.raises(ValueError):
        registry.resolve_mission(Path("mission1.csv"), 2)
    with pytest.raises(ValueError):
        registry.resolve_mission(Path("out.csv"))


def test_official_columns():
    assert registry.MISSION_COLUMNS[1] == ("audio file name", "gender")
    assert registry.MISSION_COLUMNS[2] == ("audio file name", "startAt", "endAt", "speaker")
    assert registry.MISSION_COLUMNS[3] == ("label file name", "symptom")


def test_validate_frame_rejects_wrong_columns_and_empty():
    with pytest.raises(ValueError):
        registry.validate_frame(1, pd.DataFrame({"audio_file_name": ["a.wav"], "gender": ["F"]}))
    with pytest.raises(ValueError):
        registry.validate_frame(1, pd.DataFrame(columns=["audio file name", "gender"]))
    registry.validate_frame(1, pd.DataFrame({"audio file name": ["a.wav"], "gender": ["F"]}))


def test_unwired_missions_fail_loudly():
    # M1–M3 are wired (M3: tests/test_inference_mission3.py). A missing
    # checkpoint must not be the old NotImplementedError stub.
    with pytest.raises(FileNotFoundError):
        registry.load_predict(3)(Path("a"), Path("b"), Path("missing-ckpt.bin"))


def test_entry_writes_validated_csv(tmp_path, monkeypatch):
    entry = _load_entry()
    fake = types.ModuleType("dcc_inference.mission1")
    fake.predict = lambda a, l, c: pd.DataFrame(
        {"audio file name": ["x.wav", "y.wav"], "gender": ["F", "M"]}
    )
    monkeypatch.setitem(sys.modules, "dcc_inference.mission1", fake)
    ckpt = tmp_path / "m1.ckpt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "outputs" / "mission1.csv"
    rc = entry.main(
        ["--audio_dir", str(tmp_path), "--label_dir", str(tmp_path),
         "--ckpt_path", str(ckpt), "--output", str(out)]
    )
    assert rc == 0
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").splitlines() == ["audio file name,gender", "x.wav,F", "y.wav,M"]