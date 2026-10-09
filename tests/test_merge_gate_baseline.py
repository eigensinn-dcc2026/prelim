from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.merge_gate_baseline import baseline_exemptions, deselect_args

ROOT = Path(__file__).resolve().parents[1]


def entry(test: str = "tests/test_x.py::test_a", reason: str = "ruling", pr: object = 90) -> dict:
    return {"test": test, "reason": reason, "pr": pr}


def test_missing_key_means_no_exemptions():
    assert baseline_exemptions({}) == []


def test_deselect_args_map_onto_the_copied_tests_dir():
    entries = baseline_exemptions(
        {"allowed_baseline_test_failures": [entry(), entry("tests/sub/test_y.py::test_b")]}
    )
    assert deselect_args(entries, ".merge-gate-baseline-tests/") == [
        "--deselect",
        ".merge-gate-baseline-tests/test_x.py::test_a",
        "--deselect",
        ".merge-gate-baseline-tests/sub/test_y.py::test_b",
    ]
    with pytest.raises(ValueError):
        deselect_args(entries, str(ROOT / "baseline"))


@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        ["not an object"],
        [entry(test="test_x.py::test_a")],  # must start with tests/
        [entry(test="tests/test_x.py")],  # whole file is not allowed
        [entry(reason="  ")],
        [entry(pr="90")],
        [entry(), entry()],  # duplicate
    ],
)
def test_malformed_exemptions_are_rejected(bad):
    with pytest.raises(ValueError):
        baseline_exemptions({"allowed_baseline_test_failures": bad})


def test_repository_config_is_valid():
    config = json.loads((ROOT / ".github" / "merge-gate.json").read_text(encoding="utf-8"))
    for item in baseline_exemptions(config):
        path, name = str(item["test"]).split("::", 1)
        assert (ROOT / path).is_file(), item["test"]
        assert f"def {name}(" in (ROOT / path).read_text(encoding="utf-8"), item["test"]


def test_cli_skips_only_the_exempted_test(tmp_path: Path):
    tests_dir = tmp_path / "baseline"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "def test_kept():\n    assert True\n\ndef test_broken():\n    assert False\n",
        encoding="utf-8",
    )
    config = tmp_path / "gate.json"
    config.write_text(
        json.dumps({"allowed_baseline_test_failures": [entry("tests/test_sample.py::test_broken")]}),
        encoding="utf-8",
    )
    # Relative, as in CI (.merge-gate-baseline-tests).
    command = [sys.executable, str(ROOT / "scripts" / "merge_gate_baseline.py"),
               "--config", str(config), "--tests-dir", "baseline"]
    done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed, 1 deselected" in done.stdout

    config.write_text(json.dumps({}), encoding="utf-8")
    done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 1
