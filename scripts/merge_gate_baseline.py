"""Run the trusted ``main`` tests against PR code, skipping approved exemptions.

``.github/merge-gate.json`` may list ``allowed_baseline_test_failures``: main
tests that a PR is allowed to break because it changes the asserted behavior on
purpose (for example to follow an organizer ruling). Each entry needs ``test``
(a ``tests/...::name`` node id), a non-empty ``reason`` and ``pr``. CI reads the
list from the trusted ``main`` checkout, so a PR cannot exempt its own tests;
the exemption has to be merged first. Remove an entry once its PR has merged.

The command intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

CONFIG_KEY = "allowed_baseline_test_failures"


def baseline_exemptions(config: Mapping[str, object]) -> list[dict]:
    """Validated exemption entries; raises ``ValueError`` on a malformed one."""

    entries = config.get(CONFIG_KEY, [])
    if not isinstance(entries, list):
        raise ValueError(f"{CONFIG_KEY} must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{CONFIG_KEY}[{index}] must be an object")
        test = entry.get("test")
        if not isinstance(test, str) or not test.startswith("tests/") or "::" not in test:
            raise ValueError(f"{CONFIG_KEY}[{index}].test must look like tests/<file>.py::<name>")
        if not str(entry.get("reason", "")).strip():
            raise ValueError(f"{CONFIG_KEY}[{index}].reason is required")
        if not isinstance(entry.get("pr"), int):
            raise ValueError(f"{CONFIG_KEY}[{index}].pr must be the PR number")
        if test in seen:
            raise ValueError(f"{CONFIG_KEY}[{index}].test is listed twice: {test}")
        seen.add(test)
    return entries


def deselect_args(entries: Sequence[Mapping[str, object]], tests_dir: str) -> list[str]:
    """``--deselect`` arguments with ``tests/`` mapped onto the copied ``tests_dir``.

    ``tests_dir`` must be relative to the working directory: pytest silently ignores
    an absolute ``--deselect`` node id, which would drop every exemption.
    """

    if Path(tests_dir).is_absolute():
        raise ValueError(f"--tests-dir must be relative to the working directory: {tests_dir}")
    args: list[str] = []
    for entry in entries:
        node = tests_dir.rstrip("/") + "/" + str(entry["test"])[len("tests/"):]
        args += ["--deselect", node]
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tests-dir", required=True, help="Copy of the trusted tests/ folder")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    entries = baseline_exemptions(config)
    for entry in entries:
        print(f"기존 테스트 예외: {entry['test']} (PR #{entry['pr']}) — {entry['reason']}")
    command = [sys.executable, "-m", "pytest", args.tests_dir, "-q", "--disable-warnings"]
    command += deselect_args(entries, args.tests_dir)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
