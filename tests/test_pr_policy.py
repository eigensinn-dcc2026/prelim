from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.pr_policy import collect_policy_inputs, detect_secret, evaluate_policy_inputs


def config() -> dict:
    return {
        "max_file_bytes": 10 * 1024 * 1024,
        "forbidden_prefixes": ["data/", "checkpoints/", "outputs/"],
        "test_path_prefixes": ["src/", "scripts/", "tests/"],
        "test_exact_paths": ["requirements.txt"],
    }


def evaluate(
    root: Path,
    *,
    changed: list[str],
    sizes: dict[str, int] | None = None,
    lines: dict[str, list[str]] | None = None,
) -> dict:
    return evaluate_policy_inputs(
        root=root,
        changed_paths=changed,
        changed_file_sizes=sizes or {},
        added_lines=lines or {},
        config=config(),
    )


def finding_codes(result: dict) -> set[str]:
    return {finding["code"] for finding in result["findings"]}


def test_forbidden_artifact_path_fails(tmp_path: Path) -> None:
    result = evaluate(tmp_path, changed=["data/private.csv"])

    assert result["status"] == "fail"
    assert finding_codes(result) == {"forbidden-path"}


def test_file_size_limit_is_strictly_greater_than_10_mib(tmp_path: Path) -> None:
    limit = 10 * 1024 * 1024
    at_limit = evaluate(
        tmp_path,
        changed=["reports/at-limit.bin"],
        sizes={"reports/at-limit.bin": limit},
    )
    over_limit = evaluate(
        tmp_path,
        changed=["reports/too-large.bin"],
        sizes={"reports/too-large.bin": limit + 1},
    )

    assert "large-file" not in finding_codes(at_limit)
    assert "large-file" in finding_codes(over_limit)


def test_personal_absolute_path_in_added_python_fails(tmp_path: Path) -> None:
    windows_path = "C:" + "\\" + "Users" + "\\" + "alice" + "\\" + "dcc-data"
    line = "DATA_ROOT = " + repr(windows_path)
    result = evaluate(
        tmp_path,
        changed=["scripts/train.py"],
        lines={"scripts/train.py": [line]},
    )

    assert "personal-path" in finding_codes(result)


def test_valid_notebook_passes_and_malformed_notebook_fails(tmp_path: Path) -> None:
    valid_path = tmp_path / "notebooks" / "valid.ipynb"
    invalid_path = tmp_path / "notebooks" / "invalid.ipynb"
    valid_path.parent.mkdir()
    valid_path.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": ["print('ok')"]}]}),
        encoding="utf-8",
    )
    invalid_path.write_text("{not-json", encoding="utf-8")

    valid = evaluate(tmp_path, changed=["notebooks/valid.ipynb"])
    invalid = evaluate(tmp_path, changed=["notebooks/invalid.ipynb"])

    assert valid["status"] == "pass"
    assert "invalid-notebook" in finding_codes(invalid)


def test_probable_secret_in_added_line_fails(tmp_path: Path) -> None:
    fake_token = "ghp_" + ("a" * 36)
    assignment = "TO" + "KEN = " + repr(fake_token)
    result = evaluate(
        tmp_path,
        changed=["scripts/tool.py"],
        lines={"scripts/tool.py": [assignment]},
    )

    assert "probable-secret" in finding_codes(result)
    assert detect_secret("token = '${GITHUB_TOKEN}'") is None


def test_test_scope_distinguishes_docs_from_python_changes(tmp_path: Path) -> None:
    docs = evaluate(tmp_path, changed=["docs/decision.md"])
    code = evaluate(tmp_path, changed=["src/dcc_data/guards.py"])
    dependency = evaluate(tmp_path, changed=["requirements.txt"])

    assert docs["tests_required"] is False
    assert code["tests_required"] is True
    assert dependency["tests_required"] is True


def test_git_diff_collection_drives_policy_evaluation(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Merge Gate Test")
    git("config", "user.email", "merge-gate@example.invalid")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")

    script = tmp_path / "scripts" / "tool.py"
    forbidden = tmp_path / "data" / "private.csv"
    script.parent.mkdir()
    forbidden.parent.mkdir()
    personal = "C:" + "\\" + "Users" + "\\" + "alice" + "\\" + "dcc-data"
    script.write_text(f"DATA_ROOT = {personal!r}\n", encoding="utf-8")
    forbidden.write_text("private\n", encoding="utf-8")
    git("add", "scripts/tool.py", "data/private.csv")
    git("commit", "-q", "-m", "head")
    head = git("rev-parse", "HEAD")

    changed, sizes, lines = collect_policy_inputs(base, head, tmp_path)
    result = evaluate_policy_inputs(
        root=tmp_path,
        changed_paths=changed,
        changed_file_sizes=sizes,
        added_lines=lines,
        config=config(),
    )

    assert result["tests_required"] is True
    assert {"forbidden-path", "personal-path"} <= finding_codes(result)


def test_modified_file_over_limit_is_rejected_by_git_diff_collection(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Merge Gate Test")
    git("config", "user.email", "merge-gate@example.invalid")
    artifact = tmp_path / "reports" / "artifact.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"small")
    git("add", "reports/artifact.bin")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")

    artifact.write_bytes(b"x" * 17)
    git("add", "reports/artifact.bin")
    git("commit", "-q", "-m", "enlarge tracked file")
    head = git("rev-parse", "HEAD")

    changed, sizes, lines = collect_policy_inputs(base, head, tmp_path)
    test_config = config()
    test_config["max_file_bytes"] = 16
    result = evaluate_policy_inputs(
        root=tmp_path,
        changed_paths=changed,
        changed_file_sizes=sizes,
        added_lines=lines,
        config=test_config,
    )

    assert changed == ["reports/artifact.bin"]
    assert sizes == {"reports/artifact.bin": 17}
    assert "large-file" in finding_codes(result)
