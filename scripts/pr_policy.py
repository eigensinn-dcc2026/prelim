"""Evaluate deterministic repository policy for a pull request diff.

The command intentionally uses only the Python standard library.  GitHub
Actions checks out the synthetic merge commit and fetches full history, while
``--base`` and ``--head`` identify the pull request endpoints to inspect.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


PERSONAL_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+(?![<$\{])[^\\/\s\"']+[\\/]"),
    re.compile(r"(?i)(?<![\w])/(?:Users|home)/(?![<$\{])[^/\s\"']+/"),
)

SECRET_PATTERNS = (
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|passwd|token)"
            r"\s*[:=]\s*[\"']([^\"']{8,})[\"']"
        ),
    ),
)

PLACEHOLDER_MARKERS = (
    "example",
    "placeholder",
    "dummy",
    "changeme",
    "your_",
    "your-",
    "<",
    "${",
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def is_forbidden_path(path: str, prefixes: Sequence[str]) -> bool:
    normalized = normalize_path(path)
    return any(normalized.startswith(normalize_path(prefix).rstrip("/") + "/") for prefix in prefixes)


def requires_tests(path: str, config: Mapping[str, object]) -> bool:
    normalized = normalize_path(path)
    exact = {normalize_path(value) for value in config.get("test_exact_paths", [])}
    if normalized in exact:
        return True
    return any(
        normalized.startswith(normalize_path(prefix).rstrip("/") + "/")
        for prefix in config.get("test_path_prefixes", [])
    )


def detect_personal_path(line: str) -> bool:
    return any(pattern.search(line) for pattern in PERSONAL_PATH_PATTERNS)


def detect_secret(line: str) -> str | None:
    lowered = line.lower()
    for name, pattern in SECRET_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        if name == "assigned-secret":
            candidate = match.group(1).lower()
            if any(marker in candidate for marker in PLACEHOLDER_MARKERS):
                continue
        return name
    return None


def notebook_code(path: Path) -> tuple[list[str], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], str(exc)

    cells = payload.get("cells")
    if not isinstance(cells, list):
        return [], "최상위 cells 배열이 없습니다."

    code_lines: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            code_lines.extend(source.splitlines())
        elif isinstance(source, list) and all(isinstance(value, str) for value in source):
            code_lines.extend("".join(source).splitlines())
        else:
            return [], "code cell의 source 형식이 올바르지 않습니다."
    return code_lines, None


def evaluate_policy_inputs(
    *,
    root: Path,
    changed_paths: Sequence[str],
    changed_file_sizes: Mapping[str, int],
    added_lines: Mapping[str, Sequence[str]],
    config: Mapping[str, object],
    deleted_test_files: Sequence[str] = (),
    removed_test_functions: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    findings: list[Finding] = []
    forbidden_prefixes = list(config.get("forbidden_prefixes", []))
    max_bytes = int(config.get("max_file_bytes", 10 * 1024 * 1024))

    normalized_changed = sorted({normalize_path(path) for path in changed_paths})

    allowed_test_files = {
        normalize_path(path) for path in config.get("allowed_test_file_deletions", [])
    }
    for path in sorted({normalize_path(path) for path in deleted_test_files}):
        if path not in allowed_test_files:
            findings.append(
                Finding("deleted-test-file", path, "기존 테스트 파일은 삭제할 수 없습니다.")
            )

    allowed_test_functions = set(config.get("allowed_test_function_deletions", []))
    for path, names in sorted((removed_test_functions or {}).items()):
        normalized_path = normalize_path(path)
        for name in sorted(set(names)):
            identifier = f"{normalized_path}::{name}"
            if identifier not in allowed_test_functions:
                findings.append(
                    Finding(
                        "deleted-test-function",
                        normalized_path,
                        f"기존 테스트 함수 `{name}`은 삭제하거나 이름을 바꿀 수 없습니다.",
                    )
                )

    for path in normalized_changed:
        if is_forbidden_path(path, forbidden_prefixes):
            findings.append(
                Finding("forbidden-path", path, "Git에 포함할 수 없는 산출물 경로입니다.")
            )

    for path in normalized_changed:
        size = int(changed_file_sizes.get(path, 0))
        if size > max_bytes:
            findings.append(
                Finding(
                    "large-file",
                    path,
                    f"파일 크기 {size} bytes가 제한 {max_bytes} bytes를 초과합니다.",
                )
            )

    for path in normalized_changed:
        file_path = root / Path(path)
        if path.endswith(".ipynb") and file_path.is_file():
            code_lines, error = notebook_code(file_path)
            if error:
                findings.append(Finding("invalid-notebook", path, error))
            else:
                for line in code_lines:
                    if detect_personal_path(line):
                        findings.append(
                            Finding("personal-path", path, "노트북 코드 셀에 개인 절대 경로가 있습니다.")
                        )
                        break

        if path.endswith(".py"):
            for line in added_lines.get(path, []):
                if detect_personal_path(line):
                    findings.append(
                        Finding("personal-path", path, "추가된 코드에 개인 절대 경로가 있습니다.")
                    )
                    break

    for path, lines in added_lines.items():
        for line in lines:
            secret_type = detect_secret(line)
            if secret_type:
                findings.append(
                    Finding(
                        "probable-secret",
                        normalize_path(path),
                        f"추가된 내용에서 {secret_type} 형태의 비밀정보가 감지됐습니다.",
                    )
                )
                break

    tests_required = any(requires_tests(path, config) for path in normalized_changed)
    tests_changed = any(path == "tests" or path.startswith("tests/") for path in normalized_changed)
    return {
        "status": "fail" if findings else "pass",
        "tests_required": tests_required,
        "tests_changed": tests_changed,
        "changed_paths": normalized_changed,
        "findings": [asdict(finding) for finding in findings],
    }


def git_bytes(args: Sequence[str], root: Path) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def nul_paths(args: Sequence[str], root: Path) -> list[str]:
    raw = git_bytes(args, root)
    return [normalize_path(value.decode("utf-8", errors="surrogateescape")) for value in raw.split(b"\0") if value]


def test_function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()

    def visit_body(body: Sequence[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit_body(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                names.add("::".join((*parents, node.name)))

    visit_body(tree.body)
    return names


def collect_test_protection_inputs(
    base: str, head: str, root: Path
) -> tuple[list[str], dict[str, list[str]]]:
    def tracked_test_files(revision: str) -> set[str]:
        return {
            path
            for path in nul_paths(
                ["ls-tree", "-r", "--name-only", "-z", revision, "--", "tests"], root
            )
            if path.endswith(".py")
        }

    base_files = tracked_test_files(base)
    head_files = tracked_test_files(head)
    deleted_files = sorted(base_files - head_files)
    removed_functions: dict[str, list[str]] = {}

    for path in sorted(base_files & head_files):
        base_source = git_bytes(["show", f"{base}:{path}"], root).decode(
            "utf-8", errors="replace"
        )
        head_source = git_bytes(["show", f"{head}:{path}"], root).decode(
            "utf-8", errors="replace"
        )
        missing = sorted(test_function_names(base_source) - test_function_names(head_source))
        if missing:
            removed_functions[path] = missing

    return deleted_files, removed_functions


def collect_added_lines(base: str, head: str, root: Path) -> dict[str, list[str]]:
    raw = git_bytes(
        [
            "-c",
            "core.quotepath=false",
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            f"{base}...{head}",
            "--",
        ],
        root,
    ).decode("utf-8", errors="replace")

    current_path: str | None = None
    result: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if line.startswith("+++ b/"):
            current_path = normalize_path(line[6:])
            result.setdefault(current_path, [])
        elif line.startswith("+++ /dev/null"):
            current_path = None
        elif current_path and line.startswith("+") and not line.startswith("+++"):
            result[current_path].append(line[1:])
    return result


def collect_policy_inputs(
    base: str, head: str, root: Path
) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    changed_paths = nul_paths(["diff", "--name-only", "-z", f"{base}...{head}"], root)
    sizes: dict[str, int] = {}
    for path in changed_paths:
        file_path = root / Path(path)
        if file_path.is_file():
            sizes[path] = file_path.stat().st_size
    return changed_paths, sizes, collect_added_lines(base, head, root)


def write_github_output(path: Path, tests_required: bool, tests_changed: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"tests_required={'true' if tests_required else 'false'}\n")
        handle.write(f"tests_changed={'true' if tests_changed else 'false'}\n")


def print_result(result: Mapping[str, object], *, scope_only: bool) -> None:
    changed_count = len(result["changed_paths"])
    test_text = "필요" if result["tests_required"] else "생략 가능"
    print(f"변경 파일: {changed_count}개, Python 테스트: {test_text}")
    if scope_only:
        return
    findings = result["findings"]
    if findings:
        print(f"정책 위반: {len(findings)}건", file=sys.stderr)
        for finding in findings:
            print(
                f"- [{finding['code']}] {finding['path']}: {finding['message']}",
                file=sys.stderr,
            )
    else:
        print("정책 검사를 통과했습니다.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--scope-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    config = load_config(args.config)
    changed, sizes, added_lines = collect_policy_inputs(args.base, args.head, root)
    deleted_test_files, removed_test_functions = collect_test_protection_inputs(
        args.base, args.head, root
    )
    result = evaluate_policy_inputs(
        root=root,
        changed_paths=changed,
        changed_file_sizes=sizes,
        added_lines=added_lines,
        config=config,
        deleted_test_files=deleted_test_files,
        removed_test_functions=removed_test_functions,
    )
    print_result(result, scope_only=args.scope_only)

    if args.json_output:
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.github_output:
        write_github_output(
            args.github_output,
            bool(result["tests_required"]),
            bool(result["tests_changed"]),
        )
    if args.scope_only:
        return 0
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
