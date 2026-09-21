"""Compute and publish a pull request merge-readiness verdict.

This script is run only from the trusted default branch.  It never checks out
or executes pull request code; all mutable operations use GitHub's REST API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


COMMENT_MARKER = "<!-- merge-gate -->"
CHECK_DISPLAY_NAMES = {
    "PR CI / policy": "파일·보안 규칙 검사",
    "PR CI / tests": "자동 테스트",
}


@dataclass(frozen=True)
class Condition:
    name: str
    state: str
    message: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_check_states(checks: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for check in checks:
        name = str(check.get("name", ""))
        if not name:
            continue
        previous = latest.get(name)
        if previous is None or int(check.get("id") or 0) > int(previous.get("id") or 0):
            latest[name] = check
    return latest


def evaluate_snapshot(snapshot: Mapping[str, object], config: Mapping[str, object]) -> dict:
    conditions: list[Condition] = []
    expected_base = str(config.get("base_branch", "main"))
    base_branch = str(snapshot.get("base_branch", ""))

    if base_branch == expected_base:
        conditions.append(
            Condition("합칠 대상", "pass", f"팀 공용 코드인 `{expected_base}`에 합치는 PR입니다.")
        )
    else:
        conditions.append(
            Condition("합칠 대상", "block", f"`{expected_base}`에 합치는 PR이 아닙니다.")
        )

    if bool(snapshot.get("draft")):
        conditions.append(
            Condition(
                "작성 상태",
                "wait",
                "아직 작성 중(Draft)으로 표시되어 있습니다. 작성 완료 상태로 바꿔주세요.",
            )
        )
    else:
        conditions.append(Condition("작성 상태", "pass", "작성 완료 상태입니다."))

    mergeable = snapshot.get("mergeable")
    mergeable_state = str(snapshot.get("mergeable_state") or "unknown").lower()
    if mergeable is False or mergeable_state == "dirty":
        conditions.append(
            Condition("코드 충돌", "block", "같은 부분이 서로 다르게 바뀌어 자동으로 합칠 수 없습니다.")
        )
    elif mergeable is None or mergeable_state == "unknown":
        conditions.append(Condition("코드 충돌", "wait", "GitHub가 충돌 여부를 확인하고 있습니다."))
    else:
        conditions.append(Condition("코드 충돌", "pass", "자동으로 합칠 수 있습니다."))

    behind_by = int(snapshot.get("behind_by") or 0)
    if behind_by > 0:
        conditions.append(
            Condition(
                "최신 코드 반영",
                "wait",
                f"팀 공용 코드보다 {behind_by}개 변경사항이 뒤처져 있습니다. `{expected_base}`을 먼저 반영해주세요.",
            )
        )
    else:
        conditions.append(Condition("최신 코드 반영", "pass", "팀 공용 최신 코드가 반영되어 있습니다."))

    latest_checks = latest_check_states(snapshot.get("checks", []))
    for check_name in config.get("required_checks", []):
        display_name = CHECK_DISPLAY_NAMES.get(str(check_name), str(check_name))
        check = latest_checks.get(str(check_name))
        if not check or str(check.get("status", "")).lower() != "completed":
            conditions.append(
                Condition(display_name, "wait", "GitHub가 자동 검사를 실행하고 있습니다.")
            )
            continue
        conclusion = str(check.get("conclusion") or "").lower()
        if conclusion == "success":
            conditions.append(Condition(display_name, "pass", "문제가 발견되지 않았습니다."))
        else:
            conditions.append(
                Condition(
                    display_name,
                    "block",
                    "자동 검사에서 문제가 발견됐습니다. 실패한 검사 항목을 열어 내용을 확인해주세요.",
                )
            )

    states = {condition.state for condition in conditions}
    if "block" in states:
        state = "blocked"
        status_state = "failure"
        description = "정책, 테스트 또는 충돌 문제를 해결해야 합니다."
    elif "wait" in states:
        state = "waiting"
        status_state = "pending"
        description = "자동 검사 완료 또는 최신 코드 반영을 기다립니다."
    else:
        state = "ready"
        status_state = "success"
        description = "자동 검사 조건을 모두 통과했습니다. PR 작성자가 병합해도 됩니다."

    return {
        "state": state,
        "status_state": status_state,
        "description": description,
        "conditions": [condition.__dict__ for condition in conditions],
    }


def render_comment(snapshot: Mapping[str, object], verdict: Mapping[str, object]) -> str:
    icons = {"pass": "✅", "wait": "⏳", "block": "❌"}
    state_titles = {
        "ready": "병합 가능",
        "waiting": "병합 대기",
        "blocked": "수정 필요",
    }
    lines = [
        COMMENT_MARKER,
        "## PR 병합 가능성 자동 판정",
        "",
        f"**최종 상태: `{state_titles[str(verdict['state'])]}`** — {verdict['description']}",
        "",
        "| 항목 | 상태 | 설명 |",
        "|---|---:|---|",
    ]
    for condition in verdict["conditions"]:
        icon = icons[str(condition["state"])]
        message = str(condition["message"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {condition['name']} | {icon} | {message} |")

    lines.extend(
        [
            "",
            f"판정 커밋: `{str(snapshot['head_sha'])[:12]}`",
            "",
            "### 용어 설명",
            "",
            "- **main**: 팀원들의 최종 코드를 모아두는 공용 브랜치입니다.",
            "- **작성 중(Draft)**: PR 작업이 아직 끝나지 않았다는 표시입니다.",
            "- **자동 검사(CI)**: GitHub가 파일 규칙과 테스트를 자동으로 확인하는 과정입니다.",
            "- **코드 충돌**: 같은 코드가 서로 다르게 수정되어 사람이 정리해야 하는 상태입니다.",
            "",
            "> 별도 팀원 리뷰는 필수가 아닙니다. 최종 상태가 `병합 가능`이면 PR 작성자가 직접 merge할 수 있습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


class GitHubApi:
    def __init__(self, token: str, repository: str) -> None:
        self.token = token
        self.repository = repository
        self.base_url = "https://api.github.com"

    def request(self, method: str, path: str, data: object | None = None) -> object:
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "dcc-prelim-merge-gate",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc

    def request_allow_404(self, method: str, path: str, data: object | None = None) -> object | None:
        try:
            return self.request(method, path, data)
        except RuntimeError as exc:
            if "failed: 404 " in str(exc):
                return None
            raise

    def paginate(self, path: str) -> list[dict]:
        separator = "&" if "?" in path else "?"
        page = 1
        items: list[dict] = []
        while True:
            payload = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(payload, list):
                raise RuntimeError(f"Expected list from GitHub API path {path}")
            items.extend(payload)
            if len(payload) < 100:
                return items
            page += 1

    def pull_request_numbers(self, event: Mapping[str, object]) -> list[int]:
        if isinstance(event.get("pull_request"), dict):
            return [int(event["pull_request"]["number"])]
        if isinstance(event.get("workflow_run"), dict):
            pull_requests = event["workflow_run"].get("pull_requests") or []
            return [int(pr["number"]) for pr in pull_requests]
        inputs = event.get("inputs") or {}
        if isinstance(inputs, dict) and str(inputs.get("pr_number") or "").strip():
            return [int(str(inputs["pr_number"]).strip())]
        pulls = self.paginate(f"/repos/{self.repository}/pulls?state=open&base=main")
        return [int(pr["number"]) for pr in pulls]

    def snapshot(self, number: int) -> dict:
        pull = self.request("GET", f"/repos/{self.repository}/pulls/{number}")
        if not isinstance(pull, dict):
            raise RuntimeError(f"PR #{number} response is invalid")

        for _ in range(3):
            if pull.get("mergeable") is not None:
                break
            time.sleep(2)
            pull = self.request("GET", f"/repos/{self.repository}/pulls/{number}")

        head_sha = str(pull["head"]["sha"])
        base_sha = str(pull["base"]["sha"])
        compare_path = (
            f"/repos/{self.repository}/compare/"
            f"{urllib.parse.quote(base_sha, safe='')}...{urllib.parse.quote(head_sha, safe='')}"
        )
        comparison = self.request("GET", compare_path)
        checks_payload = self.request(
            "GET", f"/repos/{self.repository}/commits/{head_sha}/check-runs?per_page=100"
        )
        checks = checks_payload.get("check_runs", []) if isinstance(checks_payload, dict) else []

        return {
            "number": number,
            "html_url": pull["html_url"],
            "base_branch": pull["base"]["ref"],
            "draft": pull.get("draft", False),
            "head_sha": head_sha,
            "mergeable": pull.get("mergeable"),
            "mergeable_state": pull.get("mergeable_state"),
            "behind_by": comparison.get("behind_by", 0),
            "checks": [
                {
                    "id": check.get("id"),
                    "name": check.get("name"),
                    "status": check.get("status"),
                    "conclusion": check.get("conclusion"),
                }
                for check in checks
            ],
        }

    def ensure_labels(self, labels: Mapping[str, str]) -> None:
        definitions = {
            labels["ready"]: ("2da44e", "자동 검사 조건을 모두 통과해 병합할 수 있습니다."),
            labels["waiting"]: ("bf8700", "자동 검사 완료 또는 최신 코드 반영을 기다립니다."),
            labels["blocked"]: ("d73a4a", "병합 전에 수정해야 할 문제가 있습니다."),
        }
        for name, (color, description) in definitions.items():
            encoded = urllib.parse.quote(name, safe="")
            current = self.request_allow_404("GET", f"/repos/{self.repository}/labels/{encoded}")
            if current is None:
                try:
                    self.request(
                        "POST",
                        f"/repos/{self.repository}/labels",
                        {"name": name, "color": color, "description": description},
                    )
                except RuntimeError as exc:
                    if "failed: 422 " not in str(exc):
                        raise

    def publish(self, snapshot: Mapping[str, object], verdict: Mapping[str, object], config: Mapping[str, object]) -> None:
        number = int(snapshot["number"])
        head_sha = str(snapshot["head_sha"])
        current = self.request("GET", f"/repos/{self.repository}/pulls/{number}")
        if not isinstance(current, dict) or str(current["head"]["sha"]) != head_sha:
            print(f"PR #{number}: 새 커밋이 감지되어 오래된 판정을 게시하지 않습니다.")
            return

        self.request(
            "POST",
            f"/repos/{self.repository}/statuses/{head_sha}",
            {
                "state": verdict["status_state"],
                "context": config["status_context"],
                "description": str(verdict["description"])[:140],
                "target_url": snapshot["html_url"],
            },
        )

        labels = config["labels"]
        self.ensure_labels(labels)
        desired = labels[verdict["state"]]
        managed_labels = set(labels.values()) | set(config.get("legacy_labels", []))
        for label in managed_labels:
            encoded = urllib.parse.quote(label, safe="")
            if label == desired:
                self.request("POST", f"/repos/{self.repository}/issues/{number}/labels", {"labels": [label]})
            else:
                self.request_allow_404(
                    "DELETE", f"/repos/{self.repository}/issues/{number}/labels/{encoded}"
                )

        body = render_comment(snapshot, verdict)
        comments = self.paginate(f"/repos/{self.repository}/issues/{number}/comments")
        existing = next(
            (
                comment
                for comment in comments
                if COMMENT_MARKER in str(comment.get("body", ""))
                and str((comment.get("user") or {}).get("login", "")).lower()
                == "github-actions[bot]"
            ),
            None,
        )
        if existing:
            self.request(
                "PATCH", f"/repos/{self.repository}/issues/comments/{existing['id']}", {"body": body}
            )
        else:
            self.request("POST", f"/repos/{self.repository}/issues/{number}/comments", {"body": body})


def event_is_stale(event: Mapping[str, object], snapshot: Mapping[str, object]) -> bool:
    workflow_run = event.get("workflow_run")
    if not isinstance(workflow_run, dict):
        return False
    for pull_request in workflow_run.get("pull_requests") or []:
        if int(pull_request.get("number") or 0) != int(snapshot["number"]):
            continue
        event_sha = str((pull_request.get("head") or {}).get("sha") or "")
        return bool(event_sha and event_sha != str(snapshot["head_sha"]))
    # Some workflow_run payloads omit the PR head object. In that case the
    # current head's check-runs are authoritative and publishing remains safe.
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-path", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluate-snapshot", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    if args.evaluate_snapshot:
        snapshot = load_json(args.evaluate_snapshot)
        print(json.dumps(evaluate_snapshot(snapshot, config), ensure_ascii=False, indent=2))
        return 0

    if not args.event_path or not args.repository:
        raise SystemExit("--event-path and --repository are required when publishing")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")

    event = load_json(args.event_path)
    api = GitHubApi(token, args.repository)
    numbers = api.pull_request_numbers(event)
    if not numbers:
        print("판정할 열린 PR이 없습니다.")
        return 0

    failures = 0
    for number in sorted(set(numbers)):
        try:
            snapshot = api.snapshot(number)
            if event_is_stale(event, snapshot):
                print(f"PR #{number}: 완료된 workflow가 이전 커밋 대상이므로 무시합니다.")
                continue
            verdict = evaluate_snapshot(snapshot, config)
            api.publish(snapshot, verdict, config)
            print(f"PR #{number}: {verdict['state']}")
        except Exception as exc:  # keep refreshing other open PRs on push/manual runs
            failures += 1
            print(f"PR #{number}: 판정 실패: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
