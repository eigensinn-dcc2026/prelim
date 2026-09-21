from __future__ import annotations

from pathlib import Path

from scripts.pr_verdict import (
    COMMENT_MARKER,
    GitHubApi,
    evaluate_snapshot,
    event_is_stale,
    load_json,
    render_comment,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / ".github" / "merge-gate.json"


def config() -> dict:
    return load_json(CONFIG_PATH)


def passing_snapshot() -> dict:
    return {
        "number": 24,
        "html_url": "https://github.com/eigensinn-dcc2026/prelim/pull/24",
        "base_branch": "main",
        "draft": False,
        "head_sha": "head-sha",
        "mergeable": True,
        "mergeable_state": "clean",
        "behind_by": 0,
        "checks": [
            {
                "id": 10,
                "name": "PR CI / policy",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "id": 11,
                "name": "PR CI / tests",
                "status": "completed",
                "conclusion": "success",
            },
        ],
    }


def test_all_conditions_produce_ready_verdict() -> None:
    verdict = evaluate_snapshot(passing_snapshot(), config())

    assert verdict["state"] == "ready"
    assert verdict["status_state"] == "success"


def test_human_review_is_not_required_for_any_author_or_file() -> None:
    snapshot = passing_snapshot()
    snapshot.update(
        {
            "author": "sooyahh",
            "files": [".github/workflows/pr-ci.yml"],
            "reviews": [],
        }
    )

    verdict = evaluate_snapshot(snapshot, config())

    assert verdict["state"] == "ready"
    assert all("승인" not in condition["name"] for condition in verdict["conditions"])


def test_failed_check_or_conflict_blocks() -> None:
    failed = passing_snapshot()
    failed["checks"][0]["conclusion"] = "failure"
    conflict = passing_snapshot()
    conflict.update({"mergeable": False, "mergeable_state": "dirty"})

    assert evaluate_snapshot(failed, config())["state"] == "blocked"
    assert evaluate_snapshot(conflict, config())["state"] == "blocked"


def test_behind_branch_and_draft_wait() -> None:
    behind = passing_snapshot()
    behind["behind_by"] = 2
    draft = passing_snapshot()
    draft["draft"] = True

    assert evaluate_snapshot(behind, config())["state"] == "waiting"
    assert evaluate_snapshot(draft, config())["state"] == "waiting"


def test_stale_workflow_run_is_ignored() -> None:
    snapshot = passing_snapshot()

    old_event = {
        "workflow_run": {
            "pull_requests": [{"number": 24, "head": {"sha": "old-sha"}}]
        }
    }
    current_event = {
        "workflow_run": {
            "pull_requests": [{"number": 24, "head": {"sha": "head-sha"}}]
        }
    }

    assert event_is_stale(old_event, snapshot) is True
    assert event_is_stale(current_event, snapshot) is False


def test_comment_contains_marker_and_head_sha() -> None:
    snapshot = passing_snapshot()
    body = render_comment(snapshot, evaluate_snapshot(snapshot, config()))

    assert COMMENT_MARKER in body
    assert "head-sha" in body
    assert "병합 가능" in body
    assert "자동 검사(CI)" in body
    assert "별도 팀원 리뷰는 필수가 아닙니다" in body


def test_config_and_workflow_do_not_require_human_review() -> None:
    workflow_path = CONFIG_PATH.parent / "workflows" / "pr-verdict.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    settings = config()

    assert "required_approvals" not in settings
    assert "owner" not in settings
    assert "pull_request_review" not in workflow
    assert settings["labels"]["waiting"] == "병합-대기"


class FakeApi(GitHubApi):
    def __init__(self, comment_author: str = "github-actions[bot]") -> None:
        super().__init__("unused", "owner/repo")
        self.calls: list[tuple[str, str, object | None]] = []
        self.comment_author = comment_author

    def request(self, method: str, path: str, data: object | None = None) -> object:
        self.calls.append((method, path, data))
        if method == "GET" and path.endswith("/pulls/24"):
            return {"head": {"sha": "head-sha"}}
        return {}

    def request_allow_404(self, method: str, path: str, data: object | None = None) -> object | None:
        self.calls.append((method, path, data))
        return {}

    def paginate(self, path: str) -> list[dict]:
        if path.endswith("/comments"):
            return [
                {
                    "id": 99,
                    "body": COMMENT_MARKER + "\nold",
                    "user": {"login": self.comment_author},
                }
            ]
        return []

    def ensure_labels(self, labels: dict[str, str]) -> None:
        return None


def test_publish_updates_existing_comment_instead_of_duplicating() -> None:
    api = FakeApi()
    snapshot = passing_snapshot()
    verdict = evaluate_snapshot(snapshot, config())

    api.publish(snapshot, verdict, config())

    assert any(method == "PATCH" and path.endswith("/comments/99") for method, path, _ in api.calls)
    assert not any(
        method == "POST" and path.endswith("/issues/24/comments") for method, path, _ in api.calls
    )
    assert any(
        method == "DELETE" and path.endswith("/labels/merge-waiting")
        for method, path, _ in api.calls
    )


def test_publish_does_not_edit_a_users_marker_comment() -> None:
    api = FakeApi(comment_author="author")
    snapshot = passing_snapshot()
    verdict = evaluate_snapshot(snapshot, config())

    api.publish(snapshot, verdict, config())

    assert any(
        method == "POST" and path.endswith("/issues/24/comments") for method, path, _ in api.calls
    )
    assert not any(method == "PATCH" and path.endswith("/comments/99") for method, path, _ in api.calls)
