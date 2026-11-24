"""Shared fakes for the tests that drive the REAL graph with only GitHub stubbed.

test_graph_resume.py and test_audit_and_reports.py both need the same three
things: GitHub's network calls replaced by recorders, an issue-shaped surface the
gate can actually be reached through, and a graph on a fresh checkpointer. They
used to carry identical copies, which is how two suites quietly come to test two
different fakes; one module means one definition.

Plain functions, not fixtures: each test module wraps them in its own named
fixture, so a test requests `posted` / `issue_surface` without a module-level
name shadowing the argument.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.memory import MemorySaver

import registry
from app.graph import build_graph
from tools import github_client

ISSUE_TITLE = "OAuth callback returns 500 when state param missing"


def stub_github(monkeypatch) -> list[str]:
    """Capture what would have been written, and guarantee no real call escapes.

    Returns the list every posted comment body is appended to, in order.
    """
    sent: list[str] = []
    monkeypatch.setattr(github_client, "fetch_issue", lambda url, token=None: {
        "html_url": url,
        "title": ISSUE_TITLE,
        "body": "help",                      # sparse -> ai_investigation
        "labels": [{"name": "bug"}],
    })
    monkeypatch.setattr(github_client, "get_check_runs", lambda url, token=None: [])
    monkeypatch.setattr(github_client, "post_comment",
                        lambda url, body, token=None: (sent.append(body), {"posted": body})[1])
    monkeypatch.setattr(github_client, "add_label",
                        lambda url, label, token=None: {"labelled": label})
    return sent


class IssueShaped:
    """A `code_review` surface whose classify matches ISSUE-shaped evidence.

    The real `code_review` surface routes on `touched_areas`, which issue
    evidence does not carry -- it therefore returns `human_review` and never
    reaches the gate. That mismatch is T5; it fails safe, but it means the gate
    cannot be exercised through that surface yet.
    """

    name = "code_review"
    source_name, sink_name = "github", "pr_comment"
    diagnosis_prompt = judge_prompt = "diagnosis"

    def build_query(self, ev):
        """The issue title is the whole query: issue evidence has no diff."""
        return ev.get("title", "")

    def classify(self, ev):
        """Failing CI is deterministic; a sparse issue needs the AI path; else a human."""
        if ev.get("has_failing_ci"):
            return "deterministic"
        return "ai_investigation" if ev.get("is_information_sparse") else "human_review"


@contextmanager
def issue_surface_registered() -> Iterator[type]:
    """Register `IssueShaped` as `code_review` for the block, then put the real one back.

    Through the registry's own decorator rather than its private dict, so the swap
    is exactly what a plugged-in surface does -- and the restore is in `finally`,
    because a leaked stub would make every later test exercise the fake while
    appearing to use the real surface.
    """
    original = registry.get_surface("code_review")
    registry.surface("code_review")(IssueShaped)
    try:
        yield IssueShaped
    finally:
        registry.surface("code_review")(original)


def fresh_graph():
    """The real graph on its own in-memory checkpointer, so tests never share a thread."""
    return build_graph(checkpointer=MemorySaver())
