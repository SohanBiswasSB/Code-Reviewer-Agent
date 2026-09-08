"""Shared graph state. Surface-agnostic: `target` is a PR ref (code review) OR a
date (log review); `evidence` is whatever that surface gathered."""
from __future__ import annotations
from typing import TypedDict, Literal, Any


class TriageState(TypedDict, total=False):
    tenant: str                 # which org/project is executing (resolves prompts + policy)
    surface: Literal["code_review", "log_review", "issue_triage"]
    target: str                 # PR/commit ref, issue url, or a date (YYYY-MM-DD)
    _policy: dict               # tenant+surface guardrails, resolved at request time
    github_token: str | None
    evidence: Any               # IssueEvidence | CodeChange | LogWindow
    route: Literal["deterministic", "ai_investigation", "human_review"]
    diagnosis: Any
    review: Any
    proposed_action: dict
    approved: bool
    result: dict
    _raw: Any
    _retrieved: Any
