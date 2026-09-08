"""Shared graph state. Surface-agnostic: `target` is a PR ref (code review) OR a
date (log review); `evidence` is whatever that surface gathered.

**Every key a node writes must be declared here.** LangGraph keeps only the
channels the state schema names, so an undeclared key is silently dropped rather
than erroring — which is how `_policy_version` and `_surface_name` came back
`None` from a run whose `resolve_tenant` had definitely returned them. Worth
remembering: a missing field here fails quietly, not loudly.

Underscore-prefixed keys are internal plumbing rather than part of the request or
the result. They still travel through the checkpoint, so they must stay
serialisable — which is why `_surface_name` is a string and not the Surface
object (T15).
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class TriageState(TypedDict, total=False):
    # ── request ──────────────────────────────────────────────────────────────
    tenant: str                 # which codebase is executing (resolves prompts + policy)
    surface: Literal["code_review", "log_review", "issue_triage"]
    target: str                 # PR/commit ref, issue url, or a date (YYYY-MM-DD)
    github_token: str | None    # the approver's own OAuth token, when present

    # ── resolved once, by resolve_tenant, and pinned for the whole run ───────
    _policy: dict               # tenant+surface guardrails (thresholds, allowlists)
    _surface_name: str          # a NAME, not the object: checkpoints must serialise
    _policy_version: str        # content hash, so a resumed run cannot mix versions

    # ── produced as the run proceeds ─────────────────────────────────────────
    _raw: Any                   # unvalidated provider payload
    evidence: Any               # IssueEvidence | CodeChange | LogWindow
    route: Literal["deterministic", "ai_investigation", "human_review"]
    _retrieved: Any             # snippets, WITH their calibrated relevance scores
    diagnosis: Any
    review: Any

    # ── the gate ─────────────────────────────────────────────────────────────
    proposed_action: dict       # built ONCE, posted verbatim, never recomputed
    _interrupt_payload: dict    # what the human was actually shown
    approved: bool              # written ONLY by await_approval, from the resume
    result: dict
