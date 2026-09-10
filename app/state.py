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
    thread_id: str              # the resumable identity, mirrored from config so the
                                # audit record can name the thread a prune job must keep
    github_token: str | None    # the approver's own OAuth token, when present

    # ── resolved once, by resolve_tenant, and pinned for the whole run ───────
    _run_id: str                # correlates every audit record written for this run
    _policy: dict               # tenant+surface guardrails (thresholds, allowlists)
    _surface_name: str          # a NAME, not the object: checkpoints must serialise
    _policy_version: str        # content hash, so a resumed run cannot mix versions
    _prompts: dict              # role -> composed prompt text, pinned at run start
    _prompt_version: str        # content hash of the above (G8)

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
    _audit: dict                # did the run record persist? (G6) -- never the record itself


# ── what may be reconstructed from a checkpoint (T15) ────────────────────────
# Every non-primitive type that enters a channel above, listed as
# (module, qualname). LangGraph's serialiser refuses to rebuild a type it was
# not told about: it warns today, blocks under LANGGRAPH_STRICT_MSGPACK=true,
# and will block by default in a future release. A blocked type does not raise
# -- it comes back as a plain `dict`, so a resumed run silently loses the model
# it was written with, and `review.decision` or `evidence.has_failing_ci` turns
# into an AttributeError far from the cause.
#
# It lives next to TriageState because it is the same invariant as the module
# docstring's: a type that enters a channel must be declared, or it fails
# quietly rather than loudly. Adding a model to `evidence`/`diagnosis`/`review`
# means adding it here, and `tests/test_graph_resume.py` pins that.
#
# Nested models do NOT need their own entry -- registering the outer type is
# enough to rebuild `Diagnosis.citations[]` as `Citation` -- but they are listed
# anyway, because any of them could become a channel value directly and the
# cost of an extra line is nothing against the cost of the silent failure.
CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("schemas.evidence", "CheckRun"),
    ("schemas.evidence", "LinkedPR"),
    ("schemas.evidence", "IssueEvidence"),
    ("schemas.evidence", "CodeChange"),
    ("schemas.evidence", "LogSample"),
    ("schemas.evidence", "LogCandidate"),
    ("schemas.evidence", "LogWindow"),
    ("schemas.diagnosis", "Citation"),
    ("schemas.diagnosis", "Finding"),
    ("schemas.diagnosis", "Diagnosis"),
    ("schemas.diagnosis", "ReviewResult"),
)


def checkpoint_serde():
    """The serialiser every checkpointer must use.

    Built here rather than left to default so the allowlist above is applied
    everywhere a checkpointer is constructed -- `app/deps.py` for both backends,
    and the tests. A checkpointer created with the bare default serialiser is
    the failure mode this exists to prevent, and it is invisible until a run
    resumes.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
