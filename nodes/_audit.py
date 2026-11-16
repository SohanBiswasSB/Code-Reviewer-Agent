"""Build and write the run record (G6). Shared by `audit` and `await_approval`.

Two callers, because no single node observes every terminal outcome:

  · `nodes/audit.py` sits on the three edges that reach END normally —
    human_review, escalate_to_human, and a completed execute.
  · `await_approval` writes the DENIED record itself. A denial resumes the graph
    into `execute`, whose PermissionError is what stops it, and that guard is
    pinned by tests precisely to prove routing is not the mechanism. Routing
    denials past `execute` to reach an audit node would remove that defence in
    depth, so the node that *observes* the denial records it instead.

When G9 adds failure edges, both should collapse onto one terminal audit path.
"""
from __future__ import annotations

import os
import uuid

import config
from errors import CONFIG_ERRORS, WRITE_ERRORS
from nodes.independent_review import effective_gates


def _model(obj):
    """Pydantic model -> dict, anything else unchanged."""
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else obj


def _runtime() -> dict:
    """Which models and plugs THIS run used, resolved in the run's own process.

    Recorded because the profile is env-driven: the monitor, a later shell or a
    teammate reading the record cannot reconstruct it, and a diagnosis cannot be
    judged without knowing which model wrote it. Never raises -- a profile that
    does not resolve must not be why the record is lost.
    """
    try:
        out = {"platform": type(config.get_platform()).__name__,
               "profile": config.get_profile(),
               "llm_provider": os.getenv("LLM_PROVIDER", "lmstudio"),
               "retriever": os.getenv("RETRIEVER", "indexed"),
               "entailer": os.getenv("ENTAILER", "lexical"),
               "model_server": config.model_server()}
        out["models"] = config.get_models()
        if out["entailer"] == "cross_encoder":
            out["models"]["entailer"] = os.getenv("ENTAILER_MODEL", "")
        return out
    except CONFIG_ERRORS as e:
        # An unresolvable profile/platform is recorded, not fatal: the record
        # must still be written. Anything outside CONFIG_ERRORS is a bug and
        # propagates -- the activity log keeps its traceback.
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _gates(pol: dict) -> dict:
    """The EFFECTIVE bars the gate compared against: the policy's value, else the
    node default that `independent_review` falls back to.

    A failure string says "0.1283 < 0.25"; without this, a record read after the
    policy changed cannot say which bar the run was held to. Recording only the
    policy's keys showed "entailment >= default" for every tenant that leaves it
    unset -- i.e. it hid the one bar that has escalated every real run so far.
    """
    try:
        return effective_gates(pol)
    except (*CONFIG_ERRORS, TypeError) as e:
        # A policy value of the wrong type (e.g. a threshold given as a list)
        # must not cost the record; the bars are then reported as unknown.
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def build_record(state: dict, outcome: str) -> dict:
    """Flatten a run into the audit record described in docs/DATA_STORAGE.md.

    Reads defensively throughout. A record is most valuable when something went
    wrong, which is exactly when a field may be absent — so a missing key must
    never be the reason the record is not written.
    """
    diag = state.get("diagnosis")
    review = state.get("review")

    record = {
        "run_id": state.get("_run_id") or str(uuid.uuid4()),
        # The RESUMABLE identity, as distinct from run_id. Layer 4 prunes
        # checkpoints by thread, so a record that cannot name its thread
        # cannot authorise deleting it.
        "thread_id": state.get("thread_id"),
        "outcome": outcome,
        "tenant": state.get("tenant"),
        "surface": state.get("surface") or state.get("_surface_name"),
        "target": state.get("target"),
        "route": state.get("route"),
        "policy_version": state.get("_policy_version"),
        "prompt_version": state.get("_prompt_version"),
        # What the InputGuard removed before any model saw the evidence (G5/T12).
        # Counts and category names only -- recording the redacted values here
        # would move the leak into the audit store rather than stop it. A rising
        # `injection_suspected` is the signal worth watching, since the guard
        # deliberately reports injection phrasing instead of rewriting it.
        "guard": state.get("_guard") or {},
        # Deterministic check results (status + location + message, no code).
        "checks": list(state.get("_checks") or []),
        # The retriever's OWN scores. This is the field that earns the layer:
        # T21 -- the grounding gate inert against real retrieval -- would have
        # been one line of this, showing every score as 1.0 on every query.
        "retrieved": [
            {"id": s.get("id"), "score": s.get("score")}
            for s in (state.get("_retrieved") or [])
            if isinstance(s, dict)
        ],
        "approved": state.get("approved"),
        # Whether the write was attributable to a human, NEVER the token itself.
        "approver_identity_present": bool(state.get("github_token")),
        "action": state.get("proposed_action"),
        "result": state.get("result"),
    }

    record["gates"] = _gates(state.get("_policy") or {})
    record["runtime"] = _runtime()

    if diag is not None:
        d = _model(diag)
        findings = [f for f in (d.get("findings") or []) if isinstance(f, dict)]
        record["diagnosis"] = {
            "summary": d.get("summary"),
            "severity": d.get("severity"),
            "citations": d.get("citations", []),
            "findings_count": len(findings),
            # The findings themselves, not just their number. An ESCALATED run
            # posts nothing and writes no review file, so until this was recorded
            # the five things the model said about the code existed nowhere once
            # the process exited -- which is exactly the run a human must look at.
            # Model output over evidence the guard already redacted (T12).
            "findings": [{k: f.get(k) for k in (
                "severity", "file", "line", "issue", "suggestion", "grounded",
                "citation_ids")} for f in findings],
            "proposed_comment": d.get("proposed_comment"),
        }
    if review is not None:
        r = _model(review)
        booleans = ("groundedness_ok", "risk_ok", "permission_ok", "entailment_ok")
        record["review"] = {k: r.get(k) for k in (*booleans, "failures")}
        # The judge's free text. Recorded as an explanation for the human, never
        # as an input to `decision` (invariant 1) -- that is recomputed below
        # from the booleans alone.
        record["review"]["reasoning"] = r.get("reasoning")
        # `decision` is a PROPERTY, so it survives on a model and vanishes on a
        # dict — and state types really do come back as dicts across a resume
        # under LANGGRAPH_STRICT_MSGPACK. Recompute from the booleans rather than
        # recording None for every run that paused at the gate.
        record["decision"] = getattr(review, "decision", None) or (
            "approve" if all(r.get(k) for k in booleans) else "escalate_to_human"
        )
    return record


def write(state: dict, outcome: str, sink=None) -> dict:
    """Write one record. `sink` is injectable for tests.

    Never raises. An audit failure must not destroy the run it is describing —
    on the `executed` path the comment has already been posted, and throwing
    would discard the fact that it happened, which is strictly worse than a
    missing record. Same reasoning as PRCommentSink's partial-success handling.

    The failure is returned in the state channel rather than swallowed, so a run
    whose record did not persist says so instead of looking clean.
    """
    record = build_record(state, outcome)
    if sink is None:
        sink = config.get_audit_sink()
    try:
        sink.record(record)
    except WRITE_ERRORS as e:
        # Disk or serialisation failure: said in state, never swallowed silently.
        return {"_audit": {"written": False, "run_id": record["run_id"],
                           "error": f"{type(e).__name__}: {e}"}}
    return {"_audit": {"written": True, "run_id": record["run_id"],
                       "outcome": outcome}}
