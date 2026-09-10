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

import uuid


def _model(obj):
    """Pydantic model -> dict, anything else unchanged."""
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else obj


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

    if diag is not None:
        d = _model(diag)
        record["diagnosis"] = {
            "summary": d.get("summary"),
            "severity": d.get("severity"),
            "citations": d.get("citations", []),
            "findings_count": len(d.get("findings", []) or []),
        }
    if review is not None:
        r = _model(review)
        booleans = ("groundedness_ok", "risk_ok", "permission_ok", "entailment_ok")
        record["review"] = {k: r.get(k) for k in (*booleans, "failures")}
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
        import config
        sink = config.get_audit_sink()
    try:
        sink.record(record)
    except Exception as e:                    # noqa: BLE001 -- see docstring
        return {"_audit": {"written": False, "run_id": record["run_id"],
                           "error": f"{type(e).__name__}: {e}"}}
    return {"_audit": {"written": True, "run_id": record["run_id"],
                       "outcome": outcome}}
