"""FastAPI surface — surface-agnostic and plugin-driven.

`/run/{surface}` drives any registered surface up to the approval gate and
returns what a human needs in order to decide. `/approve` resumes that exact
run on its `thread_id`.

Endpoints are async and use `ainvoke` on purpose: `AsyncPostgresSaver` — the
only checkpointer that makes a paused approval durable — requires the async
path, so writing these synchronously would have quietly locked the service to
the in-memory saver.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Code Reviewer Agent (plug-and-play)")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/registered")
def registered():
    """What is plugged into THIS deployment: adapters, surfaces, platform, models."""
    import config
    return config.registered()


@app.get("/doctor")
def doctor():
    """The same checks as `python -m app.doctor`, over HTTP."""
    from app.doctor import Report, _check_lmstudio, _check_local_stack, _check_platform
    r = Report()
    models = _check_platform(r)
    _check_local_stack(r)
    if models:
        _check_lmstudio(r, models)
    return r.as_dict()


@app.post("/run/{surface}")
async def run_surface(surface: str, body: dict):
    """Drive a surface to the approval gate.

    body: {"tenant": "...", "target": "<pr_ref | issue_url | YYYY-MM-DD>",
           "github_token": "..."}

    Returns {thread_id, status, proposed_action, reasoning, policy_version}.

    `status` is either "awaiting_approval" (suspended at the gate — call
    /approve with the thread_id) or "ended" (the run finished without needing a
    human, which for `human_review` and `escalate_to_human` is the whole point).
    """
    import config

    if surface not in config.enabled_surfaces():
        raise HTTPException(404, f"surface '{surface}' is not served by this deployment")

    target = body.get("target")
    if not target:
        raise HTTPException(422, "body.target is required")

    # TODO(G3): authenticate the caller and authorise caller -> tenant. Right now
    # any caller may run any tenant, which is a real gap and not a stub: see
    # docs/ARCHITECTURE_GAPS.md G3. resolve_tenant enforces enabled_surfaces,
    # which is a different (and lesser) check.
    from app.deps import get_graph

    thread_id = body.get("thread_id") or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        state = await get_graph().ainvoke(
            {
                "tenant": body.get("tenant", "_default"),
                "surface": surface,
                "target": target,
                "github_token": body.get("github_token"),
            },
            config=cfg,
        )
    except PermissionError as e:
        # resolve_tenant refusing a surface the tenant has not enabled
        raise HTTPException(403, str(e)) from None

    interrupts = state.get("__interrupt__") or ()
    if not interrupts:
        return {
            "thread_id": thread_id,
            "status": "ended",
            "route": state.get("route"),
            "review": getattr(state.get("review"), "decision", None),
            "policy_version": state.get("_policy_version"),
        }

    payload = interrupts[0].value
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval",
        "route": state.get("route"),
        "proposed_action": payload["proposed_action"],
        "reasoning": payload["reasoning"],
        "policy_version": payload["policy_version"],
    }


@app.post("/approve")
async def approve(body: dict):
    """Resume a paused run. body: {"thread_id": "...", "approved": true}.

    `approved` must be a real boolean. Anything else is treated as a refusal —
    `execute` re-checks `is not True` independently, so a malformed body cannot
    become an approval by being truthy.
    """
    thread_id = body.get("thread_id")
    if not thread_id:
        raise HTTPException(422, "body.thread_id is required")
    approved = body.get("approved") is True

    from langgraph.types import Command

    from app.deps import get_graph

    cfg = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()

    snapshot = await graph.aget_state(cfg)
    if not snapshot.next:
        raise HTTPException(409, f"thread '{thread_id}' is not awaiting approval")

    try:
        state = await graph.ainvoke(Command(resume=approved), config=cfg)
    except PermissionError as e:
        # The expected path for a denial: execute refuses rather than posting.
        return {"thread_id": thread_id, "approved": False,
                "status": "refused", "detail": str(e)}

    return {
        "thread_id": thread_id,
        "approved": state.get("approved"),
        "status": "executed" if state.get("result") else "ended",
        "result": state.get("result"),
    }


@app.get("/runs/{thread_id}")
async def run_status(thread_id: str):
    """What is this run waiting on? Useful after a restart, to find pending gates."""
    from app.deps import get_graph
    snapshot = await get_graph().aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot.created_at is None:
        raise HTTPException(404, f"no such thread '{thread_id}'")
    return {
        "thread_id": thread_id,
        "next": list(snapshot.next),
        "awaiting_approval": bool(snapshot.next),
        "policy_version": snapshot.values.get("_policy_version"),
        "proposed_action": snapshot.values.get("proposed_action"),
    }

# GET /auth/github/callback -> store the approver's OAuth token on the thread, so
# execute() writes as the person who approved rather than a shared credential.
