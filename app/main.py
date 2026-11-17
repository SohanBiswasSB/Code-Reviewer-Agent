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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langgraph.types import Command

import config
from app import activity
from app.deps import get_graph, open_checkpointer, set_graph
from app.doctor import run_checks
from app.graph import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the checkpointer's lifecycle for the life of the process.

    `AsyncPostgresSaver` is an async context manager; entering it here is what
    makes the durable-approval path actually work, and closing it on shutdown is
    what returns the connections. Building the graph lazily per request could
    not do either -- see app/deps.py.
    """

    activity.capture_library_logs()
    async with open_checkpointer() as saver:
        set_graph(build_graph(checkpointer=saver))
        try:
            yield
        finally:
            set_graph(None)


app = FastAPI(title="Code Reviewer Agent (plug-and-play)", lifespan=lifespan)

CORRELATION_HEADER = "X-Correlation-Id"


@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    """One `request` record per call, correlated; the id is echoed on the response.

    The Masterclass CorrelationIdMiddleware's contract: accept an inbound
    X-Correlation-Id or mint one, bind it for everything logged downstream, and
    echo it so a caller can quote it when reporting a problem.
    """
    cid = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
    with activity.correlation(cid), activity.ActivityLogger(
            "request", f"{request.method} {request.url.path}",
            metadata={"client": request.client.host if request.client else None}) as alog:
        response = await call_next(request)
        alog.set_response(response.status_code)
        if response.status_code >= 500:
            alog.record_error(message=f"HTTP {response.status_code}", failure_source="internal")
    response.headers[CORRELATION_HEADER] = cid
    return response


@app.get("/healthz")
def healthz():
    """Liveness only: the process answers. Model server and tenants are `/doctor`'s job."""
    return {"ok": True}


@app.get("/registered")
def registered():
    """What is plugged into THIS deployment: adapters, surfaces, platform, models."""
    return config.registered()


@app.get("/doctor")
def doctor():
    """The same checks as `python -m app.doctor`, minus the gate suite, over HTTP."""
    return run_checks(gates=False).as_dict()


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

    if surface not in config.enabled_surfaces():
        raise HTTPException(404, f"surface '{surface}' is not served by this deployment")

    target = body.get("target")
    if not target:
        raise HTTPException(422, "body.target is required")

    # TODO(G3): authenticate the caller and authorise caller -> tenant. Right now
    # any caller may run any tenant, which is a real gap and not a stub: see
    # docs/ARCHITECTURE_GAPS.md G3. resolve_tenant enforces enabled_surfaces,
    # which is a different (and lesser) check.

    thread_id = body.get("thread_id") or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        state = await get_graph().ainvoke(
            {
                "tenant": body.get("tenant", "_default"),
                "surface": surface,
                "target": target,
                # Mirrored into state so the audit record can name its thread;
                # the graph itself keys on config, not on this.
                "thread_id": thread_id,
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



    cfg = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()

    snapshot = await graph.aget_state(cfg)
    if not snapshot.next:
        raise HTTPException(409, f"thread '{thread_id}' is not awaiting approval")

    activity.log_event("approval", "approval.granted" if approved else "approval.denied",
                       severity=activity.INFO if approved else activity.WARNING,
                       metadata={"thread_id": thread_id, "via": "api"})
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
