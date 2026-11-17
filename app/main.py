"""FastAPI surface: surface-agnostic, plugin-driven, versioned.

`POST /v1/run/{surface}` drives any registered surface up to the approval gate
and returns what a human needs in order to decide. `POST /v1/approve` resumes
that exact run on its `thread_id`. The request and response models, and the
reasons each is strict, are in app/api_models.py.

Every response, success or error, is the `{success, message, data}` envelope:
the exception handlers below wrap FastAPI's own errors too, so a client parses
one shape. An undeclared or mistyped field is a 400 (the house rule), not
FastAPI's default 422. Only `/healthz` sits outside /v1: a load balancer's probe
must not move when the API version does.

Endpoints are async and use `ainvoke` on purpose: `AsyncPostgresSaver` -- the
only checkpointer that makes a paused approval durable -- requires the async
path, so writing these synchronously would have quietly locked the service to
the in-memory saver.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.types import Command
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from app import activity
from app.api_models import (
    ApproveRequestV1,
    ApproveResultV1,
    DoctorReportV1,
    EnvelopeV1,
    HealthV1,
    RegisteredV1,
    RunRequestV1,
    RunResultV1,
    RunStatusV1,
)
from app.deps import get_graph, open_checkpointer, set_graph
from app.doctor import run_checks
from app.graph import build_graph

CORRELATION_HEADER = "X-Correlation-Id"


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
v1 = APIRouter(prefix="/v1")


def envelope(data=None, message: str = "") -> dict:
    """A successful body: the route's data inside the house envelope."""
    return {"success": True, "message": message, "data": data}


def _error(status: int, message: str) -> JSONResponse:
    """A failed body, in the same envelope as a successful one."""
    return JSONResponse(status_code=status,
                        content={"success": False, "message": message, "data": None})


@app.exception_handler(StarletteHTTPException)
async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Our own HTTPExceptions and Starlette's (404 route, 405 method), enveloped."""
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """A body that breaks its model: undeclared, missing or mistyped fields.

    400, not FastAPI's 422, per the house rule. The message names every field and
    why, so a caller can fix the request without reading this code.
    """
    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
                         for e in exc.errors())
    return _error(400, f"invalid request: {problems}")


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


@app.get("/healthz", name="healthz", response_model=EnvelopeV1[HealthV1])
def healthz():
    """Liveness only: the process answers. Model server and tenants are `/v1/doctor`'s job."""
    return envelope({"ok": True})


@v1.get("/registered", name="registered-v1", response_model=EnvelopeV1[RegisteredV1])
def registered_v1():
    """What is plugged into THIS deployment: adapters, surfaces, platform, models."""
    return envelope(config.registered())


@v1.get("/doctor", name="doctor-v1", response_model=EnvelopeV1[DoctorReportV1])
def doctor_v1():
    """The same checks as `python -m app.doctor`, minus the gate suite, over HTTP."""
    report = run_checks(gates=False).as_dict()
    return envelope(report, "" if not report["failed"] else
                    f"{report['failed']} check(s) failed: the agent cannot run as configured")


def _run_result(thread_id: str, state: dict) -> dict:
    """Where a run stopped: paused at the approval gate, or ended on its own."""
    interrupts = state.get("__interrupt__") or ()
    if not interrupts:
        # Ended without needing a human -- for `human_review` and
        # `escalate_to_human` that is the whole point of the route.
        return {"thread_id": thread_id, "status": "ended", "route": state.get("route"),
                "review": getattr(state.get("review"), "decision", None),
                "policy_version": state.get("_policy_version")}
    payload = interrupts[0].value
    return {"thread_id": thread_id, "status": "awaiting_approval",
            "route": state.get("route"), "proposed_action": payload["proposed_action"],
            "reasoning": payload["reasoning"], "policy_version": payload["policy_version"]}


@v1.post("/run/{surface}", name="run-surface-v1", response_model=EnvelopeV1[RunResultV1])
async def run_surface_v1(surface: str, body: RunRequestV1):
    """Drive a surface to the approval gate.

    `data.status` is "awaiting_approval" (suspended at the gate: answer it with
    POST /v1/approve and the thread_id) or "ended" (finished without a human).
    """
    if surface not in config.enabled_surfaces():
        raise HTTPException(404, f"surface '{surface}' is not served by this deployment")

    # TODO(G3): authenticate the caller and authorise caller -> tenant. Right now
    # any caller may run any tenant, which is a real gap and not a stub: see
    # docs/ARCHITECTURE_GAPS.md G3. resolve_tenant enforces enabled_surfaces,
    # which is a different (and lesser) check.
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        state = await get_graph().ainvoke(
            {"tenant": body.tenant, "surface": surface, "target": body.target,
             # Mirrored into state so the audit record can name its thread; the
             # graph itself keys on config, not on this.
             "thread_id": thread_id, "github_token": body.github_token},
            config={"configurable": {"thread_id": thread_id}})
    except PermissionError as e:
        # resolve_tenant refusing a surface the tenant has not enabled
        raise HTTPException(403, str(e)) from None
    result = _run_result(thread_id, state)
    return envelope(result, "waiting for approval" if result["status"] == "awaiting_approval"
                    else "the run ended without needing approval")


@v1.post("/approve", name="approve-v1", response_model=EnvelopeV1[ApproveResultV1])
async def approve_v1(body: ApproveRequestV1):
    """Resume a paused run with a human's answer.

    `approved` is a strict boolean (see ApproveRequestV1), and `execute`
    re-checks `is True` independently, so nothing but a real `true` can post.
    """
    cfg = {"configurable": {"thread_id": body.thread_id}}
    graph = get_graph()
    snapshot = await graph.aget_state(cfg)
    if not snapshot.next:
        raise HTTPException(409, f"thread '{body.thread_id}' is not awaiting approval")

    activity.log_event("approval", "approval.granted" if body.approved else "approval.denied",
                       severity=activity.INFO if body.approved else activity.WARNING,
                       metadata={"thread_id": body.thread_id, "via": "api"})
    try:
        state = await graph.ainvoke(Command(resume=body.approved), config=cfg)
    except PermissionError as e:
        # The expected path for a denial: execute refuses rather than posting.
        return envelope({"thread_id": body.thread_id, "approved": False,
                         "status": "refused", "detail": str(e)}, "declined: nothing was posted")
    executed = bool(state.get("result"))
    return envelope({"thread_id": body.thread_id, "approved": state.get("approved"),
                     "status": "executed" if executed else "ended",
                     "result": state.get("result")},
                    "the review was written" if executed else "the run ended")


@v1.get("/runs/{thread_id}", name="run-status-v1", response_model=EnvelopeV1[RunStatusV1])
async def run_status_v1(thread_id: str):
    """What is this run waiting on? Useful after a restart, to find pending gates."""
    snapshot = await get_graph().aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot.created_at is None:
        raise HTTPException(404, f"no such thread '{thread_id}'")
    return envelope({"thread_id": thread_id, "next": list(snapshot.next),
                     "awaiting_approval": bool(snapshot.next),
                     "policy_version": snapshot.values.get("_policy_version"),
                     "proposed_action": snapshot.values.get("proposed_action")})


app.include_router(v1)

# GET /v1/auth/github/callback -> store the approver's OAuth token on the thread,
# so execute() writes as the person who approved rather than a shared credential.
