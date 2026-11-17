"""The HTTP API's contract: one request model and one response model per route.

Three rules from the house API standard shape every class here:

  * Strict input. Request models forbid undeclared fields (`extra="forbid"`)
    and use strict types, so `{"approved": "yes"}` is a 400, not a guess.
  * Exact output. Each route declares its response model, and FastAPI sends
    exactly those fields -- a value added to the graph's state never leaks out
    by accident.
  * One envelope. Every body, success or error, is
    `{"success": bool, "message": str, "data": <the route's model> | null}`.

Names carry V1 because the paths do (/v1/...). A breaking change is a V2 model
beside these, never an edit to them.
"""
from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool

T = TypeVar("T")


class StrictModelV1(BaseModel):
    """Base for request models: undeclared fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


class EnvelopeV1(BaseModel, Generic[T]):
    """The body of every response, success or error."""

    success: bool
    message: str = ""
    data: T | None = None


# -- requests --------------------------------------------------------------------
class RunRequestV1(StrictModelV1):
    """POST /v1/run/{surface}: what to review, for which codebase."""

    tenant: str = Field(default="_default", min_length=1)
    target: str = Field(min_length=1, description="pr_ref | issue_url | YYYY-MM-DD | git range")
    github_token: str | None = None
    thread_id: str | None = Field(default=None, min_length=1)


class ApproveRequestV1(StrictModelV1):
    """POST /v1/approve: the human's answer for one paused run.

    `approved` is a StrictBool: "yes", 1 or "true" are a 400, so a malformed body
    can neither approve nor silently deny. `execute` still re-checks `is True`
    on its own, which is the gate's guarantee and does not depend on this model.
    """

    thread_id: str = Field(min_length=1)
    approved: StrictBool


# -- responses -------------------------------------------------------------------
class HealthV1(BaseModel):
    """GET /healthz: the process answers."""

    ok: bool


class SelectedV1(BaseModel):
    """The adapters this deployment's environment selects."""

    llm: str
    retriever: str
    reranker: str
    tenant_store: str
    audit: str
    guard: str
    tracer: str
    surfaces: list[str]
    platform: str
    profile: str


class HardwareV1(BaseModel):
    """Where the models run. `runtime`'s fields depend on the platform (mlx,
    cuda, cpu, a remote host), so it stays an open mapping."""

    detected: str
    runtime: dict[str, Any]
    models: dict[str, str]
    embedding_dims: int


class RegisteredV1(BaseModel):
    """GET /v1/registered: what is plugged into this deployment."""

    adapters: dict[str, list[str]]
    surfaces: list[str]
    selected: SelectedV1
    hardware: HardwareV1


class DoctorCheckV1(BaseModel):
    """One line of the environment check."""

    status: str
    check: str
    detail: str


class DoctorReportV1(BaseModel):
    """GET /v1/doctor: every check, and how many failed or warned."""

    checks: list[DoctorCheckV1]
    failed: int
    warned: int


class RunResultV1(BaseModel):
    """POST /v1/run/{surface}: where the run stopped.

    `awaiting_approval` carries what a human needs to decide (proposed_action,
    reasoning); `ended` carries the route and the review's decision.
    """

    thread_id: str
    status: Literal["awaiting_approval", "ended"]
    route: str | None = None
    review: str | None = None
    policy_version: str | None = None
    proposed_action: dict[str, Any] | None = None
    reasoning: str | None = None


class ApproveResultV1(BaseModel):
    """POST /v1/approve: what resuming the run did."""

    thread_id: str
    approved: bool | None
    status: Literal["executed", "ended", "refused"]
    result: dict[str, Any] | None = None
    detail: str | None = None


class RunStatusV1(BaseModel):
    """GET /v1/runs/{thread_id}: what a run is waiting on."""

    thread_id: str
    next: list[str]
    awaiting_approval: bool
    policy_version: str | None = None
    proposed_action: dict[str, Any] | None = None
