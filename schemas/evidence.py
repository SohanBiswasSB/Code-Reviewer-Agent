"""Validated evidence. Pydantic v2, for two reasons that are not style choices.

**Validation at the boundary (T11).** `normalize_evidence`'s own docstring said
bad provider data should fail *there, loudly*. With stdlib dataclasses it could
not: a missing field or a wrong type propagated into the graph and surfaced
somewhere unrelated. Now malformed provider JSON raises at the edge.

**Checkpoint serialisation (T15).** LangGraph warned on every state type that
"deserializing unregistered type ... will be blocked in a future version". On a
stricter release that makes **every paused run unresumable** — catastrophic for a
design whose safety property is a durable pause. Pydantic models are a type
LangGraph's serialiser handles natively.

The computed properties stay properties: `has_failing_ci` and
`is_information_sparse` are derived facts, not stored fields, and the eval suite
pins the routing that depends on them.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SPARSE_BODY_CHARS = 140  # bodies shorter than this with no repro are "information sparse"


class CheckRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    conclusion: str  # "success" | "failure" | "neutral" | ...


class LinkedPR(BaseModel):
    model_config = ConfigDict(extra="ignore")

    number: int
    state: str


class IssueEvidence(BaseModel):
    """One issue, validated.

    `extra="forbid"` on purpose: a provider field we did not expect is a signal
    that the payload shape changed, and finding that out here beats finding out
    when a downstream node reads something that is no longer there.
    """
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str = ""
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    check_runs: list[CheckRun] = Field(default_factory=list)
    linked_prs: list[LinkedPR] = Field(default_factory=list)

    @property
    def has_failing_ci(self) -> bool:
        return any(c.conclusion == "failure" for c in self.check_runs)

    @property
    def is_information_sparse(self) -> bool:
        body = (self.body or "").strip()
        has_repro = "repro" in body.lower() or "steps to reproduce" in body.lower()
        return len(body) < SPARSE_BODY_CHARS and not has_repro


class CodeChange(BaseModel):
    """A proposed change, as the code_review surface normalises it.

    Was a bare dict. Typed now for the same two reasons as above — and because
    `classify` routes on `touched_areas` and `has_failing_ci`, so a typo in
    either key would silently route everything to `human_review`.
    """
    model_config = ConfigDict(extra="forbid")

    url: str = ""
    title: str = ""
    body: str = ""
    diff: str = ""
    files: list[str] = Field(default_factory=list)
    # Already filtered to the risky areas by the surface's normalize -- classify
    # only asks whether it is empty.
    touched_areas: list[str] = Field(default_factory=list)
    has_failing_ci: bool = False
    is_trivial: bool = False
