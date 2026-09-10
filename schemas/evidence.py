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


# ── log review (Layer 0: a BOUNDED window, never the day's lines) ────────────
MAX_LINE_CHARS = 400        # one log line, truncated before it can enter state


class LogSample(BaseModel):
    """One retained example line, carrying where it came from.

    `ref` + `line_no` are what make the sample checkable: a human reading the
    report can go back to the export rather than trusting a copied string. The
    text is kept because `generate_diagnosis` needs something to reason over —
    G4's "sample only the candidates into diagnose" — but it is truncated and
    capped so a day of logs can never reach the checkpoint.
    """
    model_config = ConfigDict(extra="forbid")

    service: str = ""
    line_no: int = 0
    ref: str = ""
    text: str = ""


class LogCandidate(BaseModel):
    """An anomaly a Detector flagged, aggregated across the whole day."""
    model_config = ConfigDict(extra="forbid")

    signal: str
    severity: str = "medium"
    count: int = 0
    services: list[str] = Field(default_factory=list)
    samples: list[LogSample] = Field(default_factory=list)


class LogWindow(BaseModel):
    """A day of logs, reduced to something that fits in a checkpoint.

    This type is the whole point of Layer 0. LangGraph re-serialises a channel's
    entire value on every super-step, so a `LogWindow` holding ~2M raw lines (G4's
    figure for one day) would be rewritten repeatedly, per run. What it holds
    instead is counts, detector candidates and a capped sample — kilobytes — plus
    a `ref` to the export that remains the source of record.

    `digest` is over the stream, so re-running the same day is detectable. That is
    the natural idempotency key for G7.
    """
    model_config = ConfigDict(extra="forbid")

    date: str = ""
    ref: str = ""
    digest: str = ""
    line_count: int = 0
    services: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    candidates: list[LogCandidate] = Field(default_factory=list)
    # True when a cap was hit, so a reader knows the sample is partial rather
    # than assuming it saw everything the detectors found.
    truncated: bool = False
