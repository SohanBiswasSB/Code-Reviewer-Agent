"""Validated evidence. Pydantic v2, for two reasons that are not style choices.

**Validation at the boundary (T11).** `normalize_evidence`'s own docstring said
bad provider data should fail *there, loudly*. With stdlib dataclasses it could
not: a missing field or a wrong type propagated into the graph and surfaced
somewhere unrelated. Now malformed provider JSON raises at the edge.

**Checkpoint serialisation (T15).** LangGraph warned on every state type that
"deserializing unregistered type ... will be blocked in a future version". On a
stricter release that makes **every paused run unresumable** — catastrophic for a
design whose safety property is a durable pause. Being Pydantic is NOT enough on
its own: the fix that closed T15 was registration. Every model here is listed in
`app.state.CHECKPOINT_TYPES`, and a model added without that entry does not
raise -- it comes back from a checkpoint as a plain dict.

The computed properties stay properties: `has_failing_ci` and
`is_information_sparse` are derived facts, not stored fields, and the eval suite
pins the routing that depends on them.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SPARSE_BODY_CHARS = 140  # bodies shorter than this with no repro are "information sparse"


class CheckRun(BaseModel):
    """One CI check on the change. Only `conclusion == "failure"` routes anything."""
    model_config = ConfigDict(extra="ignore")

    name: str
    conclusion: str  # "success" | "failure" | "neutral" | ...


class LinkedPR(BaseModel):
    """A pull request linked to an issue, by number and state."""
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
        """Any check concluded "failure". `classify` routes on this first."""
        return any(c.conclusion == "failure" for c in self.check_runs)

    @property
    def is_information_sparse(self) -> bool:
        """A body under SPARSE_BODY_CHARS that names no repro steps.

        A crude keyword test ("repro", "steps to reproduce"). In the built-in
        issue-triage rules (`classify`, when no surface classifies) a sparse issue
        with green CI goes to `ai_investigation`; the eval suite pins that.
        """
        body = (self.body or "").strip()
        has_repro = "repro" in body.lower() or "steps to reproduce" in body.lower()
        return len(body) < SPARSE_BODY_CHARS and not has_repro


class CodeChange(BaseModel):
    """A proposed change, as the code_review surface normalises it -- in intent.

    Typed for the same two reasons as above, and because `classify` routes on
    `touched_areas` and `has_failing_ci`, so a typo in either key would silently
    route everything to `human_review`.

    NOT YET BUILT by the surface: `CodeReviewSurface.normalize` still returns a
    plain dict. It had drifted from that dict -- it lacked `revision`, so adopting
    it under `extra="forbid"` would have rejected every local review. `revision`
    is declared now, and `tests/test_local_review.py` validates the surface's
    real output against this model so the two cannot drift apart again. It is
    registered in `app.state.CHECKPOINT_TYPES` all the same.
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
    # Where the diff came from (`git_local`: diff_args, base/new refs, head, commit).
    # The pre-push checks re-read files at exactly this revision; {} for GitHub.
    revision: dict = Field(default_factory=dict)


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
