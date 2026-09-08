"""Validated evidence for one issue.

STDLIB dataclasses for a zero-install runnable skeleton. TODO: migrate to
Pydantic v2 (BaseModel) for real validation — the graph state should reject
malformed GitHub JSON loudly at normalize_evidence, not silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SPARSE_BODY_CHARS = 140  # bodies shorter than this with no repro are "information sparse"


@dataclass
class CheckRun:
    name: str
    conclusion: str  # "success" | "failure" | "neutral" | ...


@dataclass
class LinkedPR:
    number: int
    state: str


@dataclass
class IssueEvidence:
    url: str
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    check_runs: list[CheckRun] = field(default_factory=list)   # #5: wire this for REAL
    linked_prs: list[LinkedPR] = field(default_factory=list)

    @property
    def has_failing_ci(self) -> bool:
        return any(c.conclusion == "failure" for c in self.check_runs)

    @property
    def is_information_sparse(self) -> bool:
        body = (self.body or "").strip()
        has_repro = "repro" in body.lower() or "steps to reproduce" in body.lower()
        return len(body) < SPARSE_BODY_CHARS and not has_repro
