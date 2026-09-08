"""Structured LLM output + the review result. Dataclasses (see evidence.py note)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    snippet_id: str
    score: float          # cosine (or fused) relevance — retrieval MUST return this


@dataclass
class Diagnosis:
    summary: str
    severity: str                       # must be in SEVERITY_ALLOWLIST to pass risk_ok
    proposed_comment: str
    proposed_label: str | None
    citations: list[Citation] = field(default_factory=list)


@dataclass
class ReviewResult:
    reasoning: str          # FREE TEXT from the 2nd llm call — for a human to read
    groundedness_ok: bool   # computed in PYTHON, never parsed from the llm
    risk_ok: bool
    permission_ok: bool

    @property
    def decision(self) -> str:
        return "approve" if (self.groundedness_ok and self.risk_ok and self.permission_ok) \
            else "escalate_to_human"
