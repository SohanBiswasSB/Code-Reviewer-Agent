"""Structured LLM output and the review result. Pydantic v2 (see evidence.py).

`Finding` is new (T17): the diagnosis prompt has always documented a `findings[]`
array of per-file, per-line issues, but `Diagnosis` had no field for it — so the
richest half of the model's output was parsed and thrown away, while anyone
reading the prompt would reasonably assume it was used.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SEVERITIES = ("low", "medium", "high", "critical")


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snippet_id: str
    # The RETRIEVER's score, never the model's. The gate compares against exactly
    # this number, so a model-supplied value would make grounding self-certified.
    score: float = 0.0


class Finding(BaseModel):
    """One specific, located issue. Per-file rather than per-review.

    `grounded` is the model's own claim about whether it could cite this finding.
    It is advisory: the gate's `groundedness_ok` is computed from real citation
    membership and real relevance scores, exactly as with everything else the
    model asserts about itself.
    """
    model_config = ConfigDict(extra="ignore")

    severity: str = "medium"
    file: str = ""
    line: int = 0
    issue: str = ""
    suggestion: str = ""
    grounded: bool = False
    citation_ids: list[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    severity: str = "high"          # must be in the allowlist to pass risk_ok
    proposed_comment: str = ""
    proposed_label: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """The gate's verdict — computed, never parsed from a model.

    `reasoning` is free text from the second model, for a human to read. Every
    boolean beside it comes from Python over real data. Four now, not three:
    `entailment_ok` (T22) closes the third of the three grounding checks the
    design calls for.
    """
    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    groundedness_ok: bool = False    # cited ids are real AND clear the threshold
    risk_ok: bool = False            # severity is in the tenant's allowlist
    permission_ok: bool = False      # the action type is in the tenant's allowlist
    entailment_ok: bool = True       # the claim follows from the cited passage
    # Why it failed, when it did. A gate that only says "escalated" makes the
    # human redo the diagnosis to find out what tripped.
    failures: list[str] = Field(default_factory=list)

    @property
    def decision(self) -> str:
        ok = (self.groundedness_ok and self.risk_ok
              and self.permission_ok and self.entailment_ok)
        return "approve" if ok else "escalate_to_human"
