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
    """A retrieved passage the diagnosis rests on, by snippet id."""
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
    """What the diagnosis model proposes: a summary, a severity, a comment, citations.

    A PROPOSAL, never a verdict: the gate decides from it, and every field the
    model asserts about itself is checked rather than trusted. `severity`
    defaults to "high" so a diagnosis that never set one escalates (invariant
    12). `citations[].score` is filled from the retriever's hits, never from the
    model (T17; see `diagnosis_request_schema`).
    """
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
    # False like the other three (invariant 12): a ReviewResult built without
    # running the entailment check must not count as having passed it.
    entailment_ok: bool = False      # the claim follows from the cited passage
    # Why it failed, when it did. A gate that only says "escalated" makes the
    # human redo the diagnosis to find out what tripped.
    failures: list[str] = Field(default_factory=list)

    @property
    def decision(self) -> str:
        """Return "approve" only if all four booleans hold, else "escalate_to_human".

        Computed, so it cannot disagree with the booleans and cannot be written
        by anyone -- the graph routes on it (`app/graph.py`).
        """
        ok = (self.groundedness_ok and self.risk_ok
              and self.permission_ok and self.entailment_ok)
        return "approve" if ok else "escalate_to_human"


def diagnosis_request_schema() -> dict:
    """The JSON shape the diagnosis model is ASKED to emit (T1/T17).

    Deliberately hand-written rather than `Diagnosis.model_json_schema()`, and
    the difference is the point: **`citations[].score` is absent.** The model must
    not be asked for a relevance score, because the gate compares against exactly
    that number and `_parse_diagnosis` looks it up from the retriever's own hits.
    Generating the schema from the model would put `score` back in the request and
    invite the assumption that the model's value is used — the T17 mistake in a
    new place.

    Passed to `LLM.complete(..., json_schema=...)` for constrained decoding, which
    is not the same as asking for JSON in the prompt: prompt instructions are
    advisory and a model can still emit prose, whereas constrained decoding
    restricts token choice so the output cannot leave the schema. That is what
    makes the bare `json.loads` in `_parse_diagnosis` reasonable rather than
    reckless — and the parse still fails safe, because a provider that ignores
    `response_format` must not be able to crash the node.

    `severity` is enum-constrained so the model cannot invent a level outside
    `SEVERITIES`. An out-of-allowlist severity still escalates via `risk_ok`; this
    just removes one way to produce a value nothing can interpret.
    """
    _sev = {"type": "string", "enum": list(SEVERITIES)}
    return {
        "title": "Diagnosis",
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "severity", "proposed_comment", "citations"],
        "properties": {
            "summary": {"type": "string"},
            "severity": _sev,
            "proposed_comment": {"type": "string"},
            "proposed_label": {"type": ["string", "null"]},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["snippet_id"],
                    # No `score`. See the docstring -- this omission is load-bearing.
                    "properties": {"snippet_id": {"type": "string"}},
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["file", "issue"],
                    "properties": {
                        "severity": _sev,
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "issue": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "grounded": {"type": "boolean"},
                        "citation_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }
