"""raw provider payload -> validated evidence. Bad data should fail HERE, loudly.

Normalisation is per-SHAPE, not global: a GitHub issue and a code diff are not
interchangeable, and a surface that routes on `touched_areas` gets nothing useful
from an `IssueEvidence`. So a surface may own its own `normalize`, and the
built-in issue normaliser is the fallback for surfaces that do not.

That mismatch was not theoretical — with delegation enabled, the code_review
surface received issue-shaped evidence and routed everything to `human_review`
(safe, but useless). This is where that gets fixed properly.
"""
from __future__ import annotations

from nodes._surface import active_surface
from schemas.evidence import CheckRun, IssueEvidence


def normalize_issue(raw: dict) -> IssueEvidence:
    """The built-in issue-triage normaliser.

    GitHub's issues endpoint returns PRs mixed in; that filtering belongs
    upstream in the client. TODO(T11): Pydantic, so malformed provider JSON
    raises here rather than propagating into the graph.
    """
    return IssueEvidence(
        url=raw["html_url"],
        title=raw.get("title", ""),
        body=raw.get("body", "") or "",
        labels=[lbl["name"] for lbl in raw.get("labels", [])],
        check_runs=[CheckRun(name=c["name"], conclusion=c["conclusion"])
                    for c in raw.get("check_runs", [])],
    )


def normalize_evidence(state: dict) -> dict:
    raw = state["_raw"]
    surface = active_surface(state)
    normalizer = getattr(surface, "normalize", None) if surface else None
    return {"evidence": normalizer(raw) if callable(normalizer) else normalize_issue(raw)}
