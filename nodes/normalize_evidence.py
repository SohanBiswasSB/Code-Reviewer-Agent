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
    upstream in the client. `IssueEvidence` is a Pydantic model (T11), so
    malformed provider JSON raises HERE rather than propagating into the graph.
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
    """Turn `_raw` into the validated `evidence` the rest of the graph reads.

    Reads `_raw`, the active surface and the pinned `_policy`; writes only
    `evidence`. `_raw` was already redacted by `fetch_evidence` (T25), so nothing
    here sees an unredacted secret, and redaction does not belong here: by now
    the fetch step's checkpoint would already hold the value. Where the
    normaliser builds a Pydantic model (issues, log windows), malformed provider
    data raises here, at the boundary, rather than deeper in the graph (T11);
    code_review's normaliser returns a plain dict.
    """
    raw = state["_raw"]
    surface = active_surface(state)
    normalizer = getattr(surface, "normalize", None) if surface else None
    if not callable(normalizer):
        return {"evidence": normalize_issue(raw)}
    # The tenant's policy goes in so a surface can apply tenant VALUES -- the
    # code_review surface widens its risky areas with them (T27). The policy is
    # the one resolve_tenant pinned, so a resumed run routes as it started.
    return {"evidence": normalizer(raw, policy=state.get("_policy") or {})}
