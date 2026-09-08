"""raw JSON -> validated IssueEvidence. Bad data should fail HERE, loudly."""
from __future__ import annotations

from schemas.evidence import CheckRun, IssueEvidence


def normalize_evidence(state: dict) -> dict:
    raw = state["_raw"]
    # GitHub's issues endpoint returns PRs mixed in — filter them out upstream in
    # github_client; here we just trust normalized fields. TODO: Pydantic validation.
    ev = IssueEvidence(
        url=raw["html_url"], title=raw.get("title", ""), body=raw.get("body", "") or "",
        labels=[lbl["name"] for lbl in raw.get("labels", [])],
        check_runs=[CheckRun(c["name"], c["conclusion"]) for c in raw.get("check_runs", [])],
    )
    return {"evidence": ev}
