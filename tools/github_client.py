"""GitHub REST client. Reads filter out PRs; writes are the ONLY side effects.

All network calls are STUBS (raise NotImplementedError) — wire httpx + the real
endpoints. #5: get_check_runs is defined so fetch_evidence can populate real CI
results (the reference hardcodes these to []).
"""
from __future__ import annotations


def fetch_issue(url: str, token: str | None = None) -> dict:
    """The issue at `url` as GitHub's JSON. STUB: raises NotImplementedError.

    Called by `adapters/source_github.py` with the approver's token when there is
    one. A stub that raises rather than returning fake data, so nothing can run
    on evidence that was never fetched.
    """
    raise NotImplementedError("wire GitHub REST GET /repos/{o}/{r}/issues/{n}")


def get_check_runs(url: str, token: str | None = None) -> list[dict]:
    """The CI check runs for the target's commit. STUB: raises NotImplementedError.

    `classify` routes failing CI to `deterministic`, so a client that returned []
    instead of the real runs would hide every red build from routing.
    """
    # #5: fetch GET /repos/{o}/{r}/commits/{sha}/check-runs — do NOT hardcode [].
    raise NotImplementedError("wire GitHub Actions check-runs")


def post_comment(url: str, body: str, token: str | None = None) -> dict:
    """Post `body` as a comment, as `token`'s owner. STUB: raises NotImplementedError.

    A write, so reached only through `PRCommentSink` from `execute`, after
    approval (invariant 3).
    """
    raise NotImplementedError("wire POST /repos/{o}/{r}/issues/{n}/comments")


def add_label(url: str, label: str, token: str | None = None) -> dict:
    """Add `label`, as `token`'s owner. STUB: raises NotImplementedError.

    Labelling needs triage/write access where commenting does not, so a 403 here
    after a posted comment is a partial success for `PRCommentSink` to report,
    not a crash.
    """
    raise NotImplementedError("wire POST .../labels (403 => partial success, don't crash)")
