"""GitHub REST client. Reads filter out PRs; writes are the ONLY side effects.

All network calls are STUBS (raise NotImplementedError) — wire httpx + the real
endpoints. #5: get_check_runs is defined so fetch_evidence can populate real CI
results (the reference hardcodes these to []).
"""
from __future__ import annotations


def fetch_issue(url: str, token: str | None = None) -> dict:
    raise NotImplementedError("wire GitHub REST GET /repos/{o}/{r}/issues/{n}")


def get_check_runs(url: str, token: str | None = None) -> list[dict]:
    # #5: fetch GET /repos/{o}/{r}/commits/{sha}/check-runs — do NOT hardcode [].
    raise NotImplementedError("wire GitHub Actions check-runs")


def post_comment(url: str, body: str, token: str | None = None) -> dict:
    raise NotImplementedError("wire POST /repos/{o}/{r}/issues/{n}/comments")


def add_label(url: str, label: str, token: str | None = None) -> dict:
    raise NotImplementedError("wire POST .../labels (403 => partial success, don't crash)")
