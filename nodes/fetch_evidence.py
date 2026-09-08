"""Real GitHub REST calls -> raw dict. #5: populate check_runs for real."""
from __future__ import annotations

from tools import github_client


def fetch_evidence(state: dict) -> dict:
    url = state["target"]                # surface-agnostic: PR ref | issue url | date
    token = state.get("github_token")
    raw = github_client.fetch_issue(url, token)
    raw["check_runs"] = github_client.get_check_runs(url, token)   # not hardcoded []
    return {"_raw": raw}
