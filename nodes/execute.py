"""The ONLY node allowed to write. Refuses without approval — checked IN THE NODE."""
from __future__ import annotations

from tools import github_client


def execute(state: dict, github=github_client) -> dict:
    # Defense-in-depth check #2 (independent of graph routing): hard guard here so a
    # future shortcut edge straight to execute STILL can't bypass approval.
    if state.get("approved") is not True:
        raise PermissionError("execute() refused: state['approved'] is not True")

    action = state["proposed_action"]          # posted VERBATIM, never recomputed
    url, token = state["target"], state.get("github_token")   # approver's own OAuth
    result = {"comment": None, "label": None}
    result["comment"] = github.post_comment(url, action["comment"], token)
    if action.get("label"):
        try:
            result["label"] = github.add_label(url, action["label"], token)
        except Exception as e:               # 403 needs-triage -> partial success, no crash
            result["label"] = {"skipped": str(e)}
    return {"result": result}
