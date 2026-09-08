from registry import adapter
from tools import github_client


@adapter("sink", "pr_comment")
class PRCommentSink:
    """Post the approved review as a PR/issue comment, as the approver.

    The partial-success handling lives here rather than in `execute` because it
    is provider-specific: on GitHub, commenting is open to any authenticated
    user while labelling needs triage/write access, so a 403 on the label after
    a successful comment is a routine outcome and not an error. Raising would
    discard the fact that the comment really was posted.
    """

    def emit(self, action: dict, *, target: str, identity: str | None = None) -> dict:
        result = {"comment": None, "label": None}
        result["comment"] = github_client.post_comment(target, action["comment"], identity)
        if action.get("label"):
            try:
                result["label"] = github_client.add_label(target, action["label"], identity)
            except Exception as e:      # noqa: BLE001 -- any label failure is partial success
                result["label"] = {"skipped": str(e)}
        return result


@adapter("sink", "report")
class ReportSink:
    """Log review: the 'action' is a rendered daily report. Non-mutating.

    Deliberately has no write path to infrastructure or a database. The log
    surface's whole output is advisory; anything actionable still goes through
    the HITL gate as its own action.
    """

    def emit(self, action: dict, *, target: str, identity: str | None = None) -> dict:
        raise NotImplementedError(
            f"persist/notify the daily report for {target}; never mutates infra"

        )
