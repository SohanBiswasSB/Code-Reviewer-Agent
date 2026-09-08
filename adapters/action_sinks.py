from registry import adapter


@adapter("sink", "pr_comment")
class PRCommentSink:
    def emit(self, action: dict, *, identity: str | None = None) -> dict:
        raise NotImplementedError("post the approved review comment as `identity`'s token")

@adapter("sink", "report")
class ReportSink:
    """Log-review: 'action' is a rendered daily report -> file/notify. Non-mutating."""
    def emit(self, action: dict, *, identity: str | None = None) -> dict:
        raise NotImplementedError("persist/notify the report; never mutates infra")
