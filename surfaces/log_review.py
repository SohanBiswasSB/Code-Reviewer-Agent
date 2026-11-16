"""Daily log review: a day of exports in, a grounded advisory report out.

Owns its own `normalize` for the same reason code_review does: its evidence SHAPE
is not an issue's. Without one, `normalize_evidence` fell back to the built-in
issue normaliser, which reads `raw["html_url"]` — so this surface raised KeyError
on any real run. It was unreachable in three separate ways (see `action_type`
below and docs/DATA_STORAGE.md); this closes one of them.
"""
from registry import surface
from schemas.evidence import LogWindow


@surface("log_review")
class LogReviewSurface:
    """Advisory only. This surface never comments, labels, or mutates anything."""

    name = "log_review"
    source_name, sink_name = "s3logs", "report"
    # Node KEYS, resolved per tenant by the TenantStore (T19).
    diagnosis_prompt = "diagnosis"
    judge_prompt = "judge"

    def normalize(self, raw: dict, policy: dict | None = None) -> LogWindow:
        """Reduced source payload -> validated LogWindow. `policy` is accepted for
        the port's shape and unused: log routing reads severities, not paths.

        The reduction already happened in the SOURCE, while the lines were still
        a stream (adapters/source_s3logs.py). By the time evidence reaches here it
        is counts, candidates and a capped sample — kilobytes, safe to put in a
        checkpoint. Validating with `extra="forbid"` means a source that starts
        returning raw lines fails here, loudly, instead of quietly making every
        super-step rewrite a day of logs.
        """
        return LogWindow(**raw)

    def action_type(self, diagnosis) -> str:
        """Always `report` — this surface never comments and never labels.

        Load-bearing, not cosmetic. Every log-review policy allows only
        `["report"]`, so while the gate hardcoded the comment shape,
        `permission_ok` was false for every run of this surface: it escalated
        100% of the time and its ReportSink was unreachable.
        """
        return "report"

    def build_query(self, evidence: dict) -> str:
        """Retrieve on the SIGNALS the detectors found, not on raw log text.

        The detectors have already decided what is anomalous; the query should
        therefore ask what is known about those anomalies (runbooks, prior
        incidents) rather than re-deriving them from prose.
        """
        cands = evidence.get("candidates", []) or []
        sigs = ", ".join(c.get("signal", "") for c in cands if c.get("signal"))
        services = " ".join(evidence.get("services", []) or [])
        return f"anomalies: {sigs}\nservices: {services}".strip()

    def classify(self, evidence: dict) -> str:
        """Rule-based route, on the detectors' output.

        A quiet day is a legitimate outcome, not a failure: with nothing above
        threshold there is nothing to ground a diagnosis in, so it goes to a human
        as a healthy report rather than inviting a model to invent something.
        """
        if not (evidence.get("candidates") or []):
            return "human_review"
        return "ai_investigation"
