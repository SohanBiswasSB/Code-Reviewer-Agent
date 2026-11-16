"""Where an approved action actually goes. The ONLY write paths in the system."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from registry import adapter
from tools import github_client

REPORTS_SUBDIR = "reports"


def safe_part(part: str, fallback: str) -> str:
    """One path component from caller-supplied text (a tenant, a date, a target).

    None of them may choose where a file lands, so everything but alphanumerics
    and `-_.` is dropped and a leading dot is stripped -- no `..`, no separator,
    no hidden file.
    """
    cleaned = "".join(c for c in (part or "") if c.isalnum() or c in "-_.")
    return cleaned.lstrip(".") or fallback


@adapter("sink", "pr_comment")
class PRCommentSink:
    """Post the approved review as a PR/issue comment, as the approver.

    The partial-success handling lives here rather than in `execute` because it
    is provider-specific: on GitHub, commenting is open to any authenticated
    user while labelling needs triage/write access, so a 403 on the label after
    a successful comment is a routine outcome and not an error. Raising would
    discard the fact that the comment really was posted.
    """

    def emit(self, action: dict, *, target: str,
             identity: str | None = None, context: dict | None = None) -> dict:  # pylint: disable=unused-argument
        """Post the comment, then try the label; a label failure is partial success."""
        # `context` is run metadata (tenant, run_id). A comment goes to a URL, so
        # this sink has no use for it -- accepted to satisfy the port.
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

    **Stores the ANALYSIS, not the logs.** The raw lines already live in S3,
    which is the source of record — copying them into a second store would
    double the PII surface and buy nothing. Sampled lines should be cited by
    S3 key + offset once the G4 detect/aggregate/sample pipeline exists; until
    then a report carries what the diagnosis produced.

    Two artefacts per (tenant, date): `.json` to analyse and compare over time,
    `.md` for a human to read. See docs/DATA_STORAGE.md.
    """

    def __init__(self, root: str | None = None):
        self.root = root or os.getenv("CRA_DATA_DIR", "var")

    def path_for(self, tenant: str, target: str, ext: str) -> str:
        """`<root>/reports/<tenant>/<date>.<ext>`, both parts sanitised."""
        return os.path.join(self.root, REPORTS_SUBDIR, safe_part(tenant, "_unknown"),
                            f"{safe_part(target, 'undated')}.{ext}")

    def _render_md(self, report: dict) -> str:
        cites = report.get("citations") or []
        lines = [
            f"# Daily log review — {report['target']}",
            "",
            f"- tenant: `{report['tenant']}`",
            f"- severity: **{report.get('severity', 'unknown')}**",
            f"- run: `{report.get('run_id')}`",
            f"- policy: `{report.get('policy_version')}` · "
            f"prompt: `{report.get('prompt_version')}`",
            "",
            "## Analysis",
            "",
            report.get("analysis", "") or "_(no analysis produced)_",
        ]
        if cites:
            lines += ["", "## Cited evidence", ""]
            lines += [f"- `{c.get('snippet_id')}` (relevance {c.get('score')})" for c in cites]
        lines += ["", "---", "",
                  "Advisory only. Nothing here mutated infrastructure or a database."]
        return "\n".join(lines) + "\n"

    def emit(self, action: dict, *, target: str,
             identity: str | None = None, context: dict | None = None) -> dict:  # pylint: disable=unused-argument
        """Write the report as .json and .md. `identity` is unused: nothing is posted."""
        ctx = context or {}
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "tenant": ctx.get("tenant") or "_unknown",
            "surface": ctx.get("surface") or "log_review",
            "target": target,                      # the date, for this surface
            "run_id": ctx.get("run_id"),
            "policy_version": ctx.get("policy_version"),
            "prompt_version": ctx.get("prompt_version"),
            "severity": ctx.get("severity"),
            "analysis": action.get("comment", ""),  # posted VERBATIM, as approved
            "citations": ctx.get("citations") or [],
            # Populated once G4's detect -> aggregate -> sample stage exists.
            "candidates": ctx.get("candidates") or [],
            "counts": ctx.get("counts") or {},
        }

        json_path = self.path_for(report["tenant"], target, "json")
        md_path = self.path_for(report["tenant"], target, "md")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._render_md(report))

        return {"report": {"json": json_path, "markdown": md_path,
                           "tenant": report["tenant"], "target": target}}


@adapter("sink", "review_file")
class ReviewFileSink:
    """Write the approved code review to a Markdown file instead of a PR comment.

        CRA_SINK_CODE_REVIEW=review_file     (app/review.py sets this for you)

    The local counterpart of `pr_comment`, for reviewing a change that has no PR
    yet -- and the only working write path for code_review while
    tools/github_client.py is stubbed. Still reached only through `execute`, so
    it writes only what a human approved, verbatim.

    One file per RUN, never overwritten: the same target reviewed twice is two
    reviews, and replacing the first would erase what was approved the first time.
    """

    SUBDIR = "reviews"

    def __init__(self, root: str | None = None):
        self.root = root or os.getenv("CRA_DATA_DIR", "var")

    def path_for(self, tenant: str, target: str, run_id: str | None) -> str:
        """`<root>/reviews/<tenant>/<target>-<run>.md`, every part sanitised."""
        safe = safe_part
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{safe(target, 'change')}-{safe(run_id or stamp, stamp)}.md"
        return os.path.join(self.root, self.SUBDIR, safe(tenant, "_unknown"), name)

    def emit(self, action: dict, *, target: str,
             identity: str | None = None, context: dict | None = None) -> dict:  # pylint: disable=unused-argument
        """Write the review; return where it went. `identity` is unused locally."""
        ctx = context or {}
        cites = ctx.get("citations") or []
        lines = [
            f"# Code review — `{target}`",
            "",
            f"- tenant: `{ctx.get('tenant') or '_unknown'}`",
            f"- severity: **{ctx.get('severity') or 'unknown'}**",
            f"- run: `{ctx.get('run_id')}`",
            f"- policy: `{ctx.get('policy_version')}` · prompt: `{ctx.get('prompt_version')}`",
        ]
        if action.get("label"):
            lines.append(f"- proposed label: `{action['label']}`")
        lines += ["", "## Review", "", action.get("comment", "") or "_(empty)_"]
        if cites:
            lines += ["", "## Grounded in", ""]
            lines += [f"- `{c.get('snippet_id')}` (relevance {c.get('score')})" for c in cites]
        lines += ["", "---", "", "Approved by a human before it was written. Advisory only."]

        path = self.path_for(ctx.get("tenant") or "_unknown", target, ctx.get("run_id"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return {"review": {"markdown": path, "target": target}}
