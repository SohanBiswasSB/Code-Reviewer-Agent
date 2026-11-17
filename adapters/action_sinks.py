"""Where an approved action actually goes. The ONLY write paths in the system."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from app.activity import ERROR, log_event
from datadir import data_dir
from errors import ProviderError
from registry import adapter
from tools import github_client

REPORTS_SUBDIR = "reports"

# What a label call can fail with once the comment is already posted. OSError
# covers the network and the provider: ProviderError is a ConnectionError and a
# missing-permission 403 is a PermissionError. RuntimeError covers the stub's
# NotImplementedError and an HTTP client that reports a refused request as a
# RuntimeError (the shape eval/harness.py's 403 stand-in raises). ValueError
# covers a refused key (ProviderAuthError is a ConfigError) and a response that
# will not parse. Anything outside this family is a bug in the label path, and a
# bug propagates.
LABEL_ERRORS: tuple[type[BaseException], ...] = (OSError, RuntimeError, ValueError)


def safe_part(part: str, fallback: str) -> str:
    """One path component from caller-supplied text (a tenant, a date, a target).

    None of them may choose where a file lands, so everything but alphanumerics
    and `-_.` is dropped and a leading dot is stripped -- no `..`, no separator,
    no hidden file.
    """
    cleaned = "".join(c for c in (part or "") if c.isalnum() or c in "-_.")
    return cleaned.lstrip(".") or fallback


def _label_source(exc: BaseException) -> str:
    """`external` for a provider/permission/network failure, else `internal` (a bug)."""
    expected = (ProviderError, PermissionError, OSError, NotImplementedError)
    return "external" if isinstance(exc, expected) else "internal"


@adapter("sink", "pr_comment")
class PRCommentSink:
    """Post the approved review as a PR/issue comment, as the approver.

    The partial-success handling lives here rather than in `execute` because it
    is provider-specific: on GitHub, commenting is open to any authenticated
    user while labelling needs triage/write access, so a 403 on the label after
    a successful comment is a routine outcome and not an error. Raising would
    discard the fact that the comment really was posted.
    """

    def emit(self, action: dict, *, target: str, identity: str | None = None,
             **kwargs) -> dict:
        """Post the comment, then try the label; a label failure is partial success.

        `context` (run metadata: tenant, run_id) arrives in `kwargs` and is not
        read: a comment goes to a URL, so this sink has no use for it.
        """
        result = {"comment": None, "label": None}
        result["comment"] = github_client.post_comment(target, action["comment"], identity)
        if action.get("label"):
            try:
                result["label"] = github_client.add_label(target, action["label"], identity)
            except LABEL_ERRORS as e:
                # The comment above is already POSTED -- an irreversible write.
                # Letting a label failure escape would crash `execute` after that
                # write, and with no failure edges (G9) the run would lose the
                # audit record of a write that really happened (invariant 9). So
                # every expected label failure (LABEL_ERRORS) is a partial
                # success, pinned by the safety gate `partial_success_reported`
                # -- and is logged at ERROR with its full traceback, so it is
                # loud rather than lost.
                result["label"] = {"skipped": f"{type(e).__name__}: {e}"}
                log_event("external_api", "github.add_label", severity=ERROR,
                          exception=e, failure_source=_label_source(e),
                          metadata={"target": target, "label": action.get("label")})
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
        """`root` overrides `CRA_DATA_DIR` (tests point it at a temp dir)."""
        self.root = root or str(data_dir())

    def path_for(self, tenant: str, target: str, ext: str) -> str:
        """`<root>/reports/<tenant>/<date>.<ext>`, both parts sanitised."""
        return os.path.join(self.root, REPORTS_SUBDIR, safe_part(tenant, "_unknown"),
                            f"{safe_part(target, 'undated')}.{ext}")

    def _render_md(self, report: dict) -> str:
        """The human-readable twin of the `.json` report, built from the same dict.

        Rendered from the stored report rather than from `action` and `context`
        separately, so the two artefacts cannot disagree about what was approved.
        """
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

    def emit(self, action: dict, *, target: str, context: dict | None = None,
             **kwargs) -> dict:
        """Write the report as .json and .md.

        `identity` arrives in `kwargs` and is not read: nothing is posted, so no
        approver credential is needed.
        """
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
        """`root` overrides `CRA_DATA_DIR`, as for `ReportSink`."""
        self.root = root or str(data_dir())

    def path_for(self, tenant: str, target: str, run_id: str | None) -> str:
        """`<root>/reviews/<tenant>/<target>-<run>.md`, every part sanitised."""
        safe = safe_part
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{safe(target, 'change')}-{safe(run_id or stamp, stamp)}.md"
        return os.path.join(self.root, self.SUBDIR, safe(tenant, "_unknown"), name)

    def emit(self, action: dict, *, target: str, context: dict | None = None,
             **kwargs) -> dict:
        """Write the review; return where it went.

        `identity` arrives in `kwargs` and is not read: a local file needs no
        approver credential.
        """
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
