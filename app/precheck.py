"""The check you run by hand before a push: what to fix, what you fixed, and whether it is clear.

    python -m app.precheck --tenant merger_app --root ~/Dev/<repo>   # first run; root remembered
    python -m app.precheck --tenant merger_app                       # every time after
    python -m app.precheck --tenant merger_app --target staged       # only what is staged
    python -m app.precheck --tenant merger_app --dismiss 1a2b3c4d --reason "renderer adds it"
    python -m app.precheck --tenant merger_app --undismiss 1a2b3c4d
    python -m app.precheck --tenant merger_app --list

The user's loop (2026-11-24): write code -> run this -> it says what to fix ->
fix it in the repo -> run it again -> it says what is fixed and, once nothing is
left, "clear to push". Deliberately MANUAL: no git hook, nothing installed in
the reviewed repo, nothing fetched or pushed. Git is only read.

## What it reviews

By default everything this branch would push: `<base>...HEAD`, where `<base>` is
the tenant's `precheck_base` (merger_app: `origin/ID_AWS_Ongoing_Development_Branch`)
as of the last `git fetch` YOU ran -- this command never fetches.

## What decides "clear to push"

Only the computed pre-push checks. They are code, so "no longer flagged" really
means fixed. The model's findings are shown as advice: they are regenerated on
every run with different wording, and on 2026-11-17 both Merger findings checked
by hand were wrong, so they may not block anyone yet (TODO T32 is the
calibration that would change that). Same run, same graph as `app/review.py` --
the gate, the audit record and the monitor all see it -- but nothing is ever
offered for approval or written to a PR: a precheck that reaches the approval
gate is recorded as `abandoned`, exactly as an unattended CLI review is.

## When the model server is down

The checks still run and the verdict still stands; the AI section says it
stopped and why. A gateway 502 on 2026-11-24 used to end the whole review with a
traceback and nothing to read.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import uuid

from langgraph.checkpoint.memory import MemorySaver

import config
from app import activity, live
from app.findings_ledger import Ledger, Report
from app.graph import build_graph
from app.state import checkpoint_serde
from errors import ProviderError
from nodes import _audit
from nodes._surface import evidence_as_dict

RULE = "-" * 72


def _prepare(args, ledger: Ledger) -> str:
    """Resolve root and target; the root given once is remembered (machine-local)."""
    if args.root:
        ledger.data["root"] = os.path.expanduser(args.root)
    root = ledger.data.get("root") or os.getenv("CRA_CODEBASE_ROOT")
    if root:
        os.environ["CRA_CODEBASE_ROOT"] = root
    os.environ.setdefault("CRA_SOURCE_CODE_REVIEW", "git_local")
    os.environ.setdefault("CRA_SINK_CODE_REVIEW", "review_file")
    if args.target:
        return args.target
    policy = config.get_tenant_store().policy(args.tenant, "code_review")
    base = args.base or policy.get("precheck_base")
    if not base:
        raise SystemExit(f"{args.tenant} has no precheck_base in its policy: pass --base or "
                         f"--target")
    return f"{base}...HEAD"


def _has_changes(target: str) -> bool:
    """False when a range or the index holds nothing, so there is nothing to check.

    Asked of git directly (read-only), BEFORE the graph: an empty target is the
    normal state of a branch that is fully merged, and it used to surface as the
    source's "contains no changes" exception -- a traceback, for the answer
    "nothing to push" (2026-11-24, the first live precheck). A single revision
    is not pre-checked: a commit always has a change, and a bad one should fail
    loudly in the source as it always has.
    """
    if target == "staged":
        args = ["diff", "--cached", "--name-only"]
    elif ".." in target:
        args = ["diff", "--name-only", target]
    else:
        return True
    root = os.getenv("CRA_CODEBASE_ROOT") or "."
    out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                         check=True).stdout
    return bool(out.strip())


def _run_graph(tenant: str, target: str, thread_id: str) -> tuple[dict, str | None]:
    """One graph run. Returns its state and, if the model server failed, why.

    A ProviderError (the server is down or answered wrongly, retries exhausted) is
    the one failure handled here: the checks ran before any model call, and the
    checkpoint still holds them, so they are read back and reported. Every other
    exception is a bug or a bad target and propagates (after `_traced` has
    written its audit record).
    """
    graph = build_graph(checkpointer=MemorySaver(serde=checkpoint_serde()))
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.invoke({"tenant": tenant, "surface": "code_review", "target": target,
                              "thread_id": thread_id}, config=cfg)
    except ProviderError as e:
        # One line: a gateway answers an outage with an HTML error page.
        return graph.get_state(cfg).values, " ".join(re.sub(r"<[^>]+>", " ", str(e)).split())[:300]
    if state.get("__interrupt__"):
        _audit.write(graph.get_state(cfg).values, "abandoned")
        live.finish(thread_id, "abandoned")
    return state, None


def _scope(state: dict) -> list[str]:
    """Every path in the change, plus the head commit as `commit <sha12>`."""
    ev = evidence_as_dict(state.get("evidence") or {})
    files = [f.get("filename", "") if isinstance(f, dict) else str(f)
             for f in ev.get("files") or []]
    head = (ev.get("revision") or {}).get("head")
    return files + ([f"commit {head[:12]}"] if head else [])


def _ai_findings(state: dict) -> list[dict]:
    diag = state.get("diagnosis")
    return [f.model_dump() for f in diag.findings] if diag is not None else []


def _row(f: dict, note: str = "") -> str:
    where = f" {f['file']}" + (f":{f['line']}" if f.get("line") else "") if f.get("file") else ""
    return f"  [{f['id']}] {f.get('check', '')}{where} -- {f.get('message', '')}{note}"


def _section(title: str, rows: list[str]) -> None:
    if rows:
        print(f"\n{title} ({len(rows)})")
        print("\n".join(rows))


def _ai_status(state: dict, ai_error: str | None, report: Report) -> list[str]:
    if ai_error:
        return [f"  AI review stopped: {ai_error}",
                "  The checks above are complete; run again when the model server is back."]
    if state.get("route") != "ai_investigation":
        return [f"  AI review not run: route={state.get('route')} (the reviewer sends a change "
                "to the model only when it touches a risky area of this codebase)."]
    out = [f"  [{f['id']}] {f.get('severity', '?')} {f.get('file', '')}:{f.get('line', 0)} -- "
           f"{f.get('issue', '')}" for f in report.ai]
    review = state.get("review")
    if review is not None:
        out.append(f"  gate: {review.decision}" + (f" ({'; '.join(review.failures)})"
                                                   if review.failures else ""))
    return out or ["  AI review ran and raised nothing."]


def _print(target: str, scope: list[str], state: dict, report: Report,
           ai_error: str | None) -> None:
    """The report: to fix, fixed, then advice, then the verdict last."""
    skips = sorted({c["check"] for c in state.get("_checks") or [] if c.get("status") == "skip"})
    print(RULE)
    print(f"precheck  {target}  ({len(scope)} files/commits in the change)")
    _section("TO FIX", [_row(f, " (new)") for f in report.new]
             + [_row(f, f" (seen {f['runs']} runs)") for f in report.still_open])
    _section("FIXED SINCE LAST RUN", [_row(f) for f in report.fixed])
    _section("NO LONGER IN THE CHANGE", [_row(f) for f in report.left_change])
    _section("DISMISSED, not counted", [_row(f, f"  (you: {f['dismissed']['reason']})")
                                        for f in report.dismissed])
    print("\nAI REVIEW (advice only, never blocks)")
    print("\n".join(_ai_status(state, ai_error, report)))
    if skips:
        print(f"\nNOT CHECKED here ({len(skips)}): {', '.join(skips)}")
    print(RULE)
    if report.to_fix:
        print(f"VERDICT: {len(report.to_fix)} to fix. Fix them, then run this again.")
    else:
        print("VERDICT: clear to push -- every computed check passed or was dismissed."
              + (" (AI review did not run; see above.)" if ai_error else ""))


def run(args) -> int:
    """Review the change, reconcile with the ledger, print, save. Exit 0 = clear."""
    ledger = Ledger.load(args.tenant)
    target = _prepare(args, ledger)
    if not _has_changes(target):
        ledger.save()                          # keep a newly given --root
        print(f"Nothing to check: {target} has no changes. If you expected some, run "
              f"`git fetch` -- the base is as of your last fetch.")
        return 0
    thread_id = f"precheck-{uuid.uuid4()}"
    with activity.correlation(thread_id), activity.ActivityLogger(
            "review", "review.precheck", tenant=args.tenant,
            metadata={"target": target, "thread_id": thread_id}) as alog:
        print(f"run {thread_id} -- watch it live with: python -m app.monitor")
        state, ai_error = _run_graph(args.tenant, target, thread_id)
        flags = [c for c in state.get("_checks") or [] if c.get("status") == "flag"]
        scope = _scope(state)
        report = ledger.reconcile(flags, _ai_findings(state), scope)
        ledger.record_run(target, scope, report)
        ledger.save()
        _print(target, scope, state, report, ai_error)
        code = 1 if report.to_fix else 0
        alog.set_response(code)
        return code


def list_cmd(args) -> int:
    """`--list`: every tracked finding by status, and the last run."""
    ledger = Ledger.load(args.tenant)
    last = ledger.data.get("last_run") or {}
    print(f"last run: {last.get('at', 'never')}  target: {last.get('target', '-')}")
    by_status: dict[str, list[dict]] = {}
    for entry in ledger.findings.values():
        by_status.setdefault(entry["status"], []).append(entry)
    for status in ("open", "dismissed", "fixed", "left_change"):
        _section(status.upper(), [_row(f) for f in by_status.get(status, [])])
    _section("DISMISSED AI FINDINGS", [f"  [{d['id']}] {d['file']} -- {d['issue'][:100]}  "
                                       f"(you: {d['reason']})"
                                       for d in ledger.data.get("ai_dismissed") or []])
    return 0


def dismiss_cmd(args) -> int:
    """`--dismiss ID --reason ...`: stop showing a finding as something to fix."""
    ledger = Ledger.load(args.tenant)
    ledger.dismiss(args.dismiss, args.reason or "")
    ledger.save()
    print(f"dismissed {args.dismiss}: it will not be shown as something to fix again.")
    return 0


def undismiss_cmd(args) -> int:
    """`--undismiss ID`: lift a dismissal."""
    ledger = Ledger.load(args.tenant)
    ledger.undismiss(args.undismiss)
    ledger.save()
    print(f"{args.undismiss} is no longer dismissed; the next run decides if it is still there.")
    return 0


def main() -> None:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Check a change before you push it.")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--root", help="the repo on THIS machine (remembered after the first use)")
    ap.add_argument("--base", help="compare against this ref instead of the policy's base")
    ap.add_argument("--target", help="staged | <rev> | <a>..<b> | <a>...<b> (default: base...HEAD)")
    ap.add_argument("--dismiss", metavar="ID", help="stop showing a finding (needs --reason)")
    ap.add_argument("--reason", help="why it is dismissed")
    ap.add_argument("--undismiss", metavar="ID", help="lift a dismissal")
    ap.add_argument("--list", action="store_true", help="show every tracked finding")
    args = ap.parse_args()
    activity.capture_library_logs()
    action = (dismiss_cmd if args.dismiss else undismiss_cmd if args.undismiss
              else list_cmd if args.list else run)
    raise SystemExit(action(args))


if __name__ == "__main__":
    main()
