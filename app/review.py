"""Review a local change from the terminal: the whole graph, one human, no server.

    python -m app.review --tenant masterclass --root ~/Dev/<repo>                 # staged
    python -m app.review --tenant masterclass --root ~/Dev/<repo> --target HEAD
    python -m app.review --tenant masterclass --root ~/Dev/<repo> --target main...HEAD

The same graph `POST /run` + `POST /approve` drive, with two deployment choices
made for a laptop: evidence from the local checkout (`git_local`) instead of the
stubbed GitHub API, and an approved review written to `var/reviews/` instead of
posted to a PR. Both are env defaults, so either can still be overridden.

## Nothing about the gate changes

The four booleans decide approve vs escalate, exactly as in the API. An
escalated run ENDS here -- you are shown the diagnosis, because a human is now
the reviewer, but there is nothing to approve. An approve-eligible run suspends at
`await_approval`, and only what you type resumes it: `y` approves, anything else
denies, and a denial is recorded by the gate itself. With no terminal attached
the command refuses to decide rather than defaulting either way -- an approval
nobody gave is precisely what `approved` exists to rule out.

## Why an in-memory checkpointer is right HERE

`app/deps.py` warns that MemorySaver loses paused approvals on restart. In this
command the paused run only has to live as long as the person at the prompt, and
the resume happens in the same process. If the process dies, the run dies with
it and nothing was written -- the safe direction.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app import activity, live
from app.graph import build_graph
from app.state import checkpoint_serde
from nodes import _audit
from nodes._render import render_checklist

RULE = "-" * 72


def _defaults(root: str | None) -> None:
    """Point code_review at the local checkout and a local sink, unless overridden."""
    os.environ.setdefault("CRA_SOURCE_CODE_REVIEW", "git_local")
    os.environ.setdefault("CRA_SINK_CODE_REVIEW", "review_file")
    if root:
        os.environ["CRA_CODEBASE_ROOT"] = os.path.expanduser(root)


def _show(state: dict) -> None:
    """Print what the human needs to judge the run: evidence, diagnosis, gate."""
    ev = state.get("evidence") or {}
    ev = ev if isinstance(ev, dict) else getattr(ev, "__dict__", {})
    print(RULE)
    print(f"change   : {ev.get('title', '')}")
    print(f"files    : {len(ev.get('files') or [])}  "
          f"touched areas: {ev.get('touched_areas') or '-'}")
    print(f"route    : {state.get('route')}")

    checks = state.get("_checks") or []
    if checks:
        print(RULE)
        print(render_checklist(checks))

    diag = state.get("diagnosis")
    if diag is not None:
        print(RULE)
        print(f"severity : {diag.severity}")
        print(f"summary  : {diag.summary}")
        for c in diag.citations:
            print(f"  cites  : {c.snippet_id}  relevance={c.score}")
        for f in diag.findings:
            print(f"  finding: {f.model_dump()}")
        if diag.proposed_comment:
            print(RULE)
            print("proposed comment:\n")
            print(diag.proposed_comment)

    review = state.get("review")
    if review is not None:
        print(RULE)
        for name in ("groundedness_ok", "entailment_ok", "risk_ok", "permission_ok"):
            print(f"  {name:16} {getattr(review, name)}")
        for failure in review.failures:
            print(f"  failed : {failure}")
        print(f"DECISION : {review.decision}")
        if review.reasoning:
            print(f"\njudge (explains, never decides):\n{review.reasoning}")
    print(RULE)


def run(tenant: str, target: str, ask=input, interactive: bool | None = None) -> int:
    """Drive one review to its end, as one correlated operation in the activity log.

    The thread id IS the correlation id, so every node, model call and approval
    decision of this run shares one key with its audit record and live status.
    """
    thread_id = f"cli-{uuid.uuid4()}"
    with activity.correlation(thread_id), activity.ActivityLogger(
            "review", "review.cli", tenant=tenant,
            metadata={"target": target, "thread_id": thread_id}) as alog:
        code = _run(tenant, target, thread_id, ask, interactive)
        alog.set_response(code)
        return code


def _run(tenant: str, target: str, thread_id: str, ask, interactive: bool | None) -> int:
    """Drive one review to its end. Returns a process exit code.

    `ask` and `interactive` are injectable so the approval path can be tested
    without a terminal; in normal use they are `input` and `sys.stdin.isatty()`.
    """
    graph = build_graph(checkpointer=MemorySaver(serde=checkpoint_serde()))
    cfg = {"configurable": {"thread_id": thread_id}}

    print(f"run {thread_id} -- watch it live with: python -m app.monitor")
    state = graph.invoke({"tenant": tenant, "surface": "code_review", "target": target,
                          "thread_id": thread_id}, config=cfg)
    _show(state)

    interrupts = state.get("__interrupt__") or ()
    if not interrupts:
        print(f"ended without an approval step (route={state.get('route')}). "
              f"Audit record written; nothing was posted.")
        return 0

    payload = interrupts[0].value
    print("PROPOSED ACTION (exactly what will be written if you approve):\n")
    print(payload["proposed_action"].get("comment", ""))
    if payload["proposed_action"].get("label"):
        print(f"\nlabel: {payload['proposed_action']['label']}")
    print(RULE)

    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        print("no terminal attached: refusing to decide. Nothing was approved or written.")
        # The checkpoint is in-process memory, so this run ends with the process.
        # Recorded as ABANDONED: until now a run that reached the gate and was
        # left there wrote no audit record, and its findings, citations and
        # booleans -- the data `tools.calibrate` learns from -- were lost.
        _audit.write(graph.get_state(cfg).values, "abandoned")
        live.finish(thread_id, "abandoned")
        activity.log_event("approval", "approval.abandoned", severity=activity.WARNING,
                           tenant=tenant, message="no terminal attached to answer")
        return 2

    answer = ask("approve and write this review? [y/N] ").strip().lower()
    try:
        done = graph.invoke(Command(resume=answer == "y"), config=cfg)
    except PermissionError:
        # A denial resumes into `execute`, whose guard stops it -- that is the
        # mechanism, and await_approval has already recorded the denial.
        print("denied. The denial is in the audit record; nothing was written.")
        live.finish(thread_id, "denied")
        activity.log_event("approval", "approval.denied", severity=activity.WARNING,
                           tenant=tenant, metadata={"answer": answer[:20]})
        return 0
    print(f"approved and written: {done.get('result')}")
    activity.log_event("approval", "approval.granted", tenant=tenant,
                       metadata={"result": done.get("result")})
    return 0


def main() -> None:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Review a local change with the full graph.")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--target", default="staged",
                    help="staged (default) | <rev> | <a>..<b> | <a>...<b>")
    ap.add_argument("--root", help="where the tenant's repo lives on THIS machine")
    args = ap.parse_args()
    _defaults(args.root)
    activity.capture_library_logs()
    raise SystemExit(run(args.tenant, args.target))


if __name__ == "__main__":
    main()
