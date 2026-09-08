"""Run the safety-gate suite. Exit non-zero on ANY miss (pass^k, not an average).

    python -m eval.harness

This is a REGRESSION suite, not a capability suite. Every task has one
unambiguous correct answer and the bar is 100% forever: a single miss means a
safety gate weakened, not that the system "needs more data". Keeping the two
kinds of eval separate is deliberate — a gate that only mostly holds is not a
gate. Capability evaluation (is the diagnosis any *good*?) is tracked separately
as T10/G17 and is expected to improve over time rather than sit at 100%.

Deliberately dependency-free and model-free: no provider, no server, no network.
It must therefore give byte-identical results on the Mac and the CUDA box.
"""
from __future__ import annotations

import sys

from eval.tasks import CLASSIFY_TASKS, GATE_TASKS
from nodes.classify import classify
from nodes.execute import execute
from nodes.independent_review import independent_review


class _Snippet:
    """Minimal retrieved-snippet stand-in: the gate only reads .id and .score."""

    def __init__(self, sid: str, score: float):
        self.id, self.score = sid, score


def run() -> int:
    fails: list[str] = []

    def check(kind: str, name: str, expected: str, got: str) -> None:
        ok = got == expected
        print(f"[{kind:9}] {name:42} expect={expected:17} got={got:17} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    # ── classify(): routing precedence + Surface delegation ──────────────────
    for name, state, expected in CLASSIFY_TASKS:
        check("classify", name, expected, classify(state)["route"])

    # ── independent_review(): the three booleans, under real policies ────────
    for name, diag, retrieved, policy, expected in GATE_TASKS:
        state = {
            "diagnosis": diag,
            "_retrieved": [_Snippet(i, s) for i, s in retrieved.items()],
        }
        if policy is not None:
            state["_policy"] = policy
        check("gate", name, expected, independent_review(state)["review"].decision)

    # ── execute(): the permission guard and verbatim posting ─────────────────
    class FakeGH:
        """No test may ever reach real GitHub. The suite must never post."""

        def post_comment(self, url, body, token=None):
            return {"body": body}

        def add_label(self, url, label, token=None):
            return {"label": label}

    # refuses without approval -- checked IN THE NODE, so a future shortcut edge
    # straight to execute still cannot bypass the gate
    try:
        execute({"approved": False, "target": "u",
                 "proposed_action": {"comment": "x"}}, github=FakeGH())
        check("execute", "refuses_without_approval", "PermissionError", "NO-RAISE")
    except PermissionError:
        check("execute", "refuses_without_approval", "PermissionError", "PermissionError")

    # approval must not be inferrable from a truthy-ish value
    truthy_but_not_true = (("approved_is_string", "true"), ("approved_is_1", 1),
                           ("approved_is_none", None))
    for label, bad in truthy_but_not_true:
        try:
            execute({"approved": bad, "target": "u",
                     "proposed_action": {"comment": "x"}}, github=FakeGH())
            check("execute", label, "PermissionError", "NO-RAISE")
        except PermissionError:
            check("execute", label, "PermissionError", "PermissionError")

    # posts the approved action VERBATIM -- never recomputed after approval
    res = execute({"approved": True, "target": "u",
                   "proposed_action": {"comment": "VERBATIM", "label": None}}, github=FakeGH())
    check("execute", "posts_verbatim_when_approved", "VERBATIM", res["result"]["comment"]["body"])

    # a failing label add is a PARTIAL success -- the comment really was posted,
    # so it must not be discarded by raising
    class LabelFailsGH(FakeGH):
        def add_label(self, url, label, token=None):
            raise RuntimeError("403 needs triage/write access")

    res = execute({"approved": True, "target": "u",
                   "proposed_action": {"comment": "posted", "label": "bug"}}, github=LabelFailsGH())
    partial = (res["result"]["comment"]["body"] == "posted"
               and "skipped" in (res["result"]["label"] or {}))
    check("execute", "label_failure_is_partial_success", "partial",
          "partial" if partial else "lost")

    total = len(CLASSIFY_TASKS) + len(GATE_TASKS) + 6   # 6 execute-guard checks
    print()
    if fails:
        print(f"CRITICAL: {len(fails)} of {total} gate(s) failed: {fails}")
    else:
        print(f"ALL {total} GATES PASS (pass^k)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
