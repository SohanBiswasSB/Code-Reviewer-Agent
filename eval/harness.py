"""Run the safety-gate suite. Exit non-zero on ANY miss (pass^k, not average).

    python3 -m eval.harness
"""
from __future__ import annotations
import sys
from nodes.classify import classify
from nodes.independent_review import independent_review
from nodes.execute import execute
from eval.tasks import CLASSIFY_TASKS, GATE_TASKS


def run() -> int:
    fails = []

    for name, ev, expected in CLASSIFY_TASKS:
        got = classify({"evidence": ev})["route"]
        ok = got == expected
        print(f"[classify ] {name:28} expect={expected:16} got={got:16} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    for name, diag, retrieved, expected in GATE_TASKS:
        state = {"diagnosis": diag,
                 "_retrieved": [type('S', (), {'id': i, 'score': s})() for i, s in retrieved.items()]}
        out = independent_review(state)
        got = out["review"].decision
        ok = got == expected
        print(f"[gate     ] {name:28} expect={expected:16} got={got:16} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    # execute() permission guard — must refuse without approval, post verbatim with it
    class FakeGH:
        def post_comment(self, url, body, token=None): return {"body": body}
        def add_label(self, url, label, token=None): return {"label": label}

    try:
        execute({"approved": False, "target": "u",
                 "proposed_action": {"comment": "x"}}, github=FakeGH())
        print("[execute  ] refuses_without_approval    expect=raise         got=NO-RAISE      FAIL")
        fails.append("refuses_without_approval")
    except PermissionError:
        print("[execute  ] refuses_without_approval    expect=raise         got=PermissionError OK")

    res = execute({"approved": True, "target": "u",
                   "proposed_action": {"comment": "VERBATIM", "label": None}}, github=FakeGH())
    ok = res["result"]["comment"]["body"] == "VERBATIM"
    print(f"[execute  ] posts_verbatim_when_approved expect=VERBATIM      got={res['result']['comment']['body']:14} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append("posts_verbatim")

    print("\n" + ("ALL GATES PASS (pass^k)" if not fails else f"CRITICAL: {len(fails)} gate(s) failed: {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(run())
