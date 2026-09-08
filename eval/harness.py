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

from eval.tasks import (
    CLASSIFY_TASKS,
    GATE_TASKS,
    RETRIEVAL_GATE_TASKS,
    SURFACE_PIPELINE_TASKS,
)
from nodes.classify import classify
from nodes.execute import execute
from nodes.independent_review import independent_review


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
    for name, diagnosis, retrieved, policy, expected in GATE_TASKS:
        state = {"diagnosis": diagnosis, "_retrieved": retrieved}
        if policy is not None:
            state["_policy"] = policy
        check("gate", name, expected, independent_review(state)["review"].decision)

    # ── the Surface's own normalize -> classify chain (T5 regression) ────────
    from nodes.normalize_evidence import normalize_evidence
    from surfaces.code_review import CodeReviewSurface

    for name, raw, expected in SURFACE_PIPELINE_TASKS:
        surface = CodeReviewSurface()
        state = {"_surface": surface, "_raw": raw}
        state.update(normalize_evidence(state))
        check("pipeline", name, expected, classify(state)["route"])

    # ── the gate driven by the REAL retriever (T21 regression) ───────────────
    # The GATE_TASKS above hand-supply scores, so they pin the gate's logic but
    # cannot see whether its input is meaningful. These retrieve for real.
    # Through the ADAPTER, not tools/retrieval directly: the port's dict shape is
    # what a node actually receives, so that is what must be exercised.
    from adapters.retriever_hybrid import HybridRetriever
    from schemas.diagnosis import Citation, Diagnosis

    retriever = HybridRetriever()
    for name, query, expected in RETRIEVAL_GATE_TASKS:
        hits = retriever.retrieve(query, k=3)
        # cite the top hit, as a diagnosis grounded on this retrieval would
        cited = ([Citation(snippet_id=hits[0]["id"], score=hits[0]["score"])]
                 if hits else [])
        state = {
            # The claim paraphrases the demo corpus's auth snippet, so an
            # ON-TOPIC query grounds AND entails, while an off-topic one fails
            # the relevance check first.
            "diagnosis": Diagnosis(
                summary="auth regression", severity="medium",
                proposed_comment=("The OAuth callback returns a 500 when the state "
                                  "param is missing; fix in the auth middleware."),
                proposed_label=None, citations=cited),
            "_retrieved": hits,
        }
        check("retrieval", name, expected, independent_review(state)["review"].decision)

    # ── execute(): the permission guard and verbatim posting ─────────────────
    class FakeSink:
        """No test may ever reach a real provider. The suite must never post."""

        def __init__(self):
            self.calls = []

        def emit(self, action, *, target, identity=None):
            self.calls.append((action, target, identity))
            return {"comment": {"body": action["comment"]}, "label": action.get("label")}

    # refuses without approval -- checked IN THE NODE, so a future shortcut edge
    # straight to execute still cannot bypass the gate
    try:
        execute({"approved": False, "target": "u",
                 "proposed_action": {"comment": "x"}}, sink=FakeSink())
        check("execute", "refuses_without_approval", "PermissionError", "NO-RAISE")
    except PermissionError:
        check("execute", "refuses_without_approval", "PermissionError", "PermissionError")

    # approval must not be inferrable from a truthy-ish value
    truthy_but_not_true = (("approved_is_string", "true"), ("approved_is_1", 1),
                           ("approved_is_none", None))
    for label, bad in truthy_but_not_true:
        try:
            execute({"approved": bad, "target": "u",
                     "proposed_action": {"comment": "x"}}, sink=FakeSink())
            check("execute", label, "PermissionError", "NO-RAISE")
        except PermissionError:
            check("execute", label, "PermissionError", "PermissionError")

    # posts the approved action VERBATIM -- never recomputed after approval
    res = execute({"approved": True, "target": "u",
                   "proposed_action": {"comment": "VERBATIM", "label": None}}, sink=FakeSink())
    check("execute", "posts_verbatim_when_approved", "VERBATIM", res["result"]["comment"]["body"])

    # a failing label add is a PARTIAL success -- the comment really was posted,
    # so it must not be discarded by raising. This exercises the REAL
    # PRCommentSink, since that is where the partial-success handling lives now.
    from adapters.action_sinks import PRCommentSink
    from tools import github_client

    saved = (github_client.post_comment, github_client.add_label)
    github_client.post_comment = lambda url, body, token=None: {"body": body}

    def _label_403(url, label, token=None):
        raise RuntimeError("403 needs triage/write access")

    github_client.add_label = _label_403
    try:
        res = execute({"approved": True, "target": "u",
                       "proposed_action": {"comment": "posted", "label": "bug"}},
                      sink=PRCommentSink())
    finally:
        github_client.post_comment, github_client.add_label = saved
    partial = (res["result"]["comment"]["body"] == "posted"
               and "skipped" in (res["result"]["label"] or {}))
    check("execute", "label_failure_is_partial_success", "partial",
          "partial" if partial else "lost")

    total = (len(CLASSIFY_TASKS) + len(GATE_TASKS) + len(RETRIEVAL_GATE_TASKS)
             + len(SURFACE_PIPELINE_TASKS) + 6)   # 6 execute-guard checks
    print()
    if fails:
        print(f"CRITICAL: {len(fails)} of {total} gate(s) failed: {fails}")
    else:
        print(f"ALL {total} GATES PASS (pass^k)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
