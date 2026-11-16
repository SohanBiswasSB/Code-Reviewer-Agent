"""Run the safety-gate suite. Exit non-zero on ANY miss (pass^k, not an average).

    python -m eval.harness

This is a REGRESSION suite, not a capability suite. Every task has one
unambiguous correct answer and the bar is 100% forever: a single miss means a
safety gate weakened, not that the system "needs more data". Keeping the two
kinds of eval separate is deliberate — a gate that only mostly holds is not a
gate. Capability evaluation (is the diagnosis any *good*?) is tracked separately
as T10/G17 and is expected to improve over time rather than sit at 100%.

Deliberately model-free: no provider, no server, no network. (It does need
`requirements.txt`: it imports `config`, which imports every adapter.)
It must therefore give byte-identical results on the Mac and the CUDA box.
"""
from __future__ import annotations

import os
import sys

from eval.tasks import (
    CLASSIFY_TASKS,
    GATE_TASKS,
    GUARD_TASKS,
    PARSE_GATE_TASKS,
    RETRIEVAL_GATE_TASKS,
    SURFACE_PIPELINE_TASKS,
)
from nodes.classify import classify
from nodes.execute import execute
from nodes.independent_review import independent_review
from schemas.diagnosis import SEVERITIES

# MODEL-FREE, DECLARED rather than inherited. The LLM nodes now resolve their
# model from config like every other provider, so "no llm argument" no longer
# silently means "use a stub" -- without this line the gate suite would try to
# reach LM Studio and stop being runnable on a bare machine.
#
# `setdefault`, not assignment: an operator who deliberately exports
# LLM_PROVIDER=lmstudio to see how a real model fares against the gates can still
# do so. The DEFAULT is offline because that is what makes this suite a
# regression suite -- byte-identical on the Mac and the CUDA box, 100% bar, no
# GPU required.
#
# Safe below the imports because `config.get_llm` reads LLM_PROVIDER when it is
# CALLED, not when config is imported.
os.environ.setdefault("LLM_PROVIDER", "stub")
# Same reasoning for retrieval (T10): the default retriever is now the real
# namespaced index, which needs an embedder server. The gates run on the demo
# corpus, whose scores are known, so every gate stays byte-identical.
os.environ.setdefault("RETRIEVER", "hybrid")


# The gate tasks build synthetic states to exercise the FOUR BOOLEANS in
# isolation. They now also need a pinned judge prompt, because the judge call is
# no longer skipped when no llm is injected -- `independent_review` resolves the
# stub from config and then (correctly) refuses to run without the prompt
# `resolve_tenant` is supposed to have pinned (T6).
#
# That is a small improvement to what this suite covers: these tasks now drive the
# real judge call path, not just the boolean arithmetic after it. The prompt text
# is irrelevant -- the stub judge ignores it, and by design no judge output can
# affect the decision.
_STUB_PROMPTS = {"diagnosis": "[gate-suite] diagnosis prompt",
                 "judge": "[gate-suite] judge prompt"}


def run() -> int:
    fails: list[str] = []
    # Counted as checks RUN, not derived from the task lists. The total used to be
    # recomputed as `len(A) + len(B) + ... + 6`, which is a parallel count of the
    # same thing and drifted the first time a task list was added without updating
    # the sum -- the suite reported 36 while running 42. A number that can
    # disagree with reality is worse than no number, because
    # tests/test_docs_consistency.py trusts this one to check MODEL_CARD's claim.
    ran: list[str] = []

    def check(kind: str, name: str, expected: str, got: str) -> None:
        ran.append(name)
        ok = got == expected
        print(f"[{kind:9}] {name:42} expect={expected:17} got={got:17} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    # ── classify(): routing precedence + Surface delegation ──────────────────
    for name, state, expected in CLASSIFY_TASKS:
        check("classify", name, expected, classify(state)["route"])

    # ── independent_review(): the three booleans, under real policies ────────
    for name, diagnosis, retrieved, policy, expected in GATE_TASKS:
        state = {"diagnosis": diagnosis, "_retrieved": retrieved,
                 "_prompts": _STUB_PROMPTS}
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
            "_prompts": _STUB_PROMPTS,
        }
        check("retrieval", name, expected, independent_review(state)["review"].decision)

    # ── the InputGuard: secrets/PII out, structural injection defanged ───────
    from adapters.input_guard import RegexInputGuard
    from nodes._render import FENCE_MARKERS

    guard = RegexInputGuard(fence_markers=FENCE_MARKERS)
    for name, raw, expect in GUARD_TASKS:
        clean, report = guard.sanitize(raw, tenant="_default")
        problems = []
        for needle in expect.get("absent", []):
            if needle in clean:
                problems.append(f"leaked:{needle[:18]}")
        for needle in expect.get("present", []):
            if needle not in clean:
                problems.append(f"over-redacted:{needle[:18]}")
        if (cat := expect.get("category")) and cat not in report["redacted"]:
            problems.append(f"uncategorised:{cat}")
        if (hint := expect.get("flagged")) and hint not in report["injection_suspected"]:
            problems.append(f"unflagged:{hint}")
        if report["fence_forgery"] < expect.get("fence", 0):
            problems.append("fence-forgery-missed")
        if expect.get("no_redactions") and report["total_redactions"]:
            problems.append(f"spurious:{sorted(report['redacted'])}")
        check("guard", name, "CLEAN", "CLEAN" if not problems else ",".join(problems))

    # ── a malformed model response escalates rather than killing the run ─────
    from nodes.generate_diagnosis import _parse_diagnosis

    parse_hits = [{"id": "iss-1", "score": 0.9,
                   "text": ("OAuth callback returns 500 when state param missing; "
                            "fix in the auth middleware.")}]
    # Allowlists EVERY severity on purpose -- see the note on PARSE_GATE_TASKS.
    permissive = {"min_relevance_score": 0.3,
                  "severity_allowlist": list(SEVERITIES),
                  "allowed_actions": ["comment", "comment+label"]}
    for name, raw, expected in PARSE_GATE_TASKS:
        diag = _parse_diagnosis(raw, parse_hits)
        state = {"diagnosis": diag, "_retrieved": parse_hits,
                 "_policy": permissive, "_prompts": _STUB_PROMPTS}
        check("parse", name, expected, independent_review(state)["review"].decision)

    # ── execute(): the permission guard and verbatim posting ─────────────────
    class FakeSink:
        """No test may ever reach a real provider. The suite must never post."""

        def __init__(self):
            self.calls = []

        # `context` is run metadata (tenant, run_id) that `execute` passes to
        # per-tenant sinks. Accepted and ignored -- this fake asserts on the
        # ACTION, which is the thing that must survive approval verbatim.
        def emit(self, action, *, target, identity=None, context=None):
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

    total = len(ran)
    # A duplicated task name would make two different checks report as one and
    # hide whichever ran first. Cheap to assert, and this suite is the one place
    # where a silently-skipped check is unacceptable.
    assert len(set(ran)) == total, (
        f"duplicate gate task name(s): "
        f"{sorted({n for n in ran if ran.count(n) > 1})}"
    )
    print()
    if fails:
        print(f"CRITICAL: {len(fails)} of {total} gate(s) failed: {fails}")
    else:
        print(f"ALL {total} GATES PASS (pass^k)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
