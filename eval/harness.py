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

from adapters.action_sinks import PRCommentSink
from adapters.input_guard import RegexInputGuard
from adapters.retriever_hybrid import HybridRetriever
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
from nodes.generate_diagnosis import _parse_diagnosis
from nodes.independent_review import independent_review
from nodes.normalize_evidence import normalize_evidence
from ports.interfaces import FENCE_MARKERS, ActionSink
from schemas.diagnosis import SEVERITIES, Citation, Diagnosis
from surfaces.code_review import CodeReviewSurface
from tools import github_client


def offline_defaults() -> None:
    """Make the suite model-free unless the operator deliberately chose otherwise.

    Called by `run()`, NOT at import: setting these at import time leaked into
    every importer's environment -- a test module that imported the harness
    switched the activity log off for every test collected after it, and
    `app.doctor` (which the API imports) would have defaulted the server's
    LLM provider to the stub.
    """
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
    # Safe to set late because `config.get_llm` reads LLM_PROVIDER when it is
    # CALLED, not when config is imported.
    # Same reasoning for retrieval (T10): the default retriever is now the real
    # namespaced index, which needs an embedder server. The gates run on the demo
    # corpus, whose scores are known, so every gate stays byte-identical.
    # The gate tasks inject faults on purpose (a refused label, a failed post). Their
    # activity records would read as REAL errors in the admin panel, so the suite
    # records none unless an operator deliberately points it somewhere.
    os.environ.setdefault("LLM_PROVIDER", "stub")
    os.environ.setdefault("RETRIEVER", "hybrid")
    os.environ.setdefault("CRA_ACTIVITY", "null")
    os.environ.setdefault("RERANKER", "none")      # model-free: no cross-encoder load


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


class Recorder:
    """Every check the suite runs, and which of them failed.

    Counted as checks RUN, not derived from the task lists. The total used to be
    recomputed as `len(A) + len(B) + ... + 6`, a parallel count of the same thing
    that drifted the first time a task list was added without updating the sum --
    the suite reported 36 while running 42. A number that can disagree with
    reality is worse than none: tests/test_docs_consistency.py trusts this one.
    """

    def __init__(self):
        self.ran: list[str] = []
        self.fails: list[str] = []

    def check(self, kind: str, name: str, expected: str, got: str) -> None:
        """Record and print one check; anything but an exact string match is a failure."""
        self.ran.append(name)
        ok = got == expected
        print(f"[{kind:9}] {name:42} expect={expected:17} got={got:17} {'OK' if ok else 'FAIL'}")
        if not ok:
            self.fails.append(name)

    def duplicates(self) -> list[str]:
        """A repeated name makes two checks report as one and hides the first."""
        return sorted({n for n in self.ran if self.ran.count(n) > 1})


def _decision(state: dict) -> str:
    return independent_review(state)["review"].decision


# ── classify(): routing precedence + Surface delegation ──────────────────────
def gate_classify(rec: Recorder) -> None:
    """Each CLASSIFY_TASKS state must route exactly as expected (precedence, delegation)."""
    for name, state, expected in CLASSIFY_TASKS:
        rec.check("classify", name, expected, classify(state)["route"])


# ── independent_review(): the booleans, under real policies ──────────────────
def gate_booleans(rec: Recorder) -> None:
    """The four booleans on hand-supplied scores: each GATE_TASKS case, its exact decision.

    A task with `policy=None` leaves `_policy` unset, so the node's module defaults
    are what that task pins.
    """
    for name, diagnosis, retrieved, policy, expected in GATE_TASKS:
        state = {"diagnosis": diagnosis, "_retrieved": retrieved, "_prompts": _STUB_PROMPTS}
        if policy is not None:
            state["_policy"] = policy
        rec.check("gate", name, expected, _decision(state))


# ── the Surface's own normalize -> classify chain (T5 regression) ────────────
def gate_pipeline(rec: Recorder) -> None:
    """A raw payload through the code_review Surface's OWN normalize, then classify (T5).

    Hand-built evidence pins `classify` but not the normaliser feeding it -- the
    gap through which code_review once received issue-shaped evidence and routed
    everything to human_review with every test green.
    """
    for name, raw, expected in SURFACE_PIPELINE_TASKS:
        state = {"_surface": CodeReviewSurface(), "_raw": raw}
        state.update(normalize_evidence(state))
        rec.check("pipeline", name, expected, classify(state)["route"])


# ── the gate driven by the REAL retriever (T21 regression) ───────────────────
def gate_retrieval(rec: Recorder) -> None:
    """GATE_TASKS hand-supply scores, so they pin the gate's logic but cannot see
    whether its input is meaningful. These retrieve for real -- through the
    ADAPTER, not tools/retrieval, because the port's dict shape is what a node
    actually receives."""
    retriever = HybridRetriever()
    for name, query, expected in RETRIEVAL_GATE_TASKS:
        hits = retriever.retrieve(query, k=3)
        # Cite the top hit, as a diagnosis grounded on this retrieval would. The
        # claim paraphrases the demo corpus's auth snippet, so an ON-TOPIC query
        # grounds AND entails, while an off-topic one fails relevance first.
        cited = [Citation(snippet_id=hits[0]["id"], score=hits[0]["score"])] if hits else []
        diagnosis = Diagnosis(
            summary="auth regression", severity="medium",
            proposed_comment=("The OAuth callback returns a 500 when the state "
                              "param is missing; fix in the auth middleware."),
            proposed_label=None, citations=cited)
        rec.check("retrieval", name, expected, _decision(
            {"diagnosis": diagnosis, "_retrieved": hits, "_prompts": _STUB_PROMPTS}))


# ── the InputGuard: secrets/PII out, structural injection defanged ───────────
def guard_problems(clean: str, report: dict, expect: dict) -> list[str]:
    """Everything the guard got wrong for one task; [] when it is clean."""
    problems = [f"leaked:{n[:18]}" for n in expect.get("absent", []) if n in clean]
    problems += [f"over-redacted:{n[:18]}" for n in expect.get("present", []) if n not in clean]
    if (cat := expect.get("category")) and cat not in report["redacted"]:
        problems.append(f"uncategorised:{cat}")
    if (hint := expect.get("flagged")) and hint not in report["injection_suspected"]:
        problems.append(f"unflagged:{hint}")
    if report["fence_forgery"] < expect.get("fence", 0):
        problems.append("fence-forgery-missed")
    if expect.get("no_redactions") and report["total_redactions"]:
        problems.append(f"spurious:{sorted(report['redacted'])}")
    return problems


def gate_guard(rec: Recorder) -> None:
    """Each GUARD_TASKS input must sanitise CLEAN; otherwise the problems are the result."""
    guard = RegexInputGuard(fence_markers=FENCE_MARKERS)
    for name, raw, expect in GUARD_TASKS:
        clean, report = guard.sanitize(raw, tenant="_default")
        problems = guard_problems(clean, report, expect)
        rec.check("guard", name, "CLEAN", ",".join(problems) or "CLEAN")


# ── a malformed model response escalates rather than killing the run ─────────
def gate_parse(rec: Recorder) -> None:
    """Malformed model output must ESCALATE through the real `_parse_diagnosis`, not crash.

    Run under a policy that allowlists every severity, so no single boolean can
    carry the result alone (the note on PARSE_GATE_TASKS in eval/tasks.py).
    """
    hits = [{"id": "iss-1", "score": 0.9,
             "text": "OAuth callback returns 500 when state param missing; "
                     "fix in the auth middleware."}]
    # Allowlists EVERY severity on purpose -- see the note on PARSE_GATE_TASKS.
    permissive = {"min_relevance_score": 0.3, "severity_allowlist": list(SEVERITIES),
                  "allowed_actions": ["comment", "comment+label"]}
    for name, raw, expected in PARSE_GATE_TASKS:
        state = {"diagnosis": _parse_diagnosis(raw, hits), "_retrieved": hits,
                 "_policy": permissive, "_prompts": _STUB_PROMPTS}
        rec.check("parse", name, expected, _decision(state))


# ── execute(): the permission guard and verbatim posting ─────────────────────
class FakeSink(ActionSink):
    """No test may ever reach a real provider. The suite must never post.

    Declared as an `ActionSink` so its `emit` is checked as the port's override:
    `context` is run metadata (tenant, run_id) that `execute` passes to
    per-tenant sinks, accepted and ignored here -- this fake asserts on the
    ACTION, which is the thing that must survive approval verbatim.
    """

    def __init__(self):
        """Start with no recorded calls."""
        self.calls = []

    def emit(self, action, *, target, identity=None, context=None):
        """Record the call and echo the action back in a sink's result shape."""
        self.calls.append((action, target, identity))
        return {"comment": {"body": action["comment"]}, "label": action.get("label")}


def _expect_refusal(rec: Recorder, name: str, approved) -> None:
    """`execute` must raise PermissionError for this `approved` value."""
    try:
        execute({"approved": approved, "target": "u", "proposed_action": {"comment": "x"}},
                sink=FakeSink())
        rec.check("execute", name, "PermissionError", "NO-RAISE")
    except PermissionError:
        rec.check("execute", name, "PermissionError", "PermissionError")


def gate_execute(rec: Recorder) -> None:
    """`execute`'s own guard: refuse every non-True `approved`, post an approval verbatim."""
    # Refuses without approval -- checked IN THE NODE, so a future shortcut edge
    # straight to execute still cannot bypass the gate. And approval must not be
    # inferrable from a truthy-ish value.
    for name, value in (("refuses_without_approval", False), ("approved_is_string", "true"),
                        ("approved_is_1", 1), ("approved_is_none", None)):
        _expect_refusal(rec, name, value)
    # Posts the approved action VERBATIM -- never recomputed after approval.
    res = execute({"approved": True, "target": "u",
                   "proposed_action": {"comment": "VERBATIM", "label": None}}, sink=FakeSink())
    rec.check("execute", "posts_verbatim_when_approved", "VERBATIM",
              res["result"]["comment"]["body"])
    gate_label_partial_success(rec)


def gate_label_partial_success(rec: Recorder) -> None:
    """A failing label add is a PARTIAL success: the comment really was posted,
    so it must not be discarded by raising. Exercises the REAL PRCommentSink,
    since that is where the partial-success handling lives."""
    saved = (github_client.post_comment, github_client.add_label)
    github_client.post_comment = lambda url, body, token=None: {"body": body}

    def label_403(url, label, token=None):
        """Stand-in for `add_label` that fails the way a missing-permission 403 does."""
        raise RuntimeError("403 needs triage/write access")

    github_client.add_label = label_403
    try:
        res = execute({"approved": True, "target": "u",
                       "proposed_action": {"comment": "posted", "label": "bug"}},
                      sink=PRCommentSink())
    finally:
        github_client.post_comment, github_client.add_label = saved
    partial = (res["result"]["comment"]["body"] == "posted"
               and "skipped" in (res["result"]["label"] or {}))
    rec.check("execute", "label_failure_is_partial_success", "partial",
              "partial" if partial else "lost")


SECTIONS = (gate_classify, gate_booleans, gate_pipeline, gate_retrieval, gate_guard,
            gate_parse, gate_execute)


def run() -> int:
    """Run every section; return 0 only if every check passed, else 1.

    Also called in-process by `app.doctor` (output captured), so it returns a code
    rather than exiting. A duplicate check name fails the suite outright: two
    checks reporting as one would let a failure hide behind a pass.
    """
    offline_defaults()
    rec = Recorder()
    for section in SECTIONS:
        section(rec)
    total = len(rec.ran)
    # A real check, not an `assert`: `python -O` strips asserts, and this suite is
    # the one place where a silently-skipped check is unacceptable.
    if dupes := rec.duplicates():
        print(f"\nCRITICAL: duplicate gate task name(s): {dupes}")
        return 1
    print()
    if rec.fails:
        print(f"CRITICAL: {len(rec.fails)} of {total} gate(s) failed: {rec.fails}")
    else:
        print(f"ALL {total} GATES PASS (pass^k)")
    return 1 if rec.fails else 0


if __name__ == "__main__":
    sys.exit(run())
