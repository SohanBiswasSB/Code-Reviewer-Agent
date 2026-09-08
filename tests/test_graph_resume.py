"""The approval gate, end to end: resolve -> suspend -> resume -> execute.

These are the tests for T14 (the graph never resolved the tenant), T4 (two
approval mechanisms for one gate) and T3 (nothing ever wrote `approved`, so every
real run hit `execute`'s PermissionError).

Only the GitHub *network calls* are stubbed. Everything else — the graph, the
checkpointer, `interrupt()`, the gate booleans, the permission guard — is the
real code path.
"""
import pytest

pytest.importorskip("langgraph")

# `config` must be imported BEFORE any test overrides a registered surface.
# Importing it triggers `surfaces/__init__`, whose @surface decorators register
# the real classes. The nodes import config lazily, so without this the FIRST
# lazy import inside a run would re-register CodeReviewSurface and silently
# clobber the override -- and the test would then exercise the real surface while
# appearing to use the stub. Load-bearing, not tidy-uppable.
import config  # noqa: E402, F401
import registry  # noqa: E402
from tools import github_client  # noqa: E402


@pytest.fixture
def posted(monkeypatch):
    """Capture what would have been written, and guarantee no real call escapes."""
    sent = []
    monkeypatch.setattr(github_client, "fetch_issue", lambda url, token=None: {
        "html_url": url,
        "title": "OAuth callback returns 500 when state param missing",
        "body": "help",                      # sparse -> ai_investigation
        "labels": [{"name": "bug"}],
    })
    monkeypatch.setattr(github_client, "get_check_runs", lambda url, token=None: [])
    monkeypatch.setattr(github_client, "post_comment",
                        lambda url, body, token=None: (sent.append(body), {"posted": body})[1])
    monkeypatch.setattr(github_client, "add_label",
                        lambda url, label, token=None: {"labelled": label})
    return sent


@pytest.fixture
def issue_surface(monkeypatch):
    """A surface whose classify matches ISSUE-shaped evidence.

    The real `code_review` surface routes on `touched_areas`, which issue
    evidence does not carry — it therefore returns `human_review` and never
    reaches the gate. That mismatch is T5; it fails safe, but it means the gate
    cannot be exercised through that surface yet.
    """
    class IssueShaped:
        name = "code_review"
        source_name, sink_name = "github", "pr_comment"
        diagnosis_prompt = judge_prompt = "diagnosis"

        def build_query(self, ev):
            return ev.get("title", "")

        def classify(self, ev):
            if ev.get("has_failing_ci"):
                return "deterministic"
            return "ai_investigation" if ev.get("is_information_sparse") else "human_review"

    monkeypatch.setitem(registry._SURFACES, "code_review", IssueShaped)
    return IssueShaped


def _graph():
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph import build_graph
    return build_graph(checkpointer=MemorySaver())


def _cfg(name):
    return {"configurable": {"thread_id": name}}


def _start(graph, cfg, tenant="_default", surface="code_review", target="https://x/issues/1"):
    return graph.invoke({"tenant": tenant, "surface": surface, "target": target}, config=cfg)


# ── T4: one gate, and the checkpointer is not optional ───────────────────────
def test_build_graph_refuses_without_a_checkpointer():
    """interrupt() suspends by persisting; with nowhere to persist, the gate
    would evaporate on restart. Refuse loudly instead of compiling."""
    from app.graph import build_graph
    with pytest.raises(ValueError, match="requires a checkpointer"):
        build_graph()


def test_run_suspends_at_the_single_gate(posted, issue_surface):
    g, cfg = _graph(), _cfg("t-suspend")
    _start(g, cfg)
    assert g.get_state(cfg).next == ("await_approval",), "must suspend at await_approval"
    assert posted == [], "nothing may be written before approval"


def test_the_human_is_shown_what_they_are_approving(posted, issue_surface):
    """A gate whose contents the approver cannot see is a rubber stamp -- which
    is why interrupt() carries a payload and interrupt_before was dropped."""
    g, cfg = _graph(), _cfg("t-payload")
    out = _start(g, cfg)
    payload = out["__interrupt__"][0].value
    assert payload["proposed_action"]["comment"]
    assert payload["policy_version"], "the approver must see WHICH guardrails applied"
    assert payload["target"] == "https://x/issues/1"


# ── T14: the tenant is actually resolved ─────────────────────────────────────
def test_policy_is_resolved_into_state(posted, issue_surface):
    g, cfg = _graph(), _cfg("t-policy")
    out = _start(g, cfg)
    assert out["_policy"]["min_relevance_score"] == 0.35
    assert out["_surface_name"] == "code_review"
    assert out["_policy_version"]


def test_each_tenant_is_gated_by_its_own_policy_on_a_real_run(posted, issue_surface):
    """T14's "done when": a real graph run is judged by the TENANT's policy.

    Deliberately asserts *resolution*, not threshold arithmetic. The evidence
    here grounds at 0.5385, which clears both _default's 0.35 and project_a's
    0.40, so this run cannot show a divergent outcome — and forcing one would
    mean hunting a query that lands between the two on a 4-document Jaccard
    corpus, which would be a brittle test of the corpus rather than of the code.
    The threshold semantics are pinned precisely by the paired GATE_TASKS
    (`same_diagnosis_lenient_policy_approves` / `..._strict_policy_escalates`).

    Before T14 both runs would have shown the module default 0.35, because
    nothing ever called the TenantStore.
    """
    lenient, lenient_cfg = _graph(), _cfg("t-lenient")
    out_lenient = _start(lenient, lenient_cfg, tenant="_default")

    strict, strict_cfg = _graph(), _cfg("t-strict")
    out_strict = _start(strict, strict_cfg, tenant="project_a")

    assert out_lenient["_policy"]["min_relevance_score"] == 0.35
    assert out_strict["_policy"]["min_relevance_score"] == 0.40
    assert out_lenient["_policy_version"] != out_strict["_policy_version"], (
        "different policies must pin different versions, or a resumed run "
        "cannot prove which guardrails it was judged against"
    )
    # project_a also carries a codebase binding; _default does not
    assert out_strict["_policy"]["codebase"]["repo"] == "org/project-a"


def test_a_surface_the_tenant_has_not_enabled_is_refused_in_the_graph(posted):
    """project_b enables code_review only. Checked in the graph, not just at the
    HTTP boundary -- the scheduled job does not go through HTTP."""
    g = _graph()
    with pytest.raises(PermissionError, match="has not enabled surface"):
        _start(g, _cfg("t-blocked"), tenant="project_b",
               surface="log_review", target="2026-09-01")


# ── T3: resume actually sets `approved`, and only a human can ────────────────
def test_approval_executes_and_posts_verbatim(posted, issue_surface):
    from langgraph.types import Command

    g, cfg = _graph(), _cfg("t-approve")
    out = _start(g, cfg)
    shown = out["__interrupt__"][0].value["proposed_action"]["comment"]

    resumed = g.invoke(Command(resume=True), config=cfg)
    assert resumed["approved"] is True
    assert posted == [shown], "what executed must be exactly what was approved"


def test_denial_refuses_and_writes_nothing(posted, issue_surface):
    from langgraph.types import Command

    g, cfg = _graph(), _cfg("t-deny")
    _start(g, cfg)
    with pytest.raises(PermissionError):
        g.invoke(Command(resume=False), config=cfg)
    assert posted == []


@pytest.mark.parametrize("resume_value", ["yes please", 1, {"approved": "yes"}])
def test_truthy_is_not_approval(posted, issue_surface, resume_value):
    """A truthy resume value must not become an approval.

    These all resume the graph, so `execute`'s `is not True` check is what stops
    them — which is exactly the defence-in-depth the guard exists for.
    """
    from langgraph.types import Command

    g, cfg = _graph(), _cfg(f"t-junk-{resume_value!r}")
    _start(g, cfg)
    with pytest.raises(PermissionError):
        g.invoke(Command(resume=resume_value), config=cfg)
    assert posted == []


def test_a_falsy_resume_does_not_resume_at_all(posted, issue_surface):
    """Measured LangGraph behaviour: a falsy resume value is treated as "no
    resume provided", so the run stays parked at the gate rather than continuing.

    Also safe — nothing executes — but a *different* mechanism from the
    PermissionError above, so it is pinned separately. If a future LangGraph
    version starts resuming on a falsy value, `execute`'s guard is the backstop
    and this test will tell us the mechanism changed.

    (`Command(resume=None)` is deliberately not tested: it raises
    `UnboundLocalError: cannot access local variable 'resume_is_map'` from
    langgraph/pregel/_loop.py — an upstream bug, not our behaviour. `/approve`
    coerces to a real bool so it can never send None.)
    """
    from langgraph.types import Command

    g, cfg = _graph(), _cfg("t-falsy")
    _start(g, cfg)
    g.invoke(Command(resume={}), config=cfg)
    assert g.get_state(cfg).next == ("await_approval",), "should still be parked"
    assert posted == []


def test_dict_shaped_approval_is_accepted(posted, issue_surface):
    """A JSON body {"approved": true} is a legitimate resume shape."""
    from langgraph.types import Command

    g, cfg = _graph(), _cfg("t-dict")
    _start(g, cfg)
    resumed = g.invoke(Command(resume={"approved": True}), config=cfg)
    assert resumed["approved"] is True
    assert len(posted) == 1
