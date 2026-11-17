"""The approval gate, end to end: resolve -> suspend -> resume -> execute.

These are the tests for T14 (the graph never resolved the tenant), T4 (two
approval mechanisms for one gate) and T3 (nothing ever wrote `approved`, so every
real run hit `execute`'s PermissionError).

Only the GitHub *network calls* are stubbed. Everything else — the graph, the
checkpointer, `interrupt()`, the gate booleans, the permission guard — is the
real code path.
"""
import pytest

import registry
from tools import github_client


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


def test_a_tenant_with_a_codebase_forces_namespaced_retrieval(posted, issue_surface):
    """G1, end to end: a tenant bound to a codebase retrieves in its namespace.

    project_a's policy declares `codebase.repo = org/project-a`, so
    `generate_diagnosis` passes that as the retrieval namespace. The demo
    retriever cannot partition a four-document in-memory corpus, so it REFUSES.

    That refusal is the assertion. It proves three things at once, none of which
    was true before T14/T5: the tenant's policy is resolved into the run, the
    namespace reaches the retriever, and an adapter that cannot honour isolation
    fails loudly rather than quietly retrieving across codebases. Grounding is
    the whole safety story, so a silent cross-codebase citation would be the
    worst available failure.

    `_default` declares no codebase, so it retrieves un-namespaced and proceeds —
    which is why the other tests in this file use it.
    """
    g = _graph()
    with pytest.raises(NotImplementedError, match="namespace-scoped"):
        _start(g, _cfg("t-namespaced"), tenant="project_a")
    assert posted == []


def test_default_tenant_retrieves_unnamespaced_and_reaches_the_gate(posted, issue_surface):
    """The counterpart: no codebase binding means no namespace, so the demo
    retriever can serve it and the run reaches the approval gate."""
    g, cfg = _graph(), _cfg("t-unnamespaced")
    out = _start(g, cfg, tenant="_default")
    assert out["_policy"].get("codebase") in ({}, None)
    assert g.get_state(cfg).next == ("await_approval",)


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


# ── T15: paused runs must survive a stricter serialiser ──────────────────────
@pytest.mark.parametrize("value_factory", [
    lambda: __import__("schemas.evidence", fromlist=["x"]).IssueEvidence(url="u", title="t"),
    lambda: __import__("schemas.diagnosis", fromlist=["x"]).Citation(snippet_id="i", score=0.5),
    lambda: __import__("schemas.diagnosis", fromlist=["x"]).Diagnosis(summary="s"),
    lambda: __import__("schemas.diagnosis", fromlist=["x"]).ReviewResult(reasoning="r"),
])
def test_every_state_type_round_trips_through_the_checkpoint_serialiser(value_factory):
    """The durable pause is the basis of the approval gate, so every type that
    enters a channel must survive being written and read back.

    LangGraph warns on an unregistered type -- "deserializing unregistered type
    ... will be blocked in a future version" -- blocks it under
    LANGGRAPH_STRICT_MSGPACK=true, and will block by default eventually. A
    blocked type does NOT raise: it comes back as a plain `dict`, so a resumed
    run silently loses the model it was written with.

    The serialiser therefore comes from `app.state.checkpoint_serde()`, which
    carries the CHECKPOINT_TYPES allowlist, and is the same one `app/deps.py`
    gives both checkpointers. Constructing a bare `JsonPlusSerializer()` here
    would test the default rather than what the service actually runs -- and
    would pass under strict mode only by returning dicts.
    """
    from app.state import checkpoint_serde

    serde = checkpoint_serde()
    original = value_factory()
    restored = serde.loads_typed(serde.dumps_typed(original))
    assert restored == original
    assert type(restored) is type(original)


def test_nested_models_survive_the_round_trip():
    """Registering the OUTER type must be enough to rebuild what it contains.

    `Diagnosis.citations[]` and `.findings[]` are the values the gate and the
    audit record read; `IssueEvidence.check_runs[]` is what `classify` routes on
    via `has_failing_ci`. If a nested model came back as a dict, the failure
    would surface as an AttributeError inside a node, far from the cause.
    """
    from app.state import checkpoint_serde
    from schemas.diagnosis import Citation, Diagnosis, Finding
    from schemas.evidence import CheckRun, IssueEvidence

    serde = checkpoint_serde()

    diag = Diagnosis(summary="s", citations=[Citation(snippet_id="i", score=0.5)],
                     findings=[Finding(file="f.py")])
    back = serde.loads_typed(serde.dumps_typed(diag))
    assert type(back.citations[0]) is Citation
    assert type(back.findings[0]) is Finding
    assert back == diag

    ev = IssueEvidence(url="u", title="t",
                       check_runs=[CheckRun(name="ci", conclusion="failure")])
    back_ev = serde.loads_typed(serde.dumps_typed(ev))
    assert type(back_ev.check_runs[0]) is CheckRun
    # The computed property is what the route depends on, so assert the property
    # rather than only the field it is derived from.
    assert back_ev.has_failing_ci is True


def test_every_schema_model_is_registered_for_the_checkpoint():
    """A model added to `schemas/` but not to CHECKPOINT_TYPES fails HERE.

    This is the tripwire, and it is the reason the allowlist lists nested models
    it does not strictly need: the failure it guards against is silent, so the
    check has to be exhaustive rather than clever. Adding a model and forgetting
    to register it is exactly the kind of omission that only shows up when a
    paused run resumes -- which is the one moment it must not.
    """
    import inspect

    from pydantic import BaseModel

    import schemas.diagnosis as diagnosis
    import schemas.evidence as evidence
    from app.state import CHECKPOINT_TYPES

    registered = set(CHECKPOINT_TYPES)
    missing = []
    for module in (evidence, diagnosis):
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and obj.__module__ == module.__name__):
                if (module.__name__, name) not in registered:
                    missing.append(f"{module.__name__}.{name}")

    assert not missing, (
        f"not in app.state.CHECKPOINT_TYPES: {missing}. An unregistered model "
        f"comes back from a checkpoint as a plain dict, silently."
    )


# ── T25: nothing unredacted reaches the checkpoint ──────────────────────────
def test_a_secret_in_the_evidence_never_reaches_any_checkpoint(posted, issue_surface,
                                                              monkeypatch):
    """The CHECKPOINT, not just the prompt. A Postgres saver keeps every
    super-step, so a secret written to `_raw` by fetch_evidence would stay on
    disk even if a later node dropped it. Checked over the whole HISTORY."""
    secret, email = "AKIA" + "ABCDEFGHIJKLMNOP", "student" + "@example.com"
    monkeypatch.setattr(github_client, "fetch_issue", lambda url, token=None: {
        "html_url": url, "title": f"OAuth 500 for {email} with key {secret}",
        "body": "help", "labels": [{"name": "bug"}]})
    g, cfg = _graph(), _cfg("t-t25")
    _start(g, cfg)
    snapshots = list(g.get_state_history(cfg))
    assert len(snapshots) > 3, "the run must have checkpointed several super-steps"
    for snap in snapshots:
        dumped = repr(snap.values)
        assert secret not in dumped and email not in dumped, \
            f"unredacted evidence in the checkpoint after {snap.next}"
    final = g.get_state(cfg).values
    assert "[REDACTED:aws_access_key_id:1]" in final["_raw"]["title"]
    assert final["_raw"]["html_url"] == "https://x/issues/1", "identifiers are never redacted"
    # Counted once, in the run's report, even though the prompt pass saw only
    # the placeholder.
    assert final["_guard"]["redacted"]["aws_access_key_id"] == 1
    assert final["_guard"]["evidence"]["redacted"]["email"] == 1


# ── G9: a crashed run still writes its audit record ─────────────────────────
def test_a_node_that_crashes_still_leaves_an_audit_record(posted, issue_surface, monkeypatch):
    """No failure edges exist, so the record is written from the one wrapper every
    node runs in -- and the exception still propagates unchanged."""
    from adapters.audit_jsonl import iter_records

    def broken(url, token=None):
        raise ConnectionError("github down for token ghp_" + "A" * 36)
    monkeypatch.setattr(github_client, "fetch_issue", broken)
    with pytest.raises(ConnectionError):
        _start(_graph(), _cfg("t-g9"))
    records = [r for r in iter_records() if r.get("outcome") == "failed"]
    assert len(records) == 1, "exactly one record for the crashed run"
    err = records[0]["error"]
    assert err["node"] == "fetch_evidence"
    assert "ConnectionError" in err["error"] and "ghp_" not in err["error"], \
        "the message is recorded through the guard"
