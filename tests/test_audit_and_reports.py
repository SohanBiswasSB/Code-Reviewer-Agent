"""G6 audit records and log-review report artefacts (docs/DATA_STORAGE.md).

The graph tests here drive the REAL graph with only the GitHub network calls
stubbed, for the same reason test_graph_resume.py does: an audit trail asserted
against a hand-built state dict would not prove that a run actually reaches the
node that writes it.

`CRA_DATA_DIR` is redirected to a tmp dir by the autouse fixture in conftest.py,
so nothing here can write into the working tree.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Command

import config
from adapters.action_sinks import ReportSink
from adapters.audit_jsonl import JsonlAuditSink
from nodes._audit import build_record, write
from nodes.independent_review import independent_review
from schemas.diagnosis import Citation, Diagnosis
from surfaces.log_review import LogReviewSurface
from tests._graph_fakes import fresh_graph, issue_surface_registered, stub_github


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(name="posted")
def fixture_posted(monkeypatch):
    """GitHub's network calls recorded instead of made; the list of posted bodies."""
    return stub_github(monkeypatch)


@pytest.fixture(name="issue_surface")
def fixture_issue_surface():
    """Issue-shaped surface, as in test_graph_resume.py -- the real code_review
    surface routes on `touched_areas` and never reaches the gate (T5)."""
    with issue_surface_registered() as surface:
        yield surface


def _json(path):
    """A report file's parsed body (read whole, so no handle outlives the read)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(tenant="_default"):
    """Every audit line written for a tenant, in order."""
    path = JsonlAuditSink().path_for(tenant)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── the record is written on every terminal path ─────────────────────────────
@pytest.mark.usefixtures("issue_surface")
def test_an_executed_run_is_recorded(posted):
    """An approved, executed run leaves one record that matches what was posted."""
    g = fresh_graph()
    cfg = {"configurable": {"thread_id": "a-exec"}}
    g.invoke({"tenant": "_default", "surface": "code_review",
              "target": "https://x/issues/1"}, config=cfg)
    g.invoke(Command(resume=True), config=cfg)

    rows = _records()
    assert len(rows) == 1
    r = rows[0]
    assert r["outcome"] == "executed"
    assert r["approved"] is True
    assert r["action"]["comment"] == posted[0], "the record must match what was posted"
    assert r["result"], "the provider's response belongs in the record"
    assert r["schema_version"] == 1 and r["recorded_at"]


@pytest.mark.usefixtures("posted", "issue_surface")
def test_the_record_carries_the_retrievers_own_scores():
    """The field that earns the layer.

    T21 -- the grounding gate inert against real retrieval -- was invisible for
    weeks and would have been ONE LINE of this, showing every score as 1.0 on
    every query. Asserting the scores are real and varied is the point.
    """
    g = fresh_graph()
    cfg = {"configurable": {"thread_id": "a-scores"}}
    g.invoke({"tenant": "_default", "surface": "code_review",
              "target": "https://x/issues/1"}, config=cfg)
    g.invoke(Command(resume=True), config=cfg)

    retrieved = _records()[0]["retrieved"]
    assert retrieved, "a grounded run must record what it retrieved"
    assert all(isinstance(s["score"], float) for s in retrieved)
    assert len({s["score"] for s in retrieved}) > 1, (
        "identical scores across hits is the T21 signature -- the gate would be inert"
    )


@pytest.mark.usefixtures("issue_surface")
def test_a_denial_is_recorded_even_though_execute_raises(posted):
    """The asymmetry documented in docs/DATA_STORAGE.md.

    A denial resumes into `execute`, whose PermissionError is what stops the run
    -- so it never reaches the `audit` node. `await_approval` records it instead.
    Without this, the one outcome where a human actively said no would be the one
    outcome left unrecorded.
    """
    g = fresh_graph()
    cfg = {"configurable": {"thread_id": "a-deny"}}
    g.invoke({"tenant": "_default", "surface": "code_review",
              "target": "https://x/issues/1"}, config=cfg)
    with pytest.raises(PermissionError):
        g.invoke(Command(resume=False), config=cfg)

    rows = _records()
    assert [r["outcome"] for r in rows] == ["denied"]
    assert rows[0]["approved"] is False
    assert posted == [], "nothing may be written on a denial"


@pytest.mark.usefixtures("posted")
def test_a_run_that_never_reaches_the_gate_is_still_recorded():
    """`human_review` is an OUTCOME, not the absence of one.

    The real code_review surface returns human_review for issue-shaped evidence
    (T5). That run does real work and reaches a real decision, so it belongs in
    the record just as much as one that posted.
    """
    g = fresh_graph()
    cfg = {"configurable": {"thread_id": "a-human"}}
    g.invoke({"tenant": "_default", "surface": "code_review",
              "target": "https://x/issues/1"}, config=cfg)

    rows = _records()
    assert len(rows) == 1
    assert rows[0]["outcome"] in ("human_review", "escalated")
    assert rows[0]["route"] == "human_review"


@pytest.mark.usefixtures("posted", "issue_surface")
def test_every_record_shares_one_run_id():
    """`_run_id` is minted once by resolve_tenant, so a resumed run correlates."""
    g = fresh_graph()
    cfg = {"configurable": {"thread_id": "a-runid"}}
    first = g.invoke({"tenant": "_default", "surface": "code_review",
                      "target": "https://x/issues/1"}, config=cfg)
    resumed = g.invoke(Command(resume=True), config=cfg)
    assert first["_run_id"] == resumed["_run_id"]
    assert _records()[0]["run_id"] == resumed["_run_id"]


# ── the sink itself ──────────────────────────────────────────────────────────
def test_the_token_is_never_written_to_the_record():
    """Attributability without storing the credential."""
    rec = build_record({"tenant": "t", "github_token": "ghp_supersecret"}, "executed")
    assert rec["approver_identity_present"] is True
    assert "ghp_supersecret" not in json.dumps(rec)


def test_an_audit_failure_does_not_destroy_the_run_it_describes():
    """On the executed path the comment has ALREADY been posted. Raising here
    would discard the fact that it happened -- strictly worse than a missing
    record. Same reasoning as PRCommentSink's partial-success handling."""
    def record(_run):
        raise OSError("disk full")

    out = write({"tenant": "t"}, "executed", sink=SimpleNamespace(record=record))
    assert out["_audit"]["written"] is False
    assert "disk full" in out["_audit"]["error"], "the failure must be visible, not swallowed"


@pytest.mark.parametrize("tenant", ["../../etc", "..", "", "a/b"])
def test_a_hostile_tenant_name_cannot_choose_the_path(tenant, tmp_path):
    """`tenant` arrives in a request body and reaches a filesystem path."""
    sink = JsonlAuditSink(root=str(tmp_path))
    path = os.path.realpath(sink.path_for(tenant))
    assert path.startswith(os.path.realpath(str(tmp_path)) + os.sep)


# ── layer 3: the log-review report ───────────────────────────────────────────
def test_log_review_can_finally_be_approved():
    """The prerequisite for Layer 3 existing at all.

    `independent_review` hardcoded the action type to comment/comment+label while
    every log-review policy allows only ["report"], so permission_ok was FALSE for
    every log-review run: the surface escalated 100% of the time and its
    ReportSink was unreachable. The action type now comes from the Surface.
    """
    passage = "OAuth callback returns 500 when state param missing; fix in auth middleware"
    diag = Diagnosis(summary="s", severity="low", proposed_comment=passage,
                     citations=[Citation(snippet_id="iss-101", score=0.9)])
    state = {
        "_surface": LogReviewSurface(),
        "diagnosis": diag,
        "_retrieved": [{"id": "iss-101", "score": 0.9, "text": passage}],
        "_policy": config.get_tenant_store().policy("_default", "log_review"),
        # The judge call is no longer skipped when no llm is injected -- the node
        # resolves one from config (the stub, per conftest) and then correctly
        # refuses to run without the prompt resolve_tenant should have pinned (T6).
        "_prompts": {"diagnosis": "[test] diagnosis", "judge": "[test] judge"},
    }
    out = independent_review(state)
    assert out["review"].permission_ok is True, out["review"].failures
    assert out["review"].decision == "approve"
    assert out["proposed_action"]["type"] == "report"


def test_the_report_is_written_as_json_and_markdown(tmp_path):
    """Layer 3's artefact: the analysis and its citations, as JSON and readable markdown."""
    sink = ReportSink(root=str(tmp_path))
    result = sink.emit(
        {"type": "report", "comment": "3 timeout spikes on the celery worker."},
        target="2026-09-10",
        context={"tenant": "milesone", "surface": "log_review", "run_id": "r-1",
                 "severity": "medium",
                 "citations": [{"snippet_id": "iss-101", "score": 0.61}]},
    )
    body = _json(result["report"]["json"])
    assert body["analysis"] == "3 timeout spikes on the celery worker."
    assert body["tenant"] == "milesone" and body["target"] == "2026-09-10"
    assert body["citations"][0]["snippet_id"] == "iss-101"

    md = Path(result["report"]["markdown"]).read_text(encoding="utf-8")
    assert "3 timeout spikes" in md and "iss-101" in md


def test_the_report_stores_the_analysis_not_the_logs(tmp_path):
    """The raw lines live in S3, which is the source of record. Copying them here
    would double the PII surface and buy nothing (docs/DATA_STORAGE.md)."""
    sink = ReportSink(root=str(tmp_path))
    result = sink.emit({"type": "report", "comment": "summary only"},
                       target="2026-09-10", context={"tenant": "t"})
    body = _json(result["report"]["json"])
    assert "lines" not in body and "raw" not in body
    assert body["candidates"] == [] and body["counts"] == {}, "awaiting G4"


def test_reports_are_separated_per_tenant(tmp_path):
    """Two codebases reviewing the same date must not collide."""
    sink = ReportSink(root=str(tmp_path))
    a = sink.emit({"comment": "A"}, target="2026-09-10", context={"tenant": "project_a"})
    b = sink.emit({"comment": "B"}, target="2026-09-10", context={"tenant": "project_b"})
    assert a["report"]["json"] != b["report"]["json"]
    assert _json(a["report"]["json"])["analysis"] == "A"


def test_the_decision_survives_a_review_that_came_back_as_a_dict():
    """`ReviewResult.decision` is a property, so it vanishes when state types
    round-trip to plain dicts — which is what happens across a resume under
    LANGGRAPH_STRICT_MSGPACK. Recomputed from the booleans rather than recorded
    as None for every run that paused at the gate."""
    as_dict = {"groundedness_ok": True, "risk_ok": True, "permission_ok": True,
               "entailment_ok": True, "failures": []}
    assert build_record({"review": as_dict}, "executed")["decision"] == "approve"

    as_dict["risk_ok"] = False
    assert build_record({"review": as_dict}, "executed")["decision"] == "escalate_to_human"
