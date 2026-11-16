"""The monitor's reading of var/ -- the parts that were wrong, not just slow."""
from __future__ import annotations

import json

from app import monitor


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(tenant, rid, ts):
    span = {"span": "node.audit", "status": "ok", "duration_ms": 5.0, "ts": ts,
            "run_id": rid, "tenant": tenant, "target": "staged"}
    rec = {"run_id": rid, "tenant": tenant, "target": "staged", "outcome": "escalated",
           "recorded_at": ts, "decision": "escalate_to_human"}
    return span, rec


def test_two_tenants_audits_do_not_push_each_other_out(tmp_path):
    """Tailing audit files in path order kept only the alphabetically LAST tenant's
    records, so the other tenant's finished runs were shown as 'Stopped'."""
    var = tmp_path / "var"
    spans, a_recs, b_recs = [], [], []
    for i in range(60):                      # masterclass: older, but still recent
        s, r = _run("masterclass", f"m{i}", f"2026-11-10T10:{i:02d}:00+00:00")
        spans.append(s)
        a_recs.append(r)
    for i in range(monitor.AUDIT_TAIL + 5):  # milesone: newer, and more of them
        s, r = _run("milesone", f"x{i}", f"2026-11-11T10:{i % 60:02d}:{i // 60:02d}+00:00")
        spans.append(s)
        b_recs.append(r)
    _write(var / "audit" / "masterclass" / "2026-11.jsonl", a_recs)
    _write(var / "audit" / "milesone" / "2026-11.jsonl", b_recs)
    audits = monitor._recent_audits()
    assert {"masterclass", "milesone"} <= {r["tenant"] for r in audits.values()}
    shown = monitor.runs(spans, live=[])
    assert all(r["status"] == "escalated" for r in shown), [r["status"] for r in shown]


def test_only_the_newest_files_are_read_for_a_tail(tmp_path):
    var = tmp_path / "var" / "traces"
    for day in range(1, 31):
        _write(var / f"2026-10-{day:02d}.jsonl",
               [{"span": "node.x", "ts": f"2026-10-{day:02d}T00:00:{i:02d}+00:00"}
                for i in range(50)])
    rows = monitor._newest_rows(list(var.glob("*.jsonl")), 120)
    assert len(rows) == 120 and rows[-1]["ts"].startswith("2026-10-30")
    assert rows[0]["ts"].startswith("2026-10-28"), "oldest kept row comes from 3 files back"
    assert rows == sorted(rows, key=lambda r: r["ts"]), "oldest first"


def test_the_model_table_uses_the_adapters_strict_matcher():
    """A looser monitor-side copy called a different publisher's model 'ready'."""
    server = {"reachable": True, "url": "http://x", "models": [
        {"id": "other/qwen3-4b", "state": "loaded"},
        {"id": "qwen/qwen3-4b", "state": "loaded"},
        {"id": "text-embedding-embeddinggemma-300m-qat", "state": "loaded"}]}
    rows = {r["role"]: r for r in monitor.model_table(server, {"models": {
        "judge": "qwen3-4b", "embedding": "embeddinggemma-300m-qat"}})}
    assert rows["judge"]["state"] == "missing", "two candidates is a decision, not a match"
    assert rows["embedding"]["state"] == "ready", "the role prefix is allowlisted"


def test_the_stage_plan_matches_the_compiled_graph():
    """The monitor draws its steps from STAGE_PLAN. Every plan must be a real path
    through the graph, START to END, or the page shows steps that cannot happen."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph import STAGE_PLAN, build_graph
    edges = {(e.source, e.target) for e in build_graph(MemorySaver()).get_graph().edges}
    for route, tail in STAGE_PLAN.items():
        if route == "base":
            continue
        path = ["__start__", *STAGE_PLAN["base"], *tail, "__end__"]
        missing = [(a, b) for a, b in zip(path, path[1:], strict=False) if (a, b) not in edges]
        assert not missing, (route, missing)


def test_a_run_paused_at_approval_is_not_called_in_progress(tmp_path):
    """It writes no audit record (no outcome yet), so the page used to say
    'In progress' and then 'Stopped' for a run that passed every check."""
    var = tmp_path / "var"
    span = {"span": "node.await_approval", "status": "suspended", "duration_ms": 1.0,
            "ts": "2026-11-16T10:00:00+00:00", "run_id": "r9", "tenant": "t", "target": "x"}
    _write(var / "live" / "run-cli-1.json", [])
    (var / "live" / "run-cli-1.json").write_text(json.dumps({
        "run_id": "r9", "status": "abandoned", "stages": [{"name": "independent_review",
        "detail": {"decision": "approve", "failures": [], "checks": {
            "groundedness_ok": True, "entailment_ok": True, "risk_ok": True,
            "permission_ok": True}}}]}))
    (run,) = monitor.runs([span], live=[])
    assert run["status"] == "abandoned"
    assert run["audit"]["provisional"] and run["audit"]["decision"] == "approve"
