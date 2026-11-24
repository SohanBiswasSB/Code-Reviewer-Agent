"""The monitor's reading of var/ -- the parts that were wrong, not just slow."""
from __future__ import annotations

import json
import urllib.error

from langgraph.checkpoint.memory import MemorySaver

from app import monitor_guidance, monitor_models, monitor_runs
from app.graph import STAGE_PLAN, build_graph
from errors import ProviderAuthError


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
    for i in range(60):                      # merger_app: older, but still recent
        s, r = _run("merger_app", f"m{i}", f"2026-11-10T10:{i:02d}:00+00:00")
        spans.append(s)
        a_recs.append(r)
    for i in range(monitor_runs.AUDIT_TAIL + 5):  # milesone: newer, and more of them
        s, r = _run("milesone", f"x{i}", f"2026-11-11T10:{i % 60:02d}:{i // 60:02d}+00:00")
        spans.append(s)
        b_recs.append(r)
    _write(var / "audit" / "merger_app" / "2026-11.jsonl", a_recs)
    _write(var / "audit" / "milesone" / "2026-11.jsonl", b_recs)
    audits = monitor_runs.recent_audits()
    assert {"merger_app", "milesone"} <= {r["tenant"] for r in audits.values()}
    shown = monitor_runs.runs(spans, live=[])
    assert all(r["status"] == "escalated" for r in shown), [r["status"] for r in shown]


def test_only_the_newest_files_are_read_for_a_tail(tmp_path):
    """Re-reading every trace file per 1/s poll cost 224 ms at a year of history.
    The tail must come from the newest files only, stop once full, and still be
    returned oldest first."""
    var = tmp_path / "var" / "traces"
    for day in range(1, 31):
        _write(var / f"2026-10-{day:02d}.jsonl",
               [{"span": "node.x", "ts": f"2026-10-{day:02d}T00:00:{i:02d}+00:00"}
                for i in range(50)])
    rows = monitor_runs.newest_rows(list(var.glob("*.jsonl")), 120)
    assert len(rows) == 120 and rows[-1]["ts"].startswith("2026-10-30")
    assert rows[0]["ts"].startswith("2026-10-28"), "oldest kept row comes from 3 files back"
    assert rows == sorted(rows, key=lambda r: r["ts"]), "oldest first"


def test_the_model_table_uses_the_adapters_strict_matcher():
    """A looser monitor-side copy called a different publisher's model 'ready'."""
    server = {"reachable": True, "url": "http://x", "models": [
        {"id": "other/qwen3-4b", "state": "loaded"},
        {"id": "qwen/qwen3-4b", "state": "loaded"},
        {"id": "text-embedding-embeddinggemma-300m-qat", "state": "loaded"}]}
    rows = {r["role"]: r for r in monitor_models.model_table(server, {"models": {
        "judge": "qwen3-4b", "embedding": "embeddinggemma-300m-qat"}})}
    assert rows["judge"]["state"] == "missing", "two candidates is a decision, not a match"
    assert rows["embedding"]["state"] == "ready", "the role prefix is allowlisted"


def test_the_stage_plan_matches_the_compiled_graph():
    """The monitor draws its steps from STAGE_PLAN. Every plan must be a real path
    through the graph, START to END, or the page shows steps that cannot happen."""
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
    (run,) = monitor_runs.runs([span], live=[])
    assert run["status"] == "abandoned"
    assert run["audit"]["provisional"] and run["audit"]["decision"] == "approve"


# ── a gateway that is not LM Studio (2026-11-17) ─────────────────────────────
def test_a_gateway_without_lm_studios_api_is_reachable_not_down(monkeypatch):
    """The devops gateway serves only the OpenAI /v1 API, so /api/v0 is a 404.
    That is a server that is UP; the page said "not reachable" for one answering
    every review."""

    def no_v0(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(monitor_models.urllib.request, "urlopen", no_v0)
    monkeypatch.setattr(monitor_models, "available_models",
                        lambda url, timeout=None: ["qwen3-4b", "text-embedding-x"])
    srv = monitor_models.fetch_server_models("https://gw.example:8443")
    assert srv["reachable"] is True and srv["api"] == "openai"
    assert {m["state"] for m in srv["models"]} == {"served"}
    rows = monitor_models.model_table(srv, {"models": {"judge": "qwen3-4b"}})
    judge = next(r for r in rows if r["role"] == "judge")
    assert judge["state"] == "ready" and "OpenAI API" in judge["note"]


def test_a_refused_key_on_the_gateway_shows_as_unreachable_with_the_reason(monkeypatch):
    """The 404 fallback to the OpenAI listing must not turn a refused key into
    "reachable": the page shows it as down, and says it was the key."""
    def no_v0(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    def refused(url, timeout=None):
        raise ProviderAuthError("refused the API key (HTTP 401)")
    monkeypatch.setattr(monitor_models.urllib.request, "urlopen", no_v0)
    monkeypatch.setattr(monitor_models, "available_models", refused)
    srv = monitor_models.fetch_server_models("https://gw.example:8443")
    assert srv["reachable"] is False and "API key" in srv["error"]


# ── what the page says under a run (2026-11-17) ──────────────────────────────
def _escalated(failures, findings, **review):
    """A run shaped like the real escalated Masterclass review."""
    checks = {"groundedness_ok": True, "entailment_ok": True, "risk_ok": True,
              "permission_ok": True, **review}
    return {"run_id": "r1", "tenant": "merger_app", "status": "escalated",
            "target": "origin/main...feature", "steps": [],
            "audit": {"review": {**checks, "failures": failures},
                      "diagnosis": {"findings": findings},
                      "gates": {"min_entailment_score": 0.25},
                      "runtime": {"profile": "mac_max", "entailer": "lexical"},
                      "checks": []}}


def _steps(g, who):
    return [s["text"] for s in g["steps"] if s["who"] == who]


def test_the_reason_given_is_the_gates_own_not_a_guess_from_citations():
    """The real run: finding 1 cited its own file and failed at 0.2174 < 0.25;
    finding 2 cited a migration instead of models.py and PASSED. The old advice
    blamed finding 2's citations for the failure."""
    run = _escalated(
        ["finding 1 (utils/csv.py) not entailed by its premise (score 0.2174 < 0.25)"],
        [{"file": "utils/csv.py", "citation_ids": ["utils/csv.py:10-20:ab12cd34"]},
         {"file": "models.py", "citation_ids": ["migrations/0023.py:1-9:ef56ab78"]},
         {"file": "urls.py", "citation_ids": ["urls.py:1-5:00000000"]}],
        entailment_ok=False)
    g = monitor_guidance.guidance(run, {}, "2026-11-16T14:45:47+0530")
    assert g["why"] == ["Finding 1 (utils/csv.py) scored 0.217 on Entailed against a bar "
                        "of 0.25 (0.033 short): its claim did not clearly follow from "
                        "the code it cites."]
    assert any("Read finding 1 (utils/csv.py)" in t and "other 2 passed" in t
               for t in _steps(g, "reviewer"))
    off = next(s for s in g["steps"] if s.get("items") and "cites" in s["items"][0])
    assert off["items"] == ["Finding 2 (models.py) cites migrations/0023.py — passed the gate"]
    assert "2026-11-16" in off["text"] and off["command"].endswith("--tenant merger_app")


def test_several_failed_findings_are_one_step_listing_them():
    """Seven near-identical "read finding N" steps were noise an admin had to read
    seven times; several failures are one step that lists them."""
    fail = "finding {n} ({f}) not entailed by its premise (score 0.1 < 0.25)"
    findings = [{"file": f, "citation_ids": [f"{f}:1-2:00000000"]}
                for f in ("a.py", "b.py", "c.py")]
    run = _escalated([fail.format(n=1, f="a.py"), fail.format(n=3, f="c.py")], findings,
                     entailment_ok=False)
    (step,) = [s for s in monitor_guidance.guidance(run, {})["steps"]
               if s["text"].startswith("Read")]
    assert step["text"] == ("Read these 2 findings yourself. The gate could not verify them; "
                            "the other 1 passed.")
    assert step["items"] == ["Finding 1 (a.py)", "Finding 3 (c.py)"]


def test_a_lexical_entailer_miss_offers_the_semantic_retry_command():
    """The word-overlap entailer also fails correct paraphrases, so an Entailed
    miss under it offers the exact cross-encoder re-run, plus the calibration."""
    run = _escalated(["finding 1 (a.py) not entailed by its premise (score 0.2 < 0.25)"],
                     [{"file": "a.py", "citation_ids": ["a.py:1-2:00000000"]}],
                     entailment_ok=False)
    cmds = [s.get("command", "") for s in monitor_guidance.guidance(run, {})["steps"]]
    assert ("CRA_PROFILE=mac_max ENTAILER=cross_encoder python -m app.review "
            "--tenant merger_app --target origin/main...feature") in cmds
    assert "python -m tools.calibrate --tenant merger_app entailment" in cmds


def test_a_severity_escalation_is_policy_not_a_fault():
    """A severity outside the allowlist is the policy working: the page says so,
    names the policy file to change, and offers no entailer re-run."""
    run = _escalated(["severity 'high' not in ['low', 'medium']"], [], risk_ok=False)
    g = monitor_guidance.guidance(run, {})
    assert "rated high severity" in g["why"][0]
    assert any("working as intended" in t for t in _steps(g, "reviewer"))
    assert any("severity_allowlist in tenants/merger_app/policy.json" in t
               for t in _steps(g, "admin"))
    assert not any("ENTAILER" in s.get("command", "") for s in g["steps"])


def test_pre_push_flags_are_grouped_by_file_and_check():
    """Pre-push flags are code-computed facts; one step per file and check with a
    count, not one line per flag, and passing checks are not listed."""
    run = _escalated([], [])
    flag = {"status": "flag", "file": "api.postman_collection.json"}
    run["audit"]["checks"] = [{**flag, "check": "postman_structure"}] * 6 + \
        [{**flag, "check": "postman_hand_edit"}, {"status": "pass", "check": "ascii"}]
    (step,) = [s for s in monitor_guidance.guidance(run, {})["steps"] if s.get("tab") == "prepush"]
    assert step["text"].startswith("Fix the 7 pre-push flags")
    assert step["items"] == ["api.postman_collection.json: postman structure ×6, "
                             "postman hand edit ×1"]


def test_a_crash_hint_names_the_server_the_run_used():
    """A remote gateway's run must not be told to `lms server start` on this
    machine, and the specific API-key errors win over the generic connection one."""
    gw = "https://gw.example:8443/v1"
    assert "lms server start" in monitor_guidance.crash_hint("Connection refused",
                                                    "http://localhost:1234/v1")
    remote = monitor_guidance.crash_hint("APIConnectionError: Connection error.", gw)
    assert gw in remote and "lms server start" not in remote
    assert "refused the API key" in monitor_guidance.crash_hint(
        "ProviderAuthError: ... refused the API key (HTTP 401)", gw)
    assert "none was loaded" in monitor_guidance.crash_hint(
        "requires an API key (HTTP 401), and LMSTUDIO_API_KEY is not set", gw)


def test_a_gateway_counts_served_models_not_zero_loaded(monkeypatch):
    """The header read "0 loaded" for a gateway answering every review."""
    monkeypatch.setattr(monitor_models, "server_models", lambda: {
        "reachable": True, "url": "https://gw", "api": "openai",
        "models": [{"id": i, "state": "served"} for i in ("a", "b", "c")]})
    m = monitor_models.models()
    assert m["api"] == "openai" and m["served"] == 3 and m["loaded"] == []
