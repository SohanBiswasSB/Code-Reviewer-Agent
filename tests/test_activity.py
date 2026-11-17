"""The activity log: the Masterclass Logger's contract, ported.

Pins the four properties that make a log trustworthy rather than decorative:
nothing unmasked is stored, an exception is recorded but never swallowed, a
failed write never breaks the operation, and one run's records share one id.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys

import pytest
from langgraph.errors import GraphInterrupt

from adapters import llm_lmstudio
from adapters.tracer_jsonl import NullTracer
from app import activity, deps, monitor
from app.activity import ERROR, INFO, WARNING, ActivityLogger, log_event
from app.graph import _traced
from tools.ingest import iter_files


def rows(tmp_path):
    return [json.loads(line) for p in sorted((tmp_path / "var" / "activity").glob("*.jsonl"))
            for line in p.read_text().splitlines()]


# ── masking, on the way in ───────────────────────────────────────────────────
def test_secrets_are_redacted_by_key_substring():
    out = activity.sanitize({"github_token": "ghp_x", "X-Api-Key": "k",
                             "nested": [{"password": "p"}]})
    assert out == {"github_token": "[REDACTED]", "X-Api-Key": "[REDACTED]",
                   "nested": [{"password": "[REDACTED]"}]}


def test_counts_under_secret_looking_keys_are_kept():
    assert activity.sanitize({"completion_tokens": 30, "max_tokens": 2048,
                              "refresh_token": "abc"}) == {
        "completion_tokens": 30, "max_tokens": 2048, "refresh_token": "[REDACTED]"}


@pytest.mark.parametrize("key,value,masked", [
    ("email", "sohan@example.com", "s***@example.com"), ("email", "nope", "***"),
    ("phone", "+91 98765 43210", "***3210"), ("mobile", "12", "***"),
    ("first_name", "Sohan", "S***"), ("to", "a@b.com", "a@b.com"),    # `to` is not PII
])
def test_pii_is_masked_by_exact_key(key, value, masked):
    assert activity.mask_pii(key, value) == masked


def test_nothing_unmasked_reaches_the_file(tmp_path):
    with ActivityLogger("job", "x", metadata={"author_email": "dev@corp.com",
                                              "api_key": "sk-123"}) as alog:
        alog.add_metadata("phone", "9876543210")
    (r,) = rows(tmp_path)
    raw = json.dumps(r)
    assert "dev@corp.com" not in raw and "sk-123" not in raw and "9876543210" not in raw


def test_error_text_and_traceback_pass_through_the_guard(tmp_path):
    with pytest.raises(ValueError):
        with ActivityLogger("job", "x"):
            raise ValueError("leaked AKIAABCDEFGHIJKLMNOP in a message")
    (r,) = rows(tmp_path)
    assert "AKIAABCDEFGHIJKLMNOP" not in r["error_message"]
    assert "AKIAABCDEFGHIJKLMNOP" not in r["traceback"]


# ── the contract ─────────────────────────────────────────────────────────────
def test_an_exception_is_recorded_and_never_swallowed(tmp_path):
    with pytest.raises(KeyError):
        with ActivityLogger("node", "classify"):
            raise KeyError("route")
    (r,) = rows(tmp_path)
    assert r["severity"] == ERROR and r["status"] == "failure" and r["is_error"] is True
    assert r["exception_type"] == "KeyError" and "Traceback" in r["traceback"]


@pytest.mark.parametrize("status,duration,expected", [("success", 10, INFO),
                                                      ("warning", 10, WARNING)])
def test_severity_is_derived(tmp_path, status, duration, expected):
    span = ActivityLogger("node", "x").start()
    assert span.severity(duration) == WARNING        # still `started`: not yet a success
    span.status = status
    assert span.severity(duration) == expected


def test_a_slow_operation_is_a_warning(tmp_path):
    span = ActivityLogger("external_api", "chat.diagnosis").start()
    span.status = "success"
    assert span.severity(activity.SLOW_MS["external_api"] + 1) == WARNING
    assert span.metadata["slow_operation"] is True


def test_a_failed_write_never_breaks_the_operation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(activity, "log_dir", lambda: tmp_path / "file-not-dir" / "x")
    (tmp_path / "file-not-dir").write_text("a file where a directory should be")
    with ActivityLogger("job", "still-runs"):
        pass
    assert "activity-log write failed" in capsys.readouterr().err


def test_the_level_gate_drops_info(tmp_path, monkeypatch):
    monkeypatch.setenv("CRA_ACTIVITY_LEVEL", "WARNING")
    log_event("job", "quiet")
    log_event("job", "loud", severity=WARNING)
    assert [r["operation_name"] for r in rows(tmp_path)] == ["loud"]


def test_the_null_switch_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CRA_ACTIVITY", "null")
    log_event("job", "x", severity=ERROR)
    assert rows(tmp_path) == []


def test_the_gate_suite_never_writes_to_the_real_activity_log(monkeypatch):
    env = {k: v for k, v in os.environ.items() if k != "CRA_ACTIVITY"}
    out = subprocess.run([sys.executable, "-c", "import eval.harness, os; "
                          "print(os.environ['CRA_ACTIVITY'])"], capture_output=True,
                         text=True, env=env, check=True)
    assert out.stdout.strip() == "null"


def test_none_values_are_dropped(tmp_path):
    log_event("job", "x")
    (r,) = rows(tmp_path)
    assert "traceback" not in r and "tenant" not in r


# ── correlation ──────────────────────────────────────────────────────────────
def test_everything_inside_a_correlation_shares_its_id(tmp_path):
    with activity.correlation("run-42"):
        log_event("job", "a")
        with ActivityLogger("node", "b"):
            pass
    log_event("job", "outside")
    ids = {r["operation_name"]: r["correlation_id"] for r in rows(tmp_path)}
    assert ids["a"] == ids["b"] == "run-42" and ids["outside"] != "run-42"


# ── reading, for the admin page ──────────────────────────────────────────────
def test_query_filters_and_facets(tmp_path):
    with activity.correlation("c1"):
        log_event("node", "classify")
        log_event("external_api", "chat.judge", severity=ERROR, message="model went away")
    log_event("node", "audit", correlation_id="c2")
    d = activity.query()
    assert d["total"] == 3 and d["facets"]["severity"] == {"INFO": 2, "ERROR": 1}
    assert [r["operation_name"] for r in activity.query(severity="ERROR")["rows"]] == ["chat.judge"]
    assert len(activity.query(correlation_id="c1")["rows"]) == 2
    assert activity.query(q="went away")["matched"] == 1
    assert activity.query(operation_type="node")["matched"] == 2


# ── the hooks, end to end ────────────────────────────────────────────────────
def test_a_graph_node_failure_is_logged_with_its_traceback(tmp_path, monkeypatch):
    """No failure edges (G9): before this, a crashed run left no record at all."""

    def boom(state):
        raise RuntimeError("node blew up")
    wrapped = _traced("generate_diagnosis", boom, NullTracer())
    with pytest.raises(RuntimeError):
        wrapped({"thread_id": "t-1", "tenant": "merger_app"})
    (r,) = [x for x in rows(tmp_path) if x["operation_type"] == "node"]
    assert r["correlation_id"] == "t-1" and r["tenant"] == "merger_app"
    assert r["severity"] == ERROR and "node blew up" in r["traceback"]
    assert r["failure_source"] == "internal"


def test_the_approval_pause_is_waiting_not_an_error(tmp_path):

    def pause(state):
        raise GraphInterrupt(())
    with pytest.raises(GraphInterrupt):
        _traced("await_approval", pause, NullTracer())({"thread_id": "t"})
    (r,) = rows(tmp_path)
    assert r["status"] == "waiting" and r.get("is_error") is False


def test_an_llm_call_is_one_external_api_record_with_no_prompt_text(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_lmstudio, "available_models",
                        lambda url=None, timeout=2.0: ["qwen/qwen3-4b"])
    usage = type("U", (), {"prompt_tokens": 120, "completion_tokens": 30})
    choice = type("C", (), {"message": type("M", (), {"content": "ok"}), "finish_reason": "stop"})

    class Completions:
        def create(self, **kw):
            return type("R", (), {"choices": [choice], "usage": usage})
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    llm._client = type("Cl", (), {"chat": type("Ch", (), {"completions": Completions()})})
    llm.complete("SYSTEM SECRET PROMPT", "user evidence text")
    (r,) = rows(tmp_path)
    assert r["operation_type"] == "external_api" and r["operation_name"] == "chat.judge"
    assert r["metadata"]["completion_tokens"] == 30
    assert r["metadata"]["resolved_model"] == "qwen/qwen3-4b"
    assert "SYSTEM SECRET PROMPT" not in json.dumps(r) and "user evidence" not in json.dumps(r)


def test_the_monitor_serves_the_activity_log_and_clamps_its_inputs(tmp_path):
    log_event("job", "ingest", severity=ERROR, message="x")
    d = monitor.activity_view("severity=ERROR&hours=99999&type=job")
    assert d["matched"] == 1 and d["hours"] == 24 * 31
    assert monitor.activity_view("severity=<script>")["matched"] == 1   # ignored, not echoed
    assert monitor.activity_view("hours=abc")["hours"] == 24.0


def test_the_page_never_injects_log_text_as_html():
    page = monitor.page()
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page
    assert 'id="sec-activity"' in page


# ── nothing logs around the activity log ─────────────────────────────────────
@pytest.fixture
def bridge():
    activity.capture_library_logs()
    yield
    root = logging.getLogger()
    for h in [h for h in root.handlers if isinstance(h, activity.ActivityHandler)]:
        root.removeHandler(h)
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        for h in [h for h in lg.handlers if isinstance(h, activity.ActivityHandler)]:
            lg.removeHandler(h)
    logging.getLogger("uvicorn.access").disabled = False


def test_library_warnings_land_in_the_activity_log(tmp_path, bridge):
    logging.getLogger("httpx").warning("connection pool full for %s", "lmstudio")
    logging.getLogger("httpx").info("chatter below the threshold")
    (r,) = rows(tmp_path)
    assert r["operation_type"] == "library" and r["operation_name"] == "httpx"
    assert r["severity"] == WARNING
    assert r["metadata"]["message"] == "connection pool full for lmstudio"


def test_a_library_exception_keeps_its_traceback(tmp_path, bridge):
    try:
        raise ConnectionResetError("peer went away")
    except ConnectionResetError:
        logging.getLogger("openai").exception("request failed")
    (r,) = rows(tmp_path)
    assert r["severity"] == ERROR and "peer went away" in r["traceback"]


def test_uvicorn_access_log_is_replaced_by_the_request_records(tmp_path, bridge):
    logging.getLogger("uvicorn.access").warning("GET / 200")
    assert rows(tmp_path) == []


def test_the_bridge_installs_once(bridge):
    activity.capture_library_logs()
    assert sum(isinstance(h, activity.ActivityHandler) for h in logging.getLogger().handlers) == 1


def test_a_model_retry_is_on_record(tmp_path):
    attempt = llm_lmstudio.RetryAttempt("completion", "judge", "qwen3-4b", 2, 3, "server busy")
    llm_lmstudio.announce_retry(attempt, ConnectionError("busy"), 0.0, lambda s: None)
    (r,) = [x for x in rows(tmp_path) if x["operation_name"] == "retry.completion"]
    assert r["severity"] == WARNING and r["metadata"]["attempt"] == 2


def test_the_non_durable_checkpointer_is_on_record(tmp_path):
    with pytest.warns(RuntimeWarning):
        deps._memory_saver()
    assert [r["operation_name"] for r in rows(tmp_path)] == ["checkpointer.memory"]


def test_a_missing_include_path_is_on_record(tmp_path):
    iter_files(tmp_path, ["no/such/dir"])
    (r,) = rows(tmp_path)
    assert r["operation_name"] == "ingest.include_missing" and r["severity"] == WARNING


def test_the_service_itself_never_uses_stdlib_logging():
    """Only the bridge (for libraries) touches `logging`; our own code logs through
    the activity log. A stray logger.info() would go somewhere nobody reads."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for p in root.rglob("*.py"):
        if any(part in p.parts for part in ("tests", ".venv", "var")) or p.name == "activity.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "import logging" in text or "getLogger(" in text:
            offenders.append(str(p.relative_to(root)))
    assert offenders == []


# ── retention ────────────────────────────────────────────────────────────────
def test_retention_deletes_only_expired_day_files_and_only_when_asked(tmp_path, monkeypatch):
    from datetime import date

    from jobs import prune_activity
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("CRA_ACTIVITY", "null")
    d = tmp_path / "var" / "activity"
    d.mkdir(parents=True)
    for name in ("2026-01-01.jsonl", "2026-11-01.jsonl", "notes.jsonl"):
        (d / name).write_text("{}\n")
    today = date(2026, 11, 17)
    dry = prune_activity.prune(30, apply=False, today=today)
    assert [p.name for p in dry] == ["2026-01-01.jsonl"]
    assert (d / "2026-01-01.jsonl").exists(), "a dry run deletes nothing"
    prune_activity.prune(30, apply=True, today=today)
    assert sorted(p.name for p in d.iterdir()) == ["2026-11-01.jsonl", "notes.jsonl"]
    with pytest.raises(ValueError):
        prune_activity.prune(0, apply=True, today=today)
