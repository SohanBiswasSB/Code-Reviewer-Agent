"""Layer 0 (evidence by reference), Layer 2 (tracing), Layer 4 (retention).

See docs/DATA_STORAGE.md. `CRA_DATA_DIR` is redirected by conftest, so nothing
here writes into the working tree.
"""
import json
import os
from datetime import UTC, datetime, timedelta

import pytest

import config
from adapters.detectors import SignalDetector
from adapters.source_s3logs import MAX_SAMPLES_PER_SIGNAL, S3LogSource
from adapters.tracer_jsonl import JsonlTracer, NullTracer, hash_text
from jobs.prune_checkpoints import expired_threads
from schemas.evidence import MAX_LINE_CHARS, LogWindow
from surfaces.log_review import LogReviewSurface


def _reader(lines):
    def read():
        for i, line in enumerate(lines):
            yield ("backend", f"s3://bucket/daily/2026-09-10/backend-{i // 100}.log", line)
    return read


# ══ Layer 0 ═══════════════════════════════════════════════════════════════════
def test_detectors_find_signals_and_first_match_wins():
    hits = SignalDetector().detect({"service": "backend", "ref": "k", "start": 10, "lines": [
        "2026-09-10 INFO all good",
        "2026-09-10 ERROR Traceback (most recent call last)",
        "2026-09-10 TimeoutError: read timed out",
    ]})
    assert [h["signal"] for h in hits] == ["traceback", "timeout"]
    # ordered table: a traceback line also matches /ERROR/, and must not be
    # counted twice or the aggregate over-reports the noisiest signal
    assert hits[0]["line_no"] == 11, "offsets must be absolute within the export"


def test_a_day_of_logs_is_reduced_to_something_a_checkpoint_can_hold():
    """The entire point of Layer 0.

    LangGraph re-serialises a channel's whole value every super-step, so evidence
    has to be bounded BEFORE it enters state. 50k lines in, kilobytes out.
    """
    lines = ["2026-09-10 INFO healthy request"] * 49_000
    lines += ["2026-09-10 ERROR OperationalError: could not connect to server"] * 1_000
    raw = S3LogSource().gather("2026-09-10", reader=_reader(lines), detector=SignalDetector())

    assert raw["line_count"] == 50_000
    assert raw["counts"]["db_error"] == 1_000, "counts must cover EVERY line, not the sample"
    for candidate in raw["candidates"]:
        assert len(candidate["samples"]) <= MAX_SAMPLES_PER_SIGNAL

    window = LogReviewSurface().normalize(raw)
    assert isinstance(window, LogWindow)
    size = len(json.dumps(window.model_dump()))
    assert size < 20_000, f"a day of logs must not reach the checkpoint (got {size} bytes)"
    assert raw["truncated"] is True, "a capped sample must say so"


def test_the_sample_cites_where_each_line_came_from():
    """Kept checkable: a human can go back to the export instead of trusting a
    copied string (docs/DATA_STORAGE.md)."""
    lines = ["2026-09-10 ERROR Lock wait timeout exceeded"] * 3
    raw = S3LogSource().gather("2026-09-10", reader=_reader(lines), detector=SignalDetector())
    sample = raw["candidates"][0]["samples"][0]
    assert sample["ref"].startswith("s3://") and sample["service"] == "backend"
    assert isinstance(sample["line_no"], int)


def test_one_pathological_line_cannot_bloat_the_window():
    huge = "2026-09-10 ERROR " + ("x" * 100_000)
    raw = S3LogSource().gather("2026-09-10", reader=_reader([huge]), detector=SignalDetector())
    assert len(raw["candidates"][0]["samples"][0]["text"]) <= MAX_LINE_CHARS


def test_many_short_objects_never_buffer_the_whole_day():
    """Each stream flushed only at WINDOW_LINES, so objects shorter than that all
    stayed buffered to EOF -- 252 MB for 2M lines. The global cap bounds it, and
    every line still reaches the detector exactly once, at its own offset."""
    from adapters import source_s3logs as s3
    state = {"yielded": 0, "detected": 0, "worst": 0}
    seen: dict[str, list[int]] = {}

    class Spy:
        def detect(self, window):
            state["detected"] += len(window["lines"])
            state["worst"] = max(state["worst"], state["yielded"] - state["detected"]
                                 + len(window["lines"]))
            seen.setdefault(window["ref"], []).append(window["start"])
            return []

    def reader():
        for obj in range(40):
            for _ in range(s3.WINDOW_LINES - 1):
                state["yielded"] += 1
                yield "api", f"obj-{obj}", "line"

    S3LogSource().gather("2026-09-10", reader=reader, detector=Spy())
    assert state["detected"] == state["yielded"] == 40 * (s3.WINDOW_LINES - 1)
    assert state["worst"] <= s3.MAX_BUFFERED_LINES, state["worst"]
    assert all(starts == sorted(starts) for starts in seen.values()), "offsets stay ordered"


def test_the_digest_is_stable_and_content_addressed():
    """The natural idempotency key for G7: re-running a day is detectable."""
    lines = ["2026-09-10 ERROR boom", "2026-09-10 INFO fine"]
    a = S3LogSource().gather("2026-09-10", reader=_reader(lines), detector=SignalDetector())
    b = S3LogSource().gather("2026-09-10", reader=_reader(lines), detector=SignalDetector())
    c = S3LogSource().gather("2026-09-10", reader=_reader(lines + ["extra"]),
                             detector=SignalDetector())
    assert a["digest"] == b["digest"] != c["digest"]


def test_a_source_that_returns_raw_lines_fails_loudly():
    """LogWindow uses extra="forbid", so a regression that starts passing the
    day's lines through cannot quietly make every super-step rewrite them."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="lines"):
        LogReviewSurface().normalize({"date": "2026-09-10", "lines": ["...2M of these..."]})


def test_a_quiet_day_goes_to_a_human_not_to_a_model():
    """Nothing above threshold means nothing to ground a diagnosis in."""
    raw = S3LogSource().gather("2026-09-10", reader=_reader(["INFO ok"] * 10),
                               detector=SignalDetector())
    assert LogReviewSurface().classify(raw) == "human_review"


def test_the_log_surface_survives_the_full_normalize_classify_query_chain():
    """The chain that was broken three separate ways: no normalize (KeyError on
    html_url), nested models breaking build_query, and action_type."""
    from nodes.classify import classify
    from nodes.generate_diagnosis import _fallback_query  # noqa: F401  (import guard)

    surface = LogReviewSurface()
    raw = S3LogSource().gather(
        "2026-09-10",
        reader=_reader(["ERROR Traceback (most recent call last)"] * 5),
        detector=SignalDetector())
    state = {"_surface": surface, "_raw": raw}

    from nodes.normalize_evidence import normalize_evidence
    state.update(normalize_evidence(state))
    assert isinstance(state["evidence"], LogWindow)
    assert classify(state)["route"] == "ai_investigation"

    from nodes._surface import evidence_as_dict
    query = surface.build_query(evidence_as_dict(state["evidence"]))
    assert "traceback" in query, "nested candidates must arrive as dicts, not models"


# ══ Layer 2 ═══════════════════════════════════════════════════════════════════
def test_spans_are_written_with_timing_and_identity(tmp_path):
    tracer = JsonlTracer(root=str(tmp_path))
    with tracer.span("node.classify", run_id="r-1", tenant="t"):
        pass
    row = json.loads(open(tracer.path_for(), encoding="utf-8").readline())
    assert row["span"] == "node.classify" and row["status"] == "ok"
    assert row["run_id"] == "r-1" and isinstance(row["duration_ms"], float)


def test_a_failing_block_is_recorded_and_still_raises(tmp_path):
    tracer = JsonlTracer(root=str(tmp_path))
    with pytest.raises(PermissionError), tracer.span("node.execute", run_id="r-2"):
        raise PermissionError("refused")
    row = json.loads(open(tracer.path_for(), encoding="utf-8").readline())
    assert row["status"] == "error" and "refused" in row["error"]


def test_the_approval_gate_is_suspended_not_failed(tmp_path):
    """`interrupt()` suspends by RAISING, so a naive tracer would mark every run
    that reached the approval gate — the normal path — as a failure."""
    class GraphInterrupt(Exception):
        pass

    tracer = JsonlTracer(root=str(tmp_path))
    with pytest.raises(GraphInterrupt), tracer.span("node.await_approval"):
        raise GraphInterrupt()
    assert json.loads(open(tracer.path_for(), encoding="utf-8").readline())["status"] == "suspended"


def test_tracing_never_breaks_the_thing_it_traces(tmp_path, monkeypatch):
    """Observability that can take down the system it observes is worse than none."""
    tracer = JsonlTracer(root=str(tmp_path))
    monkeypatch.setattr(tracer, "_write", lambda row: (_ for _ in ()).throw(OSError("disk full")))
    with tracer.span("node.classify"):
        value = 42
    assert value == 42


def test_prompts_are_not_traced_by_default(monkeypatch):
    """T12 is open, so nothing redacts PII out of evidence. Storing every prompt
    verbatim would durably copy exactly what the guard is meant to remove."""
    from adapters import tracer_jsonl

    monkeypatch.delenv("CRA_TRACE_PROMPTS", raising=False)
    assert tracer_jsonl.prompts_enabled() is False
    monkeypatch.setenv("CRA_TRACE_PROMPTS", "1")
    assert tracer_jsonl.prompts_enabled() is True
    assert hash_text("secret prompt") != "secret prompt"


def test_every_node_is_traced_without_any_node_knowing(monkeypatch):
    """Wrapped in build_graph, so a node added later is covered automatically."""
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver

    import registry
    from app.graph import build_graph
    from tools import github_client

    spans = []

    class Recorder(NullTracer):
        def span(self, name, **attrs):
            spans.append(name)
            return super().span(name, **attrs)

    monkeypatch.setattr(github_client, "fetch_issue", lambda url, token=None: {
        "html_url": url, "title": "t", "body": "x" * 400, "labels": []})
    monkeypatch.setattr(github_client, "get_check_runs", lambda url, token=None: [])

    class Surf:
        name = "code_review"
        source_name, sink_name = "github", "pr_comment"
        diagnosis_prompt = judge_prompt = "diagnosis"

        def build_query(self, ev):
            return "q"

        def classify(self, ev):
            return "human_review"

    monkeypatch.setitem(registry._SURFACES, "code_review", Surf)
    g = build_graph(checkpointer=MemorySaver(), tracer=Recorder())
    g.invoke({"tenant": "_default", "surface": "code_review", "target": "u"},
             config={"configurable": {"thread_id": "trace-1"}})
    assert "node.resolve_tenant" in spans and "node.classify" in spans
    assert "node.audit" in spans, "the audit node must be traced like any other"


# ══ Layer 4 ═══════════════════════════════════════════════════════════════════
def _rec(thread_id, outcome, age_days, **extra):
    at = datetime.now(UTC) - timedelta(days=age_days)
    return {"thread_id": thread_id, "outcome": outcome,
            "recorded_at": at.isoformat(), "tenant": "t", **extra}


def test_old_finished_threads_expire():
    rows = expired_threads([_rec("old", "executed", 40), _rec("new", "executed", 2)], 30)
    assert [r["thread_id"] for r in rows] == ["old"]


def test_a_thread_with_no_terminal_record_never_expires():
    """A run that is still parked at the approval gate has no terminal record, so
    it is not a candidate at all — deleting it would discard a pending decision."""
    assert expired_threads([_rec("pending", "awaiting", 400)], 30) == []


def test_an_unreadable_age_never_authorises_a_deletion():
    """"I cannot tell how old this is" must not mean "delete it"."""
    assert expired_threads([_rec("x", "executed", 40, recorded_at="not-a-date")], 30) == []
    assert expired_threads([{"outcome": "executed", "recorded_at": "2020-01-01"}], 30) == []


def test_the_newest_record_decides_a_threads_fate():
    """A thread can hold several records — a denial, then a later attempt. An old
    terminal record must not expire a thread that has been active since."""
    records = [_rec("t1", "executed", 400), _rec("t1", "executed", 1)]
    assert expired_threads(records, 30) == []


@pytest.mark.parametrize("outcome", ["executed", "denied", "escalated", "human_review"])
def test_every_terminal_outcome_is_prunable(outcome):
    """A denial counts as finished: the human answered."""
    assert len(expired_threads([_rec("t", outcome, 90)], 30)) == 1


def test_the_registry_reports_the_new_plugs():
    plugged = config.registered()
    assert "signals" in plugged["adapters"]["detector"]
    assert "jsonl" in plugged["adapters"]["tracer"]
    assert plugged["selected"]["tracer"] == os.getenv("CRA_TRACE", "jsonl")
