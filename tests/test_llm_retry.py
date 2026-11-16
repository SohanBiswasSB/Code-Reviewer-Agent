"""The model-call retry policy (adapters/llm_lmstudio.py::with_retry).

What these pin: a TRANSIENT failure is retried -- including LM Studio's
backend crash, which arrives as HTTP 400 and was what killed a real run -- while
a genuinely bad request fails on the first attempt, and an exhausted retry
re-raises the original error rather than a wrapper.
"""
from __future__ import annotations

import json

import httpx
import openai
import pytest

from adapters import llm_lmstudio
from adapters.llm_lmstudio import transient_reason, with_retry

REQ = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
CRASH = ("Error code: 400 - {'error': 'Error in iterating prediction stream: Exception: "
         "Encountered fatal exception in the backend generation thread: Traceback ...'}")


def bad_request(msg: str) -> openai.BadRequestError:
    return openai.BadRequestError(msg, response=httpx.Response(400, request=REQ), body=None)


def flaky(*errors, result="ok"):
    """A call that raises each of `errors` in turn, then returns `result`."""
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] <= len(errors):
            raise errors[calls["n"] - 1]
        return result
    return call, calls


@pytest.fixture
def waits():
    slept: list[float] = []
    return slept


def test_the_backend_crash_400_is_retried_and_recovers(waits):
    call, calls = flaky(bad_request(CRASH))
    assert with_retry(call, what="completion", sleep=waits.append) == "ok"
    assert calls["n"] == 2 and waits == [5.0]


def test_a_genuine_bad_request_is_not_retried(waits):
    call, calls = flaky(bad_request("Invalid model identifier 'qwen3-4b'"))
    with pytest.raises(openai.BadRequestError, match="Invalid model identifier"):
        with_retry(call, what="completion", sleep=waits.append)
    assert calls["n"] == 1 and waits == []


@pytest.mark.parametrize("err", [
    openai.APIConnectionError(request=REQ),
    openai.APITimeoutError(request=REQ),
    openai.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None),
])
def test_transport_and_server_errors_are_transient(err):
    assert transient_reason(err)


def test_exhausted_retries_reraise_the_original_error(waits, monkeypatch):
    monkeypatch.setenv("CRA_LLM_RETRIES", "2")
    call, calls = flaky(*[bad_request(CRASH)] * 5)
    with pytest.raises(openai.BadRequestError, match="prediction stream"):
        with_retry(call, what="completion", sleep=waits.append)
    assert calls["n"] == 3 and waits == [5.0, 15.0]


def test_retries_can_be_turned_off_and_backoff_tuned(waits, monkeypatch):
    monkeypatch.setenv("CRA_LLM_RETRIES", "0")
    call, calls = flaky(bad_request(CRASH))
    with pytest.raises(openai.BadRequestError):
        with_retry(call, what="completion", sleep=waits.append)
    assert calls["n"] == 1
    monkeypatch.setenv("CRA_LLM_RETRIES", "3")
    monkeypatch.setenv("CRA_LLM_RETRY_BACKOFF", "1,2")
    call, _ = flaky(*[openai.APIConnectionError(request=REQ)] * 3)
    assert with_retry(call, what="completion", sleep=waits.append) == "ok"
    assert waits == [1.0, 2.0, 2.0], "the last wait repeats once the list runs out"


def test_a_retry_is_written_to_the_trace(tmp_path, waits):
    call, _ = flaky(bad_request(CRASH))
    with_retry(call, what="completion", role="judge", model="qwen/qwen3-4b",
               sleep=waits.append)
    rows = [json.loads(line) for p in (tmp_path / "var" / "traces").glob("*.jsonl")
            for line in p.read_text().splitlines()]
    (row,) = [r for r in rows if r["span"] == "llm.retry"]
    assert row["role"] == "judge" and row["attempt"] == 2 and row["of"] == 3
    assert "crashed" in row["reason"] and "prediction stream" in row["error"]


def test_the_adapter_routes_its_call_through_the_policy(monkeypatch):
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    monkeypatch.setattr(llm, "resolved_model", lambda: "qwen/qwen3-4b")
    seen = {}

    def fake_retry(call, **kw):
        seen.update(kw)
        return type("R", (), {"choices": [type("C", (), {"message": type(
            "M", (), {"content": "fine"})()})()]})()
    monkeypatch.setattr(llm_lmstudio, "with_retry", fake_retry)
    assert llm.complete("sys", "user") == "fine"
    assert seen == {"what": "completion", "role": "judge", "model": "qwen/qwen3-4b"}
    assert llm._get_client().max_retries == 0, "a second, hidden retry loop multiplies"


def test_the_monitor_places_a_retry_inside_the_step_it_happened_in(tmp_path):
    from app import monitor
    span = {"span": "node.independent_review", "status": "ok", "duration_ms": 30000.0,
            "ts": "2026-11-16T10:00:30+00:00", "run_id": "r1", "tenant": "t", "target": "staged"}
    inside = {"span": "llm.retry", "ts": "2026-11-16T10:00:10+00:00", "attempt": 2, "of": 3,
              "reason": "LM Studio's backend crashed mid-generation", "wait_s": 5.0}
    before = {**inside, "ts": "2026-11-16T09:59:00+00:00"}
    (run,) = monitor.runs([span, inside, before])
    (step,) = run["steps"]
    assert [r["ts"] for r in step["retries"]] == [inside["ts"]]
    run["audit"] = {"runtime": {"profile": "mac_max", "entailer": "cross_encoder"}}
    assert monitor.retry_command(run, {}) == (
        "CRA_PROFILE=mac_max ENTAILER=cross_encoder python -m app.review --tenant t")
