"""The model-call retry policy (adapters/llm_lmstudio.py::with_retry).

What these pin: a TRANSIENT failure is retried -- including LM Studio's
backend crash, which arrives as HTTP 400 and was what killed a real run -- while
a genuinely bad request fails on the first attempt, and an exhausted retry
raises a ProviderError chained to the original error, so the outage is labelled
the provider's rather than "internal".
"""
from __future__ import annotations

import json

import httpx
import openai
import pytest

from adapters import llm_lmstudio
from adapters.llm_lmstudio import transient_reason, with_retry
from app import monitor_guidance, monitor_runs
from errors import ProviderError
from tests._llm_fakes import chat_client, chat_reply

REQ = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
CRASH = ("Error code: 400 - {'error': 'Error in iterating prediction stream: Exception: "
         "Encountered fatal exception in the backend generation thread: Traceback ...'}")


def bad_request(msg: str) -> openai.BadRequestError:
    """A real openai 400 carrying `msg`, as the client raises it."""
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


@pytest.fixture(name="waits")
def fixture_waits():
    """The backoff waits, recorded instead of slept (pass `waits.append` as `sleep`)."""
    slept: list[float] = []
    return slept


def test_the_backend_crash_400_is_retried_and_recovers(waits):
    """LM Studio's backend crash arrives as a 400 and is retried after the first backoff."""
    call, calls = flaky(bad_request(CRASH))
    assert with_retry(call, what="completion", sleep=waits.append) == "ok"
    assert calls["n"] == 2 and waits == [5.0]


def test_a_genuine_bad_request_is_not_retried(waits):
    """Any 400 outside the crash allowlist fails at once: retrying it only costs the backoff."""
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
    """Connection errors, timeouts and 5xx are always worth another attempt."""
    assert transient_reason(err)


def test_exhausted_retries_raise_a_provider_error_chained_to_the_original(waits, monkeypatch):
    """After CRA_LLM_RETRIES, an exhausted TRANSIENT error becomes a ProviderError.

    So the graph labels the outage as the provider's, not "internal"; the original
    error stays reachable as its cause and its text stays in the message.
    """
    monkeypatch.setenv("CRA_LLM_RETRIES", "2")
    call, calls = flaky(*[bad_request(CRASH)] * 5)
    with pytest.raises(ProviderError, match="model server unavailable after 3 attempt") as info:
        with_retry(call, what="completion", sleep=waits.append)
    assert isinstance(info.value.__cause__, openai.BadRequestError)
    assert "prediction stream" in str(info.value)
    assert calls["n"] == 3 and waits == [5.0, 15.0]


def test_retries_can_be_turned_off_and_backoff_tuned(waits, monkeypatch):
    """CRA_LLM_RETRIES=0 disables retrying; CRA_LLM_RETRY_BACKOFF sets the waits.

    With retries off, the one transient failure is already exhausted, so it too is
    a ProviderError -- the outage is labelled the provider's, not "internal".
    """
    monkeypatch.setenv("CRA_LLM_RETRIES", "0")
    call, calls = flaky(bad_request(CRASH))
    with pytest.raises(ProviderError, match="after 1 attempt"):
        with_retry(call, what="completion", sleep=waits.append)
    assert calls["n"] == 1
    monkeypatch.setenv("CRA_LLM_RETRIES", "3")
    monkeypatch.setenv("CRA_LLM_RETRY_BACKOFF", "1,2")
    call, _ = flaky(*[openai.APIConnectionError(request=REQ)] * 3)
    assert with_retry(call, what="completion", sleep=waits.append) == "ok"
    assert waits == [1.0, 2.0, 2.0], "the last wait repeats once the list runs out"


def test_a_retry_is_written_to_the_trace(tmp_path, waits):
    """Every retry is a `llm.retry` trace row, so the monitor can show it on its step."""
    call, _ = flaky(bad_request(CRASH))
    with_retry(call, what="completion", role="judge", model="qwen/qwen3-4b",
               sleep=waits.append)
    rows = [json.loads(line) for p in (tmp_path / "var" / "traces").glob("*.jsonl")
            for line in p.read_text().splitlines()]
    (row,) = [r for r in rows if r["span"] == "llm.retry"]
    assert row["role"] == "judge" and row["attempt"] == 2 and row["of"] == 3
    assert "crashed" in row["reason"] and "prediction stream" in row["error"]


def test_the_adapter_routes_its_call_through_the_policy(monkeypatch):
    """complete() goes through with_retry, and the openai client's own retries are off."""
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    monkeypatch.setattr(llm, "resolved_model", lambda: "qwen/qwen3-4b")
    seen = {}

    def fake_retry(_call, **kw):
        seen.update(kw)
        return chat_reply("fine")
    monkeypatch.setattr(llm_lmstudio, "with_retry", fake_retry)
    assert llm.complete("sys", "user") == "fine"
    assert seen == {"what": "completion", "role": "judge", "model": "qwen3-4b"}
    assert llm.client().max_retries == 0, "a second, hidden retry loop multiplies"


def test_the_monitor_places_a_retry_inside_the_step_it_happened_in():
    """A retry belongs to the step whose time window holds it; one before it does not."""
    span = {"span": "node.independent_review", "status": "ok", "duration_ms": 30000.0,
            "ts": "2026-11-16T10:00:30+00:00", "run_id": "r1", "tenant": "t", "target": "staged"}
    inside = {"span": "llm.retry", "ts": "2026-11-16T10:00:10+00:00", "attempt": 2, "of": 3,
              "reason": "LM Studio's backend crashed mid-generation", "wait_s": 5.0}
    before = {**inside, "ts": "2026-11-16T09:59:00+00:00"}
    (run,) = monitor_runs.runs([span, inside, before])
    (step,) = run["steps"]
    assert [r["ts"] for r in step["retries"]] == [inside["ts"]]
    run["audit"] = {"runtime": {"profile": "mac_max", "entailer": "cross_encoder"}}
    assert monitor_guidance.retry_command(run, {}) == (
        "CRA_PROFILE=mac_max ENTAILER=cross_encoder python -m app.review --tenant t")


def test_the_model_id_lookup_is_retried_but_an_unknown_model_is_not(waits, monkeypatch):
    """The judge resolves its model id minutes into a run. A refused lookup there
    is transient; a model that is simply not served is not, and must say so."""
    calls = {"n": 0}

    def lookup(_url=None, _timeout=2.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("LM Studio is not reachable")
        return ["qwen/qwen3-4b"]
    monkeypatch.setattr(llm_lmstudio, "available_models", lookup)
    # The reply echoes the model id it was sent, so the assertion sees the resolved id.
    client = chat_client(lambda **kw: chat_reply(kw["model"]))
    monkeypatch.setattr(llm_lmstudio, "OpenAI", lambda **_kw: client)
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    monkeypatch.setattr(llm_lmstudio.time, "sleep", waits.append)
    assert llm.complete("s", "u") == "qwen/qwen3-4b" and calls["n"] == 2

    monkeypatch.setattr(llm_lmstudio, "available_models", lambda *a, **k: ["other/model"])
    with pytest.raises(LookupError):
        llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge").complete("s", "u")


@pytest.mark.parametrize("flag,stored", [("", False), ("1", True)])
def test_each_call_traces_a_prompt_hash_and_text_only_when_asked(tmp_path, monkeypatch,
                                                                flag, stored):
    """CRA_TRACE_PROMPTS was documented but wired to nothing. Hash always; the
    (already sanitised) text only when an operator turns it on to debug."""
    monkeypatch.setenv("CRA_TRACE_PROMPTS", flag)
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    monkeypatch.setattr(llm, "resolved_model", lambda: "qwen/qwen3-4b")
    client = chat_client(lambda **_kw: chat_reply("ok", finish_reason="stop"))
    monkeypatch.setattr(llm_lmstudio, "OpenAI", lambda **_kw: client)
    assert llm.complete("SYSTEM-TEXT", "USER-TEXT") == "ok"
    rows = [json.loads(line) for p in (tmp_path / "var" / "traces").glob("*.jsonl")
            for line in p.read_text().splitlines()]
    (row,) = [r for r in rows if r["span"] == "llm.call"]
    assert row["role"] == "judge" and row["prompt_sha"].startswith("sha256:")
    assert ("USER-TEXT" in json.dumps(row)) is stored
