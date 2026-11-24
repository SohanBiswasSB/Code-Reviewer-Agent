"""LLM role binding and the judge-independence invariant.

No model is called here. These tests cover the wiring that makes it *impossible*
to accidentally review a diagnosis with the model that produced it.
"""
import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from adapters import llm_lmstudio
from adapters.embed_lmstudio import LMStudioEmbedder
from adapters.platforms import base
from app import doctor
from app import main as api
from app.api_models import RunRequestV1
from errors import ConfigError, ProviderAuthError
from nodes.generate_diagnosis import generate_diagnosis
from nodes.independent_review import independent_review
from schemas.diagnosis import Citation, Diagnosis
from tests._llm_fakes import chat_client, chat_reply


@pytest.fixture(autouse=True)
def _pin_platform(monkeypatch):
    """One platform and profile for every test, so model ids never depend on the host."""
    monkeypatch.setenv("CRA_PLATFORM", "cuda")
    monkeypatch.setenv("CRA_PROFILE", "portable")


def test_roles_bind_to_distinct_models():
    """Invariant 2: diagnosis and judge resolve to different models."""
    diagnosis = config.get_llm("diagnosis")
    judge = config.get_llm("judge")
    assert diagnosis.model != judge.model, (
        "same model for both roles: independent_review would grade its own output"
    )


def test_unknown_role_is_refused():
    """get_llm is role-bound; an unknown role must not fall back to some default model."""
    with pytest.raises(ValueError, match="unknown LLM role"):
        config.get_llm("reviewer")


def test_independence_assertion_passes_for_shipped_profiles():
    """The guard must not fire on the profiles that actually ship."""
    config.assert_judge_is_independent()


def test_independence_assertion_catches_a_collapsed_profile(monkeypatch):
    """Guard the guard: if a profile edit points both roles at one model, fail."""
    collapsed = base.ModelSet(
        diagnosis="same-model", judge="same-model", utility="u",
        weights_gb={"diagnosis": 1.0, "judge": 1.0},
    )
    monkeypatch.setitem(base.PROFILES, "collapsed", collapsed)
    monkeypatch.setattr(
        config.registry.get_adapter("platform", "cuda"), "profiles",
        ("portable", "collapsed"),
    )
    monkeypatch.setenv("CRA_PROFILE", "collapsed")
    with pytest.raises(ValueError, match="same model"):
        config.assert_judge_is_independent()


def test_diagnosis_runs_deterministic_by_default():
    """Its output feeds a schema and then a citation check; sampling noise there
    buys nothing and risks a malformed field.

    This assertion was briefly REVERSED and is now restored, which is the useful
    part of its history. The first live run produced unparseable JSON, a
    temperature sweep appeared to fix it (0/3 parsed at 0, 3/3 at 0.1), and the
    default was moved to 0.1 on that evidence.

    The sweep had been run against a SYNTHETIC two-line prompt. Against the real
    3757-char diagnosis prompt the same model still failed at 0.1 and 0.3. The
    cause was never the sampling -- it was the model. Swapping mac_max's
    diagnosis model to ministral-3-14b-reasoning gives 3/3 parses at temperature
    0 with byte-identical output across runs, so the original reasoning holds
    unchanged.

    Determinism matters more than it first appears: `severity` and `citations`
    feed `risk_ok` and `groundedness_ok`, so a non-deterministic diagnosis means
    the same PR can be approved on one run and escalated on the next.
    """
    assert config.get_llm("diagnosis").temperature == 0


def test_judge_stays_deterministic():
    """The judge keeps temperature 0, which is why temperature is per-role.

    Its output is free-text `reasoning` a human reads and which CANNOT affect the
    decision, so there is nothing for sampling to corrupt -- and it showed no
    repetition loop at 0. Flipping one global knob would have changed it for no
    reason.
    """
    assert config.get_llm("judge").temperature == 0


def test_temperature_is_overridable_per_role(monkeypatch):
    """A role-specific override wins; the shared knob covers roles without one."""
    monkeypatch.setenv("CRA_TEMPERATURE_DIAGNOSIS", "0.7")
    assert config.get_llm("diagnosis").temperature == 0.7
    # The shared knob still works for anything without a role-specific override.
    monkeypatch.delenv("CRA_TEMPERATURE_DIAGNOSIS")
    monkeypatch.setenv("CRA_TEMPERATURE", "0.4")
    assert config.get_llm("judge").temperature == 0.4


def test_reachability_error_is_actionable():
    """`doctor` must be able to say WHAT to do, not just that a socket failed."""
    with pytest.raises(ConnectionError, match="lms server start"):
        llm_lmstudio.available_models("http://127.0.0.1:9/v1", timeout=0.2)


@pytest.mark.parametrize(
    "wanted,loaded,expected",
    [
        ("qwen3-8b", ["qwen3-8b"], "qwen3-8b"),                       # exact
        ("qwen3-8b", ["qwen/qwen3-8b"], "qwen/qwen3-8b"),             # publisher prefix
        ("qwen3-8b", ["qwen3-8b@q4_k_m"], "qwen3-8b@q4_k_m"),         # quant suffix
        ("qwen3-8b", ["gemma-4-e4b"], None),                          # absent
    ],
)
def test_model_id_matching_tolerates_lm_studio_naming(wanted, loaded, expected):
    """LM Studio reports ids that carry publisher and/or quantisation; a
    configured id that omits them must still match."""
    assert llm_lmstudio.resolves_to(wanted, loaded) == expected


@pytest.mark.parametrize("wanted,loaded", [
    # T26: a different model whose name merely CONTAINS the configured id
    ("qwen3-8b", ["deepseek/deepseek-r1-0528-qwen3-8b"]),
    ("qwen3-4b", ["qwen/qwen3-4b-thinking"]),
    # two candidates is a decision the matcher must not make silently
    ("qwen3-4b", ["qwen/qwen3-4b", "lmstudio-community/qwen3-4b"]),
])
def test_model_id_matching_refuses_substrings_and_ambiguity(wanted, loaded):
    """T26: a contained name or two candidates must resolve to nothing, never a guess."""
    assert llm_lmstudio.resolves_to(wanted, loaded) is None


@pytest.mark.parametrize("wanted,loaded,expected", [
    ("bge-m3", ["text-embedding-bge-m3"], "text-embedding-bge-m3"),          # role prefix
    ("qwen3-4b", ["qwen/qwen3-4b@8bit"], "qwen/qwen3-4b@8bit"),                # both affixes
    ("mistralai/ministral-3-14b-reasoning", ["mistralai/ministral-3-14b-reasoning"],
     "mistralai/ministral-3-14b-reasoning"),                                  # fully qualified
])
def test_model_id_matching_keeps_the_affixes_that_must_work(wanted, loaded, expected):
    """The T26 tightening must not break role-prefix, publisher and quant affixes."""
    assert llm_lmstudio.resolves_to(wanted, loaded) == expected


def test_the_adapter_sends_the_resolved_id_not_the_configured_one(monkeypatch):
    """LM Studio rejects `qwen3-4b`; it serves `qwen/qwen3-4b`. Found when the
    first configured shell ran doctor: the call failed with 400 while doctor's
    presence check (which resolved) said OK."""
    monkeypatch.setattr(llm_lmstudio, "available_models",
                        lambda url=None, timeout=2.0: ["qwen/qwen3-4b", "qwen/qwen3-1.7b"])
    sent = {}

    def create(**kw):
        sent.update(kw)
        return chat_reply("ok")

    # The adapter builds its client from the module's OpenAI with keywords only.
    monkeypatch.setattr(llm_lmstudio, "OpenAI", lambda **_kw: chat_client(create))
    llm = llm_lmstudio.LMStudioLLM("qwen3-4b", role="judge")
    assert llm.complete("sys", "user") == "ok"
    assert sent["model"] == "qwen/qwen3-4b"


def test_an_unresolvable_model_fails_with_the_near_misses(monkeypatch):
    """The error names what IS served, so the fix (load it, or name it exactly) is obvious."""
    monkeypatch.setattr(llm_lmstudio, "available_models",
                        lambda url=None, timeout=2.0: ["deepseek/deepseek-r1-0528-qwen3-8b"])
    with pytest.raises(LookupError, match="deepseek-r1-0528-qwen3-8b"):
        llm_lmstudio.LMStudioLLM("qwen3-8b", role="judge").resolved_model()


def test_doctor_gate_check_is_model_free_even_in_a_configured_shell(monkeypatch):
    """The shell that RUNS the agent exports LLM_PROVIDER=lmstudio; doctor's gate
    check must still use the stub, and must leave the shell's value alone."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setattr(llm_lmstudio.LMStudioLLM, "complete",
                        lambda *a, **k: pytest.fail("the gate check reached a real model"))
    r = doctor.Report()
    doctor.check_gates(r)
    assert r.lines[-1][0] == doctor.OK
    assert os.environ["LLM_PROVIDER"] == "lmstudio"


def test_a_server_that_accepts_but_never_answers_is_not_reachable():
    """A read timeout is a bare TimeoutError, not a URLError; it must still come
    back as the actionable ConnectionError rather than crash doctor."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    held = []
    threading.Thread(target=lambda: held.append(srv.accept()), daemon=True).start()
    try:
        with pytest.raises(ConnectionError, match="lms server start"):
            llm_lmstudio.available_models(f"http://127.0.0.1:{srv.getsockname()[1]}/v1",
                                          timeout=0.3)
    finally:
        srv.close()


# ── the wiring itself (T1) ───────────────────────────────────────────────────
def test_diagnosis_node_resolves_its_model_from_config(monkeypatch):
    """The node must ASK config for a model when none is injected.

    This is the regression test for the defect that made T1 drag: every provider
    -- retriever, entailer, source, sink -- resolved from config when its argument
    was None, but the two LLM arguments fell through to a hardcoded stub built
    inside the node. So `config.get_llm()` had no caller in the entire codebase
    and the graph could not reach a real model however many were installed.

    It asserts the CALL, not the output. Checking the text would pass just as well
    against a stub hardcoded back inside the node, which is precisely the thing
    that was wrong.
    """
    asked: list[str] = []
    real = config.get_llm
    monkeypatch.setattr(config, "get_llm",
                        lambda role="diagnosis": (asked.append(role), real(role))[1])

    def fixed_hits(_query, **_kwargs):
        """One fixed hit for any query (the node passes `k=` and `namespace=`)."""
        return [{"id": "iss-1", "score": 0.9, "text": "auth regression"}]

    generate_diagnosis(
        {"evidence": {"title": "t", "diff": "d"}, "tenant": "_default",
         "_prompts": {"diagnosis": "sys"}},
        retriever=SimpleNamespace(retrieve=fixed_hits))

    assert asked == ["diagnosis"], "generate_diagnosis must resolve its own model"


def test_judge_node_resolves_its_model_from_config(monkeypatch):
    """Same for the judge, and the role must be 'judge' -- not 'diagnosis'.

    Asserting the ROLE matters: `get_llm` is role-bound precisely so a caller
    cannot reuse one model for both and quietly lose the independence property
    `independent_review` exists for. Passing the wrong role here would compile,
    run, and silently destroy that guarantee.
    """
    asked: list[str] = []
    real = config.get_llm
    monkeypatch.setattr(config, "get_llm",
                        lambda role="diagnosis": (asked.append(role), real(role))[1])

    passage = "OAuth callback returns 500 when state param missing"
    independent_review({
        "diagnosis": Diagnosis(summary="s", severity="low", proposed_comment=passage,
                               citations=[Citation(snippet_id="iss-1", score=0.9)]),
        "_retrieved": [{"id": "iss-1", "score": 0.9, "text": passage}],
        "_prompts": {"judge": "sys"},
    })

    assert asked == ["judge"], "independent_review must resolve the JUDGE role"


def test_the_gate_suite_is_offline_by_default():
    """`eval/harness.py` must not need a model server, on any machine.

    The suite is a safety regression suite at a permanent 100% bar and has to give
    byte-identical results on the Mac and the CUDA box. Now that the nodes resolve
    from config, that property is only true because the harness DECLARES itself
    model-free -- so assert the declaration, since losing it would be silent until
    someone ran the gates on a machine with no GPU.
    """
    # Located rather than imported: reading its source needs no module execution.
    source = Path(find_spec("eval.harness").origin).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("LLM_PROVIDER", "stub")' in source


# ── the gateway key (LMSTUDIO_API_KEY) ───────────────────────────────────────
KEY = "sk-test-gateway-key-0123456789"


class _Gateway(BaseHTTPRequestHandler):
    """A model server that wants KEY, like a hosted gateway, and records what it saw."""

    seen: list = []

    def serve_get(self):
        """Answer GET /v1/models: 200 with one model for the right key, else 401."""
        auth = self.headers.get("Authorization")
        _Gateway.seen.append(auth)
        if auth != f"Bearer {KEY}":
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"data": [{"id": "qwen3-4b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    # http.server routes a request to the attribute "do_" + its HTTP method.
    do_GET = serve_get

    def log_message(self, *args):
        pass


@pytest.fixture(name="gateway")
def fixture_gateway():
    """A live _Gateway on a free local port; yields its /v1 base URL."""
    _Gateway.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Gateway)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_the_model_listing_sends_the_key_as_a_bearer_token(monkeypatch, gateway):
    """With the key set, the stdlib model listing authenticates as a bearer token."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", KEY)
    assert llm_lmstudio.available_models(gateway) == ["qwen3-4b"]
    assert _Gateway.seen == [f"Bearer {KEY}"]


def test_without_a_key_nothing_is_sent_and_bare_lm_studio_is_unchanged(monkeypatch, gateway):
    """No key means no Authorization header: a bare LM Studio sees what it always did."""
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    assert llm_lmstudio.api_key() == llm_lmstudio.PLACEHOLDER_KEY
    headers = llm_lmstudio.auth_headers()
    assert isinstance(headers, dict) and not headers
    # A gateway that wants a key, reached with none: the message must say none
    # was set (the key file was not loaded), not that a key was refused.
    with pytest.raises(ProviderAuthError, match="is not set") as e:
        llm_lmstudio.available_models(gateway)
    assert "refused" not in str(e.value)
    assert _Gateway.seen == [None]


def test_a_refused_key_says_so_and_is_not_retried(monkeypatch, gateway):
    """A 401 is a reached server with the wrong key: "start LM Studio" would send
    the reader the wrong way, and retrying a bad key only costs the backoff."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", "wrong")
    with pytest.raises(ProviderAuthError, match="LMSTUDIO_API_KEY") as e:
        llm_lmstudio.available_models(gateway)
    assert "lms server start" not in str(e.value)
    assert llm_lmstudio.transient_reason(e.value) is None
    # Not a PermissionError: in this graph that means "tenant refused" or "human
    # denied", and both /run and /approve catch it as such.
    assert isinstance(e.value, ConfigError) and not isinstance(e.value, PermissionError)


def test_run_with_a_refused_key_is_not_reported_as_a_tenant_refusal(monkeypatch, gateway):
    """/run maps PermissionError to "403: the tenant has not enabled this surface".
    A refused gateway key raised mid-review must not come back looking like that."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", "wrong")

    async def ainvoke(*_args, **_kwargs):
        """Stands in for the graph run, whatever payload and config /run passes."""
        # Where a real run hits the gateway: resolving the model id.
        llm_lmstudio.available_models(gateway)

    graph = SimpleNamespace(ainvoke=ainvoke)
    monkeypatch.setattr(api, "get_graph", lambda: graph)
    with pytest.raises(ProviderAuthError):
        asyncio.run(api.run_surface_v1("code_review", RunRequestV1(target="HEAD")))


def test_the_chat_client_and_the_embedder_carry_the_key(monkeypatch):
    """Both OpenAI clients are built with the key, and neither repr exposes it."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", KEY)
    llm = llm_lmstudio.LMStudioLLM(model="qwen3-4b", role="judge")
    emb = LMStudioEmbedder()
    assert llm.client().api_key == KEY
    assert emb.client().api_key == KEY
    # The key never appears where a record or a page could pick it up.
    assert KEY not in repr(llm) and KEY not in repr(emb)


def test_doctor_never_prints_the_key(monkeypatch, gateway):
    """doctor output gets pasted into chats and issues; the key must not be in it."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", KEY)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", gateway)
    r = doctor.Report()
    doctor.check_lmstudio(r, {"judge": "qwen3-4b"})
    assert r.failed == 0
    assert KEY not in r.render() and KEY not in json.dumps(r.as_dict())


def test_doctor_reports_a_refused_key_as_a_failure_not_a_crash(monkeypatch, gateway):
    """A 401 is one FAIL line naming the refused key, not a traceback out of doctor."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", "wrong")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", gateway)
    r = doctor.Report()
    doctor.check_lmstudio(r, {"judge": "qwen3-4b"})
    assert r.lines == [(doctor.FAIL, "lm studio server", r.lines[0][2])]
    assert "refused the API key" in r.lines[0][2]


@pytest.mark.parametrize(("url", "key", "expect"), [
    ("https://gw.example/v1", KEY, "OK"),
    ("http://203.0.113.5:1234/v1", KEY, "WARN"),
    ("http://203.0.113.5:1234/v1", "", None),     # no key: nothing to leak
])
def test_doctor_warns_when_the_key_would_travel_in_clear_text(monkeypatch, url, key, expect):
    """A key over plain http is a WARN, over https an OK; with no key there is no line."""
    monkeypatch.setenv("LMSTUDIO_API_KEY", key)
    r = doctor.Report()
    doctor.check_key_transport(r, url)
    got = [s.strip() for s, c, _ in r.lines if c == "api key transport"]
    assert got == ([expect] if expect else [])
