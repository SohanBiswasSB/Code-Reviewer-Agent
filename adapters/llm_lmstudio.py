"""LM Studio — local models over its OpenAI-compatible server.

LM Studio speaks the OpenAI wire format, so the same client shape drives a hosted
provider and a local one; only `base_url` changes. That is why this is a separate
adapter rather than a flag on the OpenAI one: the model ids, the temperature
policy and the reachability failure mode are all different.

Two things worth knowing about this file:

  · `available_models()` uses **urllib from the stdlib, not the openai client**,
    on purpose. `doctor` must be able to tell you "LM Studio is not running" or
    "that model id is not loaded" in an environment where nothing from
    requirements-local.txt is installed. A diagnostic that needs the dependency
    it is diagnosing is not much of a diagnostic.

  · The `openai` import is lazy, inside the call. Importing this module must not
    require the package, or `config` — and therefore the dependency-free eval
    suite — stops importing on a bare environment.
"""
from __future__ import annotations

import json as _json
import os
import urllib.error
import urllib.request

from registry import adapter

DEFAULT_BASE_URL = "http://localhost:1234/v1"
# LM Studio ignores the key but the OpenAI client requires a non-empty one.
PLACEHOLDER_KEY = "lm-studio"


def base_url() -> str:
    return os.getenv("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def available_models(url: str | None = None, timeout: float = 2.0) -> list[str]:
    """Model ids LM Studio is currently serving, via GET /v1/models.

    Raises ConnectionError with an actionable message when the server is down —
    "start it with `lms server start`" is more useful than a stack trace out of
    urllib. Stdlib only; see the module docstring.
    """
    endpoint = f"{url or base_url()}/models"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"LM Studio is not reachable at {endpoint} ({e.reason}). "
            f"Start it with `lms server start`."
        ) from None
    return [m.get("id", "") for m in payload.get("data", [])]


def resolves_to(wanted: str, loaded: list[str]) -> str | None:
    """Match a configured model id against what LM Studio reports.

    Exact match first, then a prefix/suffix match, because LM Studio's reported
    id often carries the publisher or the quantisation that the configured id
    omits ("qwen3-8b" vs "qwen/qwen3-8b" vs "qwen3-8b@q4_k_m"). Returns the
    loaded id, or None.
    """
    if wanted in loaded:
        return wanted
    w = wanted.lower()
    for got in loaded:
        g = got.lower()
        if g == w or g.endswith("/" + w) or g.startswith(w + "@") or w in g:
            return got
    return None


DEFAULT_TEMPERATURE = {
    # Temperature 0 everywhere, which RESTORES the original decision after briefly
    # reversing it. The story is worth keeping, because the reversal looked
    # justified and was not.
    #
    # The first live run produced unparseable JSON from the diagnosis model, and a
    # temperature sweep "fixed" it -- 0 of 3 parsed at temperature 0, 3 of 3 at
    # 0.1. On that basis the diagnosis default was moved to 0.1.
    #
    # That measurement was taken against a SYNTHETIC two-line prompt rather than
    # the real one. Against the actual 3757-char diagnosis prompt the same model
    # still failed at 0.1 and 0.3, and needed 0.5 before it would terminate. The
    # convenient fixture had produced a convenient answer -- the same mistake this
    # project has now hit six times (T21, T5, T22, log_review, demo_multitenant).
    #
    # The real cause was the MODEL, not the sampling. Swapping the diagnosis model
    # to ministral-3-14b-reasoning gives 3 of 3 parses at temperature 0 with
    # BYTE-IDENTICAL output across runs -- so the original reasoning stands
    # exactly as written: the diagnosis output feeds a schema and then a citation
    # check, and sampling noise there buys nothing.
    #
    # Determinism is worth more here than it looks. `severity` and `citations`
    # feed `risk_ok` and `groundedness_ok`, so a non-deterministic diagnosis means
    # the same PR can be approved on one run and escalated on the next.
    "diagnosis": 0.0,
    "judge": 0.0,
    "utility": 0.0,
}


def _temperature_for(role: str | None) -> float:
    """Per-role temperature: CRA_TEMPERATURE_<ROLE>, else CRA_TEMPERATURE, else the default.

    Per role rather than one global knob because the roles genuinely differ: the
    diagnosis model needs a nudge off greedy decoding to terminate, and the judge
    does not. A single `CRA_TEMPERATURE` would have forced one compromise on both.
    """
    if role:
        specific = os.getenv(f"CRA_TEMPERATURE_{role.upper()}")
        if specific is not None:
            return float(specific)
    shared = os.getenv("CRA_TEMPERATURE")
    if shared is not None:
        return float(shared)
    return DEFAULT_TEMPERATURE.get(role or "", 0.0)


@adapter("llm", "lmstudio")
class LMStudioLLM:
    """One bound local model.

    Temperature is PER ROLE, and the diagnosis default is non-zero. See
    `_temperature_for` -- that default reverses an earlier decision, on evidence.
    """

    def __init__(self, model: str, runtime: dict | None = None, role: str | None = None):
        self.model = model
        self.runtime = runtime or {}
        self.role = role
        self.base_url = base_url()
        self.temperature = _temperature_for(role)
        # Leave room for the response inside the platform's context budget.
        self.max_tokens = int(os.getenv("CRA_MAX_TOKENS", "2048"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "the openai package is required to CALL a model "
                    "(pip install -r requirements-local.txt). Note that "
                    "`doctor` deliberately works without it."
                ) from None
            self._client = OpenAI(base_url=self.base_url, api_key=PLACEHOLDER_KEY)
        return self._client

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """Return raw text. Parsing and validation belong to the caller.

        This adapter never interprets the response — `generate_diagnosis` owns
        turning it into a Diagnosis, and `independent_review` treats the judge's
        reply as opaque human-readable text. Keeping interpretation out of here
        is what stops a provider adapter from quietly becoming a decision-maker.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                # The untrusted material (diff, log lines, issue text) arrives in
                # the USER turn, never the system turn. System content is trusted
                # operator instruction; evidence must never carry that authority.
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                    "strict": True,
                },
            }
        resp = self._get_client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:  # shows up in doctor output and tracebacks
        return f"LMStudioLLM(model={self.model!r}, base_url={self.base_url!r})"
