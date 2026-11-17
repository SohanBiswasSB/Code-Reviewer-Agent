"""LM Studio — local models over its OpenAI-compatible server.

LM Studio speaks the OpenAI wire format, so the same client shape drives a hosted
provider and a local one; only `base_url` changes. That is why this is a separate
adapter rather than a flag on the OpenAI one: the model ids, the temperature
policy and the reachability failure mode are all different.

Two things worth knowing about this file:

  · `available_models()` uses **urllib from the stdlib, not the openai client**.
    It was written so `doctor` could say "LM Studio is not running" or "that
    model id is not loaded" with nothing from requirements-local.txt installed.
    That no longer holds for this module: under the imports-at-top rule `openai`
    moved to requirements.txt and is imported at the top of this file, and
    `doctor` imports it through the `adapters` package. What the urllib path
    still buys is a plain `ConnectionError` with an actionable message, which
    `with_retry` classifies as transient when the lookup runs inside a model call.

  · **The model id SENT is the id LM Studio reports, resolved strictly (T26).**
    Profiles name models briefly (`qwen3-4b`); LM Studio's API rejects anything
    but its own identifier (`qwen/qwen3-4b`). The adapter used to send the
    configured id verbatim, so `doctor` (which matched loosely) reported OK while
    every real call failed with "Invalid model identifier". Resolving makes the
    matcher load-bearing for WHICH WEIGHTS RUN -- which is why it is strict: the
    old `w in g` clause would have bound `qwen3-8b` to a DeepSeek-R1 distill and
    then actually called it.
"""
from __future__ import annotations

import json as _json
import os
import sys
import time
import urllib.error
import urllib.request

import openai
from openai import OpenAI

from adapters.tracer_jsonl import hash_text, prompts_enabled
from app.activity import WARNING, ActivityLogger, log_event
from registry import adapter

DEFAULT_BASE_URL = "http://localhost:1234/v1"
# LM Studio ignores the key but the OpenAI client requires a non-empty one.
PLACEHOLDER_KEY = "lm-studio"


def base_url() -> str:
    """`LMSTUDIO_BASE_URL` (default localhost:1234/v1), without a trailing slash."""
    return os.getenv("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


# ── retry ────────────────────────────────────────────────────────────────────
# A model call is a pure function of its prompt (temperature 0, no side effects),
# so repeating one is safe -- the question is only WHICH failures are worth it.
#
# The OpenAI client already retried transport errors, 429 and 5xx twice, silently.
# What it never retried is the failure that actually killed a real run
# (2026-11-16, 2:59 PM): LM Studio's MLX backend crashed mid-generation and the
# server reported that as **HTTP 400** -- "Error in iterating prediction stream:
# ... fatal exception in the backend generation thread". A 400 means "your request
# is wrong" everywhere else, so nothing retried it, and three minutes of diagnosis
# were thrown away by a one-off crash in the judge call.
#
# So the policy lives here, in one place, with the client's own retries turned
# off (`max_retries=0`) -- two nested retry loops multiply, and the hidden one
# never shows up in a trace.
#
# The 400 case is an ALLOWLIST of observed crash text, like ROLE_PREFIXES: a
# genuinely bad request (unknown model id, schema rejected) must fail on the
# first attempt, because retrying it only triples the time to the same error.
CRASH_SIGNATURES = (
    "Error in iterating prediction stream",
    "backend generation thread",
)
DEFAULT_RETRIES = 2                 # i.e. up to 3 attempts
DEFAULT_BACKOFF_S = (5.0, 15.0)     # long enough for LM Studio to reload a crashed model


def transient_reason(e: BaseException) -> str | None:
    """Why `e` is worth another attempt, or None if it must propagate now."""
    if isinstance(e, ConnectionError):
        # From `available_models()` -- the model-id lookup, which is stdlib
        # urllib (see module docstring), not the openai client.
        return "could not reach the model server to resolve the model id"
    if isinstance(e, openai.APITimeoutError):
        return "the model server timed out"
    if isinstance(e, openai.APIConnectionError):
        return "could not reach the model server"
    if isinstance(e, (openai.InternalServerError, openai.RateLimitError)):
        return f"the model server answered {type(e).__name__}"
    if isinstance(e, openai.BadRequestError) and any(s in str(e) for s in CRASH_SIGNATURES):
        return "LM Studio's backend crashed mid-generation"
    return None


def retry_policy() -> tuple[int, tuple[float, ...]]:
    """(retries, backoff seconds) from CRA_LLM_RETRIES / CRA_LLM_RETRY_BACKOFF."""
    retries = max(0, int(os.getenv("CRA_LLM_RETRIES", str(DEFAULT_RETRIES))))
    raw = os.getenv("CRA_LLM_RETRY_BACKOFF")
    waits = tuple(float(x) for x in raw.split(",") if x.strip()) if raw else ()
    return retries, waits or DEFAULT_BACKOFF_S


def _announce_retry(what: str, role, model, next_attempt: int, total: int, why: str,
                    error: BaseException, wait: float, sleep) -> None:
    """Say so on stderr and in the trace, then wait. A retry nobody can see is how
    a flaky server stays flaky: the run "just took longer"."""
    # Terminal notice for the person watching, AND a record in the activity log
    # -- before this the retry existed only on a stderr nobody kept.
    print(f"[retry] {what} ({role or '-'}, {model}): {why}. "
          f"Attempt {next_attempt} of {total} in {wait:g}s.", file=sys.stderr, flush=True)
    log_event("external_api", f"retry.{what}", severity=WARNING, exception=error,
              failure_source="external", message=why,
              metadata={"role": role, "model": model, "attempt": next_attempt,
                        "of": total, "wait_s": wait})
    try:
        # Function-level: `config` imports `adapters`, so this is circular at
        # module level.
        import config
        tracer = config.get_tracer()
    except (LookupError, ValueError, ImportError):
        # No tracer configured (registry miss) or config unresolvable: the retry
        # still happens and is still announced on stderr.
        tracer = None
    if tracer is None:
        sleep(wait)
        return
    with tracer.span("llm.retry", what=what, role=role, model=model, attempt=next_attempt,
                     of=total, reason=why, wait_s=wait,
                     error=f"{type(error).__name__}: {error}"[:500]):
        sleep(wait)


def with_retry(call, *, what: str, role: str | None = None, model: str | None = None,
               sleep=None):
    """Run `call()`; on a TRANSIENT failure wait and try again, up to the policy.

    Anything not in `transient_reason` propagates on the first attempt, and the
    last transient one propagates unchanged once attempts run out -- the caller
    sees the real error, never a wrapper that hides its type.
    """
    sleep = sleep or time.sleep           # resolved per call, so it can be patched
    retries, waits = retry_policy()
    for attempt in range(retries + 1):
        try:
            return call()
        except (openai.OpenAIError, ConnectionError) as e:
            why = transient_reason(e)
            if why is None or attempt == retries:
                raise
            _announce_retry(what, role, model, attempt + 2, retries + 1, why, e,
                            waits[min(attempt, len(waits) - 1)], sleep)
    raise AssertionError("unreachable")   # the loop always returns or raises


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
    except (urllib.error.URLError, OSError) as e:
        # OSError too: a server that ACCEPTS and then answers slowly raises a bare
        # TimeoutError while reading, which urlopen does not wrap in URLError -- a
        # busy LM Studio, or the Studio over Tailscale, crashed doctor instead.
        raise ConnectionError(
            f"LM Studio is not reachable at {endpoint} ({getattr(e, 'reason', e)}). "
            f"Start it with `lms server start`."
        ) from None
    return [m.get("id", "") for m in payload.get("data", [])]


# Role prefixes LM Studio puts in front of a model's own name. An ALLOWLIST, not
# a pattern: each entry is a naming convention someone has actually observed.
ROLE_PREFIXES = ("text-embedding-",)


def _core(model_id: str) -> str:
    """A reported id with ONE publisher prefix and any quantisation suffix removed.

    `qwen/qwen3-4b@8bit` -> `qwen3-4b`. Only the outermost of each is stripped, and
    nothing else: what remains must then equal the configured id EXACTLY.
    """
    core = model_id.lower()
    if "/" in core:
        core = core.split("/", 1)[1]
    return core.split("@", 1)[0]


def resolves_to(wanted: str, loaded: list[str]) -> str | None:
    """The one id LM Studio serves for `wanted`, or None. Strict, by design (T26).

    Matches: the exact id; the id behind one publisher prefix (`qwen/qwen3-4b`);
    the id with a quantisation suffix (`qwen3-4b@8bit`); the id behind an
    allowlisted role prefix (`text-embedding-bge-m3`). Never a substring --
    `qwen3-8b` must not resolve to `deepseek/deepseek-r1-0528-qwen3-8b`, a
    different model fine-tuned FROM it.

    Two different candidates is None, not "the first one": choosing between two
    models is a decision, and silently making it is how a profile ends up
    labelling results with weights that never ran.
    """
    if wanted in loaded:
        return wanted
    w = wanted.lower()
    forms = {w} | {prefix + w for prefix in ROLE_PREFIXES}
    matches = sorted({got for got in loaded if got.lower() == w or _core(got) in forms})
    return matches[0] if len(matches) == 1 else None


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

    Per role so the roles CAN differ when a model needs it -- every default is
    0.0 today (see DEFAULT_TEMPERATURE for why the diagnosis role went back to
    greedy decoding), but a diagnosis model that only terminates with sampling
    noise can be given it without touching the judge.
    """
    if role:
        specific = os.getenv(f"CRA_TEMPERATURE_{role.upper()}")
        if specific is not None:
            return float(specific)
    shared = os.getenv("CRA_TEMPERATURE")
    if shared is not None:
        return float(shared)
    return DEFAULT_TEMPERATURE.get(role or "", 0.0)


def _trace_call(role, model, prompt_sha: str, system: str, user: str):
    """A trace span per model call: role, model, prompt HASH -- and the prompt
    text only under CRA_TRACE_PROMPTS=1 (docs/DATA_STORAGE.md, Layer 2).

    The hash alone answers "did the same prompt produce a different result",
    which is most of what a prompt trace is for. The text is sanitised already
    (T12) but stays opt-in: a durable verbatim copy of every prompt is a larger
    surface than a hash.
    """
    # Function-level: `config` imports the adapters, so this is circular at
    # module level (same as `_announce_retry`).
    import config
    attrs = {"role": role, "model": model, "prompt_sha": prompt_sha}
    if prompts_enabled():
        attrs.update(system_prompt=system, user_prompt=user)
    return config.get_tracer().span("llm.call", **attrs)


@adapter("llm", "lmstudio")
class LMStudioLLM:
    """One bound local model.

    Temperature is PER ROLE and 0.0 for every role by default; see
    DEFAULT_TEMPERATURE for the measurement that settled it.
    """

    def __init__(self, model: str, role: str | None = None):
        """Bind one model id to one role; nothing touches the network yet.

        `model` is the profile's id, resolved to LM Studio's own identifier on the
        first call (`resolved_model`). `role` picks the temperature and labels every
        trace, activity record and retry notice. (A `runtime` argument used to be
        passed and stored here, and nothing ever read it; the context budget is
        LM Studio's load setting, not this client's.) `CRA_MAX_TOKENS` (default
        2048) caps the reply.
        """
        self.model = model
        self.role = role
        self.base_url = base_url()
        self.temperature = _temperature_for(role)
        # Leave room for the response inside the platform's context budget.
        self.max_tokens = int(os.getenv("CRA_MAX_TOKENS", "2048"))
        self._client = None
        self._resolved: str | None = None

    def _get_client(self):
        """The OpenAI-compatible client, built on first use with its retries OFF."""
        if self._client is None:
            # max_retries=0: the retry policy is `with_retry`, in one visible place.
            self._client = OpenAI(base_url=self.base_url, api_key=PLACEHOLDER_KEY,
                                  max_retries=0)
        return self._client

    def resolved_model(self) -> str:
        """The identifier LM Studio actually serves for `self.model`. Cached per instance.

        Raises rather than guessing when the configured id matches nothing, or more
        than one thing, and names the near misses so the fix is obvious.
        """
        if self._resolved is None:
            loaded = available_models(self.base_url)
            got = resolves_to(self.model, loaded)
            if got is None:
                near = [m for m in loaded if self.model.lower() in m.lower()]
                raise LookupError(
                    f"model {self.model!r} (role {self.role}) does not resolve to exactly "
                    f"one model served at {self.base_url}. Near misses: {near or 'none'}. "
                    f"Matching is strict on purpose (T26): load it with `lms load`, or "
                    f"name the exact identifier in the profile.")
            self._resolved = got
        return self._resolved

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """Return raw text. Parsing and validation belong to the caller.

        This adapter never interprets the response — `generate_diagnosis` owns
        turning it into a Diagnosis, and `independent_review` treats the judge's
        reply as opaque human-readable text. Keeping interpretation out of here
        is what stops a provider adapter from quietly becoming a decision-maker.

        The `LLM` port. `json_schema` becomes a strict `response_format`, which
        fixes the reply's structure and nothing about what goes inside a string.
        Transient failures are retried by `with_retry`; an id that resolves to no
        single served model raises `LookupError` on the first attempt. Returns ""
        when the server sends no content.
        """
        kwargs: dict = {
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
        # The model-id lookup is INSIDE the retried call. It is a network round
        # trip too, and for the judge it happens minutes into a run, right after
        # the diagnosis: outside the retry, one busy moment of LM Studio there
        # threw the whole run away (and, with no failure edges, its record).
        # One activity record per call: which model, how long, how many tokens,
        # and -- when it fails after its retries -- the error and traceback.
        # Prompt TEXT is never logged: it carries evidence.
        prompt_sha = hash_text(f"{system}\n{user}")
        with ActivityLogger("external_api", f"chat.{self.role or 'llm'}",
                            target_service="lmstudio",
                            metadata={"model": self.model, "temperature": self.temperature,
                                      "max_tokens": self.max_tokens,
                                      "structured": json_schema is not None,
                                      "prompt_chars": len(system) + len(user),
                                      "prompt_sha": prompt_sha}) as alog, \
                _trace_call(self.role, self.model, prompt_sha, system, user):
            resp = with_retry(
                lambda: self._get_client().chat.completions.create(
                    model=self.resolved_model(), **kwargs),
                what="completion", role=self.role, model=self.model)
            alog.add_metadata("resolved_model", self._resolved)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                alog.add_metadata("prompt_tokens", getattr(usage, "prompt_tokens", None))
                alog.add_metadata("completion_tokens", getattr(usage, "completion_tokens", None))
            alog.add_metadata("finish_reason", getattr(resp.choices[0], "finish_reason", None))
            return resp.choices[0].message.content or ""

    def __repr__(self) -> str:  # shows up in doctor output and tracebacks
        return f"LMStudioLLM(model={self.model!r}, base_url={self.base_url!r})"
