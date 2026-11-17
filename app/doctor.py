"""Environment check: is this machine actually able to run the agent?

    python -m app.doctor
    python -m app.doctor --json

Every failure this catches would otherwise surface at the FIRST LLM call, deep
inside a graph run, as a confusing provider error. Checking up front turns
"why did the diagnosis node explode" into "gemma-4-e4b is not loaded".

Two design rules:

  · **Stdlib at import time.** The module imports only the stdlib plus the
    stdlib-only `errors` and `schemas.policy`, so `python -m app.doctor` always
    starts. The checks themselves import `config`, which imports every adapter --
    and since the imports-at-top rule moved openai, sentence-transformers and
    torch into `requirements.txt`, a machine without that file installed fails
    with an ImportError from `_check_platform` rather than a report line.
  · **FAIL vs WARN is meaningful.** FAIL means the agent cannot run at all and
    the exit code is non-zero (so CI can gate on it). WARN means degraded but
    usable -- a tenant whose index is not built is a WARN (its runs escalate,
    safely); an unreachable model server costs everything.

It also prints the memory arithmetic behind the residency verdict, because
adapters/platforms/mac.py is currently REASONED rather than measured. The first
real Mac run either confirms those numbers or shows exactly which one is wrong.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform as _platform
import sys

from errors import CONFIG_ERRORS, ProviderAuthError
from schemas.policy import policy_errors  # stdlib-only, like errors

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


class Report:
    """Accumulated (status, check, detail) lines; the exit code is derived from them.

    Checks ADD lines rather than raising, so one broken check cannot hide the
    others -- a doctor that stops at the first problem makes you run it N times.
    """

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        """Record one result. `status` is OK, WARN or FAIL."""
        self.lines.append((status, check, detail))

    @property
    def failed(self) -> int:
        """FAIL count -- any FAIL means the agent cannot run, and exits non-zero."""
        return sum(1 for s, _, _ in self.lines if s == FAIL)

    @property
    def warned(self) -> int:
        """WARN count -- degraded but usable; never changes the exit code."""
        return sum(1 for s, _, _ in self.lines if s == WARN)

    def render(self) -> str:
        """Aligned human-readable lines, ASCII only (Windows cp1252 consoles)."""
        out = []
        for status, check, detail in self.lines:
            out.append(f"[{status}] {check:38} {detail}")
        return "\n".join(out)

    def as_dict(self) -> dict:
        """The same results for `--json`, with the status padding stripped."""
        return {
            "checks": [{"status": s.strip(), "check": c, "detail": d} for s, c, d in self.lines],
            "failed": self.failed,
            "warned": self.warned,
        }


def _check_host(r: Report) -> None:
    """OS, CPU architecture and Python version: the context for every other line."""
    r.add(OK, "host", f"{_platform.system()} {_platform.machine()} / "
                      f"python {_platform.python_version()}")


def _check_platform(r: Report) -> dict:
    """Platform + profile + the residency arithmetic."""
    import config
    from adapters.platforms import base

    detected = config.detect_platform()
    forced = os.getenv("CRA_PLATFORM", "auto")
    plat = config.get_platform()
    profile = config.get_profile()

    r.add(OK, "platform", f"{plat.name} (detected={detected}, CRA_PLATFORM={forced})")

    try:
        models = config.get_models()
        rt = plat.runtime(profile)
    except (ValueError, KeyError) as e:
        r.add(FAIL, "profile", f"{profile}: {e}")
        return {}

    mem = rt["memory"]
    verdict = (f"{rt['residency']} -- hot {mem['hot_weights_gb']}GB + kv {mem['kv_reserve_gb']}GB "
               f"= {mem['needed_gb']}GB vs budget {mem['budget_gb']}GB")
    r.add(OK, "profile", f"{profile} (supported: {list(plat.profiles)})")
    r.add(OK, "residency (computed)", verdict)
    r.add(OK, "runtime", f"{rt['llm_backend']}/{rt['quant']} ctx={rt['context']} "
                         f"kv={rt['kv_cache_dtype']} threads={rt['threads']}")

    ms = base.PROFILES[profile]
    if ms.families_are_independent():
        r.add(OK, "judge independence", f"{models['diagnosis']} vs {models['judge']}")
    else:
        r.add(FAIL, "judge independence",
              f"{models['diagnosis']} and {models['judge']} look like the same family -- "
              f"independent_review would exhibit self-preference bias")
    try:
        config.assert_judge_is_independent()
    except ValueError as e:
        r.add(FAIL, "judge is a distinct model", str(e))
    return models


def _check_lmstudio(r: Report, models: dict) -> None:
    """Server reachable, and every role's model actually loaded."""
    from adapters import llm_lmstudio

    try:
        loaded = llm_lmstudio.available_models()
    except (ConnectionError, ProviderAuthError) as e:
        # ProviderAuthError: reached, but the key was refused (HTTP 401/403).
        r.add(FAIL, "lm studio server", str(e))
        return
    r.add(OK, "lm studio server", f"{llm_lmstudio.base_url()} ({len(loaded)} model(s) available)")
    _check_key_transport(r, llm_lmstudio.base_url())

    for role in ("diagnosis", "judge", "utility", "embedding"):
        wanted = models.get(role)
        if not wanted:
            continue
        got = llm_lmstudio.resolves_to(wanted, loaded)
        if got:
            note = f"{wanted}" if got == wanted else f"{wanted} -> {got}"
            r.add(OK, f"model: {role}", note)
        else:
            # WARN not FAIL: with residency=jit_swap the model is loaded on
            # demand, so "not currently loaded" is expected rather than broken.
            near = [m for m in loaded if wanted.lower() in m.lower()]
            hint = (f"ambiguous or near-miss: {near} -- matching is strict (T26)"
                    if near else f"not available (fetch: lms get {wanted})")
            r.add(WARN, f"model: {role}", f"{wanted} {hint}")


def _check_key_transport(r: Report, url: str) -> None:
    """WARN when `LMSTUDIO_API_KEY` would travel as a Bearer token over plain http.

    A WARN, not a FAIL: plain http over a tailnet is still encrypted by
    WireGuard, and localhost never leaves the machine. Doctor cannot tell a
    tailnet from the open internet, so it says so and leaves the call to the
    reader. Reports nothing when no key is set, since there is nothing to leak.
    """
    if not os.getenv("LMSTUDIO_API_KEY"):
        return
    if url.lower().startswith("https://"):
        r.add(OK, "api key transport", "LMSTUDIO_API_KEY is sent over https")
    else:
        r.add(WARN, "api key transport",
              f"LMSTUDIO_API_KEY is sent in clear text to {url}. Use https:// unless "
              f"this host is localhost or on a tailnet.")


def _check_local_stack(r: Report) -> None:
    """The pip side: the client needed to CALL a model, and the reranker's library.

    Both now come with `requirements.txt` and are imported by `config`, which
    `_check_platform` loads first -- so in practice these report OK. The reranker
    is also OFF by default (T13, on measurement), so its absence no longer costs
    a default run anything.
    """
    if importlib.util.find_spec("openai"):
        r.add(OK, "openai client", "installed")
    else:
        r.add(FAIL, "openai client", "missing -- pip install -r requirements-local.txt")

    if importlib.util.find_spec("sentence_transformers"):
        r.add(OK, "sentence-transformers", "installed (reranker available)")
    else:
        r.add(WARN, "sentence-transformers",
              "missing -- reranking disabled, retrieval precision degraded (T13)")


def _check_tenants(r: Report) -> None:
    """Every tenant's policy parses and its prompts resolve."""
    import config
    from adapters.platforms import base

    try:
        store = config.get_tenant_store()
        tenants = store.tenants()
    except CONFIG_ERRORS as e:          # missing dir, bad JSON, unknown store
        r.add(FAIL, "tenant store", f"{type(e).__name__}: {e}")
        return
    r.add(OK, "tenant store", f"{len(tenants)} codebase(s): {tenants}")

    for t in tenants + ["_default"]:
        for surface in config.enabled_surfaces():
            try:
                pol = store.policy(t, surface)
            except CONFIG_ERRORS as e:      # unreadable or malformed policy.json
                r.add(FAIL, f"policy: {t}/{surface}", f"{type(e).__name__}: {e}")
                continue
            if surface not in pol.get("enabled_surfaces", []):
                continue          # tenant legitimately does not run this surface
            for problem in policy_errors(pol):      # G11, the same check resolve_tenant runs
                r.add(FAIL, f"policy: {t}/{surface}", problem)
            missing = [k for k in ("min_relevance_score", "severity_allowlist", "allowed_actions")
                       if k not in pol]
            if missing:
                r.add(WARN, f"policy: {t}/{surface}", f"no {missing} (inherits _default)")
            _check_relevance_bar(r, f"{t}/{surface}", pol.get("min_relevance_score"),
                                 base)
            for node in ("diagnosis", "judge"):
                try:
                    store.prompt(t, surface, node)
                except FileNotFoundError:
                    r.add(FAIL, f"prompt: {t}/{surface}/{node}",
                          "missing, and no _default fallback")



def _check_relevance_bar(r: Report, where: str, bar, base) -> None:
    """Can this policy's relevance bar fail at all, for the pinned embedder?

    This used to be one blanket WARN that every bar predated the embedder. That
    stopped being true when the real tenants moved to a measured 0.75, and a
    warning that is always on teaches people to ignore it. Now each bar is checked
    against the floor `tools.calibrate` measured: below it, the relevance half of
    groundedness passes nearly every hit, which is T21's shape -- a gate that cannot fail.
    """
    if not isinstance(bar, int | float):
        return                      # absent: inherits _default, checked there
    floor = base.MEASURED_COSINE_FLOOR
    if bar < floor:
        r.add(WARN, f"relevance bar: {where}",
              f"{bar} is below {base.EMBEDDING_MODEL}'s measured cosine floor (~{floor}), "
              f"so it rejects almost nothing -- measure with `python -m tools.calibrate`")
    else:
        r.add(OK, f"relevance bar: {where}", f"{bar} (at or above the measured floor {floor})")


def _check_indexes(r: Report) -> None:
    """Is each tenant's retrieval index built, and by the embedder that will query it?

    The indexed retriever returns `[]` for an unbuilt namespace, which is honest
    and escalates via `groundedness_ok` -- and is also exactly the "fails safe,
    goes inert, nobody notices" shape this project keeps finding. A tenant whose
    every run escalates because nobody ran ingest should show up HERE, not be
    discovered from a month of escalations.

    Stdlib only: reads `manifest.json`, never the `.npy`. An index built by a
    different embedder is a FAIL, because the retriever refuses it and every
    run for that tenant would raise.
    """
    import config
    from adapters.platforms import base
    from adapters.vector_index_local import MANIFEST, _root, namespace_dirname

    retriever = os.getenv("RETRIEVER", "indexed")
    if retriever != "indexed":
        r.add(OK, "retrieval index", f"RETRIEVER={retriever} -- no index needed")
        return
    try:
        store = config.get_tenant_store()
        tenants = store.tenants()
    except CONFIG_ERRORS:
        return                      # _check_tenants already reported it
    root = _root()
    for t in tenants:
        try:
            repo = (store.policy(t, "code_review").get("codebase") or {}).get("repo")
        except CONFIG_ERRORS:
            continue                    # reported by _check_tenants
        if not repo:
            continue
        path = root / namespace_dirname(repo) / MANIFEST
        if not path.exists():
            r.add(WARN, f"index: {t}",
                  f"not built -- every run escalates. python -m tools.ingest --tenant {t}")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("model") != base.EMBEDDING_MODEL:
            r.add(FAIL, f"index: {t}",
                  f"built by {m.get('model')!r}, but EMBEDDING_MODEL is "
                  f"{base.EMBEDDING_MODEL!r}: the retriever will refuse it. Re-ingest.")
            continue
        r.add(OK, f"index: {t}",
              f"{m.get('chunks')} chunks / {m.get('files')} files, built {m.get('built_at')}")


# What "model-independent" has to mean for the gate check: forced, not defaulted.
# eval/harness.py only `setdefault`s these, so an operator can point the suite at a
# real model on purpose -- but doctor runs inside whatever shell it is given, and
# once someone exports LLM_PROVIDER=lmstudio to use the agent, the "model-free"
# gate check started calling the real judge (and crashed doctor when the id did
# not resolve). Found 2026-11-16, the first time doctor ran in a configured shell.
OFFLINE_GATE_ENV = {"LLM_PROVIDER": "stub", "RETRIEVER": "hybrid", "RERANKER": "none",
                    "CRA_ACTIVITY": "null"}


def _check_gates(r: Report) -> None:
    """The safety gates must pass on every machine, with no model server."""
    import contextlib
    import io

    saved = {k: os.environ.get(k) for k in OFFLINE_GATE_ENV}
    os.environ.update(OFFLINE_GATE_ENV)
    try:
        try:
            from eval.harness import run
        except (ImportError, SyntaxError) as e:
            r.add(FAIL, "safety gate suite", f"import failed: {type(e).__name__}: {e}")
            return
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if code == 0:
        r.add(OK, "safety gate suite", "all gates pass (pass^k, model-independent)")
    else:
        r.add(FAIL, "safety gate suite", "one or more gates FAILED -- run python -m eval.harness")


def run_checks(*, gates: bool = True) -> Report:
    """Every check, in order -- the ONE list the CLI and `GET /doctor` share.

    The HTTP endpoint used to import four private `_check_*` functions and run a
    subset of them while its docstring claimed "the same checks", so it silently
    skipped tenants and indexes. `gates=False` is the only difference allowed:
    the gate suite prints and takes ~0.5 s, which is wrong inside a request.
    """
    r = Report()
    _check_host(r)
    models = _check_platform(r)
    _check_local_stack(r)
    if models:
        _check_lmstudio(r, models)
    _check_tenants(r)
    _check_indexes(r)
    if gates:
        _check_gates(r)
    return r


def main() -> int:
    """Run every check, print the report, and return 1 if anything FAILED."""
    ap = argparse.ArgumentParser(description="check this machine can run the agent")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    r = run_checks()

    if args.json:
        print(json.dumps(r.as_dict(), indent=2))
    else:
        print(r.render())
        print()
        if r.failed:
            print(f"{r.failed} FAILED, {r.warned} warning(s) -- the agent cannot run as configured")
        elif r.warned:
            print(f"0 failed, {r.warned} warning(s) -- usable, but degraded")
        else:
            print("all checks pass")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
