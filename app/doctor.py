"""Environment check: is this machine actually able to run the agent?

    python -m app.doctor
    python -m app.doctor --json

Every failure this catches would otherwise surface at the FIRST LLM call, deep
inside a graph run, as a confusing provider error. Checking up front turns
"why did the diagnosis node explode" into "gemma-4-e4b is not loaded".

Two design rules:

  · **Stdlib only.** This must run before `requirements-local.txt` is installed,
    because "the dependency is missing" is one of the things it reports.
  · **FAIL vs WARN is meaningful.** FAIL means the agent cannot run at all and
    the exit code is non-zero (so CI can gate on it). WARN means degraded but
    usable — a missing reranker costs retrieval precision; a missing model
    server costs everything.

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

from errors import CONFIG_ERRORS

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
    except ConnectionError as e:
        r.add(FAIL, "lm studio server", str(e))
        return
    r.add(OK, "lm studio server", f"{llm_lmstudio.base_url()} ({len(loaded)} model(s) available)")

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


def _check_local_stack(r: Report) -> None:
    """The pip side: the client needed to CALL a model, and the reranker."""
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
            missing = [k for k in ("min_relevance_score", "severity_allowlist", "allowed_actions")
                       if k not in pol]
            if missing:
                r.add(WARN, f"policy: {t}/{surface}", f"no {missing} (inherits _default)")
            for node in ("diagnosis", "judge"):
                try:
                    store.prompt(t, surface, node)
                except FileNotFoundError:
                    r.add(FAIL, f"prompt: {t}/{surface}/{node}",
                          "missing, and no _default fallback")

    r.add(WARN, "threshold calibration",
          f"min_relevance_score values predate {base.EMBEDDING_MODEL} "
          f"({base.EMBEDDING_DIMS} dims) -- re-measure before trusting the gate (T10)")


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
OFFLINE_GATE_ENV = {"LLM_PROVIDER": "stub", "RETRIEVER": "hybrid", "CRA_ACTIVITY": "null"}


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
