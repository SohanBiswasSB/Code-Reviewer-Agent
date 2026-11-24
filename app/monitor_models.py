"""The monitor's model table: which model serves each role, where, and is it ready.

Split out of `app/monitor.py` (2026-11-24, which had grown past pylint's 1000-line
module limit); `app.monitor.status()` puts `models()` and `model_table()` on the page.

## Which models: what the RUN used, not what this shell would pick

The profile is env-driven, and the monitor is usually started in a different
shell from the review. Reading `CRA_PROFILE` here would describe a run that never
happened. The model table therefore prefers the `runtime` block the last audit
record wrote (in the run's own process) and says when it fell back to this
shell's config instead.

`urllib` for LM Studio's model list, no client library of its own. The matcher
is the adapter's own strict one (T26), not a copy.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import config
from adapters.llm_lmstudio import auth_headers, available_models, base_url, resolves_to
from errors import CONFIG_ERRORS, ConfigError

SERVER_TTL_S = 5.0         # how long one answer from LM Studio's model list is reused

# What each role is FOR, in one line. The page shows these next to the model, so
# "3 loaded of 14" turns into "the thing that writes the review is up".
ROLES = {
    "diagnosis": "Reads the diff plus related code from the index and writes the review: "
                 "summary, severity, findings.",
    "judge": "Independent second model. Explains whether the review holds up -- "
             "free text only, it never decides.",
    "utility": "Small helper for cheap side tasks. Not called by a code review today.",
    "embedding": "Turns code chunks (at ingest) and each diff (at review) into vectors, "
                 "so related code can be found.",
    "reranker": "Re-scores the top fused passages with a cross-encoder (T13). Off by "
                "default: on measurement it separated relevant passages worse than "
                "the cosine.",
    "entailer": "Scores whether the review's claims follow from the passages it cites. "
                "Drives the entailment check.",
}
LM_STUDIO_ROLES = ("diagnosis", "judge", "utility", "embedding")

_SERVER: dict[str, tuple[float, dict]] = {}


def server_models() -> dict:
    """Everything LM Studio has on disk, and which of it is LOADED, via its REST API.

    Reused for SERVER_TTL_S. Asked on every 1/s poll, an unreachable Studio host
    over the tailnet blocked each poll for the full 1.5 s timeout (measured), so
    the page lagged most when the thing it reports on was down.
    """
    root = base_url().rsplit("/v1", 1)[0]
    hit = _SERVER.get(root)
    if hit and time.monotonic() < hit[0]:
        return hit[1]
    value = fetch_server_models(root)
    _SERVER[root] = (time.monotonic() + SERVER_TTL_S, value)
    return value


def fetch_server_models(root: str) -> dict:
    """One uncached call to LM Studio's `/api/v0/models`, reduced to what the page shows.

    Never raises: an unreachable or garbled server is a `reachable: False` value
    the page renders, with a 1.5 s timeout so it cannot hold up a poll for long.
    It sends `LMSTUDIO_API_KEY` when one is set. The key goes in the request
    header only, never into the value the page receives. A gateway that does not
    expose LM Studio's `/api/v0` shows as unreachable here, which is what the page
    should report.
    """
    request = urllib.request.Request(f"{root}/api/v0/models", headers=auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data", [])
    except urllib.error.HTTPError as e:
        # Caught before URLError, which it subclasses. A 404 is a server that is UP
        # but is not LM Studio -- the devops gateway (2026-11-17) serves only the
        # OpenAI-compatible /v1. Showing it as "not reachable" sent the reader to
        # restart a server that was answering every review.
        if e.code == 404:
            return _fetch_openai_models(root)
        return {"reachable": False, "url": root, "error": str(e)[:200], "models": []}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"reachable": False, "url": root, "error": str(e)[:200], "models": []}
    return {"reachable": True, "url": root, "models": [{
        "id": m.get("id"), "type": m.get("type"), "state": m.get("state"),
        "quantization": m.get("quantization"), "format": m.get("compatibility_type"),
        "context": m.get("loaded_context_length") or m.get("max_context_length"),
    } for m in data]}


def _fetch_openai_models(root: str) -> dict:
    """The standard `GET /v1/models` list, for a server with no LM Studio `/api/v0`.

    The OpenAI listing says which models are SERVED, not whether they are loaded,
    so each gets `state: "served"` -- the honest value. A hosted gateway loads on
    demand or keeps models resident; either way a served model answers a call.
    Never raises, like `fetch_server_models`; the key goes in the header only.
    """
    try:
        ids = available_models(f"{root}/v1", timeout=3.0)
    except (ConnectionError, ConfigError) as e:   # ProviderAuthError is a ConfigError
        return {"reachable": False, "url": root, "error": str(e)[:200], "models": []}
    return {"reachable": True, "url": root, "api": "openai",
            "models": [{"id": m, "type": None, "state": "served", "quantization": None,
                        "format": None, "context": None} for m in ids]}


def resolve_runtime(recent: list[dict]) -> dict:
    """The newest runtime a run recorded, else this shell's own resolution."""
    for r in recent:
        rt = (r.get("audit") or {}).get("runtime") or {}
        if rt.get("models"):
            return {**rt, "source": "run", "as_of": r.get("ended_at")}
    try:
        return {"platform": type(config.get_platform()).__name__,
                "profile": config.get_profile(),
                "entailer": os.getenv("ENTAILER", "lexical"),
                "reranker": os.getenv("RERANKER", "none"),
                "models": config.get_models(), "model_server": base_url(),
                "source": "shell"}
    except CONFIG_ERRORS as e:
        # An unresolvable profile is SHOWN as an error on the page, not a crash.
        return {"source": "shell", "error": f"{type(e).__name__}: {e}"[:200], "models": {}}


def _reranker_row(runtime: dict, assigned: dict) -> dict:
    """The reranker's row, from what the run (or this shell) actually selected.

    It used to be hard-coded "unused -- dense + BM25 only", which became false the
    day T13 shipped: a run with RERANKER=cross_encoder was shown as not reranking.
    A record written before `reranker` was recorded reads as off, which is what
    every such run was.
    """
    selected = runtime.get("reranker") or "none"
    row = {"role": "reranker", "does": ROLES["reranker"],
           "model": assigned.get("reranker") or "—", "where": "in-process (this machine)"}
    if selected == "none":
        return {**row, "state": "off",
                "note": "RERANKER=none (default): order is dense + BM25 fusion, and the "
                        "gate reads the cosine. RERANKER=cross_encoder opts in."}
    return {**row, "state": "ready",
            "note": f"RERANKER={selected}: rescoring the top fused candidates; the gate "
                    f"reads min_rerank_score for these hits"}


def model_table(server: dict, runtime: dict) -> list[dict]:
    """One row per role: which model, where it runs, what it does, is it ready."""
    assigned = runtime.get("models") or {}
    on_server = server.get("models") or []
    by_id = {m["id"]: m for m in on_server}
    # The ADAPTER'S matcher, not a copy: it is strict on purpose (T26), and a
    # looser monitor-side copy reported "ready" for ids the adapter refuses.
    def served(model_id: str):
        """The server's entry for this model id, by the adapter's rules; None if absent."""
        return by_id.get(resolves_to(model_id, list(by_id))) if model_id else None

    rows = []
    for role in ("diagnosis", "judge", "embedding", "utility"):
        want = assigned.get(role) or ""
        state, note = _role_state(server, role, want, served(want))
        rows.append({"role": role, "does": ROLES[role], "model": want or "—",
                     "where": f"LM Studio · {server.get('url')}", "state": state,
                     "note": note})
    rows += [_reranker_row(runtime, assigned), _entailer_row(runtime, assigned)]
    used = {h["id"] for h in (served(m) for m in assigned.values() if m) if h}
    return rows + _unused_loaded_rows(server, on_server, used)


def _role_state(server: dict, role: str, want: str, hit: dict | None) -> tuple[str, str]:
    """(state, note) for one model role, from what the server reports for it."""
    if not server.get("reachable"):
        return "unknown", "model server not reachable"
    if hit is None:
        return "missing", f"not on this server -- lms get {want}"
    if hit["state"] == "served":
        return "ready", f"served by {server.get('url')} (OpenAI API; load state not exposed)"
    if hit["state"] == "loaded":
        return "ready", ", ".join(str(x) for x in (
            hit.get("format"), hit.get("quantization"),
            hit.get("context") and f"ctx {hit['context']:,}") if x)
    if role == "utility":
        # Never called by a code review, so "not loaded" is the right state for
        # it, not a warning to act on.
        return "unused", f"not needed for a code review -- lms load {hit['id']}"
    return "not_loaded", f"downloaded, not loaded -- lms load {hit['id']}"


def _entailer_row(runtime: dict, assigned: dict) -> dict:
    """The entailer's row: in-process, so always ready; the note says how strong."""
    ent = runtime.get("entailer") or "lexical"
    return {"role": "entailer", "does": ROLES["entailer"],
            "model": assigned.get("entailer") or (
                "none -- word overlap" if ent == "lexical" else ent),
            "where": "in-process (this machine)", "state": "ready",
            "note": ("ENTAILER=lexical: cheap but strict on paraphrase; "
                     "ENTAILER=cross_encoder is stronger") if ent == "lexical" else ""}


def _unused_loaded_rows(server: dict, on_server: list[dict], used: set[str]) -> list[dict]:
    """Models loaded in LM Studio that this profile never calls: memory held for nothing."""
    return [{"role": "(none)", "does": "Loaded in LM Studio but not used by this profile "
             "-- it only holds memory.", "model": m["id"],
             "where": f"LM Studio · {server.get('url')}", "state": "extra",
             "note": f"lms unload {m['id']}"}
            for m in on_server if m["state"] == "loaded" and m["id"] not in used]


def models() -> dict:
    """Server reachability plus the per-role table, in one block for the page.

    `api` says which listing answered. LM Studio's own API reports load state
    ("loaded"); a plain OpenAI-compatible gateway only reports what it serves
    ("served"). Counting only "loaded" read "0 loaded" for a gateway that was
    answering every review.
    """
    srv = server_models()
    loaded = [m for m in srv["models"] if m["state"] == "loaded"]
    return {"reachable": srv["reachable"], "url": srv["url"], "error": srv.get("error"),
            "api": srv.get("api", "lmstudio"), "loaded": loaded,
            "served": sum(1 for m in srv["models"] if m["state"] == "served"),
            "available": len(srv["models"]), "_server": srv}
