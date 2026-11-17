"""A live view of the agent: what is running, which models serve it, what each run did.

    python -m app.monitor                 # then open http://127.0.0.1:8765
    python -m app.monitor --port 9000

Reads only what the agent already writes under CRA_DATA_DIR -- `live/` (stage
starts and ends, ingest progress), `traces/` (every stage of every run, errors
included), `audit/` (the verdict, the four checks, the findings) and `activity/`
(the activity log, through `app.activity.query`) -- plus LM Studio's model list.
It starts nothing, changes nothing, and can be opened and closed at any time.

## A run is traces JOINED with audit, not audit alone

The first version listed audit records only, and a run that crashed mid-graph
wrote none (there are no error edges, so `audit` is never reached). The one run
that most needed looking at -- LM Studio dying inside `independent_review` -- was
therefore the one run the page could not show. History is now built from the
trace spans, keyed by `run_id`, with the audit record attached where one exists.

Since G9 a crash does leave a `failed` / `interrupted` record (written by
`app/graph.py::_traced`), but the join stays: the spans are where each stage's
status, duration and retries come from, and a run paused at the approval gate
still has no audit record at all.

## Which models: what the RUN used, not what this shell would pick

The profile is env-driven, and the monitor is usually started in a different
shell from the review. Reading `CRA_PROFILE` here would describe a run that never
happened. The model table therefore prefers the `runtime` block the last audit
record wrote (in the run's own process) and says when it fell back to this
shell's config instead.

## Localhost only

Binds 127.0.0.1 and refuses anything else. The page shows tenant names, targets,
citation ids, escalation reasons and the diagnosis text -- evidence-DERIVED data,
the same trust level as `var/` itself (docs/DATA_STORAGE.md). There is no
authentication, so it must not be reachable from another machine.

## No server framework of its own

`http.server` and `urllib` for the serving and for LM Studio's model list. The
project imports are the things that must not be copied: `config` to name the
models a run would use when no audit record says so, the adapter's strict model
matcher (T26), `app.activity`'s reader, and `STAGE_PLAN` from `app.graph`. Through
`config` and `app.graph` it loads every adapter and langgraph, so it needs
`requirements.txt` like everything else -- just nothing beyond it.
"""
from __future__ import annotations

import argparse
import bisect
import errno
import json
import os
import pathlib
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from adapters.llm_lmstudio import auth_headers, available_models, base_url, resolves_to
from adapters.vector_index_local import MANIFEST, namespace_dirname
from adapters.vector_index_local import _root as index_root
from app import activity
from app import live as live_files  # aliased: runs()/snapshot() take a `live` job list
from app.graph import STAGE_PLAN
from datadir import data_dir
from errors import CONFIG_ERRORS, ConfigError

HOST = "127.0.0.1"
RECENT_RUNS = 20
# The page lives in its own file so it can be edited as HTML, not as a string --
# and is re-read per request, so an edit shows on browser refresh without
# restarting a monitor someone left running in another terminal.
PAGE_PATH = pathlib.Path(__file__).with_name("monitor.html")
PAGE = PAGE_PATH.read_text(encoding="utf-8")


def page() -> str:
    """The page as it is on disk now; the import-time copy if the file vanished."""
    try:
        return PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        return PAGE


TRACE_TAIL = 4000          # trace rows read for history and duration estimates
AUDIT_TAIL = RECENT_RUNS * 4   # audit records read PER TENANT
ORPHAN_GAP_S = 5.0         # a span with no run_id belongs to the run that follows it
STALE_S = 900              # an unaudited, error-free run older than this was killed
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


def _data_dir() -> pathlib.Path:
    return data_dir()


# ── reading var/ ─────────────────────────────────────────────────────────────
# The page polls once a second. The first version re-read and re-parsed EVERY
# trace and audit file on every poll, so its cost grew with history: measured on
# a synthetic year of traces, 224 ms per poll. Two fixes, both here:
#   · files are read NEWEST first and reading stops once the tail is full, so a
#     year of history costs what the last day costs;
#   · parsed rows are cached per file by (mtime, size). A past day's trace file
#     and a past month's audit file never change, so a poll re-parses only the
#     one file currently being appended to.
_ROWS: dict[pathlib.Path, tuple[int, int, list[dict]]] = {}


def _rows(path: pathlib.Path) -> list[dict]:
    """Every parseable JSON line of one file, cached until the file changes."""
    try:
        st = path.stat()
        hit = _ROWS.get(path)
        if hit and hit[:2] == (st.st_mtime_ns, st.st_size):
            return hit[2]
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    _ROWS[path] = (st.st_mtime_ns, st.st_size, rows)
    return rows


def _newest_rows(paths: list[pathlib.Path], limit: int) -> list[dict]:
    """The last `limit` rows across date-named files, oldest first.

    Files are named by date (`2026-11-16.jsonl`, `2026-11.jsonl`), so name order
    is time order, and reading stops at the first file that fills the tail.
    """
    chunks, n = [], 0
    for p in sorted(paths, key=lambda q: q.name, reverse=True):
        chunk = _rows(p)[-(limit - n):]
        chunks.append(chunk)
        n += len(chunk)
        if n >= limit:
            break
    return [r for c in reversed(chunks) for r in c]


def _trace_rows() -> list[dict]:
    return _newest_rows(list((_data_dir() / "traces").glob("*.jsonl")), TRACE_TAIL)


def _recent_audits() -> dict[str, dict]:
    """Recent audit records across ALL tenants, by run_id.

    Read per tenant, then merged. The first version tailed every tenant's files
    in path order and kept the last N rows overall, so the alphabetically last
    tenant's records pushed every other tenant's out of the window -- measured:
    with two tenants, ten finished runs of the first were shown as "Stopped".
    """
    root = _data_dir() / "audit"
    rows: list[dict] = []
    for d in (root.iterdir() if root.is_dir() else ()):
        if d.is_dir():
            rows += _newest_rows(list(d.glob("*.jsonl")), AUDIT_TAIL)
    return {r["run_id"]: r for r in rows if r.get("run_id")}


def _parse_ts(ts) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


# ── models ───────────────────────────────────────────────────────────────────
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
    value = _fetch_server_models(root)
    _SERVER[root] = (time.monotonic() + SERVER_TTL_S, value)
    return value


def _fetch_server_models(root: str) -> dict:
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
    Never raises, like `_fetch_server_models`; the key goes in the header only.
    """
    try:
        ids = available_models(f"{root}/v1", timeout=3.0)
    except (ConnectionError, ConfigError) as e:   # ProviderAuthError is a ConfigError
        return {"reachable": False, "url": root, "error": str(e)[:200], "models": []}
    return {"reachable": True, "url": root, "api": "openai",
            "models": [{"id": m, "type": None, "state": "served", "quantization": None,
                        "format": None, "context": None} for m in ids]}


def _runtime(recent: list[dict]) -> dict:
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


# ── history ──────────────────────────────────────────────────────────────────
def estimates(rows: list[dict] | None = None) -> dict[str, float]:
    """Median ms per stage over recent SUCCESSFUL spans -- the 'what is left' basis."""
    rows = _trace_rows() if rows is None else rows
    by_stage: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") == "ok" and str(r.get("span", "")).startswith("node."):
            by_stage.setdefault(r["span"][5:], []).append(float(r.get("duration_ms", 0)))
    return {k: round(statistics.median(v), 1) for k, v in by_stage.items() if v}


def _group_spans(rows: list[dict]) -> dict[str, list[dict]]:
    """node.* spans by run_id. `resolve_tenant` runs before a run_id exists, so a
    span without one is given to the next span (same tenant and target, within a
    few seconds) that has one."""
    spans = sorted((r for r in rows if str(r.get("span", "")).startswith("node.")),
                   key=lambda r: r.get("ts") or "")
    by_run: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for r in spans:
        rid = r.get("run_id")
        if not rid:
            orphans.append(r)
            continue
        for o in orphans:
            t0, t1 = _parse_ts(o.get("ts")), _parse_ts(r.get("ts"))
            if (o.get("tenant"), o.get("target")) == (r.get("tenant"), r.get("target")) \
                    and t0 and t1 and (t1 - t0).total_seconds() <= ORPHAN_GAP_S:
                by_run.setdefault(rid, []).append(o)
        orphans = []
        by_run.setdefault(rid, []).append(r)
    return by_run


RETRY_FIELDS = ("attempt", "of", "reason", "wait_s", "role", "model", "error", "ts")


class _RetryIndex:
    """`llm.retry` rows sorted by time, timestamps parsed ONCE per poll.

    Retries come from the adapter, which knows its role but not the run, so they
    are placed by time: a retry belongs to the stage whose span encloses it. The
    first version re-parsed every retry timestamp for every span of every run
    (~200k parses per poll at a year of history); this bisects instead.
    """

    def __init__(self, trace_rows: list[dict]):
        """Keep only `llm.retry` rows with a parseable timestamp, sorted by it."""
        pairs = sorted((t.timestamp(), r) for r in trace_rows
                       if r.get("span") == "llm.retry" and (t := _parse_ts(r.get("ts"))))
        self.keys = [k for k, _ in pairs]
        self.rows = [r for _, r in pairs]

    def between(self, start: float, end: float) -> list[dict]:
        """Retries in [start, end] (epoch seconds, inclusive), trimmed to RETRY_FIELDS."""
        lo, hi = bisect.bisect_left(self.keys, start), bisect.bisect_right(self.keys, end)
        return [{k: r.get(k) for k in RETRY_FIELDS} for r in self.rows[lo:hi]]

    def within(self, span: dict) -> list[dict]:
        """The retries inside one node span (which records its END and duration)."""
        end = _parse_ts(span.get("ts"))
        if end is None or not self.keys:
            return []
        stop = end.timestamp()
        return self.between(stop - float(span.get("duration_ms") or 0) / 1000, stop)


def retry_command(run: dict, runtime: dict, override: dict | None = None) -> str:
    """The exact command that re-runs this review, for the page's Retry button.

    The monitor stays read-only: it hands you the command rather than launching
    it. A page with no authentication that can start processes is a page any
    other local site can start processes through.
    """
    rt = (run.get("audit") or {}).get("runtime") or runtime or {}
    env = {"CRA_PROFILE": rt["profile"]} if rt.get("profile") else {}
    if rt.get("entailer") and rt["entailer"] != "lexical":
        env["ENTAILER"] = rt["entailer"]
    if rt.get("reranker") and rt["reranker"] != "none":
        env["RERANKER"] = rt["reranker"]
    env.update(override or {})
    target = run.get("target") or "staged"
    args = f"--tenant {run.get('tenant') or '<tenant>'}"
    if target != "staged":
        args += f" --target {target}"
    return " ".join([*(f"{k}={v}" for k, v in env.items()), "python -m app.review", args])


# ── what the page says under a run ───────────────────────────────────────────
# The gate's own failure strings (nodes/independent_review.py) are the source of
# truth for WHY a run escalated. The page used to infer the reason from the
# findings' citations instead, and on 2026-11-17 it blamed a finding that had
# PASSED while the one that failed sat just under the bar.
_ENTAIL_FAIL = re.compile(r"^(?P<label>.+?) not entailed by its premise "
                          r"\(score (?P<score>[0-9.]+) < (?P<bar>[0-9.]+)\)$")
_FINDING_LABEL = re.compile(r"^finding (?P<n>\d+) \((?P<file>.*)\)$")
_SEVERITY_FAIL = re.compile(r"^severity '(?P<sev>[^']*)' not in (?P<allowed>.*)$")
_ACTION_FAIL = re.compile(r"^action '(?P<action>[^']*)' not in (?P<allowed>.*)$")


def _entailment_text(m: re.Match) -> tuple[str, int | None]:
    """An Entailed failure in words, and the finding number it names, if any."""
    score, bar = float(m["score"]), float(m["bar"])
    f = _FINDING_LABEL.match(m["label"])
    what = f"Finding {f['n']} ({f['file']})" if f else f"The {m['label']}"
    return (f"{what} scored {score:.3f} on Entailed against a bar of {bar:.2f} "
            f"({bar - score:.3f} short): its claim did not clearly follow from the "
            f"code it cites."), (int(f["n"]) if f else None)


# (does it match?, what to say) for the gate's other failure strings, in order.
_FAILURE_TEXT = (
    (lambda t: t == "no citations",
     lambda t, _m: ("The diagnosis cited no retrieved code, so nothing it says could be "
                    "checked against the codebase.")),
    (lambda t: t.startswith("citations not retrieved:"),
     lambda t, _m: ("The model cited passages that retrieval never returned"
                    f"{t[len('citations not retrieved'):]}. A citation that was not "
                    "retrieved is usually invented.")),
    (lambda t: t.startswith("citations below relevance:"),
     lambda t, _m: ("Cited passages scored under the relevance bar"
                    f"{t[len('citations below relevance'):]}.")),
    (_SEVERITY_FAIL.match,
     lambda t, m: (f"The change was rated {m['sev']} severity. This codebase's policy "
                   f"lets only {m['allowed']} through without a human.")),
    (_ACTION_FAIL.match,
     lambda t, m: (f"The proposed action '{m['action']}' is not on this codebase's "
                   f"allowed list {m['allowed']}.")),
)


def _explain_failure(text: str) -> tuple[str, int | None]:
    """One gate failure in plain words, plus the finding number it names, if any."""
    m = _ENTAIL_FAIL.match(text)
    if m:
        return _entailment_text(m)
    for matches, say in _FAILURE_TEXT:
        hit = matches(text)
        if hit:
            return say(text, hit), None
    return text, None


def _cited_files(finding: dict) -> list[str]:
    """The files a finding's citations come from, in order, without duplicates."""
    return list(dict.fromkeys(str(c).split(":", 1)[0] for c in finding.get("citation_ids") or []))


def _unreachable_hint(server: str, local: bool) -> str:
    """What to do when the model server did not answer: local and remote differ."""
    return ("Could not reach the model server. Start it with: lms server start" if local
            else f"Could not reach the model server at {server}. Check that it is up and "
            "that LMSTUDIO_BASE_URL is right.")


def _model_missing_hint(server: str, _local: bool) -> str:
    """What to do when the server does not serve a role's model."""
    return (f"The model this role needs is not served at {server}. "
            "Run `python -m app.doctor` to see which.")


# (substrings that identify the failure, the hint or a function of (server, local)).
# First match wins, so the specific key errors come before the generic connection one.
_CRASH_HINTS = (
    (("LMSTUDIO_API_KEY is not set",),
     "The model server needs an API key and none was loaded. Load your key file in the "
     "shell that runs the review, then retry."),
    (("refused the API key",),
     "The model server refused the API key. Update it in your key file, check it with "
     "doctor, then retry."),
    (("prediction stream", "generation thread", "mlx_engine"),
     "LM Studio's model backend crashed while generating, usually from memory pressure or "
     "a prompt longer than the loaded context. Reload the model, then retry."),
    (("Connection refused", "ConnectionError", "URLError", "APIConnectionError",
      "not reachable"), _unreachable_hint),
    (("context length", "context window", "n_ctx", "too many tokens"),
     "The prompt is longer than the model's context window."),
    (("timed out", "Timeout"),
     "The model call timed out. Retry; if it repeats, the server is overloaded."),
    (("Invalid model identifier", "no model", "No model"), _model_missing_hint),
)


def crash_hint(message: str, server: str) -> str:
    """What to do about a crashed step, for the model server this run used."""
    m = message or ""
    local = any(h in server for h in ("localhost", "127.0.0.1"))
    for needles, hint in _CRASH_HINTS:
        if any(n in m for n in needles):
            return hint(server, local) if callable(hint) else hint
    return "Read the full error on the failed step (Steps tab), fix it, then retry."


def _index_built_at(tenant: str) -> str | None:
    """When this tenant's code index was built, or None if unknown or unbuilt."""
    try:
        policy = config.get_tenant_store().policy(tenant, "code_review")
    except CONFIG_ERRORS:
        return None
    repo = (policy.get("codebase") or {}).get("repo")
    if not repo:
        return None
    path = index_root() / namespace_dirname(repo) / MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("built_at")
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class _RunView:
    """What the guidance reads from one run, parsed once.

    `failed` maps a finding number the gate named to its file; `shortfall` is the
    smallest gap under the entailment bar (None if Entailed did not fail).
    """

    run: dict
    runtime: dict
    audit: dict
    review: dict
    tenant: str
    findings: list
    failed: dict[int, str]
    shortfall: float | None

    @property
    def entailer(self) -> str:
        """The entailer this run used ('lexical' unless the run recorded otherwise)."""
        return ((self.audit.get("runtime") or self.runtime or {}).get("entailer")
                or "lexical")

    def failures_matching(self, pattern: re.Pattern) -> re.Match | None:
        """The first gate failure `pattern` matches, or None."""
        return next((pattern.match(t) for t in self.review.get("failures") or []
                     if pattern.match(t)), None)


def _read_run(run: dict, runtime: dict) -> tuple[_RunView, list[str]]:
    """The run's view, plus one plain-language 'why' line per gate failure."""
    a = run.get("audit") or {}
    review = a.get("review") or {}
    findings = (a.get("diagnosis") or {}).get("findings") or []
    why, failed, gaps = [], {}, []
    for text in review.get("failures") or []:
        line, n = _explain_failure(text)
        why.append(line)
        if n is not None:
            failed[n] = (findings[n - 1].get("file") if 0 < n <= len(findings) else None) or "?"
        m = _ENTAIL_FAIL.match(text)
        if m:
            gaps.append(float(m["bar"]) - float(m["score"]))
    view = _RunView(run, runtime, a, review, run.get("tenant") or "<tenant>", findings,
                    failed, min(gaps) if gaps else None)
    return view, why


def _step(who: str, text: str, **extra) -> dict:
    """One next step: who acts, what to do, and any command, tab or list to show."""
    return {"who": who, "text": text, **{k: v for k, v in extra.items() if v}}


def _reviewer_escalated(v: _RunView) -> list[dict]:
    """What the person reading an escalated change does next."""
    others = len(v.findings) - len(v.failed)
    rest = f"; the other {others} passed." if others else "."
    if len(v.failed) == 1:
        ((n, file),) = v.failed.items()
        steps = [_step("reviewer", f"Read finding {n} ({file}) yourself. It is the one the "
                       "gate could not verify" + rest, tab="findings")]
    else:
        # One step listing them: seven near-identical "read finding N" steps were
        # noise an admin had to read seven times.
        steps = [_step("reviewer", f"Read these {len(v.failed)} findings yourself. The gate "
                       "could not verify them" + rest, tab="findings",
                       items=[f"Finding {n} ({file})" for n, file in v.failed.items()])] \
            if v.failed else []
    if v.review.get("groundedness_ok") is False:
        steps.append(_step("reviewer", "Open Evidence to see which citations were missing or "
                           "scored under the bar.", tab="evidence"))
    if v.review.get("risk_ok") is False:
        steps.append(_step("reviewer", "Escalating this severity is the policy working as "
                           "intended: a human reads it before anything is posted."))
    steps.append(_step("reviewer", "Nothing was posted. If the findings hold up, post them "
                       "yourself; the drafted comment is at the bottom of the Findings tab.",
                       tab="findings"))
    return steps


def _admin_entailment(v: _RunView) -> list[dict]:
    """Admin steps for an Entailed failure: a semantic retry, then a measured bar."""
    steps = []
    if v.entailer == "lexical":
        near = f" The closest miss was {v.shortfall:.3f} under the bar." \
            if v.shortfall is not None else ""
        steps.append(_step("admin", "Re-run with the meaning-based entailer. The word-overlap "
                           "entailer this run used also fails correct paraphrases." + near,
                           command=retry_command(v.run, v.runtime,
                                                 {"ENTAILER": "cross_encoder"})))
    bar = (v.audit.get("gates") or {}).get("min_entailment_score")
    steps.append(_step("admin", f"If good findings keep failing Entailed, measure {v.tenant}'s "
                       f"entailment bar ({bar if bar is not None else 'default'}) instead of "
                       "guessing it.",
                       command=f"python -m tools.calibrate --tenant {v.tenant} entailment"))
    return steps


def _admin_policy(v: _RunView) -> list[dict]:
    """Admin steps for Grounded, Risk and Permission failures."""
    steps = []
    if v.review.get("groundedness_ok") is False:
        steps.append(_step("admin", "If the cited passages look relevant but scored under the "
                           f"bar, measure {v.tenant}'s relevance bar.",
                           command=f"python -m tools.calibrate --tenant {v.tenant} relevance"))
    m = v.failures_matching(_SEVERITY_FAIL)
    if m:
        steps.append(_step("admin", f"To let {m['sev']} severity through without a human, add "
                           f"it to severity_allowlist in tenants/{v.tenant}/policy.json. That "
                           "is a policy decision, not a fix."))
    m = v.failures_matching(_ACTION_FAIL)
    if m:
        steps.append(_step("admin", f"If '{m['action']}' should be allowed here, add it to "
                           f"allowed_actions in tenants/{v.tenant}/policy.json."))
    return steps


def _guide_escalated(v: _RunView, _why: list[str]) -> list[dict]:
    """Escalated: reviewer steps first, then the admin's. `why` is already the gate's."""
    admin = _admin_entailment(v) if v.review.get("entailment_ok") is False else []
    return _reviewer_escalated(v) + admin + _admin_policy(v)


def _guide_crashed(v: _RunView, why: list[str]) -> list[dict]:
    """Crashed: say where, then the server-aware hint and the retry."""
    err = v.run.get("error") or {}
    why.append(f"Crashed at {err.get('step')}: {_first_line(err.get('message'))}")
    server = (v.audit.get("runtime") or v.runtime or {}).get("model_server") or base_url()
    return [_step("admin", crash_hint(err.get("message") or "", server), tab="steps"),
            _step("admin", "Then re-run it.", command=retry_command(v.run, v.runtime))]


def _guide_incomplete(v: _RunView, why: list[str]) -> list[dict]:
    """Stopped without a record: name the last step, offer the retry."""
    last = (v.run.get("steps") or [{}])[-1].get("name") or "start"
    why.append(f"Stopped after {last} without writing a record. The process was "
               "probably killed (Ctrl-C, a closed terminal or a crash outside Python).")
    return [_step("admin", "Re-run it.", command=retry_command(v.run, v.runtime))]


def _guide_awaiting(_v: _RunView, why: list[str]) -> list[dict]:
    """Paused at the approval prompt."""
    why.append("Passed the checks and is paused at the approval prompt.")
    return [_step("reviewer", "Answer the y/n prompt in the terminal that started this run. "
                  "Nothing is written until you do.")]


def _guide_abandoned(v: _RunView, why: list[str]) -> list[dict]:
    """Nobody answered the prompt."""
    why.append("Stopped at the approval prompt with nobody to answer it. Nothing was written.")
    return [_step("reviewer", "Re-run it and answer the prompt.",
                  command=retry_command(v.run, v.runtime))]


def _guide_executed(v: _RunView, why: list[str]) -> list[dict]:
    """Approved and written: say where."""
    why.append("Passed all four checks and was approved.")
    path = ((v.audit.get("result") or {}).get("review") or {}).get("markdown")
    return [_step("reviewer", "Read the written review" + (f": {path}" if path else "."))]


def _guide_denied(_v: _RunView, why: list[str]) -> list[dict]:
    """Declined at the prompt: nothing to do."""
    why.append("Passed the checks, but was declined at the approval prompt. "
               "Nothing was written.")
    return []


def _guide_human_review(_v: _RunView, why: list[str]) -> list[dict]:
    """Routed to a human by rule."""
    why.append("Routed straight to a human by rule. No model looked at it.")
    return [_step("reviewer", "Review the change yourself; the agent deliberately skips "
                  "this kind of change.")]


_STATUS_GUIDES = {"escalated": _guide_escalated, "crashed": _guide_crashed,
                  "incomplete": _guide_incomplete, "awaiting_approval": _guide_awaiting,
                  "abandoned": _guide_abandoned, "executed": _guide_executed,
                  "denied": _guide_denied, "human_review": _guide_human_review}


def _off_target_steps(v: _RunView, index_built: str | None) -> list[dict]:
    """Findings citing only other files: a re-ingest hint, with which passed or failed."""
    off = [(n, f) for n, f in enumerate(v.findings, start=1)
           if f.get("file") and _cited_files(f) and f["file"] not in _cited_files(f)]
    if not off:
        return []
    built = f" ({index_built})" if index_built else ""
    return [_step("admin", "Some findings cite only other files: retrieval returned no code "
                  "from the file they are about. If those files are new or changed since the "
                  f"index was built{built}, re-ingest it.",
                  items=[f"Finding {n} ({f['file']}) cites {', '.join(_cited_files(f))}"
                         + (" — failed the gate" if n in v.failed else " — passed the gate")
                         for n, f in off],
                  command=f"python -m tools.ingest --tenant {v.tenant}")]


def _prepush_steps(v: _RunView) -> list[dict]:
    """The pre-push flags, grouped by file and check, as one reviewer step."""
    flags = [c for c in v.audit.get("checks") or [] if c.get("status") == "flag"]
    if not flags:
        return []
    by_file: dict[str, Counter] = {}
    for c in flags:
        where = c.get("file") or "(whole change)"
        by_file.setdefault(where, Counter())[c.get("check") or "?"] += 1
    plural = "s" if len(flags) > 1 else ""
    return [_step("reviewer", f"Fix the {len(flags)} pre-push flag{plural}. Code computed "
                  "these, not a model, so they are facts about the change.", tab="prepush",
                  items=[f"{file}: " + ", ".join(f"{k.replace('_', ' ')} ×{n}"
                                                 for k, n in counts.items())
                         for file, counts in by_file.items()])]


def guidance(run: dict, runtime: dict, index_built: str | None = None) -> dict:
    """Why a run ended the way it did, and the next steps, split by who acts.

    `why` holds plain-language facts read from the audit record. `steps` holds
    actions, each with `who` ("reviewer": whoever reads this change; "admin":
    whoever runs the agent and owns its config) and optionally a `command` to
    copy, a `tab` of the run panel to open, and `items` to list.
    """
    view, why = _read_run(run, runtime)
    guide = _STATUS_GUIDES.get(run.get("status"))
    steps = guide(view, why) if guide else []
    return {"why": why,
            "steps": steps + _off_target_steps(view, index_built) + _prepush_steps(view)}


def _first_line(message) -> str:
    """The first line of an error, without its traceback, capped for the page."""
    text = str(message or "")
    return text.split("\n", 1)[0].split("Traceback", 1)[0].strip()[:240]


def runs(trace_rows: list[dict] | None = None, live: list[dict] | None = None,
         retries: _RetryIndex | None = None) -> list[dict]:
    """The RECENT_RUNS newest runs: each stage it went through and how that went,
    plus the audit record (verdict, checks, findings) when it wrote one."""
    trace_rows = _trace_rows() if trace_rows is None else trace_rows
    live = jobs() if live is None else live
    retries = retries or _RetryIndex(trace_rows)
    audits = _recent_audits()
    grouped = _group_spans(trace_rows)
    for rid in audits:                          # audited but trace rotated away
        grouped.setdefault(rid, [])
    return [_build_run(rid, grouped[rid], audits.get(rid) or {}, retries)
            for rid in _shown_run_ids(grouped, audits, live)]


def _shown_run_ids(grouped: dict, audits: dict, live: list[dict]) -> list[str]:
    """The RECENT_RUNS newest run ids that are not in flight.

    Picked BEFORE building any run: building every run in the window to keep 20
    was most of the poll's cost at scale. A run still in flight is shown under
    "Now running" instead -- unless its snapshot stopped updating long ago, which
    is a killed process that must not hide its run forever.
    """
    live_ids = {j.get("run_id") for j in live if j.get("status") in ("running", "waiting")
                and (j.get("age_s") or 0) < STALE_S}

    def last_seen(rid: str) -> str:
        """Sort key: when the run was recorded, else its newest span's end."""
        spans = grouped[rid]
        return (audits.get(rid) or {}).get("recorded_at") or (spans[-1].get("ts") if spans else "")

    return sorted((rid for rid in grouped if rid not in live_ids),
                  key=last_seen, reverse=True)[:RECENT_RUNS]


def _run_status(rid: str, spans: list[dict], audit: dict, failed: bool) -> tuple[str, dict]:
    """A run's status, and the record to show for it (a provisional one if paused)."""
    if audit:
        return audit.get("outcome") or "ended", audit
    if failed:
        return "crashed", audit
    if spans and spans[-1].get("status") == "suspended":
        # Paused at the approval gate. That writes NO audit record (it is not an
        # outcome yet), so without this the page called a run that passed all
        # four checks "In progress", then "Stopped".
        snap = _snapshot_for(rid)
        status = "abandoned" if snap.get("status") == "abandoned" else "awaiting_approval"
        return status, _provisional_record(snap)
    ended = _parse_ts(spans[-1].get("ts")) if spans else None
    age = (datetime.now(UTC) - ended).total_seconds() if ended else 0
    return ("incomplete" if age > STALE_S else "in_progress"), audit


def _build_run(rid: str, spans: list[dict], audit: dict, retries: _RetryIndex) -> dict:
    """One run as the page shows it: its steps, its status and its record."""
    steps = [{"name": s["span"][5:], "status": s.get("status"),
              "duration_ms": s.get("duration_ms"), "ended_at": s.get("ts"),
              "error": s.get("error"), "retries": retries.within(s)} for s in spans]
    err = next((s for s in steps if s["status"] == "error"), None)
    ended = audit.get("recorded_at") or (spans[-1].get("ts") if spans else None)
    first = spans[0] if spans else audit
    status, audit = _run_status(rid, spans, audit, err is not None)
    review = audit.get("review") or {}
    return {
        "run_id": rid, "thread_id": audit.get("thread_id"),
        "tenant": first.get("tenant"), "surface": first.get("surface"),
        "target": first.get("target"),
        "route": audit.get("route") or next(
            (s.get("route") for s in spans if s.get("route")), None),
        "status": status, "decision": audit.get("decision"),
        "outcome": audit.get("outcome"), "ended_at": ended,
        "total_ms": round(sum(float(s.get("duration_ms") or 0) for s in spans), 1),
        "steps": steps, "error": err and {"step": err["name"], "message": err["error"]},
        "audit": audit or None,
        # Kept flat for callers that read the old shape.
        "checks": {k: v for k, v in review.items() if k.endswith("_ok")},
        "failures": review.get("failures") or [],
    }


def _snapshot_for(run_id: str) -> dict:
    """The live snapshot of one run, however old (jobs() skips stale ones)."""
    for p in (_data_dir() / "live").glob("run-*.json"):
        snap = live_files.read_json(p)
        if snap.get("run_id") == run_id:
            return snap
    return {}


def _provisional_record(snap: dict) -> dict:
    """What a run that stopped at approval DID compute, from its live snapshot:
    the four checks, the decision and the retrieved passages -- ids and scores
    only, never model text (app/live.py). Marked provisional: it is not an audit
    record, because the run has no outcome yet."""
    rec: dict = {"provisional": True}
    for st in snap.get("stages") or []:
        d = st.get("detail") or {}
        if "checks" in d:
            rec["review"] = {**d["checks"], "failures": d.get("failures") or []}
            rec["decision"] = d.get("decision")
        if "retrieved" in d:
            rec["retrieved"] = d["retrieved"]
        if "route" in d:
            rec["route"] = d["route"]
    return rec if len(rec) > 1 else {}


def jobs() -> list[dict]:
    """Every live snapshot, newest first, with its age so a dead job is visible."""
    out = []
    now = datetime.now(UTC)
    for p in (_data_dir() / "live").glob("*.json"):
        # A run or job snapshot untouched for STALE_S is finished or killed; the
        # page drops it anyway, so do not parse thousands of them per poll. A
        # job-<id> file is one per job (calibrate, ...), so they pile up like runs.
        # Ingest snapshots (one per tenant) are always kept.
        try:
            if p.name.startswith(("run-", "job-")) and \
                    time.time() - p.stat().st_mtime > STALE_S:
                continue
        except OSError:
            continue
        snap = live_files.read_json(p)
        if not snap:
            continue
        try:
            updated = datetime.fromisoformat(snap.get("updated_at") or snap["started_at"])
            snap["age_s"] = round((now - updated).total_seconds(), 1)
        except (KeyError, ValueError):
            snap["age_s"] = None
        out.append(snap)
    return sorted(out, key=lambda s: s.get("updated_at") or "", reverse=True)


def status() -> dict:
    """Everything the page shows, in one response."""
    trace_rows = _trace_rows()
    live = jobs()                     # once per poll; runs() used to read it again
    retries = _RetryIndex(trace_rows)
    recent = runs(trace_rows, live, retries)
    m = models()
    runtime = _runtime(recent)
    srv = m.pop("_server")
    m["table"] = model_table(srv, runtime)
    m["runtime"] = runtime
    for r in recent:
        r["retry_command"] = retry_command(r, runtime)
    built = {t: _index_built_at(t) for t in {r.get("tenant") for r in recent} if t}
    for r in recent:
        r["guidance"] = guidance(r, runtime, built.get(r.get("tenant")))
        if r.get("error"):
            server = ((r.get("audit") or {}).get("runtime") or {}).get("model_server")
            r["error"]["hint"] = crash_hint(r["error"].get("message") or "",
                                            server or base_url())
    _attach_live_retries(live, retries)
    return {"now": datetime.now(UTC).isoformat(), "data_dir": str(_data_dir().resolve()),
            "models": m, "estimates": estimates(trace_rows), "jobs": live,
            "recent": recent, "stage_plan": STAGE_PLAN}


def _attach_live_retries(live: list[dict], retries: _RetryIndex) -> None:
    """Retries for stages still in flight, so a 3-minute diagnosis that is on its
    second attempt says so while it happens, not only after it finishes."""
    now = datetime.now(UTC)
    for job in live:
        for st in job.get("stages") or []:
            start = _parse_ts(st.get("started_at"))
            if start is None:
                continue
            end = start + timedelta(milliseconds=float(st["duration_ms"])) \
                if st.get("duration_ms") is not None else now
            st["retries"] = retries.between(start.timestamp(), end.timestamp())


def activity_view(query: str) -> dict:
    """`/api/activity`: the activity log, filtered by the page's controls.

    Every parameter is clamped or whitelisted -- the page is read-only and a query
    string is caller input. Unknown values are ignored rather than echoed back.
    """
    params = urllib.parse.parse_qs(query)

    def one(key: str) -> str | None:
        """The first value of a query parameter, capped at 200 chars; None if empty."""
        vals = params.get(key)
        return vals[0][:200] if vals and vals[0] else None

    try:
        hours = min(max(float(one("hours") or 24), 0.25), 24 * 31)
    except ValueError:
        hours = 24.0
    sev = one("severity")
    return activity.query(hours=hours,
                          severity=sev if sev in ("DEBUG", "INFO", "WARNING", "ERROR") else None,
                          operation_type=one("type"), q=one("q"),
                          correlation_id=one("cid"), limit=300)


class Handler(BaseHTTPRequestHandler):
    """`/` is the page; `/api/status` and `/api/activity` its data. No writes."""

    def serve_get(self):
        """Serve the page, the status JSON or the activity JSON; 404 everything else."""
        if self.path in ("/", "/index.html"):
            body, kind = page().encode("utf-8"), "text/html; charset=utf-8"
        elif self.path == "/api/status":
            body, kind = json.dumps(status(), default=str).encode("utf-8"), "application/json"
        elif self.path.split("?", 1)[0] == "/api/activity":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            body = json.dumps(activity_view(query), default=str).encode("utf-8")
            kind = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # http.server routes a request to the attribute named "do_" + its HTTP method,
    # so the stdlib fixes this name; the handler itself keeps a normal one.
    do_GET = serve_get

    def log_request(self, code="-", size="-"):
        """Drop the access line `http.server` prints for every request.

        The page polls once a second, so that line is pure noise. This is the
        stdlib's hook for exactly that line. `log_message` stays as it is, so a
        request that fails (`log_error`) still reaches stderr.
        """


def serve(port: int) -> None:
    """Run until Ctrl-C, bound to localhost only."""
    try:
        server = ThreadingHTTPServer((HOST, port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        # Almost always an older monitor left running in another terminal --
        # which also serves stale code, so a new port alone is the wrong fix.
        raise SystemExit(
            f"monitor: port {port} is already in use, most likely by an older monitor.\n"
            f"  stop it:  kill $(lsof -tiTCP:{port} -sTCP:LISTEN)\n"
            f"  or use another port:  python -m app.monitor --port {port + 1}") from None
    print(f"monitor: http://{HOST}:{port}   (reading {_data_dir().resolve()})  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Live view of the agent's runs and ingests.")
    ap.add_argument("--port", type=int, default=8765)
    serve(ap.parse_args().port)


if __name__ == "__main__":
    main()
