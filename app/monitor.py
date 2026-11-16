"""A live view of the agent: what is running, which models serve it, what each run did.

    python -m app.monitor                 # then open http://127.0.0.1:8765
    python -m app.monitor --port 9000

Reads only what the agent already writes under CRA_DATA_DIR -- `live/` (stage
starts and ends, ingest progress), `traces/` (every stage of every run, errors
included) and `audit/` (the verdict, the four checks, the findings) -- plus LM
Studio's model list. It starts nothing, changes nothing, and can be opened and
closed at any time.

## A run is traces JOINED with audit, not audit alone

The first version listed audit records only, and a run that crashed mid-graph
writes none (G9: there are no error edges, so `audit` is never reached). The one
run that most needed looking at -- LM Studio dying inside `independent_review` --
was therefore the one run the page could not show. History is now built from the
trace spans, keyed by `run_id`, with the audit record attached where one exists.

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

## No third-party dependency of its own

`http.server` and `urllib` for the serving; `config` only to name the models a
run would use when no audit record says so.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import pathlib
import statistics
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from adapters.llm_lmstudio import base_url

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
ORPHAN_GAP_S = 5.0         # a span with no run_id belongs to the run that follows it
STALE_S = 900              # an unaudited, error-free run older than this was killed

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
    "reranker": "Re-orders retrieved passages by true relevance. Planned (T13) -- "
                "not called yet.",
    "entailer": "Scores whether the review's claims follow from the passages it cites. "
                "Drives the entailment check.",
}
LM_STUDIO_ROLES = ("diagnosis", "judge", "utility", "embedding")


def _data_dir() -> pathlib.Path:
    return pathlib.Path(os.getenv("CRA_DATA_DIR", "var"))


def _read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _tail_jsonl(paths: list[pathlib.Path], limit: int) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(paths):
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows[-limit:]


def _parse_ts(ts) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


# ── models ───────────────────────────────────────────────────────────────────
def server_models() -> dict:
    """Everything LM Studio has on disk, and which of it is LOADED, via its REST API."""
    root = base_url().rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data", [])
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"reachable": False, "url": root, "error": str(e)[:200], "models": []}
    return {"reachable": True, "url": root, "models": [{
        "id": m.get("id"), "type": m.get("type"), "state": m.get("state"),
        "quantization": m.get("quantization"), "format": m.get("compatibility_type"),
        "context": m.get("loaded_context_length") or m.get("max_context_length"),
    } for m in data]}


def _same_model(a: str, b: str) -> bool:
    """`qwen3-4b` (a profile id) and `qwen/qwen3-4b` (LM Studio's id) are one model."""
    return bool(a and b) and (a == b or a.split("/")[-1] == b.split("/")[-1])


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
                "models": config.get_models(), "model_server": base_url(),
                "source": "shell"}
    except Exception as e:                    # noqa: BLE001 -- a view must not crash on config
        return {"source": "shell", "error": f"{type(e).__name__}: {e}"[:200], "models": {}}


def model_table(server: dict, runtime: dict) -> list[dict]:
    """One row per role: which model, where it runs, what it does, is it ready."""
    assigned = runtime.get("models") or {}
    on_server = server.get("models") or []
    rows = []
    for role in ("diagnosis", "judge", "embedding", "utility"):
        want = assigned.get(role) or ""
        hit = next((m for m in on_server if _same_model(m["id"], want)), None)
        if not server.get("reachable"):
            state, note = "unknown", "model server not reachable"
        elif hit is None:
            state, note = "missing", f"not on this server -- lms get {want}"
        elif hit["state"] == "loaded":
            state = "ready"
            note = ", ".join(str(x) for x in (
                hit.get("format"), hit.get("quantization"),
                hit.get("context") and f"ctx {hit['context']:,}") if x)
        elif role == "utility":
            # Never called by a code review, so "not loaded" is the right state
            # for it, not a warning to act on.
            state, note = "unused", f"not needed for a code review -- lms load {hit['id']}"
        else:
            state, note = "not_loaded", f"downloaded, not loaded -- lms load {hit['id']}"
        rows.append({"role": role, "does": ROLES[role], "model": want or "—",
                     "where": f"LM Studio · {server.get('url')}", "state": state,
                     "note": note})
    rows.append({"role": "reranker", "does": ROLES["reranker"],
                 "model": assigned.get("reranker") or "—", "where": "in-process (this machine)",
                 "state": "unused", "note": "T13 -- retrieval uses dense + BM25 fusion only"})
    ent = runtime.get("entailer") or "lexical"
    rows.append({"role": "entailer", "does": ROLES["entailer"],
                 "model": assigned.get("entailer") or (
                     "none -- word overlap" if ent == "lexical" else ent),
                 "where": "in-process (this machine)",
                 "state": "ready",
                 "note": ("ENTAILER=lexical: cheap but strict on paraphrase; "
                          "ENTAILER=cross_encoder is stronger") if ent == "lexical" else ""})
    used = [m for m in assigned.values() if m]
    for m in on_server:
        if m["state"] == "loaded" and not any(_same_model(m["id"], u) for u in used):
            rows.append({"role": "(none)", "does": "Loaded in LM Studio but not used by "
                         "this profile -- it only holds memory.", "model": m["id"],
                         "where": f"LM Studio · {server.get('url')}", "state": "extra",
                         "note": f"lms unload {m['id']}"})
    return rows


def models() -> dict:
    """Server reachability plus the per-role table, in one block for the page."""
    srv = server_models()
    loaded = [m for m in srv["models"] if m["state"] == "loaded"]
    return {"reachable": srv["reachable"], "url": srv["url"], "error": srv.get("error"),
            "loaded": loaded, "available": len(srv["models"]), "_server": srv}


# ── history ──────────────────────────────────────────────────────────────────
def estimates(rows: list[dict] | None = None) -> dict[str, float]:
    """Median ms per stage over recent SUCCESSFUL spans -- the 'what is left' basis."""
    if rows is None:
        rows = _tail_jsonl(list((_data_dir() / "traces").glob("*.jsonl")), TRACE_TAIL)
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


def _retries_within(span: dict, retry_rows: list[dict]) -> list[dict]:
    """The model-call retries that happened INSIDE this node span.

    `llm.retry` rows come from the adapter, which knows its role but not the run,
    so they are placed by time: a retry belongs to the node whose span encloses
    it. Runs are sequential on one machine, so the enclosing node is unambiguous.
    """
    end = _parse_ts(span.get("ts"))
    if end is None or not retry_rows:
        return []
    start = end.timestamp() - float(span.get("duration_ms") or 0) / 1000
    out = []
    for r in retry_rows:
        t = _parse_ts(r.get("ts"))
        if t is not None and start <= t.timestamp() <= end.timestamp():
            out.append({k: r.get(k) for k in (
                "attempt", "of", "reason", "wait_s", "role", "model", "error", "ts")})
    return out


def retry_command(run: dict, runtime: dict) -> str:
    """The exact command that re-runs this review, for the page's Retry button.

    The monitor stays read-only: it hands you the command rather than launching
    it. A page with no authentication that can start processes is a page any
    other local site can start processes through.
    """
    rt = (run.get("audit") or {}).get("runtime") or runtime or {}
    env = [f"CRA_PROFILE={rt['profile']}"] if rt.get("profile") else []
    if rt.get("entailer") and rt["entailer"] != "lexical":
        env.append(f"ENTAILER={rt['entailer']}")
    target = run.get("target") or "staged"
    args = f"--tenant {run.get('tenant') or '<tenant>'}"
    if target != "staged":
        args += f" --target {target}"
    return " ".join([*env, "python -m app.review", args])


def runs(trace_rows: list[dict] | None = None) -> list[dict]:
    """Every recent run, newest first: each stage it went through and how that
    went, plus the audit record (verdict, checks, findings) when it wrote one."""
    if trace_rows is None:
        trace_rows = _tail_jsonl(list((_data_dir() / "traces").glob("*.jsonl")), TRACE_TAIL)
    audits = {a.get("run_id"): a for a in _tail_jsonl(
        list((_data_dir() / "audit").glob("*/*.jsonl")), RECENT_RUNS * 4) if a.get("run_id")}
    # In flight -> shown under "Now running" instead. A snapshot that stopped
    # updating long ago is a killed process, and must not hide its run forever.
    live_ids = {j.get("run_id") for j in jobs() if j.get("status") in ("running", "waiting")
                and (j.get("age_s") or 0) < STALE_S}
    now = datetime.now(UTC)
    grouped = _group_spans(trace_rows)
    retry_rows = [r for r in trace_rows if r.get("span") == "llm.retry"]
    for rid in audits:                          # audited but trace rotated away
        grouped.setdefault(rid, [])
    out = []
    for rid, spans in grouped.items():
        if rid in live_ids:
            continue                            # shown under "Now running"
        audit = audits.get(rid) or {}
        first = spans[0] if spans else audit
        steps = [{"name": s["span"][5:], "status": s.get("status"),
                  "duration_ms": s.get("duration_ms"), "ended_at": s.get("ts"),
                  "error": s.get("error"), "retries": _retries_within(s, retry_rows)}
                 for s in spans]
        err = next((s for s in steps if s["status"] == "error"), None)
        ended = audit.get("recorded_at") or (spans[-1].get("ts") if spans else None)
        if audit:
            run_status = audit.get("outcome") or "ended"
        elif err:
            run_status = "crashed"
        else:
            age = (now - _parse_ts(ended)).total_seconds() if _parse_ts(ended) else 0
            run_status = "incomplete" if age > STALE_S else "in_progress"
        out.append({
            "run_id": rid, "thread_id": audit.get("thread_id"),
            "tenant": first.get("tenant"), "surface": first.get("surface"),
            "target": first.get("target"),
            "route": audit.get("route") or next(
                (s.get("route") for s in spans if s.get("route")), None),
            "status": run_status, "decision": audit.get("decision"),
            "outcome": audit.get("outcome"), "ended_at": ended,
            "total_ms": round(sum(float(s.get("duration_ms") or 0) for s in spans), 1),
            "steps": steps, "error": err and {"step": err["name"], "message": err["error"]},
            "audit": audit or None,
            # Kept flat for callers that read the old shape.
            "checks": {k: v for k, v in (audit.get("review") or {}).items()
                       if k.endswith("_ok")},
            "failures": (audit.get("review") or {}).get("failures") or [],
        })
    out.sort(key=lambda r: r.get("ended_at") or "", reverse=True)
    return out[:RECENT_RUNS]


def jobs() -> list[dict]:
    """Every live snapshot, newest first, with its age so a dead job is visible."""
    out = []
    now = datetime.now(UTC)
    for p in (_data_dir() / "live").glob("*.json"):
        snap = _read_json(p)
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
    trace_rows = _tail_jsonl(list((_data_dir() / "traces").glob("*.jsonl")), TRACE_TAIL)
    recent = runs(trace_rows)
    m = models()
    runtime = _runtime(recent)
    srv = m.pop("_server")
    m["table"] = model_table(srv, runtime)
    m["runtime"] = runtime
    for r in recent:
        r["retry_command"] = retry_command(r, runtime)
    live = jobs()
    _attach_live_retries(live, [r for r in trace_rows if r.get("span") == "llm.retry"])
    return {"now": datetime.now(UTC).isoformat(), "data_dir": str(_data_dir().resolve()),
            "models": m, "estimates": estimates(trace_rows), "jobs": live,
            "recent": recent}


def _attach_live_retries(live: list[dict], retry_rows: list[dict]) -> None:
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
            st["retries"] = [
                {k: r.get(k) for k in ("attempt", "of", "reason", "wait_s", "role", "ts")}
                for r in retry_rows if (t := _parse_ts(r.get("ts"))) and start <= t <= end]


class Handler(BaseHTTPRequestHandler):
    """`/` is the page, `/api/status` its data. Nothing else, and no writes."""

    def do_GET(self):  # noqa: N802 -- the http.server naming contract
        """Serve the page or the status JSON; 404 everything else."""
        if self.path in ("/", "/index.html"):
            body, kind = page().encode("utf-8"), "text/html; charset=utf-8"
        elif self.path == "/api/status":
            body, kind = json.dumps(status(), default=str).encode("utf-8"), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 -- a poll a second is noise
        """Silence per-request logging."""


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
