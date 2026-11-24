"""A live view of the agent: what is running, which models serve it, what each run did.

    python -m app.monitor                 # then open http://127.0.0.1:8765
    python -m app.monitor --port 9000

Reads only what the agent already writes under CRA_DATA_DIR -- `live/` (stage
starts and ends, ingest progress), `traces/` (every stage of every run, errors
included), `audit/` (the verdict, the four checks, the findings) and `activity/`
(the activity log, through `app.activity.query`) -- plus LM Studio's model list.
It starts nothing, changes nothing, and can be opened and closed at any time.

This module is the entry point, the HTTP handler and `status()`, the one
response the page polls. The work behind it lives beside it (split 2026-11-24,
when this file passed pylint's 1000-line module limit):

- `app/monitor_runs.py` -- the run history: traces JOINED with audit, and why.
- `app/monitor_guidance.py` -- what the page says under a run: the gate's failure
  strings in plain words, the next steps, the retry command, the crash hint.
- `app/monitor_models.py` -- the model table: which model serves each role, from
  what the RUN used rather than what this shell would pick.

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
import errno
import json
import pathlib
import urllib.parse
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adapters.llm_lmstudio import base_url
from app import activity
from app.graph import STAGE_PLAN
from app.monitor_guidance import crash_hint, guidance, index_built_at, retry_command
from app.monitor_models import model_table, models, resolve_runtime
from app.monitor_runs import (
    RetryIndex,
    attach_live_retries,
    estimates,
    jobs,
    read_trace_rows,
    runs,
)
from datadir import data_dir

HOST = "127.0.0.1"
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


def status() -> dict:
    """Everything the page shows, in one response."""
    trace_rows = read_trace_rows()
    live = jobs()                     # once per poll; runs() used to read it again
    retries = RetryIndex(trace_rows)
    recent = runs(trace_rows, live, retries)
    m = models()
    runtime = resolve_runtime(recent)
    srv = m.pop("_server")
    m["table"] = model_table(srv, runtime)
    m["runtime"] = runtime
    for r in recent:
        r["retry_command"] = retry_command(r, runtime)
    built = {t: index_built_at(t) for t in {r.get("tenant") for r in recent} if t}
    for r in recent:
        r["guidance"] = guidance(r, runtime, built.get(r.get("tenant")))
        if r.get("error"):
            server = ((r.get("audit") or {}).get("runtime") or {}).get("model_server")
            r["error"]["hint"] = crash_hint(r["error"].get("message") or "",
                                            server or base_url())
    attach_live_retries(live, retries)
    return {"now": datetime.now(UTC).isoformat(), "data_dir": str(data_dir().resolve()),
            "models": m, "estimates": estimates(trace_rows), "jobs": live,
            "recent": recent, "stage_plan": STAGE_PLAN}


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
    print(f"monitor: http://{HOST}:{port}   (reading {data_dir().resolve()})  Ctrl-C to stop")
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
