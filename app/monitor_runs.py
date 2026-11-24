"""The monitor's run history: what each recent run did, read from var/.

Split out of `app/monitor.py` (2026-11-24, which had grown past pylint's 1000-line
module limit). Reads `live/` (stage starts and ends, ingest progress), `traces/`
(every stage of every run, errors included) and `audit/` (the verdict, the four
checks, the findings) under CRA_DATA_DIR. It writes nothing.

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
"""
from __future__ import annotations

import bisect
import json
import pathlib
import statistics
import time
from datetime import UTC, datetime, timedelta

from app import live as live_files  # aliased: runs()/snapshot() take a `live` job list
from datadir import data_dir

RECENT_RUNS = 20
TRACE_TAIL = 4000          # trace rows read for history and duration estimates
AUDIT_TAIL = RECENT_RUNS * 4   # audit records read PER TENANT
ORPHAN_GAP_S = 5.0         # a span with no run_id belongs to the run that follows it
STALE_S = 900              # an unaudited, error-free run older than this was killed


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


def newest_rows(paths: list[pathlib.Path], limit: int) -> list[dict]:
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


def read_trace_rows() -> list[dict]:
    """The newest TRACE_TAIL trace rows, oldest first: one read shared by a whole poll."""
    return newest_rows(list((data_dir() / "traces").glob("*.jsonl")), TRACE_TAIL)


def recent_audits() -> dict[str, dict]:
    """Recent audit records across ALL tenants, by run_id.

    Read per tenant, then merged. The first version tailed every tenant's files
    in path order and kept the last N rows overall, so the alphabetically last
    tenant's records pushed every other tenant's out of the window -- measured:
    with two tenants, ten finished runs of the first were shown as "Stopped".
    """
    root = data_dir() / "audit"
    rows: list[dict] = []
    for d in (root.iterdir() if root.is_dir() else ()):
        if d.is_dir():
            rows += newest_rows(list(d.glob("*.jsonl")), AUDIT_TAIL)
    return {r["run_id"]: r for r in rows if r.get("run_id")}


def _parse_ts(ts) -> datetime | None:
    """An ISO timestamp as a datetime; None for a missing or malformed one."""
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


# ── history ──────────────────────────────────────────────────────────────────
def estimates(rows: list[dict] | None = None) -> dict[str, float]:
    """Median ms per stage over recent SUCCESSFUL spans -- the 'what is left' basis."""
    rows = read_trace_rows() if rows is None else rows
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


class RetryIndex:
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


def runs(trace_rows: list[dict] | None = None, live: list[dict] | None = None,
         retries: RetryIndex | None = None) -> list[dict]:
    """The RECENT_RUNS newest runs: each stage it went through and how that went,
    plus the audit record (verdict, checks, findings) when it wrote one."""
    trace_rows = read_trace_rows() if trace_rows is None else trace_rows
    live = jobs() if live is None else live
    retries = retries or RetryIndex(trace_rows)
    audits = recent_audits()
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
        paused = "abandoned" if snap.get("status") == "abandoned" else "awaiting_approval"
        return paused, _provisional_record(snap)
    ended = _parse_ts(spans[-1].get("ts")) if spans else None
    age = (datetime.now(UTC) - ended).total_seconds() if ended else 0
    return ("incomplete" if age > STALE_S else "in_progress"), audit


def _build_run(rid: str, spans: list[dict], audit: dict, retries: RetryIndex) -> dict:
    """One run as the page shows it: its steps, its status and its record."""
    steps = [{"name": s["span"][5:], "status": s.get("status"),
              "duration_ms": s.get("duration_ms"), "ended_at": s.get("ts"),
              "error": s.get("error"), "retries": retries.within(s)} for s in spans]
    err = next((s for s in steps if s["status"] == "error"), None)
    ended = audit.get("recorded_at") or (spans[-1].get("ts") if spans else None)
    first = spans[0] if spans else audit
    run_status, audit = _run_status(rid, spans, audit, err is not None)
    review = audit.get("review") or {}
    return {
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
        "checks": {k: v for k, v in review.items() if k.endswith("_ok")},
        "failures": review.get("failures") or [],
    }


def _snapshot_for(run_id: str) -> dict:
    """The live snapshot of one run, however old (jobs() skips stale ones)."""
    for p in (data_dir() / "live").glob("run-*.json"):
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
    for p in (data_dir() / "live").glob("*.json"):
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


def attach_live_retries(live: list[dict], retries: RetryIndex) -> None:
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
