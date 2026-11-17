"""Live status: what is running RIGHT NOW, for `python -m app.monitor` to show.

Traces (adapters/tracer_jsonl.py) are written when a span ENDS, which is right for
"why was it slow" and useless for "what is it doing": during a 3-minute diagnosis
call the trace says nothing at all. This module writes the other half -- one
small JSON snapshot per job, overwritten as each stage STARTS and ENDS.

    <CRA_DATA_DIR>/live/run-<id>.json       a graph run: every stage, its status
    <CRA_DATA_DIR>/live/ingest-<tenant>.json an ingest: phase, done/total, rate

## Observability must never break the thing it observes

Every write swallows OSError. A full disk or a permissions problem costs you the
progress view, never the run -- and never the audit record, which is a different
artefact with a different durability promise.

## What goes in a snapshot

Ids, scores, counts, booleans, durations. Never diff text, snippet text or model
output: `var/live/` is read by a web page, and a page is a wider surface than an
audit file. The audit record already holds the full detail for a finished run.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from datadir import data_dir


def live_dir() -> pathlib.Path:
    """`<CRA_DATA_DIR>/live`, created on demand."""
    return data_dir() / "live"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_.").lstrip(".") or "unknown"


def _write(path: pathlib.Path, data: dict) -> None:
    """Atomic replace, so the monitor never reads half a file. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def read_json(path: pathlib.Path) -> dict:
    """A JSON file as a dict; {} when it is missing, half-written or not JSON.
    The monitor reads snapshots through this too -- one reader, one failure rule."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ── graph runs ───────────────────────────────────────────────────────────────
def summarize(stage: str, out) -> dict:
    """A few safe facts from a node's output: ids, scores, counts, booleans only."""
    if not isinstance(out, dict):
        return {}
    detail: dict = {}
    if "route" in out:
        detail["route"] = out["route"]
    raw = out.get("_raw")
    if isinstance(raw, dict) and isinstance(raw.get("files"), list):
        detail["files"] = len(raw["files"])
    if out.get("_retrieved") is not None:
        detail["retrieved"] = [{"id": h.get("id"), "score": h.get("score")}
                               for h in out["_retrieved"]][:8]
    diag = out.get("diagnosis")
    if diag is not None and hasattr(diag, "severity"):
        detail["severity"] = diag.severity
        detail["findings"] = len(getattr(diag, "findings", []) or [])
    review = out.get("review")
    if review is not None and hasattr(review, "decision"):
        detail.update(decision=review.decision, failures=list(review.failures),
                      checks={k: getattr(review, k) for k in (
                          "groundedness_ok", "entailment_ok", "risk_ok", "permission_ok")})
    if out.get("_checks") is not None:
        checks = out["_checks"]
        detail["checks_summary"] = {s: sum(1 for c in checks if c.get("status") == s)
                                    for s in ("flag", "pass", "skip")}
        detail["flagged"] = [{"check": c["check"], "file": c.get("file"), "line": c.get("line")}
                             for c in checks if c.get("status") == "flag"][:20]
    if "approved" in out:
        detail["approved"] = out["approved"]
    return detail


@dataclass(frozen=True)
class StageOutcome:
    """What a stage transition reports beyond its status: how long the stage
    took, the safe facts `summarize` drew from its output, and the error text
    when it failed. All optional -- a `running` transition has none of them."""

    duration_ms: float | None = None
    detail: dict | None = None
    error: str | None = None


def stage(state: dict, name: str, status: str, **outcome) -> None:
    """Record one stage transition (`running` | `ok` | `error` | `waiting`).

    `outcome` is `StageOutcome`'s fields by keyword (`duration_ms`, `detail`,
    `error`); an unknown keyword is a TypeError, as it was when they were named.
    """
    result = StageOutcome(**outcome)
    key = state.get("thread_id") or state.get("_run_id")
    if not key:
        return
    path = live_dir() / f"run-{_safe(str(key))}.json"
    snap = read_json(path) or {"kind": "run", "id": key, "started_at": _now(), "stages": []}
    snap.update(tenant=state.get("tenant"), target=state.get("target"),
                surface=state.get("surface") or state.get("_surface_name"),
                run_id=state.get("_run_id") or snap.get("run_id"), updated_at=_now())
    stages = snap["stages"]
    entry = next((s for s in reversed(stages) if s["name"] == name
                  and s["status"] == "running"), None)
    if entry is None:
        entry = {"name": name, "started_at": _now()}
        stages.append(entry)
    entry["status"] = status
    if result.duration_ms is not None:
        entry["duration_ms"] = round(result.duration_ms, 1)
    if result.detail:
        entry["detail"] = result.detail
    if result.error:
        entry["error"] = result.error[:300]
    snap["status"] = status if status in ("error", "waiting") else (
        "finished" if name == "audit" and status == "ok" else "running")
    _write(path, snap)


def finish(state_or_key, status: str) -> None:
    """Mark a run finished from outside the graph (the CLI after a denial, say)."""
    key = state_or_key if isinstance(state_or_key, str) else (
        state_or_key.get("thread_id") or state_or_key.get("_run_id"))
    path = live_dir() / f"run-{_safe(str(key))}.json"
    snap = read_json(path)
    if snap:
        snap.update(status=status, updated_at=_now())
        _write(path, snap)


# ── long jobs that are not graph runs (calibration, ...) ─────────────────────
class JobProgress:
    """Phase and progress of a long non-graph job, rewritten as it goes.

    Added 2026-11-17 because a 15-minute `tools.calibrate` showed nothing on the
    monitor: "Now running: 0" while the gateway was busy, which reads as idle.
    Same shape as `IngestProgress` (status, phase, done/total, ETA), plus a `kind`
    and a human `title`, so the page can draw any job with one card. Written to
    `job-<id>.json`, never `run-*` -- the page draws run files as graph runs.
    """

    def __init__(self, kind: str, job_id: str, title: str, **meta):
        """Write the `starting` snapshot at once, so the job shows immediately."""
        self.path = live_dir() / f"job-{_safe(job_id)}.json"
        self.started = time.time()
        self.data = {"kind": kind, "id": job_id, "title": title, "started_at": _now(),
                     "updated_at": _now(), "status": "running", "phase": "starting",
                     "done": 0, "total": 0, **meta}
        _write(self.path, self.data)

    def phase(self, name: str, **extra) -> None:
        """Enter a named phase."""
        self.data.update(phase=name, updated_at=_now(), **extra)
        _write(self.path, self.data)

    def progress(self, done: int, total: int) -> None:
        """Items done of total, with a rate and ETA measured since the job began."""
        elapsed = max(time.time() - self.started, 1e-6)
        rate = done / elapsed
        self.data.update(done=done, total=total, rate=round(rate, 2), updated_at=_now(),
                         eta_s=round((total - done) / rate) if rate else None)
        _write(self.path, self.data)

    def end(self, status: str, **extra) -> None:
        """`finished`, `error` or `interrupted`, with the total wall time."""
        self.data.update(status=status, phase=status, updated_at=_now(),
                         elapsed_s=round(time.time() - self.started, 1), **extra)
        _write(self.path, self.data)


# ── ingest ───────────────────────────────────────────────────────────────────
class IngestProgress:
    """Phase and progress of one `tools.ingest` run, rewritten as it goes."""

    def __init__(self, tenant: str, namespace: str):
        """Write the `starting` snapshot at once, so the monitor shows the job immediately.

        One file per TENANT, not per run: a new ingest overwrites the last one's.
        """
        self.path = live_dir() / f"ingest-{_safe(tenant)}.json"
        self.started = time.time()
        self.data = {"kind": "ingest", "id": f"ingest-{tenant}", "tenant": tenant,
                     "namespace": namespace, "started_at": _now(), "status": "running",
                     "phase": "starting", "done": 0, "total": 0}
        _write(self.path, self.data)

    def phase(self, name: str, **extra) -> None:
        """Enter a phase (scanning, chunking, redacting, embedding, writing)."""
        self.data.update(phase=name, updated_at=_now(), **extra)
        _write(self.path, self.data)

    def progress(self, done: int, total: int) -> None:
        """Embedding progress, with a rate and ETA measured since embedding began."""
        if "embed_started" not in self.data:
            self.data["embed_started"] = time.time()
        elapsed = max(time.time() - self.data["embed_started"], 1e-6)
        rate = done / elapsed
        self.data.update(done=done, total=total, rate=round(rate, 1), updated_at=_now(),
                         eta_s=round((total - done) / rate) if rate else None)
        _write(self.path, self.data)

    def end(self, status: str, **extra) -> None:
        """`finished` or `error`, with the total wall time."""
        self.data.update(status=status, phase=status, updated_at=_now(),
                         elapsed_s=round(time.time() - self.started, 1), **extra)
        _write(self.path, self.data)
