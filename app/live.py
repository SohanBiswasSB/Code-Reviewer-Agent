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
import os
import pathlib
import time
from datetime import UTC, datetime


def live_dir() -> pathlib.Path:
    """`<CRA_DATA_DIR>/live`, created on demand."""
    return pathlib.Path(os.getenv("CRA_DATA_DIR", "var")) / "live"


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


def _read(path: pathlib.Path) -> dict:
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


def stage(state: dict, name: str, status: str, *, duration_ms: float | None = None,
          detail: dict | None = None, error: str | None = None) -> None:
    """Record one stage transition (`running` | `ok` | `error` | `waiting`)."""
    key = state.get("thread_id") or state.get("_run_id")
    if not key:
        return
    path = live_dir() / f"run-{_safe(str(key))}.json"
    snap = _read(path) or {"kind": "run", "id": key, "started_at": _now(), "stages": []}
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
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms, 1)
    if detail:
        entry["detail"] = detail
    if error:
        entry["error"] = error[:300]
    snap["status"] = status if status in ("error", "waiting") else (
        "finished" if name == "audit" and status == "ok" else "running")
    _write(path, snap)


def finish(state_or_key, status: str) -> None:
    """Mark a run finished from outside the graph (the CLI after a denial, say)."""
    key = state_or_key if isinstance(state_or_key, str) else (
        state_or_key.get("thread_id") or state_or_key.get("_run_id"))
    path = live_dir() / f"run-{_safe(str(key))}.json"
    snap = _read(path)
    if snap:
        snap.update(status=status, updated_at=_now())
        _write(path, snap)


# ── ingest ───────────────────────────────────────────────────────────────────
class IngestProgress:
    """Phase and progress of one `tools.ingest` run, rewritten as it goes."""

    def __init__(self, tenant: str, namespace: str):
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
