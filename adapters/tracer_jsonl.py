"""Span every node to append-only JSONL (T23).

The argument for this layer is one sentence from docs/DECISIONS.md: **T21 would
have been obvious in a trace showing every score as 1.0.** The grounding gate was
inert for weeks because no run ever recorded the numbers it decided on.

Distinct from the audit sink, deliberately. An audit record is one immutable row
per RUN, kept forever, and answers "what did the agent do". A trace is many rows
per run, kept for weeks, and answers "why was it slow / where did it fail".
Merging them would make the compliance artefact churn at debug volume.

Stdlib only, same rule as the audit sink.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime

from registry import adapter

SCHEMA_VERSION = 1
_LOCK = threading.Lock()


def _interrupt_names() -> tuple[str, ...]:
    """Exception names that mean "suspended", not "failed".

    `interrupt()` suspends by raising, so a span that recorded that as an error
    would mark every single run reaching the approval gate as failed — which is
    the normal, correct path.
    """
    return ("GraphInterrupt", "Interrupt", "GraphBubbleUp")


def prompts_enabled() -> bool:
    """Off by default. T12 (InputGuard) is open, so nothing redacts PII or
    secrets out of evidence — storing every prompt verbatim would create a
    durable copy of exactly what the guard is meant to remove."""
    return os.getenv("CRA_TRACE_PROMPTS", "").lower() in ("1", "true", "yes")


def hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:16]


@adapter("tracer", "jsonl")
class JsonlTracer:
    """var/traces/<YYYY-MM-DD>.jsonl, one row per span."""

    def __init__(self, root: str | None = None):
        self.root = root or os.getenv("CRA_DATA_DIR", "var")

    def path_for(self, when: datetime | None = None) -> str:
        when = when or datetime.now(UTC)
        return os.path.join(self.root, "traces", f"{when:%Y-%m-%d}.jsonl")

    def _write(self, row: dict) -> None:
        path = self.path_for()
        line = json.dumps(row, default=repr, ensure_ascii=False) + "\n"
        with _LOCK:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        """Time a block and record it. NEVER changes the outcome of that block.

        A tracing failure must not break the thing being traced — observability
        that can take down the system it observes is worse than none. The span is
        best-effort; the wrapped exception always propagates unchanged.
        """
        started = time.perf_counter()
        status, error = "ok", None
        try:
            yield self
        except BaseException as e:                      # noqa: BLE001 -- re-raised below
            if type(e).__name__ in _interrupt_names():
                status = "suspended"                    # the approval gate, not a failure
            else:
                status, error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            row = {
                "schema_version": SCHEMA_VERSION,
                "ts": datetime.now(UTC).isoformat(),
                "span": name,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                **{k: v for k, v in attrs.items() if v is not None},
            }
            if error:
                row["error"] = error
            try:
                self._write(row)
            except Exception:                           # noqa: BLE001 -- see docstring
                pass


@adapter("tracer", "null")
class NullTracer:
    """Records nothing. Registered rather than improvised, so CRA_TRACE=null is
    visible in `config.registered()` instead of being a no-op somebody has to
    discover by reading code."""

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        yield self
