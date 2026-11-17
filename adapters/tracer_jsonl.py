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
import sys
import threading
import time
from datetime import UTC, datetime

from datadir import data_dir
from errors import WRITE_ERRORS
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


def span_status(exc: BaseException | None) -> tuple[str, str | None]:
    """(status, error) for a span that ended with `exc` in flight, or cleanly.

    `ok` when nothing was raised; `suspended` for LangGraph's interrupt
    exceptions (the approval gate, not a failure); otherwise `error`, with the
    exception named. Any BaseException counts -- a Ctrl-C mid-node is still a
    span, and still an error.
    """
    if exc is None:
        return "ok", None
    if type(exc).__name__ in _interrupt_names():
        return "suspended", None
    return "error", f"{type(exc).__name__}: {exc}"


def prompts_enabled() -> bool:
    """Off by default. T12 now redacts secrets and PII before a prompt is built,
    but the guard is pattern-based, and a durable verbatim copy of every prompt
    is still a larger surface than a hash. Hash always; text only while debugging."""
    return os.getenv("CRA_TRACE_PROMPTS", "").lower() in ("1", "true", "yes")


def hash_text(text: str) -> str:
    """`sha256:` plus 16 hex chars: enough to tell two prompts apart, not to read one."""
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:16]


@adapter("tracer", "jsonl")
class JsonlTracer:
    """var/traces/<YYYY-MM-DD>.jsonl, one row per span."""

    def __init__(self, root: str | None = None):
        """`root` overrides `CRA_DATA_DIR` (tests point it at a temp dir)."""
        self.root = root or str(data_dir())

    def path_for(self, when: datetime | None = None) -> str:
        """The day file a span lands in; `when` defaults to now (UTC)."""
        when = when or datetime.now(UTC)
        return os.path.join(self.root, "traces", f"{when:%Y-%m-%d}.jsonl")

    def _write(self, row: dict) -> None:
        """Append one row under `_LOCK`; unserialisable values fall back to `repr`."""
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
        best-effort; the wrapped exception always propagates unchanged. "Failure"
        means the `WRITE_ERRORS` family (a full disk, an unserialisable value),
        which costs the span silently; any other error from the write is a bug and
        propagates, per the errors rule (errors.py).

        Status is `ok`, `error` (with the exception named), or `suspended` for
        LangGraph's interrupt exceptions -- the approval gate, not a failure.
        """
        started = time.perf_counter()
        finished = False
        try:
            yield self
            finished = True
        finally:
            # Nothing is caught: the block's exception, if any, keeps propagating
            # unchanged once this `finally` ends. `sys.exception()` names it
            # without catching it -- every BaseException, so a Ctrl-C mid-node is
            # still a span. Read only when the block did NOT finish: on a clean
            # exit it would report an exception the CALLER is handling around
            # this span, and mark a successful block as failed.
            status, error = span_status(None if finished else sys.exception())
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
            except WRITE_ERRORS:
                # A full disk or an unserialisable attribute costs the span,
                # never the run. Any other error is a bug and propagates.
                pass


@adapter("tracer", "null")
class NullTracer:
    """Records nothing. Registered rather than improvised, so CRA_TRACE=null is
    visible in `config.registered()` instead of being a no-op somebody has to
    discover by reading code."""

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        """Run the block; record nothing."""
        yield self
