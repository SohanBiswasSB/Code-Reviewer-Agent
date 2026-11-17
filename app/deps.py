"""Settings, the checkpointer, and the process-wide graph.

The checkpointer is the load-bearing piece. `interrupt()` suspends by persisting
state, so the checkpointer *is* the approval gate's durability. Getting it wrong
does not fail loudly — it fails by losing pending approvals on a restart, which
nobody notices until someone asks why their approval vanished.

## The Postgres path needs an async lifecycle (fixed 2026-09-10)

`AsyncPostgresSaver.from_conn_string(url)` returns an async CONTEXT MANAGER, not
a saver. `get_graph()` used to hand that object straight to `build_graph`, so
with `DATABASE_URL` set the graph was compiled with a context manager as its
checkpointer — which fails at first use, far from the cause, and only in the
configuration that matters. Memory-only development never touched it.

`open_checkpointer()` now owns the lifecycle and yields a real saver for both
backends. `app/main.py` enters it once in the FastAPI lifespan. `get_graph()`
serves the in-memory development case and REFUSES the Postgres one rather than
rebuilding the same bug.
"""
from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager

from app.activity import WARNING, log_event

_MEMORY_WARNING = (
    "No DATABASE_URL: using MemorySaver. Paused approvals will NOT survive a "
    "restart, and nothing is ever evicted. Development only."
)


def _memory_saver():
    """The development checkpointer: MemorySaver with the T15 serde, announced loudly.

    Every in-memory saver comes through here, so the "approvals are not durable"
    warning cannot be skipped by one of its two callers (`open_checkpointer`
    without DATABASE_URL, and `get_graph`).
    """
    from langgraph.checkpoint.memory import MemorySaver

    from app.state import checkpoint_serde
    # Kept as a Python warning (a developer running this sees it) AND recorded:
    # "approvals are not durable" is exactly what an operator must find later.
    warnings.warn(_MEMORY_WARNING, RuntimeWarning, stacklevel=3)
    log_event("config", "checkpointer.memory", severity=WARNING, message=_MEMORY_WARNING,
              failure_source="config")
    return MemorySaver(serde=checkpoint_serde())


@asynccontextmanager
async def open_checkpointer():
    """Yield a REAL saver for whichever backend is configured, and own its close.

    Postgres, specifically NOT SQLite or a file: on a container with ephemeral
    disk a recycle silently discards every paused run. Memory is worse again and
    is offered only for local development — hence the warning rather than a quiet
    fallback, because "it worked locally" is exactly how this reaches production.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        yield _memory_saver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "DATABASE_URL is set but langgraph-checkpoint-postgres is not "
            "installed (pip install -r requirements.txt). Refusing to fall "
            "back to in-memory: that would silently make paused approvals "
            "non-durable."
        ) from None

    from app.state import checkpoint_serde

    async with AsyncPostgresSaver.from_conn_string(
        url, serde=checkpoint_serde()
    ) as saver:
        await saver.setup()          # idempotent; creates the checkpoint tables
        yield saver


_GRAPH = None


def set_graph(graph) -> None:
    """Install the process-wide graph. Called once from the app lifespan."""
    global _GRAPH
    _GRAPH = graph


def get_graph():
    """The process-wide compiled graph.

    Built once, deliberately: every request must share one checkpointer, or a
    `/approve` could land on a worker that has never heard of the thread it is
    resuming. (The multi-worker version of this needs Postgres, not memory —
    which is the same reason `DATABASE_URL` exists.)
    """
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    if os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL is set, so the graph must be built inside "
            "`async with app.deps.open_checkpointer()` — AsyncPostgresSaver has "
            "an async lifecycle that a lazy sync getter cannot own. The FastAPI "
            "lifespan in app/main.py does this; call set_graph() if you are "
            "embedding the graph elsewhere."
        )
    from app.graph import build_graph
    _GRAPH = build_graph(checkpointer=_memory_saver())
    return _GRAPH
