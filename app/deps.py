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
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.activity import WARNING, log_event
from app.graph import build_graph
from app.state import checkpoint_serde

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

    # The Postgres saver is imported at module level (imports-at-top, 2026-11-24):
    # a missing driver now fails at import, loudly, instead of on the first run
    # with DATABASE_URL set -- and there is no in-memory fallback to drift into.
    # psycopg-binary is in requirements.txt so libpq is never a system install.
    #
    # The exit stack owns the saver's close: entering the library's context
    # manager through it means the connection is released when THIS generator is
    # closed, whichever way it ends.
    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()))
        await saver.setup()          # idempotent; creates the checkpoint tables
        yield saver


# The process-wide graph, in a one-slot dict rather than a rebound module
# global: `set_graph` and `get_graph` mutate the slot, so neither needs a
# `global` statement, and the one place the graph lives stays this module.
_GRAPH: dict[str, object] = {}


def set_graph(graph) -> None:
    """Install the process-wide graph (None uninstalls it). Called from the app lifespan."""
    if graph is None:
        _GRAPH.pop("graph", None)
    else:
        _GRAPH["graph"] = graph


def get_graph():
    """The process-wide compiled graph.

    Built once, deliberately: every request must share one checkpointer, or a
    `/approve` could land on a worker that has never heard of the thread it is
    resuming. (The multi-worker version of this needs Postgres, not memory —
    which is the same reason `DATABASE_URL` exists.)
    """
    graph = _GRAPH.get("graph")
    if graph is not None:
        return graph
    if os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL is set, so the graph must be built inside "
            "`async with app.deps.open_checkpointer()` — AsyncPostgresSaver has "
            "an async lifecycle that a lazy sync getter cannot own. The FastAPI "
            "lifespan in app/main.py does this; call set_graph() if you are "
            "embedding the graph elsewhere."
        )
    graph = _GRAPH["graph"] = build_graph(checkpointer=_memory_saver())
    return graph
