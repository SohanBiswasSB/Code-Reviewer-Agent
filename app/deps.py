"""Settings, the checkpointer, and the process-wide graph.

The checkpointer is the load-bearing piece. `interrupt()` suspends by persisting
state, so the checkpointer *is* the approval gate's durability. Getting it wrong
does not fail loudly — it fails by losing pending approvals on a restart, which
nobody notices until someone asks why their approval vanished.
"""
from __future__ import annotations

import os
import warnings

# Read straight from the environment. Anything platform- or model-related lives
# in config.py / adapters/platforms instead -- this is only external services.
_SETTING_KEYS = (
    "OPENAI_API_KEY", "GITHUB_TOKEN", "PINECONE_INDEX", "DATABASE_URL",
    "LMSTUDIO_BASE_URL", "MIN_RELEVANCE_SCORE",
)


def settings() -> dict:
    return {k: os.getenv(k) for k in _SETTING_KEYS}


def make_checkpointer():
    """A durable checkpointer when `DATABASE_URL` is set, else in-memory.

    Postgres, specifically NOT SQLite or a file: on a container with ephemeral
    disk a recycle silently discards every paused run. Memory is worse again and
    is offered only for local development — hence the warning rather than a quiet
    fallback, because "it worked locally" is exactly how this reaches production.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "DATABASE_URL is set but langgraph-checkpoint-postgres is not "
                "installed (pip install -r requirements.txt). Refusing to fall "
                "back to in-memory: that would silently make paused approvals "
                "non-durable."
            ) from None
        # Caller owns the lifecycle: `async with AsyncPostgresSaver.from_conn_string(url)`
        return AsyncPostgresSaver.from_conn_string(url)

    from langgraph.checkpoint.memory import MemorySaver
    warnings.warn(
        "No DATABASE_URL: using MemorySaver. Paused approvals will NOT survive a "
        "restart. Development only.",
        RuntimeWarning,
        stacklevel=2,
    )
    return MemorySaver()


_GRAPH = None


def get_graph():
    """The process-wide compiled graph.

    Built once, deliberately: every request must share one checkpointer, or a
    `/approve` could land on a worker that has never heard of the thread it is
    resuming. (The multi-worker version of this needs Postgres, not memory —
    which is the same reason `DATABASE_URL` exists.)
    """
    global _GRAPH
    if _GRAPH is None:
        from app.graph import build_graph
        _GRAPH = build_graph(checkpointer=make_checkpointer())
    return _GRAPH
