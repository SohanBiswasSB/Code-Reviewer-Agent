"""Settings + client factories + the checkpointer. Skeleton."""
from __future__ import annotations
import os


def settings() -> dict:
    return {k: os.getenv(k) for k in
            ("OPENAI_API_KEY", "GITHUB_TOKEN", "PINECONE_INDEX", "DATABASE_URL", "MIN_RELEVANCE_SCORE")}


def make_checkpointer():
    """PROD: AsyncPostgresSaver(DATABASE_URL) — durable paused runs across restarts.
    Dev: MemorySaver(). Never SQLite/file on ephemeral disk."""
    raise NotImplementedError("wire AsyncPostgresSaver; fall back to MemorySaver in dev")
