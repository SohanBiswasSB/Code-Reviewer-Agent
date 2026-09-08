"""Settings + client factories + the checkpointer. Skeleton."""
from __future__ import annotations

import os

# Read straight from the environment. Anything platform- or model-related lives
# in config.py / adapters/platforms instead -- this is only external services.
_SETTING_KEYS = (
    "OPENAI_API_KEY", "GITHUB_TOKEN", "PINECONE_INDEX", "DATABASE_URL",
    "LMSTUDIO_BASE_URL", "MIN_RELEVANCE_SCORE",
)


def settings() -> dict:
    return {k: os.getenv(k) for k in _SETTING_KEYS}


def make_checkpointer():
    """PROD: AsyncPostgresSaver(DATABASE_URL) — durable paused runs across
    restarts. Dev: MemorySaver(). Never SQLite/file on ephemeral disk."""
    raise NotImplementedError("wire AsyncPostgresSaver; fall back to MemorySaver in dev")
