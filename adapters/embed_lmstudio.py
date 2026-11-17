"""The embedder, finally called by something.

`POST /v1/embeddings` against the same LM Studio server the LLM adapter uses, so
one running server covers every model role. Written because `EMBEDDING_MODEL` was
pinned, doctor-verified and dimensionally asserted while retrieval's "dense" half
was Jaccard word overlap -- the pinned model had no caller anywhere.

## Why `kind`

Modern embedders are asymmetric: they expect a query and a passage to be encoded
differently, and ignoring that costs real recall. EmbeddingGemma uses explicit
prefixes, so `kind="query"` and `kind="passage"` are not cosmetic -- embedding a
query as a passage silently degrades every score the gate then thresholds.

## Determinism

Embeddings are deterministic for a given model, so an index built once stays
comparable. That is also why `EMBEDDING_MODEL` is pinned outside the profiles:
a different embedder is a different vector space, and every
`min_relevance_score` would change meaning without a single line of code moving.
"""
from __future__ import annotations

import os

from openai import OpenAI

from adapters.llm_lmstudio import api_key, with_retry
from adapters.platforms import base
from app.activity import ActivityLogger
from registry import adapter

# EmbeddingGemma's documented task prefixes. A model without them is unharmed --
# the text is simply embedded as written.
_PREFIX = {
    "query": "task: search result | query: ",
    "passage": "title: none | text: ",
}


class _NoSpan:
    """Stand-in context for calls that roll up into a parent record."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@adapter("embedder", "lmstudio")
class LMStudioEmbedder:
    """Batch text -> vectors via the OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str | None = None, dims: int | None = None):
        """Defaults to the pinned `EMBEDDING_MODEL` / `EMBEDDING_DIMS` (platforms/base.py).

        `config.get_embedder()` passes neither, and should not: the pin is what keeps
        an index and its `min_relevance_score` meaningful. The client is created on
        first use, against `LMSTUDIO_BASE_URL` with `LMSTUDIO_API_KEY`. Both are read
        here, so `config.get_retriever` keys its shared instance on both.
        """
        self.model = model or base.EMBEDDING_MODEL
        self.dims = dims or base.EMBEDDING_DIMS
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self._api_key = api_key()
        self._client = None

    def _get_client(self):
        if self._client is None:
            # Same policy as the LLM adapter: one visible retry loop, not two.
            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key,
                                  max_retries=0)
        return self._client

    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        """Embed a batch. Raises on a dimensionality mismatch rather than returning junk.

        The `Embedder` port: one vector per input text, in order, `[]` for no texts.
        `kind` picks the EmbeddingGemma task prefix ("query" or "passage"; anything
        else is embedded unprefixed). The call goes through `with_retry`, so only
        transient failures (connection, timeout, 5xx, 429, LM Studio's backend-crash
        400) are retried; every other error propagates on the first attempt.

        The dimensionality check is cheap and guards the thing that fails silently: a vector of
        the wrong width still produces a cosine, just a meaningless one, and the
        gate would threshold it without complaint.
        """
        if not texts:
            return []
        prefix = _PREFIX.get(kind, "")
        # An ingest is thousands of these; one transient failure must not cost
        # the whole build.
        # A review embeds one query: worth a record. An ingest embeds ~550
        # batches: those roll up into the ingest job's own record instead of
        # flooding the log (tools/ingest.py).
        span = ActivityLogger("external_api", "embed.query", target_service="lmstudio",
                              metadata={"model": self.model, "texts": len(texts)}) \
            if kind == "query" else None
        with span or _NoSpan():
            resp = with_retry(lambda: self._get_client().embeddings.create(
                model=self.model, input=[prefix + t for t in texts]),
                what="embedding", role="embedding", model=self.model)
        vectors = [d.embedding for d in resp.data]
        got = len(vectors[0])
        if got != self.dims:
            raise ValueError(
                f"embedder '{self.model}' returned {got} dims, but the profile "
                f"pins EMBEDDING_DIMS={self.dims}. An index built at one width and "
                f"queried at another yields silent nonsense, and every "
                f"min_relevance_score changes meaning. Fix the pin or the model."
            )
        return vectors

    def __repr__(self) -> str:
        return f"LMStudioEmbedder(model={self.model!r}, dims={self.dims})"
