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

from adapters.platforms import base
from registry import adapter

# EmbeddingGemma's documented task prefixes. A model without them is unharmed --
# the text is simply embedded as written.
_PREFIX = {
    "query": "task: search result | query: ",
    "passage": "title: none | text: ",
}


@adapter("embedder", "lmstudio")
class LMStudioEmbedder:
    """Batch text -> vectors via the OpenAI-compatible embeddings endpoint."""

    def __init__(self, model: str | None = None, dims: int | None = None):
        self.model = model or base.EMBEDDING_MODEL
        self.dims = dims or base.EMBEDDING_DIMS
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key="lm-studio")
        return self._client

    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        """Embed a batch. Raises on a dimensionality mismatch rather than returning junk.

        The check is cheap and guards the thing that fails silently: a vector of
        the wrong width still produces a cosine, just a meaningless one, and the
        gate would threshold it without complaint.
        """
        if not texts:
            return []
        prefix = _PREFIX.get(kind, "")
        resp = self._get_client().embeddings.create(
            model=self.model, input=[prefix + t for t in texts])
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
