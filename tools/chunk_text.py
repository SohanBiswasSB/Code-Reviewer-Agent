"""What each retriever half sees of a chunk, side by side so the choice is deliberate.

Both prefix the chunk's location (path and qualified name) to its body, because
those are the most specific words a code chunk has. They differ on purpose:

  · `embed_text` is baked into every stored VECTOR. Changing a single character
    here changes the vector space, so it invalidates every index already built
    and silently shifts the meaning of each tenant's `min_relevance_score` --
    the same reason EMBEDDING_MODEL is pinned.
  · `bm25_text` is tokenised at query time, so it can change freely.

Neither is what a citation shows or what entailment reads: the stored `text`
stays exactly the source.
"""
from __future__ import annotations


def embed_text(chunk: dict) -> str:
    """What is EMBEDDED for a chunk: its location, then its body.

    A function body alone is often generic -- `if not user: return None` could be
    anywhere. Prefixing the path and qualified name gives the embedding the
    context a reader would have from the file tab ("contextual retrieval",
    without paying an LLM call per chunk).
    """
    return f"{chunk.get('path', '')} :: {chunk.get('name', '')}\n{chunk['text']}"


def bm25_text(meta: dict) -> str:
    """What BM25 indexes for a chunk: its location as well as its body.

    `payments/webhooks.py::verify_signature` says more than the body's `self`,
    `return` and `if`; a query naming a module should be able to reach it.
    """
    return f"{meta.get('path', '')} {meta.get('name', '')}\n{meta.get('text', '')}"
