"""Reciprocal Rank Fusion: combine several rankings into one ORDER.

Its own module because the real retriever (`adapters/retriever_indexed.py`) and
the offline demo (`tools/retrieval.py`) both use it, and it used to live inside
the demo's module -- next to a four-document corpus and "swap for Pinecone"
notes -- so the production path imported from a stub file.

RRF is a function of rank POSITION and carries no magnitude, which is why it
orders results and never gates them (T21; see tools/retrieval.py).
"""
from __future__ import annotations

RRF_K = 60  # standard RRF constant


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse several ranked id-lists. score(d) = sum 1/(k + rank_in_list)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused
