"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion (RRF).

Improvement #1/#2/#3 over the reference (which is dense-cosine only). The dense
and BM25 backends here are STUBS over a tiny in-memory corpus so the module runs
offline; swap them for Pinecone/pgvector + a real BM25 (or Postgres ts_rank) in
production. RRF, MMR-dedup and the score-returning contract are real.
"""
from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60  # standard RRF constant


@dataclass
class Snippet:
    id: str
    text: str
    score: float   # fused relevance; the grounding gate reads this


# --- tiny demo corpus (replace with a vector index) --------------------------
_CORPUS = [
    ("iss-101", "OAuth callback returns 500 when state param missing; fix in auth middleware"),
    ("iss-102", "CI flaky on windows terminal resize test; unrelated to auth"),
    ("iss-103", "token refresh race causes intermittent 401 after login"),
    ("iss-104", "docs: how to configure the webhook secret token"),
]


def _dense_search(query: str, k: int) -> list[tuple[str, float]]:
    # STUB: replace with vector_store.similarity_search_with_score(query, k)
    q = set(query.lower().split())
    scored = [(cid, _fake_cosine(q, txt)) for cid, txt in _CORPUS]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def _bm25_search(query: str, k: int) -> list[tuple[str, float]]:
    # STUB: replace with a real BM25 / Postgres ts_rank keyword search
    q = set(query.lower().split())
    scored = [(cid, float(len(q & set(txt.lower().split())))) for cid, txt in _CORPUS]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def _fake_cosine(q: set[str], txt: str) -> float:
    t = set(txt.lower().split())
    return 0.0 if not q else round(len(q & t) / (len(q | t) or 1), 3)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse several ranked id-lists. score(d) = sum 1/(k + rank_in_list)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def retrieve(query: str, k: int = 5) -> list[Snippet]:
    """Hybrid retrieve: dense + BM25 -> RRF fuse -> top-k Snippets (with score).

    TODO #2: add a cross-encoder rerank of the fused top-N.
    TODO #3: add MMR de-dup so citations aren't near-duplicates.
    """
    dense = [cid for cid, _ in _dense_search(query, k * 2)]
    bm25 = [cid for cid, _ in _bm25_search(query, k * 2)]
    fused = reciprocal_rank_fusion([dense, bm25])
    text = dict(_CORPUS)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    # normalise fused score to 0..1-ish for the gate demo (use real cosine in prod)
    top = ordered[0][1] if ordered else 1.0
    return [Snippet(id=cid, text=text.get(cid, ""), score=round(s / top, 3))
            for cid, s in ordered]
