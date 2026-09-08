"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion (RRF).

## Two different numbers, and why that matters (T21)

`retrieve()` returns each snippet with **two** scores, and they are not
interchangeable:

  · `relevance`  — a calibrated similarity magnitude from the dense backend.
                   **This is what the grounding gate reads.**
  · `fusion`     — the RRF value, `sum(1/(RRF_K + rank))`. **Ordering only.**

Conflating them was a real defect here. The previous version returned
`fusion / fusion_of_top_hit` as the single score, which forces the best hit to
exactly 1.0 on every query. Measured across on-topic, off-topic and gibberish
queries, the output was identical every time — `[1.0, 0.984, 0.968]` — so
`groundedness_ok`'s `score >= min_relevance_score` could not fail at any
threshold below 0.968. The relevance half of the safety gate was inert.

The reason is structural, not a typo: RRF is a function of **rank position**, so
it carries no magnitude. Calibrating a threshold requires magnitude — you set the
bar above the off-topic noise cluster, and a rank has no noise cluster. So rank
fusion is the right tool for ordering results and the wrong tool for gating them.
Keep the two numbers separate, permanently.

## Status of the backends

The dense and BM25 backends are STUBS over a tiny in-memory corpus so this
module runs offline with no dependencies. Swap them for Pinecone/pgvector plus a
real BM25 (or Postgres `ts_rank`). RRF and the two-score contract are real.

**The demo similarity is a Jaccard proxy, NOT a cosine.** Its numbers are on a
different scale from any real embedding model's, so a threshold calibrated
against it is meaningless in production. See TODO T10: the shipped
`min_relevance_score` values must be re-measured against the actual embedder.
What the proxy *does* give correctly is the property the gate depends on —
off-topic queries score strictly lower than on-topic ones, and unrelated queries
score 0.0.
"""
from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60  # standard RRF constant


@dataclass
class Snippet:
    id: str
    text: str
    relevance: float          # calibrated similarity. THE GATE READS THIS.
    fusion: float = 0.0       # RRF rank-fusion value. Ordering only.
    rank: int = 0             # 1-based position after fusion

    @property
    def score(self) -> float:
        """Back-compat alias for `relevance`.

        The gate and the eval tasks refer to `.score`. Deliberately aliased to
        `relevance` and NOT to `fusion`, so that any existing caller reading
        `.score` gets the gateable number rather than the rank artefact.
        """
        return self.relevance


# --- tiny demo corpus (replace with a vector index) --------------------------
_CORPUS = [
    ("iss-101", "OAuth callback returns 500 when state param missing; fix in auth middleware"),
    ("iss-102", "CI flaky on windows terminal resize test; unrelated to auth"),
    ("iss-103", "token refresh race causes intermittent 401 after login"),
    ("iss-104", "docs: how to configure the webhook secret token"),
]


def _demo_similarity(q: set[str], txt: str) -> float:
    """Jaccard overlap as a stand-in for cosine similarity.

    NOT a cosine and not on a cosine's scale — see the module docstring. It is
    here because it has the one property the gate needs from a real embedder:
    it falls to 0.0 for unrelated text, so there is a noise floor to calibrate
    against.
    """
    t = set(txt.lower().split())
    if not q:
        return 0.0
    return round(len(q & t) / (len(q | t) or 1), 4)


def _dense_search(query: str, k: int) -> list[tuple[str, float]]:
    # STUB: replace with vector_store.similarity_search_with_score(query, k)
    q = set(query.lower().split())
    scored = [(cid, _demo_similarity(q, txt)) for cid, txt in _CORPUS]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def _bm25_search(query: str, k: int) -> list[tuple[str, float]]:
    # STUB: replace with a real BM25 / Postgres ts_rank keyword search
    q = set(query.lower().split())
    scored = [(cid, float(len(q & set(txt.lower().split())))) for cid, txt in _CORPUS]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse several ranked id-lists. score(d) = sum 1/(k + rank_in_list)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def relevance_of(query: str) -> dict[str, float]:
    """Calibrated similarity for every candidate, keyed by id.

    Separate from `_dense_search` because a document surfaced by BM25 alone
    still needs a relevance number for the gate — otherwise a keyword-only hit
    would arrive with no gateable score and pass by default. A real backend
    embeds the query once and scores the union of candidates the same way.
    """
    q = set(query.lower().split())
    return {cid: _demo_similarity(q, txt) for cid, txt in _CORPUS}


def retrieve(query: str, k: int = 5) -> list[Snippet]:
    """Hybrid retrieve: dense + BM25 -> RRF fuse for ORDER -> top-k.

    Each snippet carries its calibrated `relevance` (what the gate reads) and
    its `fusion` value (ordering only). See the module docstring for why those
    must not be the same number.

    TODO T13: cross-encoder rerank of the fused top-N, whose calibrated score
              should then supersede the dense relevance for gating.
    TODO: MMR de-dup so citations are not near-duplicates.
    """
    dense = [cid for cid, _ in _dense_search(query, k * 2)]
    bm25 = [cid for cid, _ in _bm25_search(query, k * 2)]
    fused = reciprocal_rank_fusion([dense, bm25])

    text = dict(_CORPUS)
    relevance = relevance_of(query)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    return [
        Snippet(
            id=cid,
            text=text.get(cid, ""),
            relevance=relevance.get(cid, 0.0),
            fusion=round(fusion_score, 6),
            rank=position,
        )
        for position, (cid, fusion_score) in enumerate(ordered, start=1)
    ]
