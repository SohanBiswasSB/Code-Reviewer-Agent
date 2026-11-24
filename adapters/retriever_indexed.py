"""Hybrid retrieval over a real, namespaced index (T10).

The retriever the multi-tenant path has been waiting for. Until this existed,
`HybridRetriever` (the demo) refused every namespaced request -- correctly, since
it searches four hardcoded strings -- which meant every tenant that declares a
codebase (`milesone`, `project_a`, `project_b`) could not retrieve at all. G1 was
enforced by refusal, and the multi-tenant path was inert.

    dense   EmbeddingGemma cosine over the namespace's index      (adapters/embed_lmstudio)
    sparse  Okapi BM25 over the same namespace's chunk texts       (tools/bm25)
    fuse    Reciprocal Rank Fusion, for ORDER only                 (tools/fusion)
    gate    the dense COSINE, computed for every fused candidate

## The two-number contract is kept exactly (T21)

`score` is the cosine -- a calibrated magnitude with a noise floor, which is what
`groundedness_ok` thresholds. `fusion` is the RRF value and carries no magnitude.
A chunk surfaced by BM25 alone gets its cosine looked up explicitly, so a keyword
hit never arrives without a gateable number. See tools/retrieval.py for the
defect this contract exists to prevent.

## Refusals, and why each is a refusal rather than a degraded answer

  · **No namespace.** There is no global index to fall back on and there must
    never be one (G1). The demo retriever served un-namespaced calls; this one
    cannot, so a tenant without `codebase.repo` fails loudly here instead of
    escalating every run for a reason nobody can see.
  · **The index was built by a different embedder.** A query vector from model A
    against vectors from model B yields cosines that look normal and are noise.
    The gate would threshold the noise without complaint.

## The reranker (T13): when on, IT supplies the gate score

With a reranker (`config.get_reranker()`, OFF by default -- measured), the top
`RERANK_CANDIDATES` fused candidates (100) are rescored by a cross-encoder and
the best k kept. Each hit's `score` is then the RERANK score and
`score_kind = "rerank"`; the cosine stays beside it as `cosine`. The gate picks
its bar by kind -- `min_rerank_score` for one, `min_relevance_score` for the
other -- so a score from one model is never compared with a bar measured for
another. Why the cosine alone is not enough: adapters/rerank_cross_encoder.py.

An index that is simply NOT BUILT returns `[]`: that is the truthful answer to
"what does this codebase contain", it escalates via `groundedness_ok`, and
`doctor` reports the unbuilt namespace so the state is visible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import registry
from registry import adapter
from tools.bm25 import BM25
from tools.bm25 import FILE as BM25_FILE
from tools.fusion import reciprocal_rank_fusion

# How many candidates each backend contributes before fusion. Wider than k so a
# chunk ranked 6th by dense and 2nd by BM25 can still win -- that is the point of
# fusing. With a reranker on (T13, built, OFF by default) each backend is widened
# to at least RERANK_CANDIDATES so the reranker has a full pool to narrow to k.
WIDE_MIN = 20
WIDE_FACTOR = 4
# How many fused candidates the reranker rescores. Wide enough that a passage
# both backends ranked mediocre can still win on a real reading of the pair.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "100"))
_UNSET = object()


@dataclass(frozen=True)
class _Query:
    """One retrieval request as the backends see it: the text (for BM25 and the
    reranker), its embedding (for the dense side), the namespace both search, and
    how many candidates each backend contributes before fusion."""

    text: str
    namespace: str
    vector: list
    wide: int


def _require_namespace(namespace: str | None) -> None:
    """Refuse an un-namespaced request (G1): there is no global index to fall
    back on, and there must never be one."""
    if not namespace:
        raise ValueError(
            "G1: IndexedRetriever needs a namespace. The tenant declares no "
            "codebase.repo, so there is no index it may search -- and there is "
            "deliberately no global one. Bind the tenant to a codebase in its "
            "policy.json, or use RETRIEVER=hybrid for the offline demo corpus.")


def _build_hits(ordered: list[tuple[str, float]], relevance: dict, meta_by_id: dict) -> list[dict]:
    """The fused candidates as hits, in fusion order, each with its cosine as `score`.

    A chunk the index cannot score gets 0.0 -- fail closed: it can be read, but
    it cannot ground anything.
    """
    out = []
    for position, (cid, fusion) in enumerate(ordered, start=1):
        meta = meta_by_id.get(cid, {})
        out.append({
            "id": cid,
            "text": meta.get("text", ""),
            "score": relevance.get(cid, 0.0),
            "score_kind": "cosine",
            "cosine": relevance.get(cid, 0.0),
            "fusion": round(fusion, 6),
            "rank": position,
            "path": meta.get("path"),
            "name": meta.get("name"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
        })
    return out


def _reranked(reranker, query: str, hits: list[dict], k: int) -> list[dict]:
    """The best `k` of `hits` in rerank order, `score` now the RERANK score.

    The cosine stays on each hit as `cosine`; `score_kind` says which number
    `score` is, so the gate picks the bar measured for that model.
    """
    reranked = reranker.rerank(query, hits, top_n=k)
    for position, hit in enumerate(reranked, start=1):
        hit.update(score=hit.pop("rerank"), score_kind="rerank", rank=position)
    return reranked


@adapter("retriever", "indexed")
class IndexedRetriever:
    """The default Retriever: dense + BM25 over one namespace, cosine as the gate score.

    With a reranker configured, the gate score is the rerank score instead, and
    each hit says which via `score_kind` (module docstring).
    """

    def __init__(self, embedder=None, index=None, reranker=_UNSET):
        """All three are injectable; anything not given resolves from `config` on first use."""
        # Lazy, and injectable for tests -- the same shape as the nodes' provider
        # arguments. Resolving at construction would make `config.get_retriever()`
        # fail on a machine without the openai client even for callers that never
        # retrieve.
        self._embedder = embedder
        self._index = index
        # `None` is a real value here (no reranking), so "not given" is _UNSET.
        self._reranker = reranker
        # namespace -> (fingerprint, BM25, metas). BM25 builds IDF over the whole
        # corpus, which is O(corpus); doing it per query would dominate latency.
        self._bm25: dict[str, tuple] = {}

    @property
    def embedder(self):
        """The injected embedder, else the configured one (`config.get_embedder`) on first use.

        Reached through `registry.provider`, not `import config`: config imports
        `adapters`, which imports this module, so importing config here was a cycle.
        """
        if self._embedder is None:
            self._embedder = registry.provider("embedder")
        return self._embedder

    @property
    def index(self):
        """The injected index, else the configured one (`config.get_index`)."""
        if self._index is None:
            self._index = registry.provider("index")
        return self._index

    @property
    def reranker(self):
        """The injected reranker (None = off), else `config.get_reranker()`."""
        if self._reranker is _UNSET:
            self._reranker = registry.provider("reranker")
        return self._reranker

    def _check_space(self, namespace: str, stats: dict) -> None:
        """Refuse when the namespace's manifest names another embedder or width.

        A ValueError naming both models and the rebuild command, because the
        alternative is a cosine across two vector spaces that looks normal.
        """
        built_by, dims = stats.get("model"), stats.get("dims")
        emb = self.embedder
        if built_by != emb.model or (dims and dims != emb.dims):
            raise ValueError(
                f"index '{namespace}' was built by {built_by!r} ({dims} dims) but "
                f"would be queried by {emb.model!r} ({emb.dims} dims). Cosines across "
                f"two embedding spaces are noise, and min_relevance_score would gate "
                f"on that noise. Rebuild: python -m tools.ingest --tenant <tenant>")

    def _sparse(self, namespace: str, fingerprint: str):
        """BM25 for one namespace: in memory, else the saved statistics, else
        built from the chunks -- and then saved, so the next PROCESS loads it.

        Building costs ~1.4 s on the Masterclass index and every CLI review is a
        new process, so without the saved copy each review paid it again. The
        file is keyed by the index fingerprint: a rebuilt index never reuses the
        old statistics.
        """
        cached = self._bm25.get(namespace)
        if cached and cached[0] == fingerprint:
            return cached[1], cached[2]
        metas = self.index.documents(namespace)
        # An index without a `sidecar` location (it is not in the VectorIndex port)
        # has no saved copy: BM25 is built in memory and nothing is saved.
        path = (self.index.sidecar(namespace, BM25_FILE) if hasattr(self.index, "sidecar")
                else None)
        bm25 = BM25.load(path, fingerprint) if path else None
        if bm25 is None or bm25.n != len(metas):
            bm25 = BM25.for_chunks(metas)
            if path:
                bm25.save(path, fingerprint)
        self._bm25[namespace] = (fingerprint, bm25, metas)
        return bm25, metas

    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]:
        """Top-k chunks of ONE codebase, each carrying its gate score as `score`.

        Reranker OFF (the default): order comes from RRF over the dense and BM25
        rankings, and `score` is the cosine (`score_kind="cosine"`). Reranker ON:
        the top RERANK_CANDIDATES fused hits are rescored and the best k returned
        in rerank order, with `score` the rerank score (`score_kind="rerank"`) and
        the cosine kept as `cosine`. Either way the gate reads `score` against the
        bar for its kind. Each hit also carries `fusion`, `rank`, `path`, `name`
        and line span. See the module docstring for the refusals; an unbuilt
        namespace returns [].
        """
        _require_namespace(namespace)

        stats = self.index.stats(namespace)
        if not stats.get("built") or not stats.get("chunks"):
            return []
        self._check_space(namespace, stats)

        reranker = self.reranker
        pool = max(k, RERANK_CANDIDATES) if reranker is not None else k
        q = _Query(text=query, namespace=namespace,
                   vector=self.embedder.embed([query], kind="query")[0],
                   wide=max(WIDE_MIN, k * WIDE_FACTOR, pool))
        hits = self._candidates(q, stats.get("fingerprint", ""), pool)
        if reranker is None:
            return hits[:k]
        return _reranked(reranker, query, hits, k)

    def _candidates(self, q: _Query, fingerprint: str, pool: int) -> list[dict]:
        """The top `pool` fused candidates as hits, each with a real cosine."""
        ordered, dense_hits, metas = self._fused(q, fingerprint, pool)
        relevance, meta_by_id = self._cosines(q, ordered, dense_hits, metas)
        return _build_hits(ordered, relevance, meta_by_id)

    def _fused(self, q: _Query, fingerprint: str, pool: int):
        """(ordered, dense_hits, metas): the RRF ranking of the dense and BM25
        lists cut to `pool`, plus what each backend returned, for `_cosines`.

        RRF supplies ORDER only; the fusion value carries no magnitude and is
        never what the gate reads.
        """
        dense_hits = self.index.search(q.namespace, q.vector, q.wide)
        bm25, metas = self._sparse(q.namespace, fingerprint)
        sparse_ids = [metas[i]["id"] for i, _ in bm25.top(q.text, q.wide)]
        fused = reciprocal_rank_fusion([[h["id"] for h in dense_hits], sparse_ids])
        ordered = sorted(fused.items(), key=lambda p: p[1], reverse=True)[:pool]
        return ordered, dense_hits, metas

    def _cosines(self, q: _Query, ordered: list, dense_hits: list[dict], metas: list[dict]):
        """(relevance, meta_by_id) for every candidate in `ordered`.

        Dense hits bring their cosine and metadata with them. Keyword-only
        winners need a real cosine too, or they would reach the gate with no
        number at all (T21's sibling), so theirs is looked up explicitly.
        """
        relevance = {h["id"]: h["score"] for h in dense_hits}
        meta_by_id = {h["id"]: h for h in dense_hits}
        missing = [cid for cid, _ in ordered if cid not in relevance]
        if missing:
            relevance.update(self.index.similarities(q.namespace, q.vector, missing))
            by_id = {m["id"]: m for m in metas}
            for cid in missing:
                meta_by_id.setdefault(cid, by_id.get(cid, {}))
        return relevance, meta_by_id
