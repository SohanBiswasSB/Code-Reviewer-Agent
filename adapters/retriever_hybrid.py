"""The offline DEMO retriever: four hardcoded documents, no index, no model.

`RETRIEVER=hybrid`, declared by the tests, the gate suite and the demo so they run
with no LM Studio and no ingest. The real retriever is `indexed` (the default,
adapters/retriever_indexed.py). The demo's "dense" score is a Jaccard proxy, not a
cosine, so no threshold measured here means anything against a real embedder
(tools/retrieval.py).
"""
from registry import adapter
from tools import retrieval  # the hybrid dense+BM25+RRF implementation


@adapter("retriever", "hybrid")          # the offline DEMO; `indexed` is the default
class HybridRetriever:
    """The `Retriever` port over tools/retrieval.py's in-memory demo corpus."""

    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]:
        """Top-k demo snippets as `{id, text, score, fusion, rank}`; refuses any namespace.

        The port's contract is kept: `score` is the calibrated-relevance half,
        `fusion` only orders. A namespaced call raises `NotImplementedError`
        instead of answering from an unpartitioned corpus (G1).
        """
        # G1: `namespace` is per-codebase isolation. The in-memory demo corpus in
        # tools/retrieval.py has no namespaces, so rather than silently ignoring the
        # argument (which would look like isolation while retrieving across tenants)
        # we refuse. A real backend adapter passes it to the index.
        if namespace is not None:
            raise NotImplementedError(
                "G1: namespace-scoped retrieval needs a real index; the demo corpus "
                "is unpartitioned. Use RETRIEVER=indexed (the default) after "
                "`python -m tools.ingest`."
            )
        # "score" is the CALIBRATED relevance, never the rank-fusion value --
        # gating on a rank makes the grounding check inert (T21). "fusion" is
        # carried for debugging and ordering transparency only.
        return [{"id": s.id, "text": s.text, "score": s.relevance,
                 "fusion": s.fusion, "rank": s.rank}
                for s in retrieval.retrieve(query, k)]

