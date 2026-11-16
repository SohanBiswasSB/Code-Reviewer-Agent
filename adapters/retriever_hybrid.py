from registry import adapter
from tools import retrieval  # the hybrid dense+BM25+RRF implementation


@adapter("retriever", "hybrid")          # the offline DEMO; `indexed` is the default
class HybridRetriever:
    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]:
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

