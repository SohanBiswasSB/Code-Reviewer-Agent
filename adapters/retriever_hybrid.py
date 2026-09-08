from registry import adapter
from tools import retrieval  # the hybrid dense+BM25+RRF implementation


@adapter("retriever", "hybrid")          # default
class HybridRetriever:
    def retrieve(self, query: str, k: int = 5, namespace: str | None = None) -> list[dict]:
        # G1: `namespace` is per-codebase isolation. The in-memory demo corpus in
        # tools/retrieval.py has no namespaces, so rather than silently ignoring the
        # argument (which would look like isolation while retrieving across tenants)
        # we refuse. A real backend adapter passes it to the index.
        if namespace is not None:
            raise NotImplementedError(
                "G1: namespace-scoped retrieval needs a real index; the demo corpus "
                "is unpartitioned. Use the pinecone/pgvector adapter."
            )
        return [{"id": s.id, "text": s.text, "score": s.score}
                for s in retrieval.retrieve(query, k)]

# @adapter("retriever", "pinecone") / ("retriever", "pgvector") -> swap the backend only.
