"""Import all adapters so they self-register. Add a module + import it here.

Each module is imported for its side effect: the `@adapter` decorators inside it
run at import and fill the registry. Listing the modules in `__all__` declares
that the package exposes them, so the import is a deliberate re-export rather
than an unused name. `platforms` registers cuda/mac/cpu/studio the same way.
"""
from . import (
    action_sinks,
    audit_jsonl,
    detectors,
    embed_lmstudio,
    entail_cross_encoder,
    entail_lexical,
    input_guard,
    llm_lmstudio,
    llm_openai,
    llm_stub,
    platforms,
    rerank_cross_encoder,
    rerank_gateway,
    retriever_hybrid,
    retriever_indexed,
    source_git_local,
    source_github,
    source_s3logs,
    tenant_fs,
    tracer_jsonl,
    vector_index_local,
)

__all__ = [
    "action_sinks",
    "audit_jsonl",
    "detectors",
    "embed_lmstudio",
    "entail_cross_encoder",
    "entail_lexical",
    "input_guard",
    "llm_lmstudio",
    "llm_openai",
    "llm_stub",
    "platforms",
    "rerank_cross_encoder",
    "rerank_gateway",
    "retriever_hybrid",
    "retriever_indexed",
    "source_git_local",
    "source_github",
    "source_s3logs",
    "tenant_fs",
    "tracer_jsonl",
    "vector_index_local",
]
