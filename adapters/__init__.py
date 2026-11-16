"""Import all adapters so they self-register. Add a module + import it here."""
from . import audit_jsonl, detectors, embed_lmstudio, vector_index_local, entail_cross_encoder, entail_lexical, input_guard, llm_lmstudio, llm_stub, llm_openai, retriever_hybrid, retriever_indexed, source_github, source_s3logs, action_sinks, tenant_fs, tracer_jsonl  # noqa
from . import platforms  # noqa  -- cuda/mac/cpu/studio self-register
