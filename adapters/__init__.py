"""Import all adapters so they self-register. Add a module + import it here."""
from . import audit_jsonl, detectors, entail_cross_encoder, entail_lexical, input_guard, llm_lmstudio, llm_openai, retriever_hybrid, source_github, source_s3logs, action_sinks, tenant_fs, tracer_jsonl  # noqa
from . import platforms  # noqa  -- cuda/mac/cpu/studio self-register
