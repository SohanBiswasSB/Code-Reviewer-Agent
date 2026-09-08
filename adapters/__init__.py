"""Import all adapters so they self-register. Add a module + import it here."""
from . import llm_openai, retriever_hybrid, source_github, source_s3logs, action_sinks, tenant_fs  # noqa
