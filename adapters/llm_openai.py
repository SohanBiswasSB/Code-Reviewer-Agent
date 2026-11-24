"""Hosted OpenAI behind the `LLM` port -- a deliberate, visible stub.

`complete` raises `NotImplementedError` rather than returning anything: a stub
that faked a reply would let a misconfigured LLM_PROVIDER=openai run the whole
graph on invented output. The local LM Studio adapter is the one exercised.
"""
from registry import adapter


@adapter("llm", "openai")
class OpenAILLM:
    """Hosted OpenAI. Same bound-model contract as the LM Studio adapter.

    Still a stub — the local path is what gets exercised. Kept so the provider
    swap stays a config change (LLM_PROVIDER=openai) rather than a code change.
    """

    def __init__(self, model: str = "gpt-4o-mini", role: str | None = None):
        """Same constructor shape as `LMStudioLLM`, so `config.get_llm` can build either."""
        self.model = model
        self.role = role

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """Not implemented; always raises `NotImplementedError`."""
        raise NotImplementedError("wire the openai client; return the text (or JSON string)")

# @adapter("llm", "claude") -> drop in more, no core change.
