from registry import adapter


@adapter("llm", "openai")
class OpenAILLM:
    """Hosted OpenAI. Same bound-model contract as the LM Studio adapter.

    Still a stub — the local path is what gets exercised. Kept so the provider
    swap stays a config change (LLM_PROVIDER=openai) rather than a code change.
    """

    def __init__(self, model: str = "gpt-4o-mini", runtime: dict | None = None,
                 role: str | None = None):
        self.model = model
        self.runtime = runtime or {}
        self.role = role

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        raise NotImplementedError("wire the openai client; return the text (or JSON string)")

# @adapter("llm", "claude") -> drop in more, no core change.
