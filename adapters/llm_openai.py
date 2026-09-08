from registry import adapter

@adapter("llm", "openai")
class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    def complete(self, system: str, user: str, *, json: bool = False) -> str:
        raise NotImplementedError("wire openai client; return the text (or JSON string)")

# @adapter("llm", "claude") / ("llm", "local") -> drop in more, no core change.
