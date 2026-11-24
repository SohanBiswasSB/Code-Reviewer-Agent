"""Stand-ins for the slice of the OpenAI client that `LMStudioLLM.complete` touches.

Shared by the LLM adapter tests, which used to build the same shape inline from
nested `type(...)` calls in five places. Only `chat.completions.create(**kw)` and
the reply's `choices[0].message.content` / `finish_reason` / `usage` are
modelled. The adapter reads `usage` with `getattr(..., None)`, so `usage=None` is
the same reply a server without usage accounting sends.
"""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace


def chat_reply(content: str, finish_reason: str | None = None,
               usage: SimpleNamespace | None = None) -> SimpleNamespace:
    """A completion response whose first choice carries `content`.

    `usage`, when given, is the token accounting (`prompt_tokens`,
    `completion_tokens`); None reads exactly like a reply that has none.
    """
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                    finish_reason=finish_reason)],
                           usage=usage)


def chat_client(create: Callable[..., SimpleNamespace]) -> SimpleNamespace:
    """A client whose `chat.completions.create(**kw)` is `create`."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
