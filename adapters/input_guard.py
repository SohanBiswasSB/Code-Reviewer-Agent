"""The InputGuard adapters: the regex guard (the default) and the null one.

The guard itself -- its patterns, and the reasoning about which of its two jobs
regex can honestly claim to do -- lives in `redaction.py`, outside `adapters/`,
because the activity log redacts with the same patterns and importing anything
under `adapters` from `app.activity` was an import cycle (adapters/__init__
imports the LLM adapter, which imports app.activity). Registered here, so
`CRA_GUARD=regex|null` resolves exactly as before.
"""
from __future__ import annotations

from redaction import INJECTION_HINTS, PATTERNS, RegexInputGuard
from registry import adapter

__all__ = ["INJECTION_HINTS", "PATTERNS", "NullInputGuard", "RegexInputGuard"]

adapter("guard", "regex")(RegexInputGuard)


@adapter("guard", "null")
class NullInputGuard:
    """Passes text through untouched. For tests that assert on raw evidence.

    Not the default, and `CRA_GUARD=null` has to be set deliberately — the same
    reasoning as the audit sink: a boundary you get by forgetting is not one.
    """

    def __init__(self, fence_markers=()):
        """Same constructor shape as the real guard, so config needs no fallback."""
        self.fence_markers = tuple(fence_markers)

    def sanitize(self, text: str, *, tenant: str = "_default") -> tuple[str, dict]:
        """The text unchanged, with a zero report marked `"guard": "null"`."""
        return text, {"tenant": tenant, "redacted": {}, "total_redactions": 0,
                      "injection_suspected": {}, "fence_forgery": 0, "guard": "null"}
