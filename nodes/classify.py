"""RULE-BASED routing — NOT an llm call.

If the active Surface is injected (state["_surface"]), delegate to its own rule-based
classify (G2). Otherwise use the built-in issue-triage precedence (also what the
safety-gate eval exercises). Precedence: failing CI beats sparse body.
"""
from __future__ import annotations


def classify(state: dict) -> dict:
    surface = state.get("_surface")
    if surface is not None and hasattr(surface, "classify"):
        ev = state["evidence"]
        ev_dict = ev if isinstance(ev, dict) else getattr(ev, "__dict__", {})
        return {"route": surface.classify(ev_dict)}

    ev = state["evidence"]
    if ev.has_failing_ci:                 # precedence locked by the eval suite
        return {"route": "deterministic"}
    if ev.is_information_sparse:
        return {"route": "ai_investigation"}
    return {"route": "human_review"}
