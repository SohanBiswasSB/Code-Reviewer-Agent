"""RULE-BASED routing — NOT an llm call.

Precedence for the built-in issue-triage path is locked by the eval suite:
failing CI beats a sparse body.

Surface delegation (G2): when a Surface is active, ITS rule-based classify drives
the route. The surface is accepted two ways on purpose:

  · `state["_surface"]` — an injected object. Used by tests, which construct a
    surface directly rather than going through the registry.
  · `state["_surface_name"]` — a string, which is what `resolve_tenant` puts in
    state. Resolved lazily here.

The name is what travels through the graph because only serialisable values may
enter a checkpoint channel — a class instance would add another unregistered type
to the deserialisation problem that would make paused runs unresumable (T15).
"""
from __future__ import annotations

from nodes._surface import active_surface, evidence_as_dict


def classify(state: dict) -> dict:
    """Pick the route from validated evidence, by rule. Zero LLM calls.

    Reads `evidence` (and the surface from `_surface` / `_surface_name`); writes
    only `route`: `deterministic`, `ai_investigation` or `human_review`. Rule-based
    on purpose: what reaches a model, and what reaches the approval gate without
    one, must not be a model's decision. With no surface active, the built-in
    issue-triage rules apply, and anything they do not match goes to a human
    (invariant 12: an ambiguous route goes to `human_review`).
    """
    surface = active_surface(state)
    if surface is not None and hasattr(surface, "classify"):
        return {"route": surface.classify(evidence_as_dict(state["evidence"]))}

    ev = state["evidence"]
    if ev.has_failing_ci:                 # precedence locked by the eval suite
        return {"route": "deterministic"}
    if ev.is_information_sparse:
        return {"route": "ai_investigation"}
    return {"route": "human_review"}
