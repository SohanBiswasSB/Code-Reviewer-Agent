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

from nodes._surface import active_surface


def _as_dict(ev) -> dict:
    """Flatten evidence for a Surface's rule-based classify.

    `__dict__` alone is wrong: it returns stored fields only, so computed
    properties are silently dropped — and `has_failing_ci` and
    `is_information_sparse` are exactly the signals a surface routes on. A
    surface asking for `has_failing_ci` would have got `None` and quietly taken
    the wrong branch.
    """
    if isinstance(ev, dict):
        return dict(ev)
    out = dict(getattr(ev, "__dict__", {}))
    for name in dir(type(ev)):
        if name.startswith("_"):
            continue
        if isinstance(getattr(type(ev), name, None), property):
            try:
                out[name] = getattr(ev, name)
            except Exception:      # a property that raises must not break routing
                pass
    return out


def classify(state: dict) -> dict:
    surface = active_surface(state)
    if surface is not None and hasattr(surface, "classify"):
        return {"route": surface.classify(_as_dict(state["evidence"]))}

    ev = state["evidence"]
    if ev.has_failing_ci:                 # precedence locked by the eval suite
        return {"route": "deterministic"}
    if ev.is_information_sparse:
        return {"route": "ai_investigation"}
    return {"route": "human_review"}
