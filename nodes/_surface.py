"""Shared surface/provider lookup for nodes.

Nodes must not import a concrete provider — that is the whole point of ports/.
They resolve what they need through here, which goes via `config` (the
composition root) and therefore through the registry.

The Surface travels through state as a NAME rather than an object, because only
serialisable values may enter a checkpoint channel (T15). This module is where
that name becomes an object again.
"""
from __future__ import annotations

import config


def active_surface(state: dict):
    """The Surface driving this run, or None for the built-in issue-triage path.

    Accepts an injected object (`_surface`) as well as a name (`_surface_name`)
    so tests can supply a surface directly without going through the registry.
    """
    surface = state.get("_surface")
    if surface is not None:
        return surface
    name = state.get("_surface_name")
    if not name:
        return None
    return config.get_surface(name)


def namespace_for(state: dict) -> str | None:
    """The retrieval namespace for this run's codebase (G1).

    Returns the tenant's bound repo, or None when the tenant declares no
    codebase. This is what keeps Project A's diagnosis from citing Project B's
    code: grounding is the entire safety story, so retrieval crossing codebases
    would undermine it silently rather than loudly.
    """
    codebase = (state.get("_policy") or {}).get("codebase") or {}
    return codebase.get("repo") or None


def evidence_as_dict(ev) -> dict:
    """Evidence -> a plain, FULLY nested dict for a Surface to read.

    `model_dump()` rather than `__dict__`, because `__dict__` on a Pydantic model
    leaves nested models as models — so a surface doing `c.get("signal")` over
    `LogWindow.candidates` would hit an AttributeError on a `LogCandidate`. The
    port's contract with a surface is dicts all the way down.

    Computed properties are added back afterwards: `model_dump()` omits them, and
    `has_failing_ci` / `is_information_sparse` are exactly the signals a surface
    routes on (the lossy-conversion bug found when delegation was enabled).
    """
    if isinstance(ev, dict):
        return dict(ev)
    dump = getattr(ev, "model_dump", None)
    out = dump() if callable(dump) else dict(getattr(ev, "__dict__", {}))
    for name in dir(type(ev)):
        if name.startswith("_"):
            continue
        if isinstance(getattr(type(ev), name, None), property):
            try:
                out[name] = getattr(ev, name)
            except (AttributeError, LookupError, TypeError, ValueError):
                # A computed property over incomplete evidence (a missing field,
                # an empty list) must not break routing; the key is just absent.
                pass
    return out
