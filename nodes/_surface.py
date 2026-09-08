"""Shared surface/provider lookup for nodes.

Nodes must not import a concrete provider — that is the whole point of ports/.
They resolve what they need through here, which goes via `config` (the
composition root) and therefore through the registry.

The Surface travels through state as a NAME rather than an object, because only
serialisable values may enter a checkpoint channel (T15). This module is where
that name becomes an object again.
"""
from __future__ import annotations


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
    import config
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
