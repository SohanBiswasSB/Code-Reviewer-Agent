"""Tiny plugin registries — the 'plug' in plug-and-play.

Adapters and surfaces self-register with a decorator; config.py resolves them by
name from env. Add a new provider/surface = drop a module in adapters/ or
surfaces/ and decorate it. Nothing in core/ changes.
"""
from __future__ import annotations
from typing import Callable, Dict, TypeVar

T = TypeVar("T")

# kind ("llm" | "retriever" | "source" | "sink" | "tenant" | ...) -> name -> class
_ADAPTERS: Dict[str, Dict[str, type]] = {}
# surface name ("code_review" | "log_review" | ...) -> class
_SURFACES: Dict[str, type] = {}


def adapter(kind: str, name: str) -> Callable[[T], T]:
    """Register a concrete provider under (kind, name). Idempotent per import."""
    def deco(cls: T) -> T:
        _ADAPTERS.setdefault(kind, {})[name] = cls
        return cls
    return deco


def surface(name: str) -> Callable[[T], T]:
    """Register a capability (code_review, log_review, ...) under `name`."""
    def deco(cls: T) -> T:
        _SURFACES[name] = cls
        return cls
    return deco


def get_adapter(kind: str, name: str) -> type:
    try:
        return _ADAPTERS[kind][name]
    except KeyError:
        have = sorted(_ADAPTERS.get(kind, {}))
        raise KeyError(f"no {kind} adapter named '{name}'. have: {have}") from None


def get_surface(name: str) -> type:
    try:
        return _SURFACES[name]
    except KeyError:
        raise KeyError(f"no surface '{name}'. have: {sorted(_SURFACES)}") from None


def list_registered() -> dict:
    """What this deployment actually has plugged in (GET /registered)."""
    return {
        "adapters": {kind: sorted(names) for kind, names in _ADAPTERS.items()},
        "surfaces": sorted(_SURFACES),
    }
