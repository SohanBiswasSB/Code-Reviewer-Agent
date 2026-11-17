"""Tiny plugin registries — the 'plug' in plug-and-play.

Adapters and surfaces self-register with a decorator; config.py resolves them by
name from env. Add a new provider/surface = drop a module in adapters/ or
surfaces/ and decorate it (and add it to that package's `__init__` import list,
which is what makes the decorator run). Nothing in nodes/ changes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# kind ("llm" | "retriever" | "source" | "sink" | "tenant" | ...) -> name -> class
_ADAPTERS: dict[str, dict[str, type]] = {}
# surface name ("code_review" | "log_review" | ...) -> class
_SURFACES: dict[str, type] = {}


def adapter(kind: str, name: str) -> Callable[[T], T]:
    """Register a concrete provider under (kind, name). Idempotent per import."""
    def deco(cls: T) -> T:
        """Record `cls` and hand it back unchanged, so the class stays importable."""
        _ADAPTERS.setdefault(kind, {})[name] = cls
        return cls
    return deco


def surface(name: str) -> Callable[[T], T]:
    """Register a capability (code_review, log_review, ...) under `name`."""
    def deco(cls: T) -> T:
        """Record `cls` and hand it back unchanged."""
        _SURFACES[name] = cls
        return cls
    return deco


def get_adapter(kind: str, name: str) -> type:
    """The class registered under (kind, name) -- not an instance; config.py builds it.

    Raises KeyError naming what IS registered for that kind, so a typo in an env
    var reads as a config mistake rather than a bare missing key.
    """
    try:
        return _ADAPTERS[kind][name]
    except KeyError:
        have = sorted(_ADAPTERS.get(kind, {}))
        raise KeyError(f"no {kind} adapter named '{name}'. have: {have}") from None


def get_surface(name: str) -> type:
    """The Surface class registered as `name`. KeyError lists the registered ones."""
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
