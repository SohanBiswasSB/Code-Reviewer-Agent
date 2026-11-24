"""Tiny plugin registries — the 'plug' in plug-and-play.

Adapters and surfaces self-register with a decorator; config.py resolves them by
name from env. Add a new provider/surface = drop a module in adapters/ or
surfaces/ and decorate it (and add it to that package's `__init__` import list,
which is what makes the decorator run). Nothing in nodes/ changes.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# kind ("llm" | "retriever" | "source" | "sink" | "tenant" | ...) -> name -> class
_ADAPTERS: dict[str, dict[str, type]] = {}
# surface name ("code_review" | "log_review" | ...) -> class
_SURFACES: dict[str, type] = {}
# provider kind ("embedder" | "tracer" | ...) -> config's zero-argument getter
_PROVIDERS: dict[str, Callable[[], object]] = {}


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


def provides(kind: str) -> Callable[[T], T]:
    """Mark a `config` getter as THE way to obtain the configured `kind` provider.

    The other half of `provider()`. It exists to break an import cycle at its
    cause instead of hiding it in a function-level import: `config` imports
    `adapters` (that is what registers them), and a few adapters need another
    configured provider -- the retriever needs the embedder, the LLM client the
    tracer. They used to `import config` inside a function. Now config hands its
    getters to this module, which both sides already import, so the dependency
    points one way: adapters -> registry <- config.
    """
    def deco(getter: T) -> T:
        """Record the getter BY NAME and hand it back unchanged.

        Looked up on its module at call time, exactly as `config.get_x()` was, so
        a test that monkeypatches config's getter still reaches the adapter.
        """
        module, name = getter.__module__, getter.__name__

        def current() -> object:
            """Call whatever `module.name` is NOW -- a monkeypatched getter included."""
            return getattr(sys.modules[module], name)()

        _PROVIDERS[kind] = current
        return getter
    return deco


def provider(kind: str):
    """The configured `kind` provider, built by the getter `config` registered.

    LookupError when nothing registered one, i.e. `config` was never imported --
    the composition root is missing, which is a wiring bug, not a provider to
    default. Callers that can run without one (the retry announcer without a
    tracer) catch LookupError, as they already did for a registry miss.
    """
    try:
        getter = _PROVIDERS[kind]
    except KeyError:
        raise LookupError(f"no '{kind}' provider: import config (the composition "
                          f"root) before using an adapter that needs one. "
                          f"have: {sorted(_PROVIDERS)}") from None
    return getter()


def list_registered() -> dict:
    """What this deployment actually has plugged in (GET /registered)."""
    return {
        "adapters": {kind: sorted(names) for kind, names in _ADAPTERS.items()},
        "surfaces": sorted(_SURFACES),
    }
