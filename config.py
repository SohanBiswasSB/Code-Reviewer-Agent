"""Resolve plugins from env -> the only place providers are chosen.

    LLM_PROVIDER=openai  RETRIEVER=hybrid  CHECKPOINTER=postgres
    SURFACES=code_review,log_review
    TENANT_STORE=fs

Importing this module imports adapters/ and surfaces/ so every @adapter/@surface
decorator has run before anything asks the registry for a plug.
"""
from __future__ import annotations
import os
import registry
import adapters   # noqa: F401  -- import side effect: adapters self-register
import surfaces   # noqa: F401  -- import side effect: surfaces self-register

DEFAULT_SURFACES = "code_review,log_review"


def get_llm():
    return registry.get_adapter("llm", os.getenv("LLM_PROVIDER", "openai"))()


def get_retriever():
    return registry.get_adapter("retriever", os.getenv("RETRIEVER", "hybrid"))()


def get_source(name: str):
    """name comes from the active Surface (surface.source_name), not from env."""
    return registry.get_adapter("source", name)()


def get_sink(name: str):
    """name comes from the active Surface (surface.sink_name), not from env."""
    return registry.get_adapter("sink", name)()


def get_tenant_store():
    return registry.get_adapter("tenant", os.getenv("TENANT_STORE", "fs"))()


def get_surface(name: str):
    return registry.get_surface(name)()


def enabled_surfaces() -> list[str]:
    """Surfaces this deployment serves. A tenant's policy narrows it further."""
    return [s.strip() for s in os.getenv("SURFACES", DEFAULT_SURFACES).split(",") if s.strip()]


def registered() -> dict:
    """GET /registered — the plugs wired in, plus the env-selected defaults."""
    out = registry.list_registered()
    out["selected"] = {
        "llm": os.getenv("LLM_PROVIDER", "openai"),
        "retriever": os.getenv("RETRIEVER", "hybrid"),
        "tenant_store": os.getenv("TENANT_STORE", "fs"),
        "surfaces": enabled_surfaces(),
    }
    return out
