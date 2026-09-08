"""Resolve plugins from env -> the only place providers are chosen.

    LLM_PROVIDER=openai  RETRIEVER=hybrid  CHECKPOINTER=postgres
    SURFACES=code_review,log_review
    TENANT_STORE=fs
    CRA_PLATFORM=auto    # auto | cuda | mac | cpu   -> HOW to run
    CRA_PROFILE=portable # portable | mac_headroom   -> WHICH models

Importing this module imports adapters/ and surfaces/ so every @adapter/@surface
decorator has run before anything asks the registry for a plug.
"""
from __future__ import annotations

import os
import platform as _platform
import shutil

import adapters  # noqa: F401  -- import side effect: adapters self-register
import registry
import surfaces  # noqa: F401  -- import side effect: surfaces self-register
from adapters.platforms import base as _profiles

DEFAULT_SURFACES = "code_review,log_review"


# Roles an LLM can be bound to. `diagnosis` and `judge` MUST resolve to
# different models — see ports.LLM and adapters/platforms/base.py.
LLM_ROLES = ("diagnosis", "judge", "utility")


def get_llm(role: str = "diagnosis"):
    """An LLM bound to the model this profile assigns to `role`.

    Takes a role rather than returning one shared client, because the design
    needs two DIFFERENT models: independent_review is only independent if the
    judge never shared a reasoning trace — or a model — with the diagnosis.
    Handing back one client for both would make that impossible to guarantee at
    the call site.
    """
    if role not in LLM_ROLES:
        raise ValueError(f"unknown LLM role '{role}'. have: {list(LLM_ROLES)}")
    models = get_models()
    model_id = models.get(role)
    if not model_id:
        raise KeyError(f"profile '{get_profile()}' defines no model for role '{role}'")
    cls = registry.get_adapter("llm", os.getenv("LLM_PROVIDER", "lmstudio"))
    return cls(model=model_id, runtime=get_platform().runtime(get_profile()))


def assert_judge_is_independent() -> None:
    """Fail loudly if the active profile points diagnosis and judge at one model.

    Cheap, but it guards the invariant the whole review step rests on, and it is
    the kind of thing a well-meaning profile edit breaks silently.
    """
    models = get_models()
    if models["diagnosis"] == models["judge"]:
        raise ValueError(
            f"profile '{get_profile()}' uses the same model "
            f"({models['diagnosis']}) for diagnosis and judge. independent_review "
            f"would be reviewing its own output."
        )


def get_retriever():
    return registry.get_adapter("retriever", os.getenv("RETRIEVER", "hybrid"))()


def get_source(name: str):
    """name comes from the active Surface (surface.source_name), not from env."""
    return registry.get_adapter("source", name)()


def get_sink(name: str):
    """name comes from the active Surface (surface.sink_name), not from env."""
    return registry.get_adapter("sink", name)()


def get_entailer():
    """The Entailer used by the gate's third grounding check (T22).

    Defaults to the LEXICAL adapter on purpose: the gate's booleans must be
    computable with no provider and no model download, or the safety suite stops
    being runnable everywhere. Set ENTAILER=cross_encoder for the stronger
    instrument once the model is present.
    """
    return registry.get_adapter("entailer", os.getenv("ENTAILER", "lexical"))()


def get_tenant_store():
    return registry.get_adapter("tenant", os.getenv("TENANT_STORE", "fs"))()


def get_surface(name: str):
    return registry.get_surface(name)()


def enabled_surfaces() -> list[str]:
    """Surfaces this deployment serves. A tenant's policy narrows it further."""
    return [s.strip() for s in os.getenv("SURFACES", DEFAULT_SURFACES).split(",") if s.strip()]


# ── hardware ─────────────────────────────────────────────────────────────────
def detect_platform() -> str:
    """Best-effort hardware detection, stdlib only (no torch import).

    Deliberately dependency-free: the eval suite and CI must be able to resolve a
    platform without installing the local-model stack.
    """
    if _platform.system() == "Darwin" and _platform.machine() == "arm64":
        return "mac"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def get_platform():
    """The active hardware profile. `CRA_PLATFORM=auto` (default) detects it.

    Nothing outside adapters/platforms/ may branch on hardware — everything else
    reads runtime()/models() off this object.
    """
    name = os.getenv("CRA_PLATFORM", "auto")
    if name == "auto":
        name = detect_platform()
    return registry.get_adapter("platform", name)()


def get_profile() -> str:
    return os.getenv("CRA_PROFILE", _profiles.DEFAULT_PROFILE)


def get_models() -> dict:
    """role -> model id, for the active platform + profile.

    Raises if the requested profile cannot fit this hardware, rather than quietly
    substituting a smaller one (which would mislabel every eval result).
    """
    return get_platform().models(get_profile())


def registered() -> dict:
    """GET /registered — the plugs wired in, plus the env-selected defaults."""
    out = registry.list_registered()
    plat = get_platform()
    out["selected"] = {
        "llm": os.getenv("LLM_PROVIDER", "lmstudio"),
        "retriever": os.getenv("RETRIEVER", "hybrid"),
        "tenant_store": os.getenv("TENANT_STORE", "fs"),
        "surfaces": enabled_surfaces(),
        "platform": plat.name,
        "profile": get_profile(),
    }
    out["hardware"] = {
        "detected": detect_platform(),
        "runtime": plat.runtime(get_profile()),
        "models": get_models(),
        "embedding_dims": _profiles.EMBEDDING_DIMS,
    }
    return out
