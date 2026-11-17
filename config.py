"""Resolve plugins from env -> the only place providers are chosen.

    LLM_PROVIDER=openai  RETRIEVER=indexed  RERANKER=none
    SURFACES=code_review,log_review
    TENANT_STORE=fs
    CRA_PLATFORM=auto    # auto | cuda | mac | cpu | studio -> HOW to run
    CRA_PROFILE=portable # portable | mac_resident | mac_max | studio -> WHICH models

`studio` is never auto-detected; it has to be named. The one provider NOT chosen
here is the checkpointer: `app/deps.py` picks Postgres when DATABASE_URL is set
and MemorySaver (development only, and it warns) otherwise.

Importing this module imports adapters/ and surfaces/ so every @adapter/@surface
decorator has run before anything asks the registry for a plug.
"""
from __future__ import annotations

import os
import platform as _platform
import shutil
import threading

import adapters  # noqa: F401 -- side effect: adapters self-register  # pylint: disable=unused-import
import registry
import surfaces  # noqa: F401 -- side effect: surfaces self-register  # pylint: disable=unused-import
from adapters.llm_lmstudio import base_url as lmstudio_base_url
from adapters.platforms import base as _profiles
from nodes._render import FENCE_MARKERS

DEFAULT_SURFACES = "code_review,log_review"


# Roles an LLM can be bound to. `diagnosis` and `judge` MUST resolve to
# different models — see ports.LLM and adapters/platforms/base.py.
LLM_ROLES = ("diagnosis", "judge", "utility")


# ── process-wide providers ───────────────────────────────────────────────────
# Providers whose value is a CACHE live once per process. The retriever holds the
# loaded index and the BM25 statistics; the cross-encoder entailer holds a model.
# Built fresh per run, both caches started empty every time: measured on the
# 17.6k-chunk Masterclass index, 1.46 s and ~360 MB of churn per review, against
# ~14 ms with one shared instance. Staleness is already handled inside them
# (index file stamp + fingerprint), and the namespace is still passed on every
# retrieve, so G1 is unchanged.
#
# Keyed on the adapter CLASS plus the env vars its construction reads, so a
# changed setting -- or a test pointing CRA_DATA_DIR somewhere new -- gets a new
# instance rather than a stale one.
_SHARED: dict[tuple, object] = {}
_SHARED_LOCK = threading.Lock()


def _shared(cls, *env_keys: str):
    """The one process-wide instance of `cls` for the current values of `env_keys`.

    `env_keys` must be every env var the class's constructor reads; one left out
    would hand back an instance built for the old setting. Double-checked under
    a lock so two threads starting reviews at once do not both pay the load.
    """
    key = (cls, *(os.getenv(k) for k in env_keys))
    inst = _SHARED.get(key)
    if inst is None:
        with _SHARED_LOCK:
            inst = _SHARED.get(key)
            if inst is None:
                inst = _SHARED[key] = cls()
    return inst


def model_server() -> str | None:
    """Where the LLM provider is served, for the audit record. None when the
    provider is not a server (stub) -- a node asking this instead of importing an
    adapter is the T5/G2 rule, and the record stops naming LM Studio for runs
    that never touched it."""
    return lmstudio_base_url() if os.getenv("LLM_PROVIDER", "lmstudio") == "lmstudio" else None


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
    # The role is passed so the adapter can apply per-role decoding settings
    # (adapters/llm_lmstudio.py::DEFAULT_TEMPERATURE). Every LLM adapter accepts
    # it. There used to be an `except TypeError` retry without `role`, which also
    # swallowed a TypeError raised INSIDE a constructor and silently dropped the
    # role -- a bug that looked like a working fallback.
    return cls(model=model_id, role=role)


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
    """The Retriever. Defaults to the REAL namespaced index (T10).

    The default used to be `hybrid` -- the four-document demo corpus -- which
    refuses any namespaced request, so every tenant that declares a codebase
    could not retrieve at all under the default configuration. Offline callers
    (the gate suite, the demo, the unit tests) now declare RETRIEVER=hybrid
    themselves, the same way they declare LLM_PROVIDER=stub: offline is a choice
    a caller makes, not a state the system defaults into.
    """
    return _shared(registry.get_adapter("retriever", os.getenv("RETRIEVER", "indexed")),
                   "CRA_DATA_DIR", "LMSTUDIO_BASE_URL", "EMBEDDER", "VECTOR_INDEX")


def get_reranker():
    """The Reranker (T13), or None when RERANKER=none -- the DEFAULT.

    OFF by default, on measurement (2026-11-17, `tools.calibrate rerank`, 176
    Masterclass queries, both instruments on the same 100-candidate pools):
    same-file vs other-file separation (Youden J) was 0.68 for the cosine and 0.31
    for bge-reranker-v2-m3; a same-file passage reached the top 3 in 98.9% of
    queries by fusion order and 81.8% after reranking; and it cost 13.5 s per
    query. "Same file" is a proxy that favours the cosine (the query IS the file's
    path and diff), but nothing measured shows the reranker earning its cost, so
    it is opt-in: RERANKER=cross_encoder.

    Shared per process like the retriever: the model is ~2.3 GB and takes ~4 s to
    load, which a CLI review would otherwise pay on every run.
    """
    name = os.getenv("RERANKER", "none")
    if name == "none":
        return None
    return _shared(registry.get_adapter("reranker", name), "RERANKER_MODEL",
                   "RERANK_MAX_LENGTH", "CRA_PLATFORM", "CRA_PROFILE")


def get_source(name: str):
    """A fresh EvidenceSource registered as `name`.

    Callers pass `source_name_for(surface)`: the surface's own choice, unless
    `CRA_SOURCE_<SURFACE>` overrides it on this machine.
    """
    return registry.get_adapter("source", name)()


def get_sink(name: str):
    """A fresh ActionSink registered as `name`.

    Callers pass `sink_name_for(surface)`: the surface's own choice, unless
    `CRA_SINK_<SURFACE>` overrides it on this machine.
    """
    return registry.get_adapter("sink", name)()


def source_name_for(surface) -> str:
    """The evidence source for a surface: its own choice unless this machine overrides.

    `CRA_SOURCE_<SURFACE>` (e.g. `CRA_SOURCE_CODE_REVIEW=git_local`) swaps WHERE
    evidence comes from without touching the surface, the same split as
    CRA_PLATFORM: the surface says what it needs, the deployment says where to
    get it. Per surface, not global, because one source cannot serve both a PR
    diff and a day of S3 logs -- a global override would break whichever surface
    it was not meant for.
    """
    return os.getenv(f"CRA_SOURCE_{surface.name.upper()}") or surface.source_name


def sink_name_for(surface) -> str:
    """The action sink for a surface, with `CRA_SINK_<SURFACE>` as the override.

    The override changes only WHERE an approved action lands. It cannot change
    whether one lands: `execute` still refuses without `approved is True`.
    """
    return os.getenv(f"CRA_SINK_{surface.name.upper()}") or surface.sink_name


def get_entailer():
    """The Entailer used by the gate's third grounding check (T22).

    Defaults to the LEXICAL adapter on purpose: the gate's booleans must be
    computable with no provider and no model download, or the safety suite stops
    being runnable everywhere. Set ENTAILER=cross_encoder for the stronger
    instrument once the model is present.
    """
    return _shared(registry.get_adapter("entailer", os.getenv("ENTAILER", "lexical")),
                   "ENTAILER_MODEL")


def get_detector():
    """The log-review Detector (G4). Rule-based, and that is the point.

    Detectors run over the FULL stream before any model sees anything, so they
    must be cheap enough to touch every line of a ~2M-line day. A model-backed
    detector could not, which is why this resolves to code.
    """
    return registry.get_adapter("detector", os.getenv("DETECTOR", "signals"))()


def get_input_guard():
    """The InputGuard every prompt's untrusted blocks pass through (G5/T12).

    Defaults to a REAL guard, not a no-op. `CRA_GUARD=null` has to be set
    deliberately, for the same reason the audit sink does: a data-governance
    boundary you get by remembering is not a boundary. Regex-based and
    model-free, so it also works with no provider reachable -- the same rule
    that keeps the lexical Entailer the default.

    The guard is wired with the fence markers so it can neutralise evidence that
    tries to close its own fence. `nodes/_render.py` owns those markers; passing
    them in keeps the adapter from importing the module it protects.
    """
    # Every guard accepts the markers (the null one ignores them). No `except
    # TypeError` fallback: that used to swallow a TypeError raised inside a
    # constructor and quietly build a guard with NO fence markers -- the
    # fence-forgery defence switched off by a typo.
    cls = registry.get_adapter("guard", os.getenv("CRA_GUARD", "regex"))
    return cls(fence_markers=FENCE_MARKERS)


def get_embedder():
    """The Embedder (T10). Defaults to the same LM Studio server the LLMs use.

    There is no stub default here, unlike the LLM and the entailer, and that is
    deliberate: an embedder that silently returned fake vectors would rebuild
    exactly the defect T10 exists to fix -- a retrieval score that looks like a
    similarity and is not. Code that needs to run offline uses the demo retriever
    instead of a fake embedder.
    """
    cls = registry.get_adapter("embedder", os.getenv("EMBEDDER", "lmstudio"))
    return cls()


def get_index():
    """The VectorIndex (T10). Local on-disk by default; no service required."""
    cls = registry.get_adapter("index", os.getenv("VECTOR_INDEX", "local"))
    return cls()


def get_tenant_store():
    """The TenantStore (policy + prompts per codebase). Filesystem by default."""
    return registry.get_adapter("tenant", os.getenv("TENANT_STORE", "fs"))()


def get_audit_sink():
    """One immutable record per run (G6). Defaults to append-only JSONL.

    Defaults to a REAL sink rather than a no-op: an agent that takes actions and
    keeps no record of them is the gap this closes, so silence has to be opted
    into (CRA_AUDIT=null) rather than being what you get by forgetting.
    """
    return registry.get_adapter("audit", os.getenv("CRA_AUDIT", "jsonl"))()


def get_tracer():
    """Spans per graph node and per LLM call, plus retries (T23). Append-only JSONL.

    ON by default, unlike most debugging tooling. The reason is specific: it was
    switched on just before the first live model runs, because a trace that has
    to be switched on after something looks wrong is a trace you do not have for
    the run that went wrong. T21 is the standing example.
    """
    return registry.get_adapter("tracer", os.getenv("CRA_TRACE", "jsonl"))()


def get_surface(name: str):
    """An instance of the named Surface. Raises for one that is not registered."""
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
    """WHICH models (CRA_PROFILE) -- orthogonal to CRA_PLATFORM, which is HOW they run."""
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
        "retriever": os.getenv("RETRIEVER", "indexed"),
        "reranker": os.getenv("RERANKER", "none"),
        "tenant_store": os.getenv("TENANT_STORE", "fs"),
        "audit": os.getenv("CRA_AUDIT", "jsonl"),
        "guard": os.getenv("CRA_GUARD", "regex"),
        "tracer": os.getenv("CRA_TRACE", "jsonl"),
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
