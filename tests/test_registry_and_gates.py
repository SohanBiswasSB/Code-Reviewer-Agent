"""Plugin registry, plus a thin wrapper that makes the safety gates fail CI.

The gate suite lives in eval/harness.py and stays runnable on its own (no pytest,
no dependencies). This wrapper exists so `pytest` alone is enough to catch a
weakened gate — the two entry points must never disagree.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import config
import registry

ROOT = Path(__file__).resolve().parent.parent


def test_expected_plugs_are_registered():
    """Every shipped adapter and surface registers itself on import (the plug works)."""
    reg = registry.list_registered()
    assert "lmstudio" in reg["adapters"]["llm"]
    assert "hybrid" in reg["adapters"]["retriever"]
    assert {"github", "s3logs"} <= set(reg["adapters"]["source"])
    assert {"pr_comment", "report"} <= set(reg["adapters"]["sink"])
    assert {"cuda", "mac", "cpu"} <= set(reg["adapters"]["platform"])
    assert {"code_review", "log_review"} <= set(reg["surfaces"])


def test_unknown_adapter_error_lists_what_is_available():
    """A typo'd provider should tell you the options, not just fail."""
    with pytest.raises(KeyError, match="have:"):
        registry.get_adapter("llm", "gpt5-turbo-ultra")


def test_unknown_surface_error_lists_what_is_available():
    """A typo'd surface names itself as missing, rather than a bare KeyError."""
    with pytest.raises(KeyError, match="no surface"):
        registry.get_surface("pr_risk_score")


def test_registered_report_includes_hardware():
    """GET /registered is how you find out what a deployment actually is.

    `embedding_dims` stays a LITERAL rather than a reference to the constant, on
    purpose. Asserting `== base.EMBEDDING_DIMS` would be tautological and could
    never fail; the literal is a tripwire, because changing the embedder silently
    re-scales every min_relevance_score in tenants/*/policy.json. Updating this
    number is meant to be part of the cost of changing the pin — which is exactly
    what it caught when nomic (768) was replaced by bge-m3, and again when bge-m3
    turned out to be unobtainable through LM Studio and was replaced by
    EmbeddingGemma-300m (2026-09-11, back to 768).

    This value is now MEASURED rather than reasoned: POST /v1/embeddings against a
    running server returns a 768-vector for the pinned id. Both previous values
    here were assumed.
    """
    out = config.registered()
    assert out["selected"]["platform"] in ("cuda", "mac", "cpu", "studio")
    assert out["hardware"]["models"]["diagnosis"]
    assert out["hardware"]["embedding_dims"] == 768


def test_retriever_refuses_a_namespace_it_cannot_honour():
    """G1: silently ignoring `namespace` would look like tenant isolation while
    retrieving across codebases. The demo corpus is unpartitioned, so it refuses."""
    retriever = config.get_retriever()
    with pytest.raises(NotImplementedError, match="namespace-scoped"):
        retriever.retrieve("anything", k=3, namespace="project_a")


def test_retriever_returns_scores():
    """The grounding gate reads .score; a retriever without it breaks the gate."""
    hits = config.get_retriever().retrieve("auth middleware 500", k=3)
    assert hits and all("score" in h and "id" in h for h in hits)


def test_safety_gates_all_pass():
    """The 100%-forever bar. A single failure here is a critical regression.

    Run as `python -m eval.harness` in its own process -- the exact command
    check.py runs -- rather than imported. Importing the harness `setdefault`s
    LLM_PROVIDER, RETRIEVER, RERANKER and CRA_ACTIVITY in os.environ for the rest
    of the process, so a top-of-file import would switch the activity log to
    `null` for every test collected after this module, and tests of the default
    would quietly be testing the stub. The child inherits this session's offline
    env (conftest's LLM_PROVIDER=stub, RETRIEVER=hybrid, the tmp CRA_DATA_DIR).
    """
    proc = subprocess.run([sys.executable, "-m", "eval.harness"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"a safety gate weakened:\n{proc.stdout}\n{proc.stderr}")
