"""Plugin registry, plus a thin wrapper that makes the safety gates fail CI.

The gate suite lives in eval/harness.py and stays runnable on its own (no pytest,
no dependencies). This wrapper exists so `pytest` alone is enough to catch a
weakened gate — the two entry points must never disagree.
"""
import pytest

import config
import registry


def test_expected_plugs_are_registered():
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
    with pytest.raises(KeyError, match="no surface"):
        registry.get_surface("pr_risk_score")


def test_registered_report_includes_hardware():
    """GET /registered is how you find out what a deployment actually is."""
    out = config.registered()
    assert out["selected"]["platform"] in ("cuda", "mac", "cpu")
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
    """The 100%-forever bar. A single failure here is a critical regression."""
    from eval.harness import run

    assert run() == 0, "a safety gate weakened -- see the output above"
