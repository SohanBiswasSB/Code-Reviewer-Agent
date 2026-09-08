"""LLM role binding and the judge-independence invariant.

No model is called here. These tests cover the wiring that makes it *impossible*
to accidentally review a diagnosis with the model that produced it.
"""
import pytest

import config
from adapters import llm_lmstudio


@pytest.fixture(autouse=True)
def _pin_platform(monkeypatch):
    monkeypatch.setenv("CRA_PLATFORM", "cuda")
    monkeypatch.setenv("CRA_PROFILE", "portable")


def test_roles_bind_to_distinct_models():
    diagnosis = config.get_llm("diagnosis")
    judge = config.get_llm("judge")
    assert diagnosis.model != judge.model, (
        "same model for both roles: independent_review would grade its own output"
    )


def test_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown LLM role"):
        config.get_llm("reviewer")


def test_independence_assertion_passes_for_shipped_profiles():
    config.assert_judge_is_independent()


def test_independence_assertion_catches_a_collapsed_profile(monkeypatch):
    """Guard the guard: if a profile edit points both roles at one model, fail."""
    from adapters.platforms import base

    collapsed = base.ModelSet(
        diagnosis="same-model", judge="same-model", utility="u",
        weights_gb={"diagnosis": 1.0, "judge": 1.0},
    )
    monkeypatch.setitem(base.PROFILES, "collapsed", collapsed)
    monkeypatch.setattr(
        config.registry.get_adapter("platform", "cuda"), "profiles",
        ("portable", "collapsed"),
    )
    monkeypatch.setenv("CRA_PROFILE", "collapsed")
    with pytest.raises(ValueError, match="same model"):
        config.assert_judge_is_independent()


def test_diagnosis_runs_deterministic_by_default():
    """Its output feeds a schema and then a citation check; sampling noise there
    buys nothing and risks a malformed field."""
    assert config.get_llm("diagnosis").temperature == 0


def test_reachability_error_is_actionable():
    """`doctor` must be able to say WHAT to do, not just that a socket failed."""
    with pytest.raises(ConnectionError, match="lms server start"):
        llm_lmstudio.available_models("http://127.0.0.1:9/v1", timeout=0.2)


@pytest.mark.parametrize(
    "wanted,loaded,expected",
    [
        ("qwen3-8b", ["qwen3-8b"], "qwen3-8b"),                       # exact
        ("qwen3-8b", ["qwen/qwen3-8b"], "qwen/qwen3-8b"),             # publisher prefix
        ("qwen3-8b", ["qwen3-8b@q4_k_m"], "qwen3-8b@q4_k_m"),         # quant suffix
        ("qwen3-8b", ["gemma-4-e4b"], None),                          # absent
    ],
)
def test_model_id_matching_tolerates_lm_studio_naming(wanted, loaded, expected):
    """LM Studio reports ids that carry publisher and/or quantisation; a
    configured id that omits them must still match."""
    assert llm_lmstudio.resolves_to(wanted, loaded) == expected


def test_calling_a_model_without_the_client_says_so_clearly(monkeypatch):
    """The failure must name requirements-local.txt, not just ImportError."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ModuleNotFoundError("no module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    llm = config.get_llm("judge")
    with pytest.raises(ModuleNotFoundError, match="requirements-local.txt"):
        llm.complete("sys", "user")
