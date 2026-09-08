"""Tenant resolution: policy deep-merge and prompt fallback.

The multi-tenant promise in docs/MULTI_TENANT.md is that a codebase overrides
only what differs and inherits the rest. These tests pin the inheritance, since
a broken merge would silently apply _default's thresholds to a stricter tenant.
"""
import pytest

import config


@pytest.fixture
def store():
    return config.get_tenant_store()


def test_tenants_are_discovered(store):
    found = store.tenants()
    assert "milesone" in found and "project_a" in found and "project_b" in found
    assert "_default" not in found, "_default is the baseline, not a tenant"


def test_policy_inherits_default_and_overrides_only_the_delta(store):
    """project_a sets min_relevance_score but not severity_allowlist."""
    default = store.policy("_default", "code_review")
    project_a = store.policy("project_a", "code_review")

    assert default["min_relevance_score"] == 0.35
    assert project_a["min_relevance_score"] == 0.40          # overridden
    assert project_a["severity_allowlist"] == default["severity_allowlist"]  # inherited


def test_unknown_tenant_falls_back_entirely_to_default(store):
    """An unknown tenant must not explode; it gets the org baseline."""
    unknown = store.policy("no_such_codebase", "code_review")
    default = store.policy("_default", "code_review")
    assert unknown["min_relevance_score"] == default["min_relevance_score"]


def test_codebase_binding_is_exposed(store):
    """The EvidenceSource needs to know WHICH repo to read."""
    for tenant, vcs in (("project_a", "github"), ("project_b", "gitlab")):
        cb = store.policy(tenant, "code_review")["codebase"]
        assert cb["vcs"] == vcs
        assert cb["repo"]


def test_enabled_surfaces_is_per_tenant(store):
    """project_b runs code_review only; the API must refuse log_review for it."""
    assert "log_review" in store.policy("project_a", "log_review")["enabled_surfaces"]
    assert "log_review" not in store.policy("project_b", "code_review")["enabled_surfaces"]


def test_prompt_falls_back_to_default_when_tenant_has_no_override(store):
    """milesone overrides the diagnosis prompt but not the judge prompt."""
    tenant_diag = store.prompt("milesone", "code_review", "diagnosis")
    default_diag = store.prompt("_default", "code_review", "diagnosis")
    tenant_judge = store.prompt("milesone", "code_review", "judge")
    default_judge = store.prompt("_default", "code_review", "judge")

    assert tenant_judge == default_judge, "judge should be inherited verbatim"
    assert tenant_diag and default_diag


def test_missing_prompt_with_no_default_raises(store):
    with pytest.raises(FileNotFoundError):
        store.prompt("project_a", "code_review", "no_such_node")


def test_every_tenant_resolves_prompts_for_its_enabled_surfaces(store):
    """A tenant enabling a surface it has no prompts for would fail mid-run."""
    for tenant in store.tenants():
        pol = store.policy(tenant, "code_review")
        for surface in pol.get("enabled_surfaces", []):
            for node in ("diagnosis", "judge"):
                assert store.prompt(tenant, surface, node), f"{tenant}/{surface}/{node}"
