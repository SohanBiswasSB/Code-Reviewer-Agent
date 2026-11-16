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


# ── T27: a tenant's risky_areas actually route ───────────────────────────────
def _route_for(tenant, filename):
    from nodes.classify import classify
    from nodes.normalize_evidence import normalize_evidence
    from surfaces.code_review import CodeReviewSurface
    policy = config.get_tenant_store().policy(tenant, "code_review")
    state = {"_surface": CodeReviewSurface(), "_policy": policy,
             "_raw": {"html_url": "u", "title": "t", "files": [{"filename": filename}],
                      "check_runs": []}}
    state.update(normalize_evidence(state))
    return classify(state)["route"]


def test_a_tenant_only_risky_area_reaches_the_investigation():
    """`subscription` is risky in Masterclass and in no default list."""
    assert _route_for("masterclass", "Subscriptions/views.py") == "ai_investigation"


def test_another_tenant_does_not_inherit_it():
    assert _route_for("_default", "Subscriptions/views.py") == "human_review"


def test_a_tenant_cannot_make_a_default_area_routine():
    """A union, not a replacement: _template lists only migrations+auth, and a
    celery change is still investigated."""
    from surfaces.code_review import risky_areas_for
    assert "celery" in risky_areas_for({"risky_areas": ["migrations"]})
    assert _route_for("project_b", "worker/celery_tasks.py") == "ai_investigation"


# ── G11: the gate-relevant policy keys are validated ─────────────────────────
from schemas.policy import policy_errors  # noqa: E402


def test_every_shipped_policy_is_valid():
    store = config.get_tenant_store()
    for t in [*store.tenants(), "_default"]:
        for surface in ("code_review", "log_review"):
            assert policy_errors(store.policy(t, surface)) == [], (t, surface)


@pytest.mark.parametrize("bad, fragment", [
    ({"min_relevance_score": "0.75"}, "must be a number"),
    ({"min_relevance_score": 75}, "between 0 and 1"),
    ({"min_rerank_score": True}, "must be a number"),
    ({"severity_allowlist": "low"}, "SUBSTRING"),
    ({"severity_allowlist": ["low", "meduim"]}, "unknown severities"),
    ({"risky_areas": ["auth", 3]}, "list of strings"),
])
def test_a_malformed_value_is_named(bad, fragment):
    assert any(fragment in e for e in policy_errors(bad))


def test_resolve_tenant_refuses_an_invalid_policy():
    from errors import ConfigError
    from nodes.resolve_tenant import resolve_tenant

    class Store:
        def policy(self, tenant, surface):
            return {"enabled_surfaces": ["code_review"], "severity_allowlist": "low"}

        def prompt(self, *a):
            return "p"
    with pytest.raises(ConfigError, match="severity_allowlist"):
        resolve_tenant({"surface": "code_review", "tenant": "x"}, store=Store())
