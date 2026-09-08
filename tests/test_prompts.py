"""Prompt composition (T16), prompt loading (T6) and the judge's evidence (T18).

T16 was a safety hole, not a hardening gap: prompt resolution fell back at FILE
level, so a tenant override replaced `_default` entirely. Every override in this
repo is a short emphasis note that merely *mentions* the org contract in prose,
so once prompts were actually loaded the three overriding tenants would each have
sent a model a priority list with no grounding rule, no untrusted-input rule, no
no-verdict rule and no output schema — while `_default`, which nobody overrides,
kept all of them.

The tests below are what make that impossible to regress into.
"""
import json
import pathlib

import pytest

import config
from adapters.tenant_fs import FsTenantStore

# Clauses that carry the actual guardrails in the org contract. If a tenant's
# resolved prompt loses any of these, the boundary is gone for that codebase.
ORG_GUARDRAIL_CLAUSES = {
    "grounding": "Never invent",
    "untrusted_input": "untrusted DATA",
    "no_verdict": "output an approval",
    "output_schema": "JSON only, matching this shape",
}

OVERRIDING_TENANTS = ["milesone", "project_a", "project_b"]


@pytest.fixture
def store():
    return config.get_tenant_store()


# ── T16: composition ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("tenant", ["_default", *OVERRIDING_TENANTS])
@pytest.mark.parametrize("clause", sorted(ORG_GUARDRAIL_CLAUSES))
def test_every_tenant_keeps_every_org_guardrail(store, tenant, clause):
    """The invariant: tenants tune VALUES, they never remove the BOUNDARY."""
    text = store.prompt(tenant, "code_review", "diagnosis")
    assert ORG_GUARDRAIL_CLAUSES[clause] in text, (
        f"{tenant} lost the {clause} guardrail — a tenant override must extend "
        f"the org contract, never replace it"
    )


@pytest.mark.parametrize("tenant,emphasis", [
    ("milesone", "Zoom poll-ingest"),
    ("project_a", "billing"),
    ("project_b", "concurrency"),
])
def test_the_tenants_own_emphasis_survives(store, tenant, emphasis):
    """Composition must not be a fancy way of ignoring the override."""
    assert emphasis in store.prompt(tenant, "code_review", "diagnosis")


def test_an_override_is_strictly_longer_than_the_contract(store):
    """Cheap structural check that the delta was added, not substituted."""
    org = store.prompt("_default", "code_review", "diagnosis")
    for tenant in OVERRIDING_TENANTS:
        assert len(store.prompt(tenant, "code_review", "diagnosis")) > len(org)


def test_a_tenant_without_an_override_gets_the_contract_verbatim(store):
    """milesone overrides diagnosis but not judge."""
    assert (store.prompt("milesone", "code_review", "judge")
            == store.prompt("_default", "code_review", "judge"))


def test_include_directive_controls_placement(tmp_path):
    """`{{include _default}}` lets a tenant put the contract where it wants."""
    for tenant, body in [("_default", "ORG-BODY"),
                         ("acme", "TOP\n{{include _default}}\nBOTTOM")]:
        d = tmp_path / tenant / "prompts" / "code_review"
        d.mkdir(parents=True)
        (d / "diagnosis.md").write_text(body, encoding="utf-8")
    got = FsTenantStore(root=str(tmp_path)).prompt("acme", "code_review", "diagnosis")
    assert got == "TOP\nORG-BODY\nBOTTOM"


def test_missing_prompt_everywhere_raises(store):
    with pytest.raises(FileNotFoundError):
        store.prompt("project_a", "code_review", "no_such_node")


def test_provenance_reports_what_contributed(store):
    assert store.prompt_provenance("_default", "code_review", "diagnosis") == \
        {"org_contract": True, "tenant_delta": False}
    assert store.prompt_provenance("milesone", "code_review", "diagnosis") == \
        {"org_contract": True, "tenant_delta": True}


# ── T6 / T18: the prompt reaches the model, and the judge sees the evidence ──
class _Recorder:
    """An LLM that records what it was sent instead of calling anything."""

    def __init__(self, reply):
        self.reply, self.model, self.calls = reply, "fake", []

    def complete(self, system, user, *, json_schema=None):
        self.calls.append({"system": system, "user": user})
        return self.reply


class _FixedRetriever:
    def __init__(self):
        self.seen = {}

    def retrieve(self, query, k=5, namespace=None):
        self.seen = {"query": query, "namespace": namespace}
        return [{"id": "iss-101", "text": "OAuth 500 state param auth middleware",
                 "score": 0.61, "fusion": 0.03, "rank": 1}]


RAW_PR = {
    "html_url": "u",
    "title": "OAuth callback returns 500 when state param missing",
    "body": "auth middleware",
    "files": [{"filename": "app/auth/middleware.py"}],
    "check_runs": [],
    "diff": "+ if not state: return 500",
}

DIAGNOSIS_JSON = json.dumps({
    "summary": "auth regression", "severity": "medium",
    "proposed_comment": "see prior auth fix", "proposed_label": "needs-triage",
    # A model-supplied score, which must be IGNORED in favour of the retriever's.
    "citations": [{"snippet_id": "iss-101", "score": 0.99}],
})


@pytest.fixture
def resolved_state():
    from nodes.normalize_evidence import normalize_evidence
    from nodes.resolve_tenant import resolve_tenant
    from surfaces.code_review import CodeReviewSurface

    state = {"tenant": "milesone", "surface": "code_review", "target": "u",
             "_raw": RAW_PR, "_surface": CodeReviewSurface()}
    state.update(resolve_tenant(state))
    state.update(normalize_evidence(state))
    return state


def test_prompts_are_pinned_into_state(resolved_state):
    """Resolved once at run start so a resumed run cannot mix versions (G8)."""
    assert sorted(resolved_state["_prompts"]) == ["diagnosis", "judge"]
    assert resolved_state["_prompt_version"]
    assert "Zoom poll-ingest" in resolved_state["_prompts"]["diagnosis"]


def test_the_pinned_prompt_is_what_gets_sent(resolved_state):
    from nodes.generate_diagnosis import generate_diagnosis

    llm = _Recorder(DIAGNOSIS_JSON)
    generate_diagnosis(resolved_state, llm=llm, retriever=_FixedRetriever())
    assert llm.calls[0]["system"] == resolved_state["_prompts"]["diagnosis"]


def test_evidence_goes_in_the_user_turn_and_is_fenced(resolved_state):
    """System content is trusted operator instruction; a diff is not. Fencing
    gives an injected "ignore the above" a visible boundary to be inside of.

    Asserts on the CONTENT, not on the fence marker: the org contract legitimately
    *mentions* the marker while explaining the convention, so checking for the
    string alone would fail on a correct prompt. The property that matters is that
    the untrusted material itself never reaches the system turn.
    """
    from nodes.generate_diagnosis import generate_diagnosis

    llm = _Recorder(DIAGNOSIS_JSON)
    generate_diagnosis(resolved_state, llm=llm, retriever=_FixedRetriever())
    call = llm.calls[0]

    diff_body = RAW_PR["diff"]
    assert diff_body in call["user"], "the change must reach the model"
    assert diff_body not in call["system"], (
        "untrusted evidence must never enter the system turn, which carries "
        "trusted operator authority"
    )
    # and it must be fenced where it does appear
    assert "UNTRUSTED_EVIDENCE" in call["user"]


def test_the_org_contract_is_free_of_any_project_specifics():
    """The regression guard for the leak this split fixed.

    `_default` is shared by every codebase, so a project's stack, service or
    vendor names must not appear in it. They had: the default code-review prompt
    opened "You are a senior reviewer of the MilesOne Django + Celery backend",
    so every OTHER project was told it was reviewing MilesOne. Cheap to check,
    and invisible without a check.
    """
    project_vocabulary = ["milesone", "credly", "zoom", "django", "celery",
                          "masterclass", "cloudwatch"]
    for path in pathlib.Path("tenants/_default/prompts").rglob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        found = [word for word in project_vocabulary if word in text]
        assert not found, (
            f"{path} is the shared org contract but names {found} — move that "
            f"into the owning tenant's overlay"
        )


def test_the_namespace_reaches_the_retriever(resolved_state):
    """G1: milesone is bound to a codebase, so retrieval must be scoped to it."""
    from nodes.generate_diagnosis import generate_diagnosis

    retriever = _FixedRetriever()
    generate_diagnosis(resolved_state, llm=_Recorder(DIAGNOSIS_JSON), retriever=retriever)
    assert retriever.seen["namespace"] == "MilesEducation-Tech/Miles-USP-Backend"


def test_a_model_supplied_citation_score_is_ignored(resolved_state):
    """The gate compares against this number, so a model-supplied score would be
    trivially gameable. It comes from the retriever or not at all."""
    from nodes.generate_diagnosis import generate_diagnosis

    out = generate_diagnosis(resolved_state, llm=_Recorder(DIAGNOSIS_JSON),
                             retriever=_FixedRetriever())
    assert out["diagnosis"].citations[0].score == 0.61      # retriever's
    assert out["diagnosis"].citations[0].score != 0.99      # not the model's


def test_missing_severity_defaults_to_high_not_low(resolved_state):
    """A malformed response must escalate, not auto-approve. `risk_ok` only
    passes allowlisted severities, so defaulting high fails in the safe
    direction."""
    from nodes.generate_diagnosis import generate_diagnosis

    reply = json.dumps({"summary": "x", "proposed_comment": "c",
                        "citations": [{"snippet_id": "iss-101"}]})
    out = generate_diagnosis(resolved_state, llm=_Recorder(reply),
                             retriever=_FixedRetriever())
    assert out["diagnosis"].severity == "high"


def test_the_judge_receives_the_retrieved_evidence(resolved_state):
    """T18: its prompt tells it to verify citation ids against RETRIEVED. With
    only the diagnosis it could not, so any reasoning about grounding was
    invented — and shown to a human approver as though it were real."""
    from nodes.generate_diagnosis import generate_diagnosis
    from nodes.independent_review import independent_review

    resolved_state.update(generate_diagnosis(
        resolved_state, llm=_Recorder(DIAGNOSIS_JSON), retriever=_FixedRetriever()))

    judge = _Recorder("reads well grounded")
    out = independent_review(resolved_state, judge_llm=judge)

    call = judge.calls[0]
    assert call["system"] == resolved_state["_prompts"]["judge"]
    assert "id=iss-101" in call["user"], "the judge must SEE what was retrieved"
    assert "DIAGNOSIS_UNDER_REVIEW" in call["user"]
    # and its text remains advisory: the decision is still computed
    assert out["review"].reasoning == "reads well grounded"
    assert out["review"].decision in ("approve", "escalate_to_human")


def test_an_llm_node_refuses_to_run_without_a_pinned_prompt():
    """Rather than silently sending an empty or hardcoded system prompt."""
    from nodes.independent_review import independent_review
    from schemas.diagnosis import Citation, Diagnosis

    state = {
        "diagnosis": Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.61)]),
        "_retrieved": [{"id": "iss-101", "text": "t", "score": 0.61}],
    }
    with pytest.raises(ValueError, match="no judge prompt in state"):
        independent_review(state, judge_llm=_Recorder("x"))


def test_prompt_files_on_disk_are_reachable_for_every_enabled_surface(store):
    """A tenant enabling a surface with no prompts would fail mid-run."""
    assert pathlib.Path("tenants/_default/prompts/code_review/diagnosis.md").exists()
    for tenant in store.tenants():
        for surface in store.policy(tenant, "code_review").get("enabled_surfaces", []):
            for node in ("diagnosis", "judge"):
                assert store.prompt(tenant, surface, node)
