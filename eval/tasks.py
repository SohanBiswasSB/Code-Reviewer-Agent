"""Safety-gate regression tasks. `pass^k` semantics: EVERY task must be correct —
one miss is a critical failure, never an average to improve.

Classes are deliberately balanced. A gate that rejects everything trivially
scores 100% and proves nothing, so roughly half of these cases *should* reach
`deterministic` / `approve`. (Anthropic's class-imbalance warning, applied to a
routing-and-gating suite rather than a capability one.)

No LLM is called anywhere in this file. That is the point: these tasks pin the
behaviour of `classify`, the three gate booleans and the `execute` guard, all of
which are plain Python. They must therefore produce identical results on the Mac
and the CUDA box, whatever models are installed.
"""
from __future__ import annotations

from schemas.diagnosis import Citation, Diagnosis
from schemas.evidence import CheckRun, IssueEvidence
from surfaces.code_review import CodeReviewSurface

# ── classify() routing ────────────────────────────────────────────────────────
# Each task is (name, state, expected_route). `state` is a full node input so the
# surface-delegation branch can be exercised the same way as the built-in one.
CLASSIFY_TASKS = [
    # -- built-in issue-triage precedence: failing CI must beat a sparse body
    ("ci_beats_sparse",
     {"evidence": IssueEvidence(url="u", title="t", body="x",
                                check_runs=[CheckRun("build", "failure")])},
     "deterministic"),
    ("sparse_no_ci",
     {"evidence": IssueEvidence(url="u", title="t", body="help")},
     "ai_investigation"),
    ("rich_no_ci",
     {"evidence": IssueEvidence(
         url="u", title="t",
         body="Full repro: steps to reproduce are 1,2,3 and expected/actual described " * 3)},
     "human_review"),

    # -- G2: when a Surface is injected, ITS rule-based classify must drive the
    # route. Untested before, so surfaces could silently stop being consulted.
    #
    # These take POST-normalise evidence: `touched_areas` is already filtered to
    # risky areas by CodeReviewSurface.normalize, so classify only asks whether
    # it is empty. (It used to filter inside classify, which meant the same field
    # name meant two different things at two points in the pipeline.)
    ("delegation_failing_ci",
     {"_surface": CodeReviewSurface(),
      "evidence": {"has_failing_ci": True, "touched_areas": [], "files": []}},
     "deterministic"),
    ("delegation_risky_area",
     {"_surface": CodeReviewSurface(),
      "evidence": {"touched_areas": ["auth"], "files": ["app/auth/views.py"]}},
     "ai_investigation"),
    ("delegation_trivial",
     {"_surface": CodeReviewSurface(),
      "evidence": {"touched_areas": [], "files": ["docs/readme.md"], "is_trivial": True}},
     "human_review"),
]

# ── the Surface's OWN normalize -> classify chain ─────────────────────────────
# (name, raw_provider_payload, expected_route)
#
# The tasks above hand-build post-normalise evidence, which pins classify but not
# the normaliser feeding it. That gap is exactly how the code_review surface came
# to receive issue-shaped evidence and route everything to human_review while
# every test stayed green (T5). These drive raw provider payloads through the
# surface's real normalize.
SURFACE_PIPELINE_TASKS = [
    ("pipeline_failing_ci_beats_risk",
     {"html_url": "u", "title": "fix auth", "files": [{"filename": "app/auth/views.py"}],
      "check_runs": [{"name": "build", "conclusion": "failure"}]},
     "deterministic"),
    ("pipeline_migration_is_risky",
     {"html_url": "u", "title": "add column",
      "files": [{"filename": "app/migrations/0002_add_col.py"}], "check_runs": []},
     "ai_investigation"),
    ("pipeline_celery_is_risky",
     {"html_url": "u", "title": "task tweak",
      "files": [{"filename": "worker/celery_tasks.py"}], "check_runs": []},
     "ai_investigation"),
    ("pipeline_docs_only_is_trivial",
     {"html_url": "u", "title": "typo",
      "files": [{"filename": "docs/setup.md"}, {"filename": "README.md"}],
      "check_runs": []},
     "human_review"),
    ("pipeline_ordinary_code_needs_a_human",
     {"html_url": "u", "title": "refactor helper",
      "files": [{"filename": "app/utils/strings.py"}], "check_runs": []},
     "human_review"),
]

# ── independent_review() gate ─────────────────────────────────────────────────
# (name, diagnosis, retrieved{id: score}, policy | None, expected_decision)
# policy=None exercises the module defaults (min 0.35, severity low|medium,
# actions comment|comment+label); a dict exercises a tenant's resolved policy.
_DEFAULT_ACTIONS = ["comment", "comment+label"]

GATE_TASKS = [
    # -- groundedness_ok
    ("grounded_approve",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, None, "approve"),
    ("weak_citation_escalate",                       # real id, below threshold
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.22)]),
     {"iss-101": 0.22}, None, "escalate_to_human"),
    ("phantom_citation_escalate",                    # id never retrieved
     Diagnosis("s", "medium", "c", None, [Citation("iss-999", 0.9)]),
     {"iss-101": 0.61}, None, "escalate_to_human"),
    ("no_citation_escalate",                         # cites nothing at all
     Diagnosis("s", "medium", "c", None, []),
     {"iss-101": 0.61}, None, "escalate_to_human"),

    # -- risk_ok
    ("high_severity_escalate",                       # grounded, still too risky
     Diagnosis("s", "critical", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, None, "escalate_to_human"),
    ("low_severity_approve",
     Diagnosis("s", "low", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, None, "approve"),

    # -- permission_ok  (T7: the third boolean had NO coverage at all)
    ("action_not_allowed_escalate",
     # proposed_label makes this a "comment+label" action; policy permits only
     # "comment", so the action allowlist must escalate it despite being
     # perfectly grounded and low severity.
     Diagnosis("s", "medium", "c", "needs-triage", [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, {"allowed_actions": ["comment"]}, "escalate_to_human"),
    ("action_allowed_approve",                       # balanced counterpart
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, {"allowed_actions": ["comment"]}, "approve"),
    ("label_action_allowed_approve",
     Diagnosis("s", "medium", "c", "needs-triage", [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, {"allowed_actions": _DEFAULT_ACTIONS}, "approve"),

    # -- the multi-tenant claim itself: ONE diagnosis, TWO policies, OPPOSITE
    # outcomes, purely from thresholds. docs/MULTI_TENANT.md asserts this; these
    # two tasks are what make the assertion testable rather than prose.
    ("same_diagnosis_lenient_policy_approves",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.36)]),
     {"iss-101": 0.36}, {"min_relevance_score": 0.35}, "approve"),
    ("same_diagnosis_strict_policy_escalates",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.36)]),
     {"iss-101": 0.36}, {"min_relevance_score": 0.40}, "escalate_to_human"),

    # -- a stricter tenant's severity allowlist
    ("tenant_severity_allowlist_escalates",
     Diagnosis("s", "medium", "c", None, [Citation("iss-101", 0.61)]),
     {"iss-101": 0.61}, {"severity_allowlist": ["low"]}, "escalate_to_human"),
]

# ── the gate driven by the REAL retriever ─────────────────────────────────────
# (name, query, expected_decision)
#
# Every GATE_TASK above hand-supplies its scores, which is correct for pinning
# the gate's LOGIC but means the suite never checks that the gate receives a
# meaningful INPUT. That blind spot hid T21: retrieval returned a rank-derived
# score that forced the top hit to 1.0 on every query, so no citation could ever
# fail the relevance check — and the suite stayed green throughout.
#
# These tasks close it by retrieving for real and citing whatever comes back. An
# off-topic query must produce an ungroundable diagnosis, which is the property
# the entire safety story rests on.
RETRIEVAL_GATE_TASKS = [
    ("real_retrieval_on_topic_approves",
     "OAuth callback returns 500 when state param missing", "approve"),
    ("real_retrieval_off_topic_escalates",
     "banana bread recipe with walnuts", "escalate_to_human"),
    ("real_retrieval_gibberish_escalates",
     "zzzz qqqq xyzzy plugh", "escalate_to_human"),
]
