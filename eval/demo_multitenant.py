"""Each tenant is a CODEBASE (its own repo + prompts + rules). Escalate-to-human is
the SHARED safety net: BOTH escalate when a run isn't groundable; BOTH can approve
when it is. The per-codebase difference is prompts/risky_areas, not the safety route.

Run: `python -m eval.demo_multitenant`

## Why this script had to be rewritten (2026-09-10)

It raised `TypeError` and had done since T11. Three separate ways, each of which
is a lesson the repo has already learned somewhere else:

1. It built `Diagnosis("s", "medium", ...)` and `Citation("iss-1", 0.5)`
   **positionally**. T11 made the schemas Pydantic v2 models, which accept
   keyword arguments only.
2. It passed retrieved snippets as ad-hoc attribute objects
   (`type("S", (), {...})()`). The `Retriever` port returns **dicts**, and that
   is the only shape a node ever sees -- `independent_review` subscripts them.
3. Its snippets carried **no `text`**, so once T22 added the entailment check
   there was nothing to entail against and every run escalated regardless of
   policy -- which would have made the demo print the opposite of its own point.

That third one is the recurring lesson in `docs/DECISIONS.md` for the fifth
time: a fixture convenient enough to construct by hand stopped resembling the
real input, and nothing was red because nothing ran this file.
"""
import os
import sys

import config
from nodes.independent_review import independent_review
from schemas.diagnosis import Citation, Diagnosis

# Offline, like eval/harness.py. This script exists to show that POLICY decides,
# so it must not need a model server. Since 2026-09-11 `independent_review`
# resolves a judge from config rather than skipping the call when none is
# injected -- and the judge's text cannot affect the outcome anyway, because four
# Python booleans do. `setdefault`, so it can still be pointed at a real provider
# deliberately.
#
# Set after the imports, and that is enough: config reads these three only when
# a provider is resolved (inside its get_* functions), never at import time. They
# must precede the first `config.get_*` call -- the tenant store just below.
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("RETRIEVER", "hybrid")   # the demo corpus, not a live index
os.environ.setdefault("RERANKER", "none")

store = config.get_tenant_store()

# The judge prompt the node now requires (T6). Content is irrelevant here: the
# stub ignores it, and no judge output may influence the decision.
PROMPTS = {"diagnosis": "[demo] diagnosis prompt", "judge": "[demo] judge prompt"}

# The passage a real retrieval would return, and a claim that actually FOLLOWS
# from it. Both are needed: `entailment_ok` scores the claim against the text of
# the snippets it cites, so a demo with empty text cannot show an approval.
PASSAGE = ("The OAuth callback returns a 500 when the state parameter is missing "
           "from the redirect.")
CLAIM = "The OAuth callback returns a 500 when the state parameter is missing."


def run(tenant: str, cited_score: float):
    """Gate one identical diagnosis under `tenant`'s real code_review policy.

    Only the policy varies between tenants; the claim, its citation and the
    passage are fixed, and `cited_score` is both the citation's and the hit's
    score. Returns (codebase name, vcs, repo, ReviewResult).
    """
    diag = Diagnosis(
        summary="likely regression in the OAuth callback",
        severity="medium",
        proposed_comment=CLAIM,
        proposed_label="needs-triage",
        citations=[Citation(snippet_id="iss-1", score=cited_score)],
    )
    # Dicts, because that is what the Retriever port hands a node.
    retrieved = [{"id": "iss-1", "score": cited_score, "text": PASSAGE}]
    policy = store.policy(tenant, "code_review")
    review = independent_review({
        "diagnosis": diag, "_retrieved": retrieved, "_policy": policy,
        "_prompts": PROMPTS,
    })["review"]
    cb = policy["codebase"]
    return cb["name"], cb["vcs"], cb["repo"], review


TENANTS = ("project_a", "project_b")


def show(header: str, cited_score: float, expect: str) -> list[str]:
    """Print each tenant's decision and WHY; return what did not match `expect`."""
    print(header)
    wrong = []
    for tenant in TENANTS:
        name, vcs, repo, review = run(tenant, cited_score)
        print(f"  {name:12} [{vcs}:{repo:18}] -> {review.decision}")
        # Print WHY, not just the verdict. A gate that only says "escalated"
        # makes the human redo the diagnosis to find out what tripped.
        for failure in review.failures:
            print(f"               · {failure}")
        if review.decision != expect:
            wrong.append(f"{tenant}: expected {expect}, got {review.decision}")
    return wrong


def main() -> int:
    """Exit non-zero unless the demo shows its own point.

    It used to print and exit 0 whatever happened, so `check.py` could only catch
    a crash -- the "nothing ran it" defect this file was rewritten for, one level
    up: a demo whose output nobody checks can drift into showing the opposite of
    its claim while the gate stays green.
    """
    threshold = store.policy("project_a", "code_review")["min_relevance_score"]
    wrong = show(f"GROUNDED diagnosis (cited score 0.50 >= {threshold}):", 0.50, "approve")
    print()
    wrong += show(f"UNGROUNDED diagnosis (cited score 0.22 < {threshold}):"
                  "  <-- shared escalate-to-human", 0.22, "escalate_to_human")
    if wrong:
        print(f"\nDEMO DOES NOT HOLD: {wrong}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
