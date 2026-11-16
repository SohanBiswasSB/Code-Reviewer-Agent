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

# Offline, like eval/harness.py. This script exists to show that POLICY decides,
# so it must not need a model server. Since 2026-09-11 `independent_review`
# resolves a judge from config rather than skipping the call when none is
# injected -- and the judge's text cannot affect the outcome anyway, because four
# Python booleans do. `setdefault`, so it can still be pointed at a real provider
# deliberately.
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("RETRIEVER", "hybrid")   # the demo corpus, not a live index

import config  # noqa: E402  -- must follow the env default above
from nodes.independent_review import independent_review  # noqa: E402
from schemas.diagnosis import Citation, Diagnosis  # noqa: E402

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


def show(header: str, cited_score: float) -> None:
    print(header)
    for tenant in ("project_a", "project_b"):
        name, vcs, repo, review = run(tenant, cited_score)
        print(f"  {name:12} [{vcs}:{repo:18}] -> {review.decision}")
        # Print WHY, not just the verdict. A gate that only says "escalated"
        # makes the human redo the diagnosis to find out what tripped.
        for failure in review.failures:
            print(f"               · {failure}")


threshold = store.policy("project_a", "code_review")["min_relevance_score"]
show(f"GROUNDED diagnosis (cited score 0.50 >= {threshold}):", 0.50)
print()
show(f"UNGROUNDED diagnosis (cited score 0.22 < {threshold}):"
     "  <-- shared escalate-to-human", 0.22)
