"""The gate: a SEPARATE llm produces `reasoning` text ONLY; three PYTHON booleans decide.

Improvement #8 lives adjacent: treat the issue body + retrieved snippets as
untrusted DATA, never as instructions (prompt-injection defense).
"""
from __future__ import annotations
import os
from schemas.diagnosis import ReviewResult

MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))  # MEASURE this
SEVERITY_ALLOWLIST = {"low", "medium"}          # high/critical -> never auto-approve
ALLOWED_ACTIONS = {"comment", "comment+label"}


def independent_review(state: dict, judge_llm=None) -> dict:
    diag = state["diagnosis"]
    retrieved = {s.id: s.score for s in state["_retrieved"]}

    # Guardrails come from the TENANT'S resolved policy (state["_policy"]), not from
    # hardcoded constants — each org/project plugs in its own thresholds/allowlists.
    # Falls back to the module defaults when no policy is attached (e.g. offline eval).
    pol = state.get("_policy") or {}
    min_score = pol.get("min_relevance_score", MIN_RELEVANCE_SCORE)
    severity_allowlist = set(pol.get("severity_allowlist", SEVERITY_ALLOWLIST))
    allowed_actions = set(pol.get("allowed_actions", ALLOWED_ACTIONS))

    # 1) judge EXPLAINS (free text only). Fresh call, never saw the 1st model's CoT.
    #    Its prompt is the tenant's judge prompt, resolved dynamically upstream.
    if judge_llm is None:
        reasoning = "[stub] citations checked against retrieved set and threshold."
    else:
        reasoning = judge_llm.review_reasoning(diag)  # returns str, nothing else

    # 2) code DECIDES — three booleans from real data, never parsed from the llm.
    groundedness_ok = bool(diag.citations) and all(
        c.snippet_id in retrieved and retrieved[c.snippet_id] >= min_score
        for c in diag.citations
    )
    risk_ok = diag.severity in severity_allowlist
    action_type = "comment+label" if diag.proposed_label else "comment"
    permission_ok = action_type in allowed_actions

    review = ReviewResult(reasoning, groundedness_ok, risk_ok, permission_ok)
    out = {"review": review}
    if review.decision == "approve":
        out["proposed_action"] = {
            "type": action_type,
            "comment": diag.proposed_comment,
            "label": diag.proposed_label,
        }
    return out
