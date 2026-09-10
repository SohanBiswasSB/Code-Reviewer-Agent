"""The gate: a SEPARATE llm produces `reasoning` text ONLY; PYTHON booleans decide.

Four booleans now, not three. The design calls for grounding to be checked in
three steps and only two were implemented:

  1. the cited id is real                  -> groundedness_ok (was present)
  2. its relevance clears a calibrated bar -> groundedness_ok (was present)
  3. the claim FOLLOWS from the passage    -> entailment_ok   (T22, new)

Steps 1 and 2 together prove only that a real, on-topic snippet was cited. They
cannot catch a claim that overreaches what its passage supports — and a citation
is exactly what makes such a claim look trustworthy. That is the most common RAG
failure, and it was ungated.

`risk_ok` and `permission_ok` are unchanged.

Evidence handling: the diagnosis text and the retrieved snippets are untrusted
DATA and are fenced before they reach the judge (see nodes/_render.py).
"""
from __future__ import annotations

import os

from nodes._render import render_for_judge
from nodes._surface import active_surface
from schemas.diagnosis import ReviewResult

MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))  # MEASURE this
MIN_ENTAILMENT_SCORE = float(os.getenv("MIN_ENTAILMENT_SCORE", "0.25"))
SEVERITY_ALLOWLIST = {"low", "medium"}          # high/critical -> never auto-approve
ALLOWED_ACTIONS = {"comment", "comment+label"}


def _action_type(state: dict, diag) -> str:
    """What KIND of action this diagnosis proposes, for the permission check.

    Asked of the Surface, because the answer is surface-specific: code review
    proposes a comment (optionally with a label), log review proposes a report.

    Hardcoding the comment shape here made `permission_ok` FALSE for every
    log-review run — every log-review policy allows only `["report"]` — so that
    surface escalated 100% of the time and its ActionSink was unreachable. It
    failed safe, and it made the whole surface inert, which is the same shape of
    defect as T5.

    Falls back to the comment shape when no surface is active, so the gate stays
    computable from a bare state dict — which eval/harness.py depends on.
    """
    surface = active_surface(state)
    fn = getattr(surface, "action_type", None) if surface is not None else None
    if callable(fn):
        return fn(diag)
    return "comment+label" if diag.proposed_label else "comment"


def _entailment(claim: str, cited_ids: list[str], retrieved: list[dict],
                entailer, min_score: float) -> tuple[bool, float]:
    """Does the claim follow from the passages it cites?

    Scored against the concatenation of the CITED passages only — not the whole
    retrieved set. A claim is supposed to follow from what it points at; letting
    it borrow support from snippets it never cited would defeat the check.

    Fail-closed on an unverifiable claim: no cited text means nothing to check
    against, and a grounding check that passes when it cannot verify is worse
    than no check, because it produces a false assurance.
    """
    passages = " ".join(
        s.get("text", "") for s in retrieved if s.get("id") in set(cited_ids)
    )
    score = entailer.score(claim, passages)
    return score >= min_score, score


def independent_review(state: dict, judge_llm=None, entailer=None) -> dict:
    diag = state["diagnosis"]
    # The Retriever PORT returns dicts, so that is the only shape a node ever
    # sees -- tools/retrieval.py's Snippet stays behind its adapter. "score"
    # is the calibrated relevance, never the rank-fusion value (T21).
    retrieved_list = list(state["_retrieved"])
    retrieved = {s["id"]: s["score"] for s in retrieved_list}

    # Guardrails come from the TENANT'S resolved policy (state["_policy"]), not
    # from hardcoded constants -- each codebase plugs in its own thresholds and
    # allowlists. Falls back to the module defaults when no policy is attached
    # (offline eval).
    pol = state.get("_policy") or {}
    min_score = pol.get("min_relevance_score", MIN_RELEVANCE_SCORE)
    min_entail = pol.get("min_entailment_score", MIN_ENTAILMENT_SCORE)
    severity_allowlist = set(pol.get("severity_allowlist", SEVERITY_ALLOWLIST))
    allowed_actions = set(pol.get("allowed_actions", ALLOWED_ACTIONS))

    # 1) judge EXPLAINS (free text only). Fresh call, never saw the 1st model's CoT.
    if judge_llm is None:
        reasoning = "[stub] citations checked against retrieved set and threshold."
    else:
        # The TENANT's judge prompt, pinned by resolve_tenant (T6).
        system = (state.get("_prompts") or {}).get("judge")
        if not system:
            raise ValueError(
                "independent_review: no judge prompt in state. resolve_tenant "
                "should have pinned it before this node ran."
            )
        # T18: the judge receives the SAME retrieved evidence, because its prompt
        # instructs it to verify citation ids against RETRIEVED. Given only the
        # diagnosis it could not perform that check, so its reasoning about
        # grounding was necessarily invented — and shown to a human approver as
        # though it were real.
        reasoning = judge_llm.complete(system, render_for_judge(diag, retrieved_list))

    # 2) code DECIDES — booleans from real data, never parsed from the llm.
    failures: list[str] = []

    cited_ids = [c.snippet_id for c in diag.citations]
    groundedness_ok = bool(diag.citations) and all(
        cid in retrieved and retrieved[cid] >= min_score for cid in cited_ids
    )
    if not groundedness_ok:
        if not diag.citations:
            failures.append("no citations")
        else:
            phantom = [c for c in cited_ids if c not in retrieved]
            weak = [c for c in cited_ids
                    if c in retrieved and retrieved[c] < min_score]
            if phantom:
                failures.append(f"citations not retrieved: {phantom}")
            if weak:
                failures.append(
                    f"citations below relevance {min_score}: {weak}")

    risk_ok = diag.severity in severity_allowlist
    if not risk_ok:
        failures.append(f"severity {diag.severity!r} not in {sorted(severity_allowlist)}")

    action_type = _action_type(state, diag)
    permission_ok = action_type in allowed_actions
    if not permission_ok:
        failures.append(f"action {action_type!r} not in {sorted(allowed_actions)}")

    if entailer is None:
        import config
        entailer = config.get_entailer()
    entailment_ok, entail_score = _entailment(
        diag.proposed_comment, cited_ids, retrieved_list, entailer, min_entail)
    if not entailment_ok:
        failures.append(
            f"claim not entailed by its cited passages (score {entail_score} < {min_entail})")

    review = ReviewResult(
        reasoning=reasoning,
        groundedness_ok=groundedness_ok,
        risk_ok=risk_ok,
        permission_ok=permission_ok,
        entailment_ok=entailment_ok,
        failures=failures,
    )
    out = {"review": review}
    if review.decision == "approve":
        out["proposed_action"] = {
            "type": action_type,
            "comment": diag.proposed_comment,
            "label": diag.proposed_label,
        }
    return out
