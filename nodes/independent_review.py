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

import config
from nodes._render import render_checklist, render_for_judge, render_review
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


def effective_gates(pol: dict) -> dict:
    """The bars the gate compares against: the tenant policy's value, else the
    module default. The ONE place this fallback lives -- the audit record calls it
    too, so the bars a record reports are by construction the bars that applied."""
    return {
        "min_relevance_score": pol.get("min_relevance_score", MIN_RELEVANCE_SCORE),
        "min_entailment_score": pol.get("min_entailment_score", MIN_ENTAILMENT_SCORE),
        "severity_allowlist": sorted(pol.get("severity_allowlist", SEVERITY_ALLOWLIST)),
        "allowed_actions": sorted(pol.get("allowed_actions", ALLOWED_ACTIONS)),
    }


def _judge(state: dict, diag, retrieved_list: list[dict], judge_llm) -> tuple[str, dict]:
    """Step 1: the judge EXPLAINS -- free text only, from a fresh call that never
    saw the diagnosis model's reasoning. Returns (reasoning, guard report).

    Resolved from config like every other provider (see generate_diagnosis for
    why this was not the case until 2026-09-11). Offline callers set
    LLM_PROVIDER=stub -- `eval/harness.py` does, because the gate suite must be
    model-free to stay byte-identical across machines.
    """
    if judge_llm is None:
        judge_llm = config.get_llm("judge")
    # The TENANT's judge prompt, pinned by resolve_tenant (T6).
    system = (state.get("_prompts") or {}).get("judge")
    if not system:
        raise ValueError(
            "independent_review: no judge prompt in state. resolve_tenant "
            "should have pinned it before this node ran."
        )
    # T18: the judge receives the SAME retrieved evidence, because its prompt
    # instructs it to verify citation ids against RETRIEVED. Given only the
    # diagnosis it could not perform that check, so its reasoning about grounding
    # was necessarily invented -- and shown to a human approver as though real.
    # The guard's report on the DIAGNOSIS block is kept, not discarded: it is
    # where injection phrasing the model carried forward from the evidence shows
    # up, and invariant 6 says the report reaches state and the audit record.
    judge_turn, judge_guard = render_for_judge(
        diag, retrieved_list, tenant=state.get("tenant") or "_default")
    return judge_llm.complete(system, judge_turn), judge_guard


def _groundedness(cited_ids: list[str], retrieved: dict[str, float],
                  min_score: float) -> tuple[bool, list[str]]:
    """Every cited id was really retrieved AND its RETRIEVER score clears the bar.
    The score is the retriever's own, never the model's (T17)."""
    ok = bool(cited_ids) and all(
        cid in retrieved and retrieved[cid] >= min_score for cid in cited_ids)
    if ok:
        return True, []
    if not cited_ids:
        return False, ["no citations"]
    failures = []
    phantom = [c for c in cited_ids if c not in retrieved]
    weak = [c for c in cited_ids if c in retrieved and retrieved[c] < min_score]
    if phantom:
        failures.append(f"citations not retrieved: {phantom}")
    if weak:
        failures.append(f"citations below relevance {min_score}: {weak}")
    return False, failures


def _approved_action(state: dict, diag, action_type: str, review_text: str) -> dict:
    """The action a human will approve: built HERE, once, and posted verbatim
    (invariant 4). The model's judged text, then the code-computed checklist."""
    return {
        "type": action_type,
        "comment": "\n\n".join(p for p in (
            review_text, render_checklist(state.get("_checks") or [])) if p),
        "label": diag.proposed_label,
    }


def independent_review(state: dict, judge_llm=None, entailer=None) -> dict:
    """The judge explains; Python decides. Four booleans over real data, never a
    verdict parsed out of model output (invariant 1)."""
    diag = state["diagnosis"]
    # The Retriever PORT returns dicts, so that is the only shape a node ever
    # sees. "score" is the calibrated relevance, never the rank-fusion value (T21).
    retrieved_list = list(state["_retrieved"])
    retrieved = {s["id"]: s["score"] for s in retrieved_list}
    # The TENANT'S thresholds and allowlists, with module defaults for offline eval.
    bars = effective_gates(state.get("_policy") or {})

    reasoning, judge_guard = _judge(state, diag, retrieved_list, judge_llm)

    # Step 2: code DECIDES -- booleans from real data, never parsed from the llm.
    cited_ids = [c.snippet_id for c in diag.citations]
    groundedness_ok, failures = _groundedness(cited_ids, retrieved,
                                              bars["min_relevance_score"])

    risk_ok = diag.severity in bars["severity_allowlist"]
    if not risk_ok:
        failures.append(f"severity {diag.severity!r} not in {bars['severity_allowlist']}")

    action_type = _action_type(state, diag)
    permission_ok = action_type in bars["allowed_actions"]
    if not permission_ok:
        failures.append(f"action {action_type!r} not in {bars['allowed_actions']}")

    # The text the gate judges is the text that will be posted: comment AND
    # findings, rendered once. Checking only `proposed_comment` would let findings
    # reach the page without ever being judged.
    review_text = render_review(diag)
    min_entail = bars["min_entailment_score"]
    entailment_ok, entail_score = _entailment(
        review_text, cited_ids, retrieved_list,
        entailer if entailer is not None else config.get_entailer(), min_entail)
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
    out = {"review": review,
           "_guard": {**(state.get("_guard") or {}), "judge": judge_guard}}
    if review.decision == "approve":
        out["proposed_action"] = _approved_action(state, diag, action_type, review_text)
    return out
