"""The gate: a SEPARATE llm produces `reasoning` text ONLY; PYTHON booleans decide.

Four booleans now, not three. The design calls for grounding to be checked in
three steps and only two were implemented:

  1. the cited id is real                  -> groundedness_ok (was present)
  2. its relevance clears a calibrated bar -> groundedness_ok (was present)
  3. the claim FOLLOWS from the passage    -> entailment_ok   (T22, new)

Steps 1 and 2 together prove only that a real, on-topic snippet was cited. They
cannot catch a claim that overreaches what its passage supports — and a citation
is exactly what makes such a claim look trustworthy. That is the most common RAG
failure, and it was ungated. Since 2026-11-17 step 3 is scored PER CLAIM: the
comment and each finding against its own premise, the lowest score deciding
(see `entailment_claims`), rather than the whole review against every passage.

`risk_ok` and `permission_ok` are unchanged.

Evidence handling: the diagnosis text and the retrieved snippets are untrusted
DATA and are fenced before they reach the judge (see nodes/_render.py).
"""
from __future__ import annotations

import os

import config
from nodes._render import render_checklist, render_for_judge, render_review
from nodes._surface import active_surface, evidence_as_dict
from schemas.diagnosis import ReviewResult
from tools.prepush.change import split_diff

MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))  # MEASURE this
MIN_ENTAILMENT_SCORE = float(os.getenv("MIN_ENTAILMENT_SCORE", "0.25"))
# The bar for a RERANKER score (T13) -- a different model's scale, so a different
# bar. Hits say which number they carry (`score_kind`); see `_bar_for`.
# Measured 2026-11-17 on Masterclass: the cut keeping 90% of same-file passages
# (Youden's cut was 0.73, J 0.31 -- weak; see config.get_reranker for why the
# reranker is off by default).
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.61"))
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


def entailment_claims(diag, evidence, retrieved: list[dict]) -> list[tuple[str, str, str]]:
    """Every claim the review makes, each with the premise it must follow from.

    Returns `(label, claim, premise)` triples. Shared with `tools.calibrate`, so
    what is measured is exactly what the gate scores.

    **Per claim, not per review** (2026-11-17). The whole review used to be one
    claim scored against every cited passage together. That let a well-supported
    finding carry a fabricated one, and it diluted every score with the words of
    the other findings: real reviews scored 0.08-0.17 whatever their quality.

    **A finding's premise is its OWN citations plus the diff of its OWN file.**
    The diff is the subject of the claim -- "this adds a unique constraint that
    breaks existing rows" is a claim about the change -- and in normal use the
    new code is not in the index yet, so a finding about it could never be
    entailed by retrieved passages alone. It is not borrowing: another file's
    diff and passages the finding did not cite are still excluded.

    The comment is a claim about the whole change, so its premise is every cited
    passage plus the whole diff. An empty review yields one empty claim, which
    scores 0.0 -- unverifiable fails closed, as before.
    """
    text = {s.get("id"): s.get("text", "") for s in retrieved}
    diff = (evidence_as_dict(evidence).get("diff") or "") if evidence is not None else ""
    hunks = split_diff(diff) if diff else {}

    def premise(ids, *extra):
        """The cited passages' text, each id once, in citation order, then `extra`."""
        parts = [text[i] for i in dict.fromkeys(ids) if i in text]
        return " ".join(p for p in (*parts, *extra) if p)

    cited = [c.snippet_id for c in getattr(diag, "citations", None) or []]
    comment = (getattr(diag, "proposed_comment", "") or "").strip()
    findings = getattr(diag, "findings", None) or []
    claims = []
    if comment or not findings:
        claims.append(("comment", comment, premise(cited, diff)))
    for n, f in enumerate(findings, start=1):
        claim = f"{f.issue} {f.suggestion}".strip()
        claims.append((f"finding {n} ({f.file or 'no file'})", claim,
                       premise(f.citation_ids, hunks.get(f.file, ""))))
    return claims


def _entailment(diag, evidence, retrieved: list[dict], entailer,
                min_score: float) -> tuple[bool, float, list[str]]:
    """Does EVERY claim follow from its premise? Returns (ok, lowest score, failures).

    The lowest score decides because a review is posted whole: one unsupported
    finding on the page is the failure this check exists to stop, however well
    the others are supported.
    """
    failures, lowest = [], 1.0
    for label, claim, premise in entailment_claims(diag, evidence, retrieved):
        score = entailer.score(claim, premise) if claim and premise else 0.0
        lowest = min(lowest, score)
        if score < min_score:
            failures.append(f"{label} not entailed by its premise "
                            f"(score {score} < {min_score})")
    return not failures, lowest, failures


def effective_gates(pol: dict) -> dict:
    """The bars the gate compares against: the tenant policy's value, else the
    module default. The ONE place this fallback lives -- the audit record calls it
    too, so the bars a record reports are by construction the bars that applied."""
    return {
        "min_relevance_score": pol.get("min_relevance_score", MIN_RELEVANCE_SCORE),
        "min_rerank_score": pol.get("min_rerank_score", MIN_RERANK_SCORE),
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


def _bar_for(hit: dict, bars: dict) -> float:
    """The relevance bar for ONE retrieved hit, chosen by the kind of score it carries.

    A reranker score and a cosine come from different models on different
    scales: the cross-encoder spreads 0..1, EmbeddingGemma packs ~0.7-0.9. One
    bar for both would make whichever kind it was not measured for either
    unfailable or unpassable -- T21's shape, a gate reading a number that means
    something other than what its bar assumes. An unknown kind gets the cosine
    bar, the one every pre-T13 hit was gated on.
    """
    if hit.get("score_kind") == "rerank":
        return bars["min_rerank_score"]
    return bars["min_relevance_score"]


def _groundedness(cited_ids: list[str], retrieved: dict[str, float],
                  min_score: float | dict[str, float]) -> tuple[bool, list[str]]:
    """Every cited id was really retrieved AND its RETRIEVER score clears the bar.
    The score is the retriever's own, never the model's (T17). `min_score` is one
    bar, or a bar per id when hits carry different kinds of score (T13)."""
    def bar(cid: str) -> float:
        """This id's bar; an id with no per-hit bar gets 1.0, the strictest."""
        return min_score.get(cid, 1.0) if isinstance(min_score, dict) else min_score

    ok = bool(cited_ids) and all(
        cid in retrieved and retrieved[cid] >= bar(cid) for cid in cited_ids)
    if ok:
        return True, []
    if not cited_ids:
        return False, ["no citations"]
    failures = []
    phantom = [c for c in cited_ids if c not in retrieved]
    weak = [c for c in cited_ids if c in retrieved and retrieved[c] < bar(c)]
    if phantom:
        failures.append(f"citations not retrieved: {phantom}")
    if weak:
        failures.append("citations below relevance: "
                        + ", ".join(f"{c} ({retrieved[c]} < {bar(c)})" for c in weak))
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
    verdict parsed out of model output (invariant 1).

    Reads `diagnosis`, `_retrieved`, `evidence`, `_checks`, the pinned `_policy`
    and `_prompts["judge"]`. Writes `review` (the booleans, `failures` and the
    judge's `reasoning`) and the judge's half of `_guard`; on `approve` only,
    also `proposed_action`, built once here and posted verbatim (invariant 4).
    `judge_llm` and `entailer` are injectable for tests.
    """
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
    groundedness_ok, failures = _groundedness(
        cited_ids, retrieved, {s["id"]: _bar_for(s, bars) for s in retrieved_list})

    risk_ok = diag.severity in bars["severity_allowlist"]
    if not risk_ok:
        failures.append(f"severity {diag.severity!r} not in {bars['severity_allowlist']}")

    action_type = _action_type(state, diag)
    permission_ok = action_type in bars["allowed_actions"]
    if not permission_ok:
        failures.append(f"action {action_type!r} not in {bars['allowed_actions']}")

    # The text that will be posted: comment AND findings, rendered once. The gate
    # judges that same comment and those same findings, each as its own claim
    # (`entailment_claims`). Checking only `proposed_comment` would let findings
    # reach the page without ever being judged.
    review_text = render_review(diag)
    entailment_ok, _, entail_failures = _entailment(
        diag, state.get("evidence"), retrieved_list,
        entailer if entailer is not None else config.get_entailer(),
        bars["min_entailment_score"])
    failures.extend(entail_failures)

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
