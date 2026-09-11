"""RAG-grounded diagnosis, retrieved through the Retriever port.

Two things this node stopped doing (T5):

  · importing `tools.retrieval` directly, which bypassed the `Retriever` port and
    made the pluggable backend decorative;
  · retrieving with **no namespace**, which is why G1 was unenforced in practice.
    The port signature had a `namespace` parameter; this caller ignored it, so a
    multi-codebase index would happily let Project A's diagnosis cite Project B's
    code. Grounding is the entire safety story, so that failure would be silent
    and total.

The query now comes from the Surface (`build_query`), so each surface decides
what a good query looks like for its own evidence shape.
"""
from __future__ import annotations

from nodes._render import render_evidence
from nodes._surface import active_surface, evidence_as_dict, namespace_for
from schemas.diagnosis import (
    Citation,
    Diagnosis,
    Finding,
    diagnosis_request_schema,
)


def _fallback_query(ev) -> str:
    title = getattr(ev, "title", "") or ""
    body = getattr(ev, "body", "") or ""
    return f"{title}\n{body}".strip()


def generate_diagnosis(state: dict, llm=None, retriever=None) -> dict:
    ev = state["evidence"]
    surface = active_surface(state)

    if surface is not None and hasattr(surface, "build_query"):
        ev_dict = evidence_as_dict(ev)
        query = surface.build_query(ev_dict) or _fallback_query(ev)
    else:
        query = _fallback_query(ev)

    if retriever is None:
        import config
        retriever = config.get_retriever()

    # G1: always namespaced when the tenant declares a codebase. A retriever that
    # cannot honour a namespace must REFUSE rather than silently retrieve across
    # codebases -- see adapters/retriever_hybrid.py.
    hits = retriever.retrieve(query, k=5, namespace=namespace_for(state))

    guard_report = None
    if llm is None:
        # Offline stub so the pipeline runs end to end without a model server.
        #
        # Two things it does deliberately:
        #   · cites the top hit with the retriever's OWN relevance score, never a
        #     model-supplied one, which would be gameable (T17);
        #   · DERIVES its claim from the cited passage's text, so the claim
        #     actually follows from its citation. The stub previously asserted
        #     "possible duplicate of a known auth regression", which entails
        #     nothing about whatever was retrieved — so once the entailment check
        #     existed (T22) every offline run escalated. A stub that models an
        #     *ungrounded* diagnosis is not a useful stand-in for a grounded one.
        top = hits[0] if hits else None
        claim = (f"Related to a known issue: {top['text']}." if top
                 else "No related prior issue was retrieved.")
        diag = Diagnosis(
            summary="[stub] likely regression in the area described by the cited issue",
            severity="medium",
            proposed_comment=claim,
            proposed_label="needs-triage",
            citations=[Citation(snippet_id=top["id"], score=top["score"])] if top else [],
        )
    else:
        # The TENANT's prompt, resolved and pinned by resolve_tenant (T6). Read
        # from state rather than the store, so a resumed run uses the prompt it
        # STARTED with even if the tenant edited the file meanwhile.
        system = (state.get("_prompts") or {}).get("diagnosis")
        if not system:
            raise ValueError(
                "generate_diagnosis: no diagnosis prompt in state. resolve_tenant "
                "should have pinned it before this node ran."
            )
        # Constrained decoding, not just a prompt asking for JSON (T1). A prompt
        # instruction is advisory; a schema restricts token choice so the response
        # cannot leave the shape. The schema deliberately omits citation scores --
        # see schemas.diagnosis.diagnosis_request_schema.
        user_turn, guard_report = render_evidence(
            ev, hits, tenant=state.get("tenant") or "_default")
        raw = llm.complete(system, user_turn,
                           json_schema=diagnosis_request_schema())
        diag = _parse_diagnosis(raw, hits)

    out = {"diagnosis": diag, "_retrieved": hits}
    if guard_report is not None:
        # Into state so it reaches the audit record and the trace (G6/T23). A
        # redaction nobody can see happened is indistinguishable from none.
        out["_guard"] = guard_report
    return out


def _parse_diagnosis(raw: str, hits: list[dict]) -> Diagnosis:
    """Turn the model's JSON into a Diagnosis.

    Parsing lives here rather than in the adapter: an adapter that interpreted
    responses would be one step away from making decisions, and the whole design
    rests on the model never doing that.

    Two deliberate choices:

    · **Citation scores are looked up from `hits`, never taken from the model.**
      A model-supplied relevance score would be trivially gameable if anything
      trusted it, and the gate compares against exactly this number (T17).

    · **An unparseable or missing severity defaults to `high`, not `low`.**
      `risk_ok` only passes severities in the allowlist, so defaulting high means
      a malformed response escalates to a human. Defaulting low would auto-approve
      whatever the model failed to express clearly — the wrong direction to fail.

    · **`findings[]` is parsed** (T17). The prompt has always documented a
      per-file, per-line findings array; `Diagnosis` had no field for it, so the
      most useful half of the model's output was read and thrown away while
      anyone reading the prompt would assume it was used.
    """
    import json as _json

    # A malformed response must ESCALATE, not crash the node. The graph has no
    # failure edges (G9), so an exception here kills the run -- and a killed run
    # writes no audit record at all, breaking the "every terminal path is
    # recorded" invariant (G6) at exactly the moment a record matters most.
    #
    # Constrained decoding (see the call site) should make this unreachable. It
    # is here because "should" is doing real work in that sentence: a provider
    # that ignores `response_format`, a truncated response from CRA_MAX_TOKENS,
    # or a model emitting a JSON preamble all produce this, and none of them
    # should be able to take down a run.
    try:
        data = _json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    except (ValueError, TypeError) as exc:
        return _unparseable(raw, exc)

    by_id = {h["id"]: h.get("score", 0.0) for h in hits}

    citations = []
    for entry in data.get("citations", []):
        cid = entry.get("snippet_id") if isinstance(entry, dict) else entry
        if cid:
            citations.append(Citation(snippet_id=cid, score=by_id.get(cid, 0.0)))

    findings = [
        Finding(**{k: v for k, v in f.items() if k in Finding.model_fields})
        for f in data.get("findings", []) if isinstance(f, dict)
    ]

    return Diagnosis(
        summary=data.get("summary", ""),
        severity=data.get("severity") or "high",
        proposed_comment=data.get("proposed_comment", ""),
        proposed_label=data.get("proposed_label"),
        citations=citations,
        findings=findings,
    )


# The model's raw text is UNTRUSTED and ends up in an audit record and in front
# of a human, so only a short prefix is kept.
_UNPARSEABLE_EXCERPT = 300


def _unparseable(raw: str, exc: Exception) -> Diagnosis:
    """A diagnosis that cannot be read becomes one that cannot be approved.

    Every route to `approve` is closed deliberately, and by three independent
    booleans rather than one:

      · no citations      -> `groundedness_ok` False
      · severity "high"   -> `risk_ok` False for every shipped policy
      · empty comment     -> `entailment_ok` False (nothing to entail)

    Relying on a single one would make this depend on a tenant's policy values;
    a tenant that allowlisted `high` would otherwise turn a parse failure into an
    approvable action. The gate's `failures` list then tells the human what
    happened, and the audit record keeps it.

    `summary` carries the reason and an excerpt, because a silent escalation
    looks identical to a legitimate one -- and a systematically unparseable model
    would otherwise show up only as a mysteriously rising escalation rate, which
    MODEL_CARD.md already warns is the signal to watch.
    """
    excerpt = (raw or "")[:_UNPARSEABLE_EXCERPT].replace("\n", " ")
    truncated = "..." if raw and len(raw) > _UNPARSEABLE_EXCERPT else ""
    return Diagnosis(
        summary=(f"[unparseable model response] {type(exc).__name__}: {exc}. "
                 f"raw excerpt: {excerpt}{truncated}"),
        severity="high",
        proposed_comment="",
        proposed_label=None,
        citations=[],
        findings=[],
    )
