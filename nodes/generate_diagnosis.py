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
from schemas.diagnosis import Citation, Diagnosis, Finding


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
        diag = _parse_diagnosis(llm.complete(system, render_evidence(ev, hits)), hits)

    return {"diagnosis": diag, "_retrieved": hits}


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

    data = _json.loads(raw)
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
