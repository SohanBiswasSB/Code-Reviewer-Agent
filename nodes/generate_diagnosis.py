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

from nodes._surface import active_surface, namespace_for
from schemas.diagnosis import Citation, Diagnosis


def _fallback_query(ev) -> str:
    title = getattr(ev, "title", "") or ""
    body = getattr(ev, "body", "") or ""
    return f"{title}\n{body}".strip()


def generate_diagnosis(state: dict, llm=None, retriever=None) -> dict:
    ev = state["evidence"]
    surface = active_surface(state)

    if surface is not None and hasattr(surface, "build_query"):
        ev_dict = ev if isinstance(ev, dict) else getattr(ev, "__dict__", {})
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
        # It cites the top hit with the retriever's OWN relevance score -- never
        # a model-supplied one, which would be gameable (see T17).
        top = hits[0] if hits else None
        diag = Diagnosis(
            summary="[stub] likely auth-middleware regression",
            severity="medium",
            proposed_comment="Possible duplicate of a known auth regression; see linked issue.",
            proposed_label="needs-triage",
            citations=[Citation(top["id"], top["score"])] if top else [],
        )
    else:
        diag = llm.diagnose(ev, hits)

    return {"diagnosis": diag, "_retrieved": hits}
