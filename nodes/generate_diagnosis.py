"""RAG-grounded diagnosis via structured output. LLM call is injectable for tests."""
from __future__ import annotations

from schemas.diagnosis import Citation, Diagnosis
from tools import retrieval


def generate_diagnosis(state: dict, llm=None) -> dict:
    ev = state["evidence"]
    snippets = retrieval.retrieve(f"{ev.title}\n{ev.body}", k=5)
    # PROD: llm.with_structured_output(Diagnosis).invoke(prompt_with(snippets))
    if llm is None:
        # offline stub: cite the top snippet so the pipeline runs end-to-end
        top = snippets[0] if snippets else None
        diag = Diagnosis(
            summary="[stub] likely auth-middleware regression",
            severity="medium",
            proposed_comment="Possible duplicate of a known auth regression; see linked issue.",
            proposed_label="needs-triage",
            citations=[Citation(top.id, top.score)] if top else [],
        )
    else:
        diag = llm.diagnose(ev, snippets)
    return {"diagnosis": diag, "_retrieved": snippets}
