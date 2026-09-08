"""Build the LangGraph StateGraph. Requires `langgraph` (only imported when you
run the server/graph — the eval suite imports nodes directly and needs nothing).

    issue/PR/log target
      -> fetch_evidence -> normalize_evidence -> classify
           deterministic     -> await_approval -> execute -> END
           human_review      -> END
           ai_investigation  -> generate_diagnosis -> independent_review
                                    approve           -> await_approval -> execute -> END
                                    escalate_to_human -> END
"""
from __future__ import annotations


def build_graph(checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    from app.state import TriageState
    from nodes.await_approval import await_approval
    from nodes.classify import classify
    from nodes.execute import execute
    from nodes.fetch_evidence import fetch_evidence
    from nodes.generate_diagnosis import generate_diagnosis
    from nodes.independent_review import independent_review
    from nodes.normalize_evidence import normalize_evidence

    g = StateGraph(TriageState)
    for name, fn in [("fetch_evidence", fetch_evidence), ("normalize_evidence", normalize_evidence),
                     ("classify", classify), ("generate_diagnosis", generate_diagnosis),
                     ("independent_review", independent_review), ("await_approval", await_approval),
                     ("execute", execute)]:
        g.add_node(name, fn)

    g.add_edge(START, "fetch_evidence")
    g.add_edge("fetch_evidence", "normalize_evidence")
    g.add_edge("normalize_evidence", "classify")
    g.add_conditional_edges("classify", lambda s: s["route"], {
        "deterministic": "await_approval",
        "ai_investigation": "generate_diagnosis",
        "human_review": END,
    })
    g.add_edge("generate_diagnosis", "independent_review")
    g.add_conditional_edges("independent_review",
                            lambda s: s["review"].decision,
                            {"approve": "await_approval", "escalate_to_human": END})
    g.add_edge("await_approval", "execute")   # resume only continues here when approved
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["execute"])
