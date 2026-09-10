"""Build the LangGraph StateGraph. Requires `langgraph` (only imported when you
run the server/graph — the eval suite imports nodes directly and needs nothing).

    request {tenant, surface, target}
      -> resolve_tenant     (tenant policy + surface pinned into state, ONCE)
      -> fetch_evidence -> normalize_evidence -> classify   (RULE-BASED, no LLM)
           deterministic     -> await_approval -> execute -> audit -> END
           human_review      -> audit -> END                 (no LLM touches it)
           ai_investigation  -> generate_diagnosis            (RAG-grounded)
                             -> independent_review            (SEPARATE llm; explains only)
                                  approve           -> await_approval -> execute -> audit -> END
                                  escalate_to_human -> audit -> END

Every terminal path passes through `audit`, which writes one immutable run record
(G6). A DENIAL is the exception and is recorded by `await_approval` itself -- see
the note on that edge below, and docs/DATA_STORAGE.md.

## One approval gate, not two (T4)

This used to compile with `interrupt_before=["execute"]` *and* route through
`await_approval`. Two mechanisms for one gate is worse than either alone: it was
ambiguous which actually held, and a change to one could silently leave the other
as the only protection.

`await_approval` now owns it, by calling `interrupt()` itself. That is the right
half to keep, because the payload matters: `interrupt(payload)` hands the human
the exact proposed action and the judge's reasoning, whereas `interrupt_before`
suspends with no explanation of what is about to happen. A gate the approver
cannot see the contents of is a rubber stamp.

`execute`'s in-node `PermissionError` is unchanged and is NOT the second
mechanism being removed — it is defence in depth at a different layer. Graph
routing decides *whether* execute is reached; the guard inside it re-checks the
precondition regardless of how it was reached.

## The checkpointer is required, not optional

`interrupt()` suspends by persisting state. With no checkpointer there is nowhere
to persist to, so the pause cannot survive anything. `build_graph` therefore
refuses rather than silently compiling a graph whose approval gate evaporates on
restart.
"""
from __future__ import annotations


def _traced(name, fn, tracer):
    """Wrap a node in a span (T23).

    Wrapping HERE rather than inside each node is deliberate: it is one insertion
    point, no node learns about tracing, and a node added later is covered without
    anyone remembering to instrument it. `nodes/` stays free of observability
    concerns the same way it is free of provider concerns.

    The run's identity comes off state, so every span joins to the audit record
    on `run_id`.
    """
    def wrapped(state, *args, **kwargs):
        with tracer.span(f"node.{name}",
                         run_id=state.get("_run_id"),
                         tenant=state.get("tenant"),
                         surface=state.get("surface") or state.get("_surface_name"),
                         target=state.get("target"),
                         route=state.get("route")):
            return fn(state, *args, **kwargs)
    wrapped.__name__ = name
    return wrapped


def build_graph(checkpointer=None, tracer=None):
    """Compile the agent graph.

    `checkpointer` is mandatory: see the module docstring. Pass MemorySaver() for
    a single-process test, AsyncPostgresSaver in production — never SQLite or a
    file on ephemeral disk, which would lose every pending approval on a restart.
    """
    if checkpointer is None:
        raise ValueError(
            "build_graph requires a checkpointer: interrupt() suspends by "
            "persisting state, so without one the approval gate cannot survive "
            "a restart. Use MemorySaver() for tests, AsyncPostgresSaver in prod."
        )

    # The tracer, unlike the checkpointer, is OPTIONAL by design: losing a span
    # costs a debugging aid, losing a checkpoint costs a pending approval.
    if tracer is None:
        import config
        tracer = config.get_tracer()

    from langgraph.graph import END, START, StateGraph

    from app.state import TriageState
    from nodes.audit import audit
    from nodes.await_approval import await_approval
    from nodes.classify import classify
    from nodes.execute import execute
    from nodes.fetch_evidence import fetch_evidence
    from nodes.generate_diagnosis import generate_diagnosis
    from nodes.independent_review import independent_review
    from nodes.normalize_evidence import normalize_evidence
    from nodes.resolve_tenant import resolve_tenant

    g = StateGraph(TriageState)
    for name, fn in [
        ("resolve_tenant", resolve_tenant),
        ("fetch_evidence", fetch_evidence),
        ("normalize_evidence", normalize_evidence),
        ("classify", classify),
        ("generate_diagnosis", generate_diagnosis),
        ("independent_review", independent_review),
        ("await_approval", await_approval),
        ("execute", execute),
        ("audit", audit),
    ]:
        g.add_node(name, _traced(name, fn, tracer))

    g.add_edge(START, "resolve_tenant")
    g.add_edge("resolve_tenant", "fetch_evidence")
    g.add_edge("fetch_evidence", "normalize_evidence")
    g.add_edge("normalize_evidence", "classify")
    # Every terminal path goes through `audit` (G6). human_review and
    # escalate_to_human are OUTCOMES, not absences of one -- a run that routed
    # straight to a human is exactly as worth recording as one that posted.
    g.add_conditional_edges("classify", lambda s: s["route"], {
        "deterministic": "await_approval",
        "ai_investigation": "generate_diagnosis",
        "human_review": "audit",
    })
    g.add_edge("generate_diagnosis", "independent_review")
    g.add_conditional_edges("independent_review",
                            lambda s: s["review"].decision,
                            {"approve": "await_approval", "escalate_to_human": "audit"})
    # await_approval suspends via interrupt(); the resume continues into execute
    # only when the human approved, because await_approval writes `approved`.
    #
    # A DENIAL never reaches `audit`: it resumes into execute, which raises. That
    # is deliberate and load-bearing -- the guard, not the routing, is what
    # refuses -- so await_approval records the denial itself. docs/DATA_STORAGE.md.
    g.add_edge("await_approval", "execute")
    g.add_edge("execute", "audit")
    g.add_edge("audit", END)
    return g.compile(checkpointer=checkpointer)
