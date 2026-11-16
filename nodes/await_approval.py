"""The human-in-the-loop gate. Builds `proposed_action` ONCE, then suspends.

`interrupt(payload)` does not "show a modal and wait for a click" — it suspends
the graph mid-run and persists state to the checkpointer. The pause is real
process state, which is why `build_graph` refuses to compile without a
checkpointer and why production must use `AsyncPostgresSaver` rather than SQLite
or a file: a container recycle on ephemeral disk would silently lose every
pending approval.

Two properties this node is responsible for:

**The action is built once, here, and never recomputed.** `execute` posts
`state["proposed_action"]` verbatim. If the action were rebuilt after approval
there would be a window in which what executes differs from what a human signed
off on — and it is exactly the kind of drift nobody notices until it matters.

**`approved` is only ever written from the resume value.** This node is the sole
writer of that flag. It cannot be inferred, defaulted, or derived from the
judge's opinion; a human's answer to `interrupt()` is the only thing that sets
it. `execute` then re-checks it independently (`is not True`, so a truthy string
or a 1 will not do), which is defence in depth rather than duplication: routing
decides whether execute is reached, the guard re-establishes the precondition
regardless of how it was reached.
"""
from __future__ import annotations

from langgraph.types import interrupt

from nodes._audit import write


def await_approval(state: dict, interrupt_fn=None, audit_sink=None) -> dict:
    """Suspend for human approval; return the decision as `approved`.

    `interrupt_fn` is injectable so the node can be unit-tested without a
    running graph. In the graph it is LangGraph's own `interrupt`.
    `audit_sink` is injectable for the same reason.
    """
    action = state.get("proposed_action")
    if action is None:
        # The deterministic branch reaches the gate without an LLM diagnosis, so
        # it builds its action here. Still human-approved: a cheap deterministic
        # route is not an excuse to skip the gate.
        action = {"type": "comment+label",
                  "comment": "[deterministic] re-run failing CI job.",
                  "label": "ci-failure"}

    payload = {
        "proposed_action": action,
        "reasoning": getattr(state.get("review"), "reasoning", ""),
        "tenant": state.get("tenant"),
        "surface": state.get("surface"),
        "target": state.get("target"),
        # So an approver can see WHICH guardrail values this run was judged
        # against, not merely that some policy applied (G8).
        "policy_version": state.get("_policy_version"),
    }

    if interrupt_fn is None:
        interrupt_fn = interrupt

    decision = interrupt_fn(payload)

    # Normalise explicitly. A resume value could arrive as a bool, or as a dict
    # from a JSON body; anything else is NOT approval. Defaulting to False here
    # keeps `execute`'s guard the backstop rather than the only check.
    if isinstance(decision, dict):
        approved = decision.get("approved") is True
    else:
        approved = decision is True

    out = {"proposed_action": action, "approved": approved,
           "_interrupt_payload": payload}

    if not approved:
        # A denial is an OUTCOME and has to be recorded, but it cannot reach the
        # `audit` node: resuming with anything that is not True continues into
        # `execute`, whose PermissionError is what stops the run. That guard is
        # the mechanism (tests/test_graph_resume.py pins it explicitly), and
        # routing denials around `execute` to reach an audit node would remove
        # the defence in depth those tests exist to prove. So the node that
        # observes the denial records it. See docs/DATA_STORAGE.md.
        out.update(write({**state, **out}, "denied", sink=audit_sink))

    return out
