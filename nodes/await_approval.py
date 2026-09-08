"""Human-in-the-loop gate. Builds proposed_action ONCE, then suspends the graph."""
from __future__ import annotations
# from langgraph.types import interrupt  # uncomment when running the real graph


def await_approval(state: dict) -> dict:
    action = state.get("proposed_action")
    if action is None:  # deterministic branch builds it here
        action = {"type": "comment", "comment": "[deterministic] re-run failing CI job.",
                  "label": "ci-failure"}
    payload = {"proposed_action": action, "reasoning": getattr(state.get("review"), "reasoning", "")}
    # REAL: decision = interrupt(payload)  -> graph SUSPENDS, checkpointed to Postgres.
    #       frontend resumes with Command(resume=True/False) on the same thread_id.
    return {"proposed_action": action, "_interrupt_payload": payload}
