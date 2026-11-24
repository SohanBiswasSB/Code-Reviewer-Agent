"""Terminal node: write the run record, then END.

In the graph rather than in `app/main.py`, for the reason `resolve_tenant` and
`execute` give for their own checks: a record that only exists when the run came
through HTTP is missing for the scheduled job and for every future caller. Put it
where the work happens.

The outcome is DERIVED from state rather than passed per edge, so a future edge
into this node cannot mislabel what happened.
"""
from __future__ import annotations

from nodes._audit import write


def _outcome(state: dict) -> str:
    """Which terminal this run reached, read from what earlier nodes wrote.

    `executed` if a sink returned a result, `escalated` if the gate said so,
    `human_review` if classify routed there, else `ended`. Checked in that order,
    so a run that executed is never labelled by an earlier step.
    """
    if state.get("result") is not None:
        return "executed"
    review = state.get("review")
    if review is not None and getattr(review, "decision", None) == "escalate_to_human":
        return "escalated"
    if state.get("route") == "human_review":
        return "human_review"
    return "ended"


def audit(state: dict, sink=None) -> dict:
    """Write the run record for a normal terminal edge (invariant 9, G6).

    Reads the whole state; writes only `_audit` (whether the record persisted,
    and its run id). A failed write is reported there, not raised -- see
    `nodes._audit.write`. `sink` is injectable for tests.
    """
    return write(state, _outcome(state), sink=sink)
