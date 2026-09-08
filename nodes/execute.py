"""The ONLY node allowed to write. Refuses without approval — checked IN THE NODE.

Writes now go through the active Surface's `ActionSink` rather than a hardcoded
GitHub client (T5), so the destination is a config choice and this node knows
nothing about any provider.

What did NOT change is the guard. Routing decides whether this node is reached;
the check below re-establishes the precondition regardless of how it was reached.
That redundancy is the point — a future refactor adding a shortcut edge straight
here still cannot bypass approval.
"""
from __future__ import annotations

from nodes._surface import active_surface


def execute(state: dict, sink=None) -> dict:
    """`sink` is injectable for tests; otherwise resolved from the surface."""
    # `is not True` is exact on purpose: a truthy string, a 1, or a populated
    # dict is NOT an approval. Only a human's explicit True is.
    if state.get("approved") is not True:
        raise PermissionError("execute() refused: state['approved'] is not True")

    action = state["proposed_action"]     # posted VERBATIM, never recomputed
    target = state["target"]

    if sink is None:
        surface = active_surface(state)
        if surface is None:
            raise ValueError("execute: no active surface, so no sink to write through")
        import config
        sink = config.get_sink(surface.sink_name)

    # identity = the approver's own token, so GitHub's permission model does the
    # real enforcement, and "the approver posted this" is verifiable from its
    # audit log rather than only from our UI.
    return {"result": sink.emit(action, target=target, identity=state.get("github_token"))}
