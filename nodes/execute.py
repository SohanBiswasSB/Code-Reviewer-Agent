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

import config
from nodes._surface import active_surface


def execute(state: dict, sink=None) -> dict:
    """Post the approved action verbatim through the surface's sink.

    Reads `approved`, `proposed_action`, `target`, `github_token` and run
    metadata; writes only `result`. Raises PermissionError unless `approved is
    True` (invariant 3), and ValueError with no active surface. `sink` is
    injectable for tests; otherwise resolved from the surface.
    """
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
        sink = config.get_sink(config.sink_name_for(surface))

    # Run METADATA, never content. The action itself is posted verbatim and is
    # not touched here; `context` only tells a per-tenant sink (the daily report)
    # which tenant and run it belongs to. Folding any of this into `action` would
    # mean mutating the payload a human approved.
    diag = state.get("diagnosis")
    context = {
        "tenant": state.get("tenant"),
        "surface": state.get("surface") or state.get("_surface_name"),
        "run_id": state.get("_run_id"),
        "policy_version": state.get("_policy_version"),
        "prompt_version": state.get("_prompt_version"),
        "severity": getattr(diag, "severity", None),
        "citations": [c.model_dump() for c in getattr(diag, "citations", [])],
    }

    # identity = the approver's own token, so GitHub's permission model does the
    # real enforcement, and "the approver posted this" is verifiable from its
    # audit log rather than only from our UI.
    return {"result": sink.emit(action, target=target,
                                identity=state.get("github_token"), context=context)}
