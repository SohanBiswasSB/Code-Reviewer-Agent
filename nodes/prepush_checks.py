"""Deterministic pre-push checks on the change under review. No LLM.

The reviewer's design bet is "detectors find, the LLM explains, code decides".
These are the detectors for a CODE change: N+1 queries, new warning-silencers,
long or non-ASCII added lines, new HTTP handlers without a docstring, an import
of a file git does not track, and -- per tenant --
whether the change carries the artefacts that codebase's own pre-push ritual
expects (doc files, Postman collection, tracker row, commit message).

Modelled on the Masterclass repo's `/prepush-ritual`, so a change reviewed here
is held to the same bar that ritual enforces before a push -- and a change that
skipped the ritual is FLAGGED for each thing it skipped.

## What a result is

    {"check": "n_plus_one", "step": 0, "status": "flag", "message": "...",
     "file": "Masterclass/views.py", "line": 88}

`status` is `pass`, `flag` (something expected is wrong or missing) or `skip`
(does not apply here, or needs a database, network or human the reviewer does
not have). A skip says WHY, so "not checked" is never mistaken for "fine".

## Flags inform; they do not route or decide

Routing stays rule-based on paths (surfaces/code_review.py) and the
approve/escalate decision stays the four booleans. Flags reach the diagnosis
prompt as computed facts, the written review as a code-generated checklist, the
audit record, and the live monitor. A check that could veto would be a fifth
boolean, and that is a design change to make on purpose, not by accident here.
"""
from __future__ import annotations

from nodes._surface import active_surface, evidence_as_dict
from tools import change_checks
from tools.codebase import codebase_root


def prepush_checks(state: dict) -> dict:
    """Run the change checks for surfaces that declare them; others get none.

    Reads `evidence` (its files and git revision) and the pinned `_policy` (the
    codebase root and the `prepush` conventions); writes only `_checks`, a list
    of result dicts. Never routes and never decides: see the module docstring.
    """
    surface = active_surface(state)
    if surface is None or not getattr(surface, "runs_change_checks", False):
        return {"_checks": []}
    policy = state.get("_policy") or {}
    evidence = evidence_as_dict(state["evidence"])
    root = codebase_root(policy.get("codebase") or {})
    config = (policy.get("prepush") or {})
    return {"_checks": change_checks.run_all(evidence, root, config)}
