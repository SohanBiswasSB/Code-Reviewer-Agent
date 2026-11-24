"""Validation for the gate-relevant part of a tenant policy (G11).

A policy is hand-edited JSON, and before this nothing checked its shape. The
failure that motivates it is quiet rather than loud: `"min_relevance_score":
"0.75"` (a string) compares as a TypeError deep in `independent_review`, but
`"severity_allowlist": "low"` (a string, not a list) makes `"low" in "low"` true
AND `"lo" in "low"` true -- membership on a string is substring, so a typo widens
the allowlist instead of failing. A bar written as 75 instead of 0.75 makes every
run escalate forever, which is the inert-surface shape this project keeps finding.

Only the keys the GATE and the ROUTER read are checked. The rest of a policy
(prepush conventions, codebase binding, free-text notes) is owned by the code
that reads it, and a tenant may carry keys this service does not know.

Not a Pydantic model on purpose: the policy is a flattened, deep-merged dict
(_default + tenant + surface), and the errors should name the KEY as the tenant
wrote it.
"""
from __future__ import annotations

SEVERITIES = frozenset({"low", "medium", "high", "critical"})
UNIT_INTERVAL = ("min_relevance_score", "min_rerank_score", "min_entailment_score")
STRING_LISTS = ("severity_allowlist", "allowed_actions", "risky_areas", "enabled_surfaces")


def policy_errors(pol: dict) -> list[str]:
    """Every problem with the gate-relevant keys, as human-readable strings."""
    errs = []
    for key in UNIT_INTERVAL:
        if key not in pol:
            continue
        v = pol[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"{key} must be a number, got {type(v).__name__} {v!r}")
        elif not 0.0 <= v <= 1.0:
            errs.append(f"{key} must be between 0 and 1, got {v} (a percentage?)")
    for key in STRING_LISTS:
        if key not in pol:
            continue
        v = pol[key]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errs.append(f"{key} must be a list of strings, got {v!r} -- a bare string "
                        f"would match by SUBSTRING and widen the rule")
    unknown = sorted(set(pol.get("severity_allowlist") or []) - SEVERITIES) \
        if isinstance(pol.get("severity_allowlist"), list) else []
    if unknown:
        errs.append(f"severity_allowlist has unknown severities {unknown}; "
                    f"known: {sorted(SEVERITIES)}")
    return errs
