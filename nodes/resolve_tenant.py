"""Resolve the tenant's policy and surface ONCE, at the head of the run.

Before this node existed, `build_graph()` took neither tenant nor surface, so
nothing between START and the gate ever called the `TenantStore`. A real run
returned state carrying `tenant` and `surface` but **no `_policy`** — which meant
`independent_review` fell back to its module constants and **every codebase was
gated by `_default`'s thresholds regardless of its own policy.json**. The
multi-tenant design was void at runtime while looking wired, because
`eval/demo_multitenant.py` passes `_policy` in by hand. (T14.)

Three deliberate choices here:

**It is a node, not a `build_graph(tenant, surface)` argument.** Running inside
the graph means the resolved policy lands in the checkpoint automatically, so a
run that pauses at the approval gate and resumes hours later is judged against
the policy it *started* with. If the tenant edits `policy.json` in between, the
resumed run does not silently mix versions. That is the substance of G8, and
binding at compile time would have thrown it away.

**Only serialisable values go into state.** `_policy` is a plain dict and
`_surface_name` is a string; the Surface *object* is resolved lazily by the nodes
that need it. Putting a class instance in a graph channel would add one more
unregistered type to the checkpoint-serialisation problem (T15), and a paused run
that cannot be deserialised destroys the durability the approval gate rests on.

**A surface the tenant has not enabled raises here, in the graph.** `app/main.py`
checks it too, but a check that lives only at the HTTP boundary is bypassed by
the scheduled job and by any future caller. Same reasoning as `execute`'s
in-node permission guard: put the check where the work happens.
"""
from __future__ import annotations

import hashlib
import json

DEFAULT_TENANT = "_default"


def _policy_version(policy: dict) -> str:
    """A short content hash of the resolved policy.

    Pinned into state so an audit record (G6) or a resumed run can state exactly
    which guardrail values it was judged against — "the tenant's policy" is not
    an answer when the file is editable mid-run.
    """
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def resolve_tenant(state: dict, store=None) -> dict:
    """Attach `_policy`, `_surface_name` and `_policy_version` to state."""
    surface_name = state.get("surface")
    if not surface_name:
        raise ValueError("resolve_tenant: state['surface'] is required")

    tenant = state.get("tenant") or DEFAULT_TENANT

    if store is None:                      # injectable for tests
        import config
        store = config.get_tenant_store()

    policy = store.policy(tenant, surface_name)

    enabled = policy.get("enabled_surfaces", [])
    if surface_name not in enabled:
        raise PermissionError(
            f"tenant '{tenant}' has not enabled surface '{surface_name}' "
            f"(enabled: {enabled})"
        )

    return {
        "_policy": policy,
        "_surface_name": surface_name,
        "_policy_version": _policy_version(policy),
    }
