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
import uuid

import config

DEFAULT_TENANT = "_default"


def _content_version(obj: dict) -> str:
    """A short, order-independent content hash.

    Pinned into state for the policy AND the composed prompts, so an audit record
    (G6) or a run resumed hours later can state exactly which guardrail values and
    which prompt text it was judged by -- "the tenant's policy" is not an answer
    when the file is editable mid-run. A prompt is production configuration too.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


_policy_version = _prompt_version = _content_version


def resolve_tenant(state: dict, store=None, surface=None) -> dict:
    """Resolve policy AND prompts once, and pin both into state.

    Prompts are resolved here rather than in the LLM nodes so that they land in
    the checkpoint alongside the policy. That is what stops a resumed run from
    mixing versions — the substance of G8, and the reason this is a node.
    """
    surface_name = state.get("surface")
    if not surface_name:
        raise ValueError("resolve_tenant: state['surface'] is required")

    tenant = state.get("tenant") or DEFAULT_TENANT

    if store is None:                      # injectable for tests
        store = config.get_tenant_store()

    policy = store.policy(tenant, surface_name)

    enabled = policy.get("enabled_surfaces", [])
    if surface_name not in enabled:
        raise PermissionError(
            f"tenant '{tenant}' has not enabled surface '{surface_name}' "
            f"(enabled: {enabled})"
        )

    # Which prompt node keys this surface needs. Resolved from the surface so a
    # new surface with different nodes needs no change here.
    if surface is None:
        surface = state.get("_surface")
        if surface is None:
            surface = config.get_surface(surface_name)

    node_keys = {
        "diagnosis": getattr(surface, "diagnosis_prompt", "diagnosis"),
        "judge": getattr(surface, "judge_prompt", "judge"),
    }
    prompts = {}
    for role, node in node_keys.items():
        # Missing prompts must fail HERE, before any model call and before the
        # human sees a gate — not deep inside generate_diagnosis.
        prompts[role] = store.prompt(tenant, surface_name, node)

    return {
        # Minted here, not in the audit sink, so every record written for this
        # run carries the same id -- a denial recorded by await_approval and a
        # completion recorded by the audit node are the same run (G6).
        "_run_id": state.get("_run_id") or str(uuid.uuid4()),
        "_policy": policy,
        "_surface_name": surface_name,
        "_policy_version": _policy_version(policy),
        "_prompts": prompts,
        "_prompt_version": _prompt_version(prompts),
    }
