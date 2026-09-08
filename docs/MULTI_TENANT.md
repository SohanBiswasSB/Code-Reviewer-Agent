# Multi-tenant: one agent, many orgs — dynamic prompts + guardrails

No prompt or threshold is hardcoded. One deployed agent serves many projects (A, B, …);
each brings **its own prompts and its own guardrail policy, per surface and per level**,
resolved at request time from the `TenantStore` port.

## What a tenant owns
```
tenants/
  _default/                         # org-level baseline (fallback for everyone)
    policy.json                     # thresholds, allowlists, allowed actions, enabled surfaces
    prompts/<surface>/<node>.md     # diagnosis.md, judge.md, ...
  milesone/                         # tenant A — its own rules
    policy.json                     # e.g. min_relevance_score 0.40, risky_areas incl. zoom/webhook
    prompts/code_review/diagnosis.md
  project_b/                        # tenant B — stricter
    policy.json                     # min_relevance_score 0.55, severity_allowlist ["low"], comment-only
    prompts/code_review/diagnosis.md
```

## Resolution (per request)
Every run carries `state["tenant"]` + `state["surface"]`.
- **Prompt:** `store.prompt(tenant, surface, node)` → tenant file, else `_default` file.
- **Policy:** `store.policy(tenant, surface)` → `_default` **deep-merged** with the tenant's,
  scoped to the surface. Attached to `state["_policy"]`; the gate reads its thresholds from
  there (`nodes/independent_review.py`), never from constants.

So the **same diagnosis** can be **approved for MilesOne** and **escalated for Project B**,
purely because Project B's policy sets a higher grounding bar and a tighter severity/action
allowlist — zero code change.

## Levels of guardrail (all per-tenant)
1. **Prompt** (per surface/node) — the tenant's own instructions & emphasis.
2. **Policy** (per surface) — `min_relevance_score`, `severity_allowlist`, `allowed_actions`,
   `risky_areas`, `anomaly_threshold`, `enabled_surfaces`.
3. **Code invariants** (org-wide, NOT tenant-overridable) — judge-explains/code-decides,
   the `execute` PermissionError guard, untrusted-input handling. Tenants tune *values*,
   never remove the *boundary*.

## Onboard a new project
1. `mkdir tenants/<name>` with a `policy.json` (override only what differs).
2. Drop any prompt overrides under `tenants/<name>/prompts/<surface>/<node>.md` (others inherit `_default`).
3. Point the request at `tenant=<name>`. Done — no deploy, no core change.

## Swap the backend
`TenantStore` is a port. `adapters/tenant_fs.py` is filesystem-backed; write a DB- or
config-service-backed adapter (`@adapter("tenant","db")`) and set `TENANT_STORE=db` to
manage prompts/policies in a UI without a redeploy.

## Clarification: a tenant IS a codebase
A "tenant" here = **one codebase/repo** the agent is plugged into (Project A = repo A,
Project B = repo B, MilesOne = the backend repo). Each tenant's `policy.json` carries a
`codebase` binding — `{vcs, repo, root, language}` — so the `EvidenceSource` knows which
repo to pull the diff/files from. Onboarding a new codebase = add `tenants/<codebase>/`
with its `codebase` binding, prompts, and rules.

**Escalate-to-human is shared, not per-project.** Every codebase routes to human review
on the same conditions — an ungroundable diagnosis, a risk/permission miss, or an explicit
`escalate_to_human: true` safety default. It's the org-wide safety net; a codebase tunes
its *thresholds and prompts*, but the escalate route is always there for A and B alike.
