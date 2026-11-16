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
2. **Policy** (per surface) — `min_relevance_score` (the bar for a COSINE),
   `min_rerank_score` (the bar for a RERANKER score, T13 -- each retrieved passage says
   which kind it carries, and the gate picks the matching bar), `min_entailment_score`,
   `severity_allowlist`, `allowed_actions`, `risky_areas`, `anomaly_threshold`,
   `enabled_surfaces`. `risky_areas` is ADDED to the surface's defaults (T27, 2026-11-17)
   -- a tenant can make `subscription` risky but cannot make `migrations` routine. These
   keys are validated when a run starts (G11, `schemas/policy.py`): a bad value fails the
   run with the key named, instead of quietly changing what the gate does.
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

The same binding names the tenant's **retrieval namespace** (`repo`) — build it with
`python -m tools.ingest --tenant <codebase>`. `codebase.index_exclude` lists path globs
(a pattern also matches every parent directory) that must never enter the index — data
exports and user-record reports that live beside the code. Redaction is pattern-based and
an index is a durable copy, so excluding such files is the only complete fix.

**Escalate-to-human is shared, not per-project.** Every codebase routes to human review
on the same conditions — an ungroundable diagnosis, a risk/permission miss, or an explicit
`escalate_to_human: true` safety default. It's the org-wide safety net; a codebase tunes
its *thresholds and prompts*, but the escalate route is always there for A and B alike.

## Pre-push checks per codebase (`code_review.prepush`)

Every code review runs the GENERIC change checks -- N+1 queries in loops, new
warning-silencing comments, line length / ASCII on added lines, new undocumented
HTTP handlers, imports of files git does not track. A codebase that has its own
pre-push conventions declares them, and the reviewer then FLAGS a change that
skipped them:

| key | what it enforces |
|---|---|
| `max_line_length`, `hygiene_suffixes`, `skip_dirs` | tune the generic checks |
| `never_commit`, `read_only` | path prefixes / globs the change must not touch |
| `docs` | per-app doc folder; which changed file needs which doc; dated change-log entry; the exact file set |
| `postman` | collection + environments; API-file triggers; generator inputs; no secret values |
| `tracker` | owner (git author) -> tracker file + key prefix; CSV header; required description blocks |
| `commit_message` | file (staged) or the commit itself; subject regex; required blocks; no attribution |
| `not_checkable` | `[name, step, why]` rows the reviewer cannot run (DB, Django, network) -- reported as `skip` |

`tenants/masterclass/policy.json` is the worked example: it mirrors that repo's
`.claude/skills/prepush-ritual`. A codebase without a block is held to the generic
checks only -- it is never flagged for lacking someone else's conventions.
