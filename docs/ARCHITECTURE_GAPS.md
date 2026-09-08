# Architecture review — gaps found (v1 recheck)

Honest audit of the current design/skeleton. P0 = would bite in production or breaks the
multi-tenant promise; P1 = needed before real use; P2 = hardening. Each: gap → risk → fix.

## P0 — must close

### G1. Retrieval is not tenant-scoped  → cross-codebase leakage
`Retriever.retrieve(query, k)` has no namespace. With many codebases sharing one index,
Project A's diagnosis could retrieve and **cite Project B's code**. Grounding is the whole
safety story, so this is critical.
**Fix:** add `namespace` (per-codebase, from `policy.codebase`) to the port + every adapter;
ingest writes per-codebase namespaces/collections. *(port signature updated in this pass.)*

### G2. Surfaces don't actually drive the graph  → plug-and-play is half-wired
The graph calls `nodes.classify` directly; the active `Surface`'s own
`classify`/`build_query`/`source`/`sink` are never invoked. Surfaces are currently
decorative.
**Fix:** inject the resolved surface into state; nodes delegate to it (source, query,
classify, prompt keys, sink). *(delegation hook added to `classify` in this pass; the
diagnose/fetch/execute nodes still need the same treatment.)*

### G3. Per-tenant credentials + API authN/Z are missing
One shared `GITHUB_TOKEN`; `/run/{surface}` has no caller authentication and doesn't check
the tenant. Any caller could run any tenant against any repo, with one blast-radius token.
**Fix:** a `CredentialProvider` port that resolves a **secret ref** from `policy.codebase`
(never a raw token in config); authenticate the caller and authorize caller→tenant on `/run`.
*(port added; `/run` now enforces `enabled_surfaces`.)*

### G4. Log surface can't feed raw logs to an LLM  → detectors not wired
A day of logs is ~2M lines. The `Detector` port exists but nothing uses it. Without a
**detect → aggregate → sample** stage, the log surface is infeasible.
**Fix:** log-review evidence pipeline = rule-based detectors first (cheap), aggregate,
sample only the candidates into `diagnose`. Register detectors like adapters.

### G5. PII / secrets leave the trust boundary to the LLM provider
Logs contain PII (emails, mobiles) and code may contain secrets. Nothing redacts before the
LLM call. Data-governance + injection surface.
**Fix:** a redaction stage (regex + allowlist) on evidence before any provider call; make it
a port so tenants can tune it.

## P1 — before real use

### G6. No audit trail
For an agent that takes actions, there's no immutable record of *what was decided, on what
evidence, cited at what scores, approved by whom, posted where*. Can't debug or prove.
**Fix:** an `AuditSink` port; write one record per run (inputs, route, diagnosis, booleans,
approver identity, action result). *(port added.)*

### G7. No idempotency  → re-runs double-post
Re-running code review on the same PR, or the daily job twice, re-posts comments / re-emits
reports. `execute` posts verbatim with no "already done" guard.
**Fix:** idempotency key = (tenant, surface, target, content-hash); check before `execute`.

### G8. Resume can use a *different* prompt/policy than the run started with
`await_approval` suspends; a human resumes hours later. If the tenant edited prompts/policy
meanwhile, the resumed run mixes versions.
**Fix:** snapshot the resolved prompt+policy (+ a version id) into the checkpointed state at
run start; pin it for the whole run.

### G9. No failure handling / fallback edges  → a node error kills the run
The graph has no error edges. If fetch/retrieval/LLM fails, there's no retry/backoff, no
fallback to `human_review`, no dead-letter.
**Fix:** per-node failure policy; on unrecoverable error route to `human_review` with the
error captured; retry transient calls with backoff + a circuit breaker.

### G10. Corrective-RAG loop (#7) has no cap  → infinite-loop / cost risk
`reject_retrieve_more` re-retrieves; without a max-iteration counter it can loop forever.
**Fix:** `max_retrieve_iters` in state + policy; escalate to human when exhausted.

### G11. enabled_surfaces + policy schema not validated
A typo'd `policy.json` key or a surface a tenant hasn't enabled fails silently at runtime.
**Fix:** validate policy against a schema at load/onboarding (fail fast); `/run` rejects a
surface not in the tenant's `enabled_surfaces`. *(enabled_surfaces check added.)*

## P2 — hardening
- **G12.** No cost/token budget or per-tenant rate limiting → runaway cost; add budgets + limits.
- **G13.** No human-notification channel (Slack/email) + no approver authZ (who may approve
  for tenant A?) → add a `Notifier` port + approval authorization. *(Notifier port added.)*
- **G14.** Tenant prompts fully replace `_default` (no inheritance) → add `{{include _default}}`
  so tenants write only their delta.
- **G15.** No prompt/policy cache + reload strategy (fs read per request; edits need a bust).
- **G16.** Scheduled log job shares web workers → run it as its own RunTask/queue (already the
  intended run mode; keep them isolated).
- **G17.** Capability eval (diagnosis quality) still absent — safety-gate suite only.

## Closed in this pass
Retriever `namespace` param (G1 port), `Surface` delegation hook in `classify` (G2 start),
`CredentialProvider`/`AuditSink`/`Notifier`/`InputGuard` ports (G3/G5/G6/G13), and
`enabled_surfaces` enforcement on `/run` (G11). The rest are tracked in ROADMAP.md.
