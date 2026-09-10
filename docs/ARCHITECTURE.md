# Architecture (as-built, v1)

A plug-and-play, multi-tenant agentic microservice. Three ideas stack:
**(1) a guarded agent graph**, **(2) ports & adapters** so every provider is swappable,
**(3) multi-tenant** so each plugged-in codebase brings its own prompts + guardrails.

## System view
```
 request: {tenant, surface, target}         (target = PR ref | issue url | YYYY-MM-DD)
        │                                     tenant  = a CODEBASE (repo A, repo B, ...)
        ▼
 ┌──────────────────────────── AGENT MICROSERVICE (FastAPI) ─────────────────────────────┐
 │  resolve tenant  ──►  TenantStore (port)                                                │
 │     · prompt(tenant, surface, node)  → tenant file, else _default                       │
 │     · policy(tenant, surface)        → _default deep-merged with tenant                 │
 │              │ prompts + policy (thresholds, allowlists, codebase repo binding)         │
 │              ▼                                                                           │
 │   CORE GRAPH (LangGraph) — depends ONLY on ports/, never a concrete provider            │
 │                                                                                         │
 │   fetch_evidence → normalize → classify(RULE-BASED) ─┬─ deterministic → await_approval  │
 │                                                      ├─ human_review  → END             │
 │                                                      └─ ai_investigation                │
 │                                                            → diagnose (RAG, tenant prompt)│
 │                                                            → judge     (tenant prompt;    │
 │                                                              EXPLAINS only)               │
 │                                                                 │ 4 python booleans use   │
 │                                                                 │ the TENANT policy       │
 │                                                     approve ────┘   escalate_to_human →END│
 │                                                            ▼         (SHARED safety net)  │
 │                                              await_approval  interrupt() → checkpoint     │
 │                                                            ▼  (resume: same thread_id)    │
 │                                              execute  (guarded; PermissionError in-node)  │
 │              │ (ports)                                                                   │
 │   ┌──────────┴──────────┬───────────────┬──────────────┬───────────────┬─────────────┐  │
 │   ▼                     ▼               ▼              ▼               ▼             ▼  │
 │  LLM                Retriever      EvidenceSource   ActionSink      TenantStore    Surface│
 │  openai|claude|…    hybrid|        github|gitlab|   pr_comment|     fs|db          code_review│
 │                     pinecone|pgv   s3logs          report                         log_review │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Layers
1. **Core graph (`nodes/`, `app/graph.py`)** — provider-agnostic. Holds the safety design:
   reasoning≠execution as code paths, judge-explains/code-decides, checkpointed HITL,
   in-node `execute` guard. These are org-wide invariants; tenants tune values, not the boundary.
2. **Ports (`ports/interfaces.py`)** — `LLM, Retriever, EvidenceSource, ActionSink, Detector,
   Checkpointer, TenantStore, Surface`. The only things core imports.
3. **Adapters (`adapters/`)** — concrete plugins; self-register via `@adapter(kind, name)`.
   Chosen by env in `config.py`. Swap GitHub↔GitLab, OpenAI↔Claude, Pinecone↔pgvector,
   fs↔db tenant store — no core change.
4. **Surfaces (`surfaces/`)** — capabilities (`code_review`, `log_review`) registered via
   `@surface(name)`; each binds a source + prompt keys + sink + a rule-based `classify`.
5. **Tenants (`tenants/`)** — one dir per **codebase**: `codebase` binding + `policy.json` +
   `prompts/<surface>/<node>.md`. `_default` is the baseline everyone inherits.

## Request flow (both surfaces)
1. Resolve `tenant` → prompts + policy (incl. which repo/logs to read).
2. `fetch_evidence` via the tenant's `EvidenceSource` (its repo diff, or its S3 logs).
3. `normalize` → validated evidence; `classify` (rule-based) picks the route.
4. `ai_investigation`: `diagnose` (RAG, tenant's diagnosis prompt) → `judge` (tenant's judge
   prompt, reasoning only) → **4 booleans read the tenant policy** (relevance threshold,
   entailment threshold,
   severity allowlist, allowed actions) → `approve` or **`escalate_to_human`**.
5. `await_approval` → `interrupt()` (durable checkpoint) → human resumes → `execute`
   (guarded, verbatim, as the approver) or the log-review report sink (advisory only).

## Run modes (one image, MilesOne-style)
- **web** — `uvicorn app.main:app` — `POST /run/{surface}`, `POST /approve`, `GET /registered`.
- **daily** — `python -m jobs.daily_log_review --date <d>` on **EventBridge→ECS RunTask**.
- **eval** — `python -m eval.harness` (safety gates, 100% bar).

## Non-negotiable invariants (not tenant-overridable)
- The judge never emits a verdict; the decision is Python booleans over real data.
- `execute` refuses without `approved` (checked in the node body, not just routing).
- Evidence (code diffs, log lines) is untrusted DATA — never instructions.
- Nothing mutates infra/DB; the agent proposes, a human approves.

## Known gaps
See **docs/ARCHITECTURE_GAPS.md** for the v1 architecture recheck (P0–P2). Ports added to
support them: `CredentialProvider` (per-tenant secrets), `InputGuard` (PII/secret redaction
+ injection neutralization), `AuditSink` (immutable run record), `Notifier` (approval
notify + authZ). `Retriever` is now namespace-scoped so retrieval never crosses codebases.
**G2 is closed** (T5): every node dispatches through the active `Surface` — source,
query, classify, prompt keys and sink — and no file under `nodes/` imports `tools/`.
**G1 is enforced** end to end: the namespace comes from `policy.codebase`, and a retriever
that cannot honour one *refuses* rather than retrieving across codebases. The open half of
G1 is ingest — which means every tenant that declares a `codebase` currently cannot
retrieve at all, so only `_default` runs (T10). The biggest open structural items are now
**G5/T12** (no redaction before evidence leaves the trust boundary) and **G9** (no failure
edges, so a node error kills the run instead of routing to a human).

## Accuracy-first stance (cost is no object)
Because this runs locally with no cost constraint, the design maximizes correctness (see
docs/BEST_PRACTICES.md). Load-bearing choices:
- **Facts before opinions:** deterministic tools/detectors produce ground truth
  (linters, mypy, semgrep, secret-scanners, tests for code; parsers + statistical detectors
  for logs). The LLM only explains/prioritizes/grounds; code decides. (`ToolRunner`,
  `Detector` ports.)
- **Spend compute for accuracy:** wide retrieval + cross-encoder **reranker ensemble**,
  multi-query + HyDE fusion, contextual retrieval, AST-aware code chunking; **self-consistency**
  + **model ensemble** + a **multi-judge jury** on the gate.
- **Prove it:** safety-gate suite (have) + capability eval (RAG-triad/G-Eval/citation-precision/
  retrieval metrics) with golden + adversarial sets, in CI. (`Evaluator` port.)
- **See everything:** full tracing (`Tracer`) + immutable audit (`AuditSink`); pin
  prompt/policy/model/index versions per run for reproducibility.
Ports added for this: `Reranker`, `ToolRunner`, `OutputGuard`, `Tracer`, `Evaluator`.
