# Roadmap / deliberate scope cuts (own these precisely)

Represent designed-but-not-built things in the SCHEMA and document each cut in the
node docstring + this file. Honesty about scope reads stronger than vague confidence.

- [ ] Real GitHub/GitLab client (currently stubbed, raises NotImplementedError)
- [x] Real vector index  *(T10: `tools/ingest.py` builds a namespaced on-disk index;
      `IndexedRetriever` is the default. Masterclass and MilesOne are indexed)*
- [~] Cross-encoder rerank + MMR  *(T13: reranker built, OFF by default on measurement --
      J 0.31 vs the cosine's 0.68; MMR still a TODO)*
- [ ] `reject_retrieve_more` corrective-RAG loop (not wired)
- [x] Prompt-injection input guard + adversarial eval class  *(GUARD_TASKS in eval/tasks.py)*
- [ ] Multi-provider capability eval (OpenAI vs Claude vs local)
- [x] AsyncPostgresSaver + real interrupt/resume  *(lifespan owns the async lifecycle;
      the Postgres path itself is written but UNVERIFIED — no Postgres on either machine)*

## Architecture gaps (from docs/ARCHITECTURE_GAPS.md — prioritized)
P0 (must close):
- [x] G1 tenant-scoped retrieval (namespace per codebase)   *(port + ENFORCEMENT: an
      adapter that cannot honour a namespace refuses; T10's namespaced ingest is real, and
      merger_app + milesone retrieve from their own index)*
- [x] G2 graph dispatches through the active Surface          *(T5: source/query/classify/prompts/sink)*
- [ ] G3 per-tenant credentials (secret ref) + API authN/Z    *(CredentialProvider port done)*
- [x] G4 log-review detect→aggregate→sample pipeline  *(adapters/detectors.py + source_s3logs)*
- [x] G5 PII/secret redaction before any LLM call (InputGuard)  *(T12: redaction.py, registered by adapters/input_guard.py;
      12 adversarial gates. Prose injection is COUNTED, not rewritten -- see the module docstring)*
P1 (before real use):
- [x] G6 AuditSink: one immutable record per run              *(jsonl adapter; docs/DATA_STORAGE.md)*
- [ ] G7 idempotency key -> no double-post on re-runs
- [ ] G8 pin resolved prompt/policy (+version) into checkpointed state at run start
- [~] G9 failure edges: retry/backoff + fallback to human_review + dead-letter  *(retry with
      backoff done; a crashed run writes a `failed` audit record. Failure EDGES still open)*
- [ ] G10 corrective-RAG loop cap (max_retrieve_iters)
- [x] G11 validate policy schema; enforce enabled_surfaces    *(`schemas/policy.py`, 2026-11-17)*
P2 (hardening):
- [ ] G12 cost/token budgets + per-tenant rate limits
- [ ] G13 Notifier + approver authorization                   *(port done)*
- [ ] G14 prompt inheritance ({{include _default}})
- [ ] G15 prompt/policy cache + reload/bust
- [ ] G16 isolate the scheduled log job from web workers
- [ ] G17 capability eval (diagnosis quality) alongside the safety-gate suite

## Accuracy track (best-practices, cost-no-object — from docs/BEST_PRACTICES.md)
Retrieval: [~] cross-encoder reranker (+ensemble; built, off) [ ] MMR [ ] contextual retrieval
[ ] multi-query+HyDE fusion [ ] AST/tree-sitter code chunking [ ] parent-doc expansion
[ ] SPLADE sparse [ ] ensemble embeddings [ ] retrieval eval (hit@k/MRR/NDCG).
Code review tools (ground truth): [ ] ruff/pylint [ ] mypy/pyright [ ] semgrep/bandit
[ ] gitleaks/trufflehog [ ] sandboxed tests + diff-coverage [ ] AST call-graph/blast-radius.
Log tools: [ ] Drain parsing [ ] statistical baselines/drift [ ] correlation-ID stitching.
Reasoning: [ ] self-consistency [ ] model ensemble [ ] multi-judge jury [ ] reflection pass
[ ] claim-verification chain [ ] debate for hard cases.
Eval/obs: [ ] RAG-triad+G-Eval [ ] adversarial+golden sets [ ] human calibration set
[ ] trajectory eval [ ] OpenTelemetry+Langfuse tracing [ ] per-run version pinning.
