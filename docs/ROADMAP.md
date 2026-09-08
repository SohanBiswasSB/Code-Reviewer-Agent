# Roadmap / deliberate scope cuts (own these precisely)

Represent designed-but-not-built things in the SCHEMA and document each cut in the
node docstring + this file. Honesty about scope reads stronger than vague confidence.

- [ ] Real GitHub/GitLab client (currently stubbed, raises NotImplementedError)
- [ ] Real vector index (retrieval.py runs on a tiny in-memory corpus)
- [ ] Cross-encoder rerank + MMR (RRF fusion is real; rerank/MMR are TODOs)
- [ ] `reject_retrieve_more` corrective-RAG loop (not wired)
- [ ] Prompt-injection input guard + adversarial eval class
- [ ] Multi-provider capability eval (OpenAI vs Claude vs local)
- [ ] AsyncPostgresSaver + real interrupt/resume (skeleton uses MemorySaver semantics)

## Architecture gaps (from docs/ARCHITECTURE_GAPS.md — prioritized)
P0 (must close):
- [ ] G1 tenant-scoped retrieval (namespace per codebase)   *(port done; wire adapters + ingest)*
- [ ] G2 graph dispatches through the active Surface          *(classify hook done; do fetch/diagnose/execute)*
- [ ] G3 per-tenant credentials (secret ref) + API authN/Z    *(CredentialProvider port done)*
- [ ] G4 log-review detect→aggregate→sample pipeline (wire the Detector port)
- [ ] G5 PII/secret redaction before any LLM call (InputGuard)
P1 (before real use):
- [ ] G6 AuditSink: one immutable record per run              *(port done; add fs/db adapter)*
- [ ] G7 idempotency key -> no double-post on re-runs
- [ ] G8 pin resolved prompt/policy (+version) into checkpointed state at run start
- [ ] G9 failure edges: retry/backoff + fallback to human_review + dead-letter
- [ ] G10 corrective-RAG loop cap (max_retrieve_iters)
- [ ] G11 validate policy schema; enforce enabled_surfaces    *(enforcement done; add schema check)*
P2 (hardening):
- [ ] G12 cost/token budgets + per-tenant rate limits
- [ ] G13 Notifier + approver authorization                   *(port done)*
- [ ] G14 prompt inheritance ({{include _default}})
- [ ] G15 prompt/policy cache + reload/bust
- [ ] G16 isolate the scheduled log job from web workers
- [ ] G17 capability eval (diagnosis quality) alongside the safety-gate suite

## Accuracy track (best-practices, cost-no-object — from docs/BEST_PRACTICES.md)
Retrieval: [ ] cross-encoder reranker (+ensemble) [ ] MMR [ ] contextual retrieval
[ ] multi-query+HyDE fusion [ ] AST/tree-sitter code chunking [ ] parent-doc expansion
[ ] SPLADE sparse [ ] ensemble embeddings [ ] retrieval eval (hit@k/MRR/NDCG).
Code review tools (ground truth): [ ] ruff/pylint [ ] mypy/pyright [ ] semgrep/bandit
[ ] gitleaks/trufflehog [ ] sandboxed tests + diff-coverage [ ] AST call-graph/blast-radius.
Log tools: [ ] Drain parsing [ ] statistical baselines/drift [ ] correlation-ID stitching.
Reasoning: [ ] self-consistency [ ] model ensemble [ ] multi-judge jury [ ] reflection pass
[ ] claim-verification chain [ ] debate for hard cases.
Eval/obs: [ ] RAG-triad+G-Eval [ ] adversarial+golden sets [ ] human calibration set
[ ] trajectory eval [ ] OpenTelemetry+Langfuse tracing [ ] per-run version pinning.
