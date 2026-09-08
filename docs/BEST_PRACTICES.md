# Engineering standards — accuracy-first agentic AI (cost is no object)

North star: **maximum accuracy and groundedness**, running locally. Because cost is not a
constraint, we deliberately spend compute to buy correctness: ensembles, multiple judges,
self-consistency, wide retrieval + heavy reranking, and real deterministic tools feeding the
LLM. Status tags: **[have]** in the skeleton · **[design]** decided, not built · **[add]** to build.

## 0. The core accuracy bet
**Detectors/tools find; the LLM explains and prioritizes; code decides.** We never ask the
model to be the source of truth. Real analyzers/tests/detectors produce facts; the LLM
grounds on them; Python booleans gate the action. This is why the whole system stays
verifiable. **[have]** (gate) / **[add]** (tools + detectors below).

## 1. Retrieval & grounding (RAG) — the full stack
- **Hybrid retrieval**: dense + sparse (BM25 now; add **SPLADE**) fused with **RRF**. **[have]**
- **Reranking**: retrieve wide (top-100) → **cross-encoder rerank** → top-n. Add a **reranker
  ensemble** (2+ cross-encoders) and average. **[add]** (`Reranker` port added)
- **MMR** diversity so citations aren't near-duplicates. **[design]**
- **Contextual retrieval** (Anthropic): prepend an LLM-written one-line context to each chunk
  at ingest. **[add]**
- **Multi-query expansion**: generate N query paraphrases + **HyDE** hypothetical answer,
  retrieve for each, fuse with RRF. (cost-free → do it.) **[add]**
- **Query decomposition** for complex asks; **self-query** metadata filters. **[design]**
- **Parent-document / small-to-big**: retrieve small chunks, expand to enclosing function/file
  for context. **[add]**
- **Code-aware chunking**: **AST / tree-sitter** chunking (by function/class), not fixed chars —
  critical for code review. **[add]**
- **Corrective RAG (CRAG)**: grade retrieval; if weak, rewrite + re-retrieve (capped loop). **[design G10]**
- **Ensemble embeddings**: 2 embedding models, fuse. **[add]**
- **Per-codebase namespace isolation** — never retrieve across tenants. **[have port G1]**
- **Retrieval eval**: hit@k, Recall@k, MRR, NDCG on a golden set; prove rerank earns its place. **[add]**

## 2. Code-review accuracy — deterministic tools FIRST (the biggest lever)
Run real tools in a sandbox and ground the review on their output:
- **Linters/formatters**: ruff, pylint, black-check, eslint (per language). **[add]**
- **Type checkers**: mypy / pyright; tsc; go vet. **[add]**
- **SAST / security**: **semgrep**, bandit; dependency CVE scan (pip-audit/osv). **[add]**
- **Secret scanners**: gitleaks / trufflehog on the diff. **[add]**
- **Tests + coverage**: run the suite in a sandbox; **diff-coverage**; ground on real failures. **[add]**
- **AST / call-graph / blast-radius**: what else calls the changed symbol. **[add]**
- **Convention learning**: per-repo style/patterns as long-term memory. **[design]**
The LLM synthesizes, dedupes, and prioritizes tool findings + adds semantic review the tools
can't (design, correctness, edge cases) — always cited. `ToolRunner`/`StaticAnalyzer` ports added.

## 3. Log-review accuracy — detectors & parsing FIRST
- **Log parsing/templating** (Drain-style) → structured events; dedup. **[add]**
- **Deterministic detectors**: thresholds, rate spikes, new error signatures, task-failure,
  "signal received but no downstream action", replica-lag, pool exhaustion. `Detector` port. **[design G4]**
- **Statistical baselines / seasonality / drift** vs prior days; anomaly scores. **[add]**
- **Cross-service correlation** via correlation/trace IDs. **[add]**
- **Aggregate + sample** candidates only into the LLM (never raw 2M lines). **[design G4]**

## 4. Reasoning accuracy (cost-free → spend it)
- **Self-consistency**: sample the diagnosis N times, aggregate/vote. **[add]**
- **Model ensemble**: run 2–3 models (e.g. GPT + Claude + a strong local), reconcile. **[add]**
- **LLM-as-jury**: the independent review is a **panel of judges** (odd number), majority on
  each boolean; still code-decides. **[design]** (extends `independent_review`)
- **Reflection / self-critique** pass before finalizing. **[design]**
- **Verification chain**: verify every claim/citation against evidence; drop unverifiable. **[add]**
- **Structured output + schema validation + repair**; constrained decoding. **[design]**
- **Confidence calibration + abstention**: low confidence → escalate, never guess. **[have gate]**
- **Debate** for hard/conflicting cases (two agents argue, judge). **[add]**

## 5. Agent design
- **Plan-and-Execute + ReAct** hybrid; a planner for multi-file changes. **[design]**
- **Supervisor + specialist sub-agents** (security / performance / migrations / tests /
  style) — orchestrator-workers; each grounded on its tools. **[design]**
- **Tools with validation**; **short + long-term memory** (per-codebase conventions). **[design]**
- **HITL** approval gate; **checkpointed** interrupts. **[have]**

## 6. Guardrails (defense in depth)
- **Input guard**: PII/secret redaction + **injection neutralization** (spotlight/delimit
  untrusted evidence). `InputGuard` port. **[have port G5]**
- **Output guard**: schema + policy + safety validation before any action. `OutputGuard`
  port added. **[add]**
- **Code invariants** (not tenant-overridable): judge-explains/code-decides; `execute`
  refuses without approval; evidence is data-not-instructions; no infra/DB mutation. **[have]**
- Optional: NeMo-Guardrails / Llama-Guard-style policy layer. **[design]**

## 7. Evaluation (accuracy is the product → eval is central)
- **Safety-gate regression suite**, `pass^k`, 100% bar, balanced classes, in CI. **[have]**
- **Capability eval**: **RAG-triad** (context-relevance, faithfulness, answer-relevance),
  **G-Eval**, citation-precision/recall, hallucination rate, retrieval metrics. **[add]**
- **Golden datasets** + **adversarial sets** (injection, ambiguity, duplicates). **[design]**
- **Multi-judge** eval + a **human-labeled calibration set** to validate the judges. **[add]**
- **Trajectory eval** (grade the path, not just the final answer). **[design]**
- **Continuous/online eval**, A/B, regression gate on every change. **[add]**

## 8. Observability & reliability
- **Full tracing** (OpenTelemetry + Langfuse/Phoenix): every node, prompt, token, tool call,
  retrieval, score, latency. `Tracer` port added. **[add]**
- **Immutable audit trail** per run. `AuditSink` port. **[have port G6]**
- **Deterministic replay** from checkpoints. **[design]**
- **Reliability**: retries + backoff, circuit breakers, fallback→human_review, dead-letter,
  **idempotency** keys. **[design G7/G9]**
- **Reproducibility/versioning**: pin prompt+policy+model+index versions **per run**,
  snapshot into checkpoint. **[design G8]**

## 9. Security & governance
- **Tenant isolation**: per-codebase namespace + **per-tenant secrets (secret ref)** +
  **API authN/Z**. **[have ports G1/G3]** / **[add]** authZ
- **Least privilege**, read-only, secrets via vault; no raw tokens in config. **[design]**
- **Data governance**: redaction before any provider call; what may leave the boundary. **[design G5]**

## 10. LLMOps
- **Registries**: prompt registry (versioned, per-tenant), model registry, **index/corpus
  versioning** (weekly re-ingest + immutable versions). **[design]**
- **CI**: eval gates on every change; canary + rollback; drift monitoring. **[add]**

## 11. Prompting standards
Role + explicit output schema + few-shot exemplars + **delimited/spotlighted untrusted
input** + chain-of-thought + self-check + refusal criteria. Prompts are **versioned data**
(per-tenant, `tenants/`), covered by eval. **[have]**

## 12. Deliberately NOT doing
- Not letting any LLM emit the final decision (code decides).
- Not mutating infra/DB/repos autonomously (human approves; writes as the approver).
- Not trading the safety-gate's 100% bar for "capability" averages.
