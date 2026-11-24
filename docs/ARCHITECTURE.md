# Architecture (as built, 2026-11-17)

A plug-and-play, multi-tenant agentic microservice. Three ideas stack:
**(1) a guarded agent graph**, **(2) ports & adapters** so every provider is swappable,
**(3) multi-tenant** so each plugged-in codebase brings its own prompts + guardrails.

The bet underneath all three: **detectors and tools find, the LLM explains and
prioritizes, and code decides.** No model output becomes an action without four Python
booleans and a human.

This file describes what the code does today. Planned work is marked as planned and
lives in `docs/ARCHITECTURE_GAPS.md` (G1–G17) and `TODO.md` (T1–T28).

## System view

```text
 request: {tenant, surface, target}   tenant = a CODEBASE   target = staged | <rev> | a..b |
        │                                                            PR ref | YYYY-MM-DD
        ▼
 ┌──────────────── ENTRY POINTS (one codebase, several run modes) ──────────────────────────┐
 │  FastAPI app/main.py   ·   CLI app/review.py   ·   jobs/*   ·   eval/harness.py           │
 └──────────────────────────────────────┬───────────────────────────────────────────────────┘
                                        ▼
 ┌──────────── CORE GRAPH (LangGraph, app/graph.py) — codes against ports/ only ────────────┐
 │                                                                                          │
 │  resolve_tenant ─ policy + prompts resolved ONCE, validated (G11), pinned into state    │
 │       ▼                                                                                  │
 │  fetch_evidence ─ EvidenceSource; whole payload REDACTED before the first write (T25)    │
 │       ▼                                                                                  │
 │  normalize_evidence ─ surface-specific shape; tenant risky_areas applied (T27)           │
 │       ▼                                                                                  │
 │  prepush_checks ─ deterministic pass/flag/skip facts (never route, never decide)        │
 │       ▼                                                                                  │
 │  classify ─ RULE-BASED, zero LLM calls ─┬─ deterministic ──────────► await_approval      │
 │                                         ├─ human_review ───────────────────────► audit   │
 │                                         └─ ai_investigation                              │
 │                                               ▼                                          │
 │                      generate_diagnosis ─ RAG per changed file, cited findings          │
 │                                               ▼                                          │
 │                      independent_review ─ a DIFFERENT model explains; PYTHON decides:    │
 │                         groundedness_ok · entailment_ok · risk_ok · permission_ok        │
 │                              ├─ approve ──────────► await_approval                       │
 │                              └─ escalate_to_human ──────────────────────────► audit      │
 │                                                                                          │
 │  await_approval ─ interrupt() → checkpoint → human resumes on the same thread_id         │
 │       ▼                                                                                  │
 │  execute ─ refuses unless approved is True (in the node body); posts VERBATIM            │
 │       ▼                                                                                  │
 │  audit ─ one immutable record per run ──► END                                            │
 │                                                                                          │
 │  Every node runs inside _traced: trace span + activity-log span + live status, and a    │
 │  crash writes a `failed` audit record before re-raising (G9).                           │
 └──────────────────────────────────────┬───────────────────────────────────────────────────┘
                                        │ config.get_*()  (env selects, registry resolves)
   ┌──────────┬──────────┬──────────────┼──────────┬───────────┬──────────┬──────────────┐
   ▼          ▼          ▼              ▼          ▼           ▼          ▼              ▼
  LLM     Retriever  EvidenceSource  ActionSink  Entailer  InputGuard  TenantStore    Surface
 lmstudio  indexed    git_local      review_file lexical    regex       fs           code_review
 openai*   hybrid     github†        pr_comment† cross_     null(test)               log_review
 stub      (demo)     s3logs†        report       encoder
           ├ Embedder: lmstudio   ├ VectorIndex: local   ├ Reranker: cross_encoder (off)
   AuditSink: jsonl · Tracer: jsonl · Detector: signals · Platform: cpu|cuda|mac|studio
   (* stub raising NotImplementedError   † adapter real, network call stubbed)
```

## Layers

| Layer | Where | What it owns |
|---|---|---|
| **Core graph** | `app/graph.py`, `nodes/`, `app/state.py` | The safety design: rule-based routing, judge-explains/code-decides, checkpointed approval, the in-node `execute` guard, one audit record per run. Provider-agnostic. |
| **Ports** | `ports/interfaces.py` | Structural `Protocol`s. Core codes against these shapes and reaches providers only through `config.get_*()`; nothing in `nodes/` imports a concrete adapter (G2, closed). |
| **The plug** | `registry.py`, `config.py` | `@adapter(kind, name)` / `@surface(name)` register at import; env vars choose (`LLM_PROVIDER`, `RETRIEVER`, `RERANKER`, `ENTAILER`, `CRA_GUARD`, …). Caches (retriever, reranker, entailer) are shared per process. |
| **Adapters** | `adapters/` | The concrete providers in the diagram above. Adding one = a module + a decorator + one line in `adapters/__init__.py`. |
| **Surfaces** | `surfaces/` | A capability bound onto the shared graph: source, `normalize`, `build_queries`, rule-based `classify`, prompt keys, sink, action type. |
| **Tenants** | `tenants/<name>/` | One directory per codebase: `policy.json` (codebase binding, thresholds, allowlists, `risky_areas`, pre-push conventions) + `prompts/<surface>/<node>.md`. `_default` is deep-merged under every tenant. |
| **Hardware** | `adapters/platforms/` | The ONLY code that knows the machine. Two axes: `CRA_PLATFORM` (how: context, KV dtype, residency, device) × `CRA_PROFILE` (which models). |
| **Schemas & errors** | `schemas/`, `errors.py` | Pydantic evidence/diagnosis models (registered for the checkpoint serialiser), policy validation, and the error FAMILIES each boundary may catch. |
| **Tools** | `tools/` | Pure helpers: chunking, BM25, RRF, ingest, calibration, pre-push checks (`tools/prepush/`), `codebase_root`. |

## The ports, and what implements them

| Port | Implemented by | Notes |
|---|---|---|
| `LLM` | `lmstudio`, `stub` (`openai` raises) | Role-bound: `get_llm("diagnosis")` and `get_llm("judge")` are different model families (invariant 2). Schema-constrained decoding; transient failures retried (`with_retry`). |
| `Retriever` | `indexed` (default), `hybrid` (offline demo) | Namespace REQUIRED (G1). Dense cosine + BM25, RRF for order, the cosine is the gate score. |
| `Embedder` / `VectorIndex` | `lmstudio` / `local` | EmbeddingGemma-300m, 768 dims, pinned. The index records its embedder in a manifest and is refused by a different one. |
| `Reranker` | `cross_encoder` | bge-reranker-v2-m3, in-process. **Off by default, on measurement** (see Retrieval). |
| `Entailer` | `lexical` (default), `cross_encoder` | Returns a score; the gate thresholds it. The NLI model scores long premises in overlapping windows, best window wins. |
| `EvidenceSource` | `git_local`, `github`, `s3logs` | Per-surface override: `CRA_SOURCE_<SURFACE>`. |
| `ActionSink` | `review_file`, `pr_comment`, `report` | Per-surface override: `CRA_SINK_<SURFACE>`. A partial failure is reported, not raised. |
| `InputGuard` | `regex` (default), `null` (tests) | Redacts secrets/PII with numbered placeholders, defangs fence forgery, COUNTS injection phrasing. |
| `AuditSink` / `Tracer` | `jsonl` (default), `null` (tests) | Real by default; silence must be opted into. |
| `Detector` | `signals` | Regex + counting over the full log stream, before any model. |
| `TenantStore` | `fs` | Prompts and policies from `tenants/`. |
| `Platform` | `cpu`, `cuda`, `mac`, `studio` | `studio` is never auto-detected. |

**Declared ahead of any implementation** — they fix the shape of a planned seam, and
nothing calls them yet: `Checkpointer` (LangGraph's savers are used directly),
`CredentialProvider`, `Notifier`, `ToolRunner`, `OutputGuard`, `Evaluator`.

## The code-review path, as built

What a review actually does today, in order (each step names the module that owns it):

1. **Evidence** — `git_local` (or `github`) reads the diff, packed into a 48k-char budget
   (code, then config, then docs; data files named but never shown); `fetch_evidence`
   redacts it before anything reaches state (T25).
2. **Pre-push checks** — `nodes/prepush_checks.py` → `tools/prepush/` (`change`,
   `n_plus_one`, `hygiene`, `conventions`): deterministic `pass` / `flag` / `skip` results that
   reach the prompt, the review and the audit record. They never route or decide.
3. **Routing** — `classify` is rule-based. A tenant's `risky_areas` is unioned with the
   surface defaults (T27); anything not risky goes to a human.
4. **Retrieval, per changed file** — `CodeReviewSurface.build_queries` makes one query per
   changed file plus one for the whole change; `IndexedRetriever` fuses dense cosine and BM25
   with RRF for order and keeps the cosine as the gate score; `generate_diagnosis.merge_hits`
   round-robins, deduplicates and caps at 8 passages. The cross-encoder reranker (T13) exists
   behind `RERANKER=cross_encoder` and is off by default, on measurement.
5. **Diagnosis** — the diagnosis model writes cited findings with schema-constrained decoding,
   from a user turn where every untrusted block (change, retrieved passages, checks) is
   guarded and fenced. Unparseable output fails safe (severity `high`).
6. **Judge** — a different model family explains; it never decides (invariant 1).
7. **Gate** — four Python booleans:
   - `groundedness_ok`: every citation was really retrieved and its retriever score clears the
     bar for ITS kind of score (`min_relevance_score` for a cosine — 0.75 for the real
     tenants — or `min_rerank_score` for a rerank score);
   - `entailment_ok`: per claim — the comment and each finding scored separately, a finding
     against its own citations plus its own file's diff; the lowest claim decides;
   - `risk_ok`: severity in the tenant's allowlist;
   - `permission_ok`: action type in the tenant's allowlist.
8. **Approval and action** — the action is built once, on approve, by `independent_review`
   (comment + findings, then the code-generated pre-push checklist, which is never
   entailed); `await_approval` interrupts and shows exactly that; `execute` re-checks
   `approved`; the sink posts it verbatim, as the approver.
9. **Record** — every path, crashes included, writes one audit record (G6, G9); every node,
   LLM call and embedding is in the activity log; the monitor shows both.

## The log-review path

Same graph, different surface (`surfaces/log_review.py`), advisory only — it never comments
or labels. Layer 0 happens in the SOURCE: `s3logs` streams a day of logs through the
`signals` detector, aggregates candidates and keeps a capped sample (measured: 200,000 lines
→ 2,041 bytes of evidence). The models explain the candidates; the action is a `report`.
The S3 read itself is still stubbed (`reader=` is injectable, which is how the figure was
measured), and `jobs/daily_log_review.py` is a stub.

## Retrieval and the index

- **Ingest** (`tools/ingest.py`, run once per codebase): AST chunking for Python, text chunks
  otherwise, tests/fixtures skipped, tenant `index_exclude` globs; the InputGuard runs BEFORE
  embedding; path + symbol name prefixed to the embedded text. Writes
  `var/index/<namespace>/` = `vectors.npy` + `chunks.jsonl` + `manifest.json` + `bm25.npz`
  (keyed by the index fingerprint). Built: Masterclass and MilesOne (15,306 chunks).
- **Query**: dense top-N + BM25 top-N → RRF for order → the cosine as `score`; a keyword-only
  hit has its cosine looked up, so nothing reaches the gate without a number (T21).
- **Calibration** (`tools/calibrate.py relevance | rerank | entailment`) measures; it never
  edits a policy. Measured: EmbeddingGemma cosines sit in ~0.7–0.9, so the old 0.40 bar could
  never fail; 0.75 is the interim bar. On the same pools the reranker separated same-file
  from other-file passages worse than the cosine (Youden J 0.31 vs 0.68) at 13.5 s/query,
  which is why it is off.

## State, storage and the trust boundary

- **Checkpoint** — LangGraph state (`app/state.py`, every channel declared; every model in
  `CHECKPOINT_TYPES`). Postgres in production, `MemorySaver` for the CLI and tests. A
  suspended thread is a pending approval and never expires (invariant 10).
- **`var/`** (`CRA_DATA_DIR`, never committed): `audit/` (one record per run), `traces/`
  (spans), `activity/` (the operational log, day files), `live/` (per-run status for the
  monitor), `index/`, `reviews/` (approved local reviews), `test_failures.jsonl`.
- **Redaction happens twice, both before storage**: on fetch (the whole evidence payload,
  identifiers untouched) and on the way into every prompt. The approver sees placeholders
  where secrets were. Nothing logs prompt or evidence text — counts and ids only.
- **Retention**: `jobs.prune_checkpoints` (terminal runs only, live state re-checked) and
  `jobs.prune_activity` (by the UTC date in the file name). Both dry-run by default.

## Observability

- **Audit record** — outcome (`executed`, `escalated`, `human_review`, `denied`, `failed`,
  `interrupted`, `abandoned`), route, policy/prompt versions, guard report, checks,
  retrieved ids with scores and score kinds, findings, the four booleans and failures,
  runtime (models, platform, profile).
- **Activity log** (`app/activity.py`) — spans for every node, LLM call, query embedding,
  approval, API request and job, one correlation id per run; secrets masked on the way in;
  library warnings bridged in.
- **Monitor** (`python -m app.monitor`, 127.0.0.1:8765 only, read-only): runs, the step list
  (drawn from `STAGE_PLAN`, tested against the compiled graph), gate results, findings, the
  model server, and the Activity log panel.
- **Doctor** (`python -m app.doctor`): can this machine run it — models served, indexes
  built by the right embedder, policies valid, thresholds sane.

## Run modes

- **web** — `uvicorn app.main:app`: `POST /v1/run/{surface}`, `POST /v1/approve`,
  `GET /v1/runs/{thread_id}`, `GET /v1/registered`, `GET /v1/doctor`, and the unversioned
  probe `GET /healthz`. One Pydantic request and response model per route
  (`app/api_models.py`); an undeclared or mistyped field is a 400; every body, errors
  included, is `{success, message, data}`. Echoes `X-Correlation-Id`.
- **local review** — `python -m app.review --tenant <t> --root <repo> [--target <rev>]`: the
  full graph over a local diff (`git_local` source → `review_file` sink, `var/reviews/`),
  approval at the terminal; without a terminal the run is recorded as `abandoned`.
- **index** — `python -m tools.ingest --tenant <t> --root <repo>` builds the tenant's
  namespaced index (vectors + `bm25.npz`) once; a review never builds it implicitly.
- **monitor** — `python -m app.monitor`: a read-only page on 127.0.0.1 over `var/` (runs,
  steps, gate, findings, model server, activity log). It launches nothing.
- **calibrate** — `python -m tools.calibrate ... relevance|entailment|rerank`: measures a gate
  threshold and prints a suggestion; it never edits a policy.
- **daily** — `python -m jobs.daily_log_review --date <d>` on **EventBridge→ECS RunTask**
  (stubbed).
- **housekeeping** — `jobs.prune_checkpoints`, `jobs.prune_activity` (dry run by default).
- **eval** — `python -m eval.harness` (safety gates, 100% bar); `python check.py` is the gate
  for every change (gate suite first, then lint, both pytest runs and the demo in parallel).

## Hardware and models

`CRA_PLATFORM` decides HOW (context budget, KV-cache dtype, residency computed from weight
cost against a memory budget, reranker device); `CRA_PROFILE` decides WHICH models. LM Studio
serves the LLMs and the embedder (locally, or the `studio` host over a tailnet — LM Studio has
no auth); the reranker and NLI entailer run in-process. The embedder and reranker are pinned
across every profile, because changing the embedder changes what every relevance bar means.

## Non-negotiable invariants (not tenant-overridable)

The full list, with the incident behind each, is in `CLAUDE.md`. In short:

- The judge never emits a verdict; the decision is Python booleans over real data.
- `diagnosis` and `judge` are different model families.
- `execute` refuses without `approved is True`, checked in the node body.
- The approved action is built once and posted verbatim.
- Evidence is untrusted DATA: redacted before storage, guarded and fenced before any model.
- Retrieval is always namespaced per codebase; a retriever that cannot, refuses.
- Nothing mutates infra or a database; a human approves, and writes run as that human.
- Every terminal path — crashes included — writes one audit record.
- The checkpointer is required; fail in the safe direction everywhere.

Tenants tune *values* (thresholds, allowlists, prompts, extra risky areas). They cannot remove
a *boundary*.

## Known gaps

See **docs/ARCHITECTURE_GAPS.md** for the full audit (P0–P2). Closed: **G1** (namespaced
retrieval, real indexes), **G2** (every node dispatches through the active `Surface`; the only
`tools/` imports in `nodes/` are pure helpers), **G5/T12** (InputGuard; injection counted,
not neutralised — the honest limit), **G6** (audit), **G8** (prompt + policy versions
pinned), **G11** (`schemas/policy.py`), and **T25** (redaction before the first state
write). **G9 is half closed**: transient model calls retry with backoff, and a crashed node
writes a `failed` audit record before re-raising, but there are still no failure *edges* to
`human_review`. Open: **G3** (no caller authN/Z on `/run` — the biggest structural item),
**G7** (idempotency: the same review can be posted twice), G10 (no corrective-RAG loop yet,
so nothing to cap), G12–G15 (budgets, Notifier + approver authZ, prompt inheritance, cache),
G17 (capability eval). The entailment bar (0.25) has not been re-measured since claims
became per-finding.

## Accuracy-first stance (cost is no object)

Because this runs locally with no cost constraint, the design maximises correctness (see
`docs/BEST_PRACTICES.md`). What is built, and what is still the plan:

**Built:** facts before opinions (pre-push checks for code, regex detectors for logs, both
before any model); hybrid retrieval with AST-aware chunking and per-file queries; a
separate-family judge; schema-constrained decoding; a per-claim entailment check; a 100%
safety-gate regression suite; full tracing, an immutable audit record, an activity log, and
prompt/policy versions pinned per run; thresholds measured with `tools.calibrate`.

**Built, not on:** the cross-encoder reranker and the NLI entailer — each waits on a
measurement showing it earns its place.

**Planned:** linters / mypy / semgrep / tests as `ToolRunner` facts; multi-query + HyDE
fusion; self-consistency, model ensembles and a multi-judge jury on the gate; capability eval
(RAG triad, G-Eval, citation precision) over golden and adversarial sets (`Evaluator`); an
`OutputGuard` on proposed actions; a human-labelled relevance set to settle the reranker.
