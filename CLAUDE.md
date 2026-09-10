# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A multi-tenant agentic **code-review microservice**. One guarded LangGraph state
machine is reused across pluggable *surfaces* (`code_review`, `log_review`); each
plugged-in codebase ("tenant") brings its own prompts and its own guardrail policy.

The design bet: **detectors and tools find, the LLM explains and prioritizes, and
code decides.** The LLM is never the source of truth and never emits the verdict.

Read `README.md` first, then `docs/DECISIONS.md` (cross-machine handoff + why
things are the way they are) on a fresh clone.

## Commands

```bash
conda activate CodeReviewerAgent      # env name on both machines

python -m app.doctor                  # can this machine run the agent? (stdlib only)
python -m app.doctor --json
python -m eval.harness                # safety-gate suite — needs NO dependencies
python -m eval.demo_multitenant       # same diagnosis, two codebases, different policies
python -c "import json, config; print(json.dumps(config.registered(), indent=2))"

pip install -r requirements-dev.txt   # tests + lint + local model stack
pytest                                # unit tests (testpaths=tests, -q --strict-markers)
pytest -m needs_server                # deselected by default; needs a live LM Studio
LANGGRAPH_STRICT_MSGPACK=true pytest  # CI also runs this — see "Checkpoint serialisation"
ruff check .                          # line-length 100, target py311

uvicorn app.main:app --reload         # the API
python -m jobs.daily_log_review --date YYYY-MM-DD
python -m jobs.prune_checkpoints --keep-days 30           # dry run by default; --apply to act
```

Requirements files are layered: `requirements.txt` (graph + API) ⊂
`requirements-local.txt` (adds LM Studio client + in-process reranker, pulls torch)
⊂ `requirements-dev.txt` (adds pytest + ruff). `eval/harness.py` deliberately needs
none of them.

## Architecture

```
request {tenant, surface, target}
  → resolve_tenant       (policy + prompts resolved ONCE, pinned into state)
  → fetch_evidence → normalize_evidence → classify    (RULE-BASED, zero LLM calls)
       deterministic     → await_approval → execute → audit → END
       human_review      → audit → END                (no LLM touches it)
       ai_investigation  → generate_diagnosis         (RAG-grounded, cited)
                         → independent_review         (SEPARATE LLM; EXPLAINS only)
                              │ decision = 4 Python booleans over real data:
                              │   groundedness_ok · entailment_ok · risk_ok · permission_ok
                              ├─ approve           → await_approval → execute → audit → END
                              └─ escalate_to_human → audit → END
```

Layered so the core never imports a provider:

| Path | Role |
|---|---|
| `ports/interfaces.py` | The contracts (Protocols). Core imports **only** these. |
| `nodes/`, `app/graph.py` | The graph. Holds the safety design. |
| `adapters/` | Concrete providers, self-registering via `@adapter(kind, name)`. |
| `surfaces/` | Capabilities, self-registering via `@surface(name)`. |
| `tenants/<name>/` | One directory per codebase: `policy.json` + `prompts/<surface>/<node>.md`. |
| `registry.py` / `config.py` | The plug: decorators register, env resolves. |
| `schemas/` | `Diagnosis`, `ReviewResult`, evidence models (Pydantic). |
| `eval/` | Safety-gate regression suite, 100% bar, `pass^k`. Dependency-free. |
| `tests/` | Unit tests. Deliberately separate from the gate suite. |
| `adapters/platforms/` | The **only** code that knows about hardware. |
| `app/doctor.py` | Environment check. Stdlib only, so it runs before setup does. |

## Invariants — do not weaken these

These are not tenant-overridable and not style preferences. Each exists because it
already failed once; the node docstrings carry the incident.

1. **The judge never emits a verdict.** `independent_review` asks a second,
   independent LLM for free-text `reasoning` only. The approve/escalate decision is
   four booleans computed in Python from retrieval scores, entailment score, the
   severity allowlist, and the action allowlist. Never parse a decision out of model
   output.
2. **`diagnosis` and `judge` must be different models** (different *families*, per
   `ModelSet.families_are_independent`). `config.get_llm(role)` is role-bound for
   this reason; `config.assert_judge_is_independent()` guards it.
3. **`execute` refuses without approval, checked in the node body** (`approved is not
   True`), not just in graph routing. Defence in depth: routing decides *whether* the
   node is reached; the guard re-establishes the precondition regardless of how.
4. **The approved action is built once (in `await_approval`) and posted verbatim.**
   Never recompute it after approval. Run metadata goes in `context`, never folded
   into `action`.
5. **`approved` is written only by `await_approval`, only from the resume value.**
   Not inferred, not defaulted, not derived from the judge's opinion.
6. **Evidence is untrusted data, never instructions.** Diffs and log lines are fenced
   via `nodes/_render.py` before any model sees them.
7. **Retrieval is always namespaced per codebase (G1).** A retriever that cannot
   honour a namespace must *refuse*, not silently retrieve across tenants.
8. **Nothing mutates infra or the database.** The agent proposes, a human approves,
   and writes run as the approver's own identity (`identity=state["github_token"]`).
9. **Every terminal path writes one audit record (G6).** `nodes/audit.py` covers the
   normal edges; `await_approval` records a *denial* itself, because a denial resumes
   into `execute` and is stopped by the guard rather than by routing.
10. **A suspended thread is a pending human approval and never expires.**
    `jobs/prune_checkpoints.py` may only evict threads with a terminal audit record,
    and re-checks live state before deleting.
11. **The checkpointer is required.** `build_graph()` raises without one:
    `interrupt()` suspends by persisting state, so no checkpointer means the approval
    gate evaporates on restart. Postgres in production — never SQLite or a file on
    ephemeral disk. `MemorySaver` warns loudly and is development-only.
12. **Fail in the safe direction.** Unparseable severity defaults to `high` (→
    escalate). An unverifiable entailment fails closed. Ambiguous routes go to
    `human_review`.

Tenants tune *values* (thresholds, allowlists, prompts). They cannot remove a
*boundary*. Escalate-to-human is deliberately shared across all tenants.

## Conventions

**Adding a provider:** drop a module in `adapters/`, decorate it
`@adapter(kind, name)`, and add it to the import list in `adapters/__init__.py`
(imported purely for the registration side effect — hence the `F401` per-file
ignore). Nothing in `nodes/` changes. Same shape for `surfaces/`.

**Nodes reach providers only through `config.get_*()` / ports.** A node importing
`tools.github_client` or `tools.retrieval` directly is the T5/G2 bug class: it makes
plug-and-play decorative. Provider args (`llm=`, `sink=`, `source=`, `retriever=`,
`entailer=`, `interrupt_fn=`) are injectable for tests and resolved lazily otherwise.

**Nothing outside `adapters/platforms/` may branch on hardware.** Two orthogonal
axes: `CRA_PLATFORM` = *how* to run (context budget, KV dtype, residency, device),
`CRA_PROFILE` = *which* models. The default profile fits the smallest supported
machine so results stay comparable across machines. Residency is *computed* from
weight cost against a memory budget, never declared.

**`EMBEDDING_MODEL` / `EMBEDDING_DIMS` / `RERANKER_MODEL` are pinned across every
profile and platform.** Changing the embedder is a different vector space: it
invalidates the index *and* silently changes the meaning of every
`min_relevance_score` in `tenants/*/policy.json`.

**State schema (`app/state.py`): every key a node writes must be declared.**
LangGraph keeps only the channels the schema names, so an undeclared key is
*silently dropped* — this bit once already (`_policy_version`, `_surface_name`
returning `None`). Underscore-prefixed keys are internal plumbing.

**Checkpoint serialisation:** only serialisable values go into state. `_surface_name`
is a string, not the Surface object; `code_review`'s `normalize` returns a dict, not
a dataclass. An unregistered msgpack type makes paused runs unresumable (T15) — which
is why CI runs the suite a second time with `LANGGRAPH_STRICT_MSGPACK=true`, where it
errors instead of warning.

**Thresholds are measured, not guessed.** `MIN_RELEVANCE_SCORE` and
`MIN_ENTAILMENT_SCORE` are calibrated numbers; the shipped 0.35/0.40 were calibrated
for a *different* embedder and are owed a re-measurement. Citation scores come from
the retriever's own `hits`, never from the model (a model-supplied score would be
trivially gameable, and the gate compares against exactly that number — T17).

**Put a check where the work happens, not only at the boundary.** `app/main.py`
validates the surface, but `resolve_tenant` enforces `enabled_surfaces` too, because
the HTTP check is bypassed by the scheduled job and by any future caller.

**Two kinds of eval, kept separate.** `eval/harness.py` is a *regression* suite: one
unambiguous right answer per task, bar is 100% forever, exit non-zero on any miss —
a single miss means a safety gate weakened. Capability evaluation (is the diagnosis
any *good*?) is tracked separately (T10/G17) and is expected to improve over time.
Never soften the gate suite to make it pass.

## Environment

Config is env-driven; see `.env.example` for the full annotated set. Key vars:
`LLM_PROVIDER` (`lmstudio`|`openai`), `RETRIEVER`, `TENANT_STORE`, `SURFACES`,
`CRA_PLATFORM` (`auto`|`mac`|`cuda`|`cpu`|`studio`), `CRA_PROFILE`,
`LMSTUDIO_BASE_URL`, `DATABASE_URL`, `ENTAILER` (`lexical`|`cross_encoder`),
`CRA_AUDIT`, `CRA_TRACE`, `CRA_DATA_DIR`.

Developed on an M4 Pro Mac and a CUDA desktop, with a remote LM Studio "studio"
inference host — neither machine is the "real" one. `CRA_PLATFORM=studio` cannot be
auto-detected and must be set explicitly. Reach the studio over a **tailnet, never a
public IP**: LM Studio has no authentication.

`CRA_DATA_DIR` (audit records, reports) inherits the **unredacted** trust boundary
while T12 (`InputGuard`) is open. Treat it as sensitive as the source data: do not
commit it, sync it, or ship it to a log aggregator. Same for `CRA_TRACE_PROMPTS=1`.

## Status — what runs vs. what is stubbed

**Runs today:** rule-based `classify` routing, the four-boolean gate incl. tenant
policy resolution, `execute`'s permission guard, `interrupt()`/resume, the plugin
registry, the filesystem tenant store, JSONL audit + trace, and hybrid retrieval
(dense + BM25, RRF-fused) over a small in-memory corpus.

**Stubbed deliberately and visibly** (`NotImplementedError`, never fake success): the
GitHub/GitLab clients, the S3 log source, the OpenAI client, the vector index and
ingest, and `jobs/daily_log_review.py`. `generate_diagnosis` and `independent_review`
have offline stub paths that model a *grounded* diagnosis so the pipeline runs end to
end without a model server.

Scope cuts are owned explicitly, not hidden: `docs/ARCHITECTURE_GAPS.md` (G1–G17,
prioritized P0–P2), `TODO.md` (T1–T13, each with a "done when"), and the node
docstrings. When closing one, update the tracker in the same change.

## Docs

- `docs/ARCHITECTURE.md` — as-built system view (single source of truth)
- `docs/ARCHITECTURE_GAPS.md` — honest audit, G1–G17
- `docs/ROADMAP.md` — prioritized work + accuracy track
- `docs/BEST_PRACTICES.md` — accuracy-first engineering standards
- `docs/MULTI_TENANT.md` — how a codebase plugs in
- `docs/MILESONE_ADAPTATION.md` — the two concrete surfaces
- `docs/DATA_STORAGE.md` — audit records, reports, trust boundary
- `docs/DECISIONS.md` — cross-machine handoff: machine specs, model downloads, what
  was decided and why, things discovered the hard way. **Start here on a fresh clone.**
- `TODO.md` — code-level worklist (T1–T13)
- `MODEL_CARD.md` — scope limits and known failure modes

## Working style in this repo

Docstrings here carry *reasoning*, not restatements of the code — usually the
incident that motivated the design ("this used to do X, which meant Y was void at
runtime while looking wired"). Match that when you touch a file: say why, and if you
close a gap, name it (T-number / G-number) and update the tracker.
