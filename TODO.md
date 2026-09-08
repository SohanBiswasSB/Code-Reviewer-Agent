# TODO — the reasoning

`ROADMAP.md` is the plan (phases, dependencies). `docs/ARCHITECTURE_GAPS.md` is the
architecture audit (G1–G17). **This file is the code-level worklist**: what is
missing, why it matters, and what "done" looks like — so a gap is never closed by
accident or left open by forgetting.

Status: `[ ]` open · `[~]` partially done · `[x]` done (kept briefly, with the commit).

---

## P0 — nothing runs end to end without these

### T1. No working LLM adapter `[~]`

`adapters/llm_openai.py::complete()` raises. Every LLM path in the graph is a
stub, so `generate_diagnosis` returns a hardcoded diagnosis and
`independent_review` a hardcoded reasoning string.

**Why it matters:** the gate is exercised by the eval suite, but the *thing being
gated* has never been produced by a model. Until a real diagnosis flows through,
`groundedness_ok` has never rejected a real hallucinated citation.

**Done when:** `@adapter("llm","lmstudio")` talks to `http://localhost:1234/v1`,
returns text for the judge and schema-constrained JSON for the diagnosis, and a
`doctor` check proves the configured model ids are actually loaded.

**Status:** adapter written and registered; `doctor` verifies reachability and
per-role model presence. Still `[~]` and not `[x]` because **no call has ever
reached a live server** — LM Studio has not been started with these models. The
request shape (schema-constrained decoding, evidence in the user turn) is
unverified against a real model.

### T2. `get_llm()` is singular, but the design needs two `[x]`

`config.get_llm()` returns one LLM and `ports.LLM.complete()` has no model
parameter. The independence invariant needs a *different model* for diagnosis and
judge — that is the whole point of `independent_review`.

**Why it matters:** with one model, "independent review" degrades to the same
model grading itself, which is the self-preference-bias failure the design exists
to avoid. The platform layer already resolves both ids; nothing consumes them.

**Done when:** `get_llm("diagnosis")` / `get_llm("judge")` resolve distinct models
from the active profile, and asking for a role the profile does not define raises.

### T3. The resume path does not exist `[ ]`

`await_approval` builds an interrupt payload but never calls `interrupt()`.
Nothing ever writes `approved` into state. `app/deps.py::make_checkpointer()`
raises. `/run` and `/approve` raise.

**Why it matters:** this is the single largest gap. With the real graph, **every
run hits `execute`'s `PermissionError`** — the safety guard fires correctly and
the pipeline is unusable. A paused run is also supposed to be durable state
(`AsyncPostgresSaver`), not a UI illusion.

**Done when:** `POST /run/{surface}` drives to a real `interrupt()`, the pause
survives a process restart, `POST /approve` resumes on the same `thread_id` with
`Command(resume=True/False)`, and approval sets `approved` so `execute` proceeds.

### T4. Double approval gate `[ ]`

`app/graph.py` compiles with `interrupt_before=["execute"]` *and* routes through
`await_approval`. Two mechanisms for one gate.

**Why it matters:** ambiguous which one actually holds; a change to either could
silently leave the other as the only gate. Pick `interrupt()` inside
`await_approval` (it carries the payload) and drop `interrupt_before`.

**Done when:** exactly one mechanism remains, and a test asserts a run suspends
before any write.

### T5. Nodes bypass their own ports (G2) `[ ]`

`fetch_evidence` imports `tools.github_client` directly; `generate_diagnosis`
imports `tools.retrieval` directly. Neither goes through `EvidenceSource` /
`Retriever`, so the active `Surface` is decorative outside `classify`.

**Why it matters:** two consequences, one cosmetic and one not. Plug-and-play is
half-wired — but worse, **G1 is unenforced in practice**: `generate_diagnosis`
retrieves with no `namespace`, so a multi-codebase index would let Project A cite
Project B's code. The port signature is fixed; the caller ignores it.

**Done when:** every node reaches its provider through a port, the resolved
surface is injected into state, and retrieval always carries the tenant's
namespace.

### T6. Tenant prompts are never loaded `[ ]`

`TenantStore.prompt()` is called by **nothing**. `policy()` is wired into the gate
(good), but both LLM nodes ignore the tenant's prompt files entirely.

**Why it matters:** `docs/MULTI_TENANT.md` claims prompts are per-tenant,
resolved at request time. Half of that claim is currently false. The prompt files
under `tenants/*/prompts/` are dead weight.

**Done when:** `generate_diagnosis` and `independent_review` load their prompts
via the store, falling back to `_default`, and a test proves a tenant override
changes the prompt actually sent.

---

## P1 — correctness and safety

### T7. The gate suite never tests `permission_ok` `[x]`

`eval/tasks.py::GATE_TASKS` covers groundedness (2 cases) and risk (1 case). The
third boolean — `permission_ok`, the action allowlist — has **no case at all**.
There is also no case that resolves a real tenant policy, and no case exercising
`classify`'s surface-delegation branch.

**Why it matters:** by the suite's own `pass^k` framing, an untested gate is an
unproven gate. A refactor could break the action allowlist and every test would
still pass green.

**Done when:** a proposed action outside `allowed_actions` escalates, a
tenant-policy-driven case passes, and the delegation branch is covered.

### T8. No pytest suite; no lint config `[x]`

`eval/harness.py` is the only executable check, and there is no `pytest.ini` or
`ruff.toml` (both present in SF-Crime-Prediction).

**Why it matters:** the safety gates and the *rest* of the code are different
kinds of test. Gates are a 100%-forever regression bar; unit tests cover platform
resolution, tenant merge, schema validation, adapter registration. Conflating
them means the gate suite slowly turns into a general test suite and loses its
meaning.

**Done when:** `pytest` runs with no dataset and no model server, `ruff check .`
is clean, and CI runs both alongside `eval.harness`.

### T9. `doctor` — no way to know the environment is sane `[x]`

Nothing verifies LM Studio is up, that the profile's model ids are actually
loaded, that the embedder's dimensionality matches the index, or that the
reranker imports.

**Why it matters:** every one of those fails at the *first LLM call*, deep inside
a graph run, as a confusing error. And `adapters/platforms/mac.py` is currently
**unverified arithmetic** — `doctor` is how those numbers become measured.

**Done when:** `python -m app.doctor` reports platform, profile, computed
residency with its arithmetic, LM Studio reachability, per-role model presence,
and exits non-zero on a real problem.

### T14. The graph never resolves the tenant `[ ]`

Measured, not inferred: a real `invoke()` returns state containing `tenant` and
`surface` but **no `_policy` and no `_surface`**. `build_graph()` takes neither,
so nothing between START and the gate ever calls the `TenantStore`.

**Why it matters:** this silently voids the multi-tenant design at runtime.
`independent_review` falls back to its module-level constants
(`MIN_RELEVANCE_SCORE = 0.35`, severity low|medium, comment|comment+label), so
**every codebase is currently gated by `_default`'s thresholds** regardless of
its `policy.json`. `eval/demo_multitenant.py` passes `_policy` in by hand, which
is exactly why it looked wired. And with no `_surface`, `classify` takes the
built-in issue-triage branch even when `surface="code_review"`.

**Done when:** `build_graph(surface, tenant)` (or a resolver node) attaches the
resolved policy and surface to state before `classify`, and a test asserts a
strict tenant's threshold actually applies to a graph run rather than to a
hand-built state dict.

### T15. Dataclasses are not checkpoint-serialisable `[ ]`

Running with a real checkpointer emits, for every state type:

    Deserializing unregistered type schemas.evidence.IssueEvidence from
    checkpoint. This will be blocked in a future version.

Affects `IssueEvidence`, `Diagnosis`, `Citation`, `ReviewResult` and
`tools.retrieval.Snippet`.

**Why it matters:** the durable pause is the whole basis of the approval gate. On
a LangGraph version that enforces this, **every suspended run becomes
unresumable** — the worst possible failure for a design whose safety property is
"a paused run is real, durable state". It also blocks T3 from being trustworthy.

**Done when:** state types are Pydantic v2 models (T11 does this) or are
explicitly registered as allowed msgpack modules, and a test round-trips a
suspended run through a checkpointer without warnings.

### T10. Retrieval is a toy `[ ]`

`tools/retrieval.py` runs on a 4-document in-memory corpus. RRF is real; the
backends are stubs. `tools/ingest.py` raises. No namespace support.

**Why it matters:** grounding is the entire safety story, and it currently
grounds against four hardcoded strings. Also: switching to
`text-embedding-nomic-embed-text-v1.5` (768 dims) from the specified
`text-embedding-3-small` at 512 dims means **every `min_relevance_score` in all
four `policy.json` files is uncalibrated** — the 0.35/0.40 values were measured
for a different vector space and now mean nothing. They must be re-derived by
measuring on-topic vs off-topic score distributions, the way the threshold was
originally chosen.

**Done when:** a real index exists, ingest is idempotent and namespaced per
codebase, and the thresholds are re-measured with the method written down.

### T11. Schemas are dataclasses; `normalize_evidence` does not validate `[ ]`

`schemas/` uses stdlib dataclasses with a TODO to move to Pydantic v2.
`normalize_evidence` "just trusts normalized fields".

**Why it matters:** its own docstring says bad data should fail *there, loudly*.
It currently cannot. Malformed provider JSON propagates into the graph.

**Done when:** evidence and diagnosis are Pydantic v2 models and malformed input
raises at the boundary. Note `eval/harness.py` constructs these directly, so it
must keep working without the API stack installed.

### T12. `InputGuard` unimplemented (G5) `[ ]`

Nothing redacts PII/secrets or neutralises injected instructions before evidence
reaches a provider. The port exists.

**Why it matters:** log lines and diffs are untrusted input by the project's own
invariant, and code can contain secrets. Right now that invariant is enforced by
prompt wording only — which the prompts themselves say is not a boundary.

**Done when:** evidence passes through a redaction + delimiting stage before any
LLM call, and an adversarial eval class covers injected instructions.

### T13. `Reranker` unimplemented `[ ]`

Port defined, no adapter. Must be in-process `sentence-transformers` — LM Studio
exposes no rerank endpoint.

**Done when:** a wide retrieve (top-100) reranks to top-n, on the device
`runtime()["reranker_device"]` names, with retrieval metrics proving it earns its
cost.

---

## P2 — from the architecture audit

Tracked in full in `docs/ARCHITECTURE_GAPS.md`; listed here so nothing lives in
only one place: G6 AuditSink `[ ]` · G7 idempotency `[ ]` · G8 pin prompt+policy
version into the checkpoint `[ ]` · G9 failure edges + retry/backoff `[ ]` ·
G10 corrective-RAG loop cap `[ ]` · G11 policy schema validation `[ ]` ·
G12 budgets/rate limits `[ ]` · G13 Notifier + approver authZ `[ ]` ·
G14 prompt inheritance `[ ]` · G15 prompt/policy cache `[ ]` · G17 capability
eval `[ ]`.

---

## Hygiene

- [x] `.env.example` — every setting, with the `CRA_` prefix convention
- [x] `MODEL_CARD.md` — scope limits, as SF-Crime has. States plainly: this agent
      never pushes code, only comments; it acts as the approving human's identity;
      it never mutates infrastructure or a database.
- [x] `.vscode/settings.json` — committed, but WITHOUT an interpreter path:
      one committed path is wrong on the other machine, so it is selected per
      machine instead. Both paths are in docs/DECISIONS.md section 1. Also sets
      python.analysis.extraPaths, which fixes the spurious import errors.
- [ ] `requirements-lock.txt` — `pip freeze` once the environment is real
- [x] `.gitattributes` (eol=lf) — 97bf3a9. Load-bearing with two machines.
- [x] Split requirements (core / -local / -dev) — 97bf3a9

---

## Deliberately NOT doing

Restating from `docs/BEST_PRACTICES.md` section 12, because these are the things a
future refactor is most likely to "helpfully" undo:

- No LLM ever emits the final decision. Code decides.
- No autonomous mutation of infrastructure, databases or repositories.
- The safety-gate suite's 100% bar is never traded for a capability average.
- `execute` never recomputes the approved action — it posts it verbatim.
