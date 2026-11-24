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

python check.py                       # RUN THIS. The local equivalent of CI, which is
                                      # commented out -- this is the only gate now.
                                      # --all to continue past a failure, --list to see steps
python -m app.doctor                  # can this machine run the agent? (stdlib only)
python -m app.doctor --json
python -m eval.harness                # safety-gate suite — no model, no server
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

# Review a LOCAL change with the full graph (git_local source -> review_file sink):
python -m tools.ingest --tenant merger_app --root ~/Dev/<repo>            # build the index once
python -m app.review --tenant merger_app --root ~/Dev/<repo>              # staged changes
python -m app.review --tenant merger_app --root ~/Dev/<repo> --target main...HEAD
python -m app.monitor                 # live view: http://127.0.0.1:8765 (localhost only)

# The user's MANUAL pre-push check (no git hook; git is only read, never fetched).
# Reviews <precheck_base>...HEAD, remembers findings in var/ledger/<tenant>.json, and
# says what is new / still open / fixed; only computed checks decide "clear to push",
# model findings are advice (T32). --root is remembered after the first run.
python -m app.precheck --tenant merger_app --root ~/Dev/<repo>
python -m app.precheck --tenant merger_app --dismiss <id> --reason "..."   # / --undismiss / --list

# Measure the gate thresholds (prints a suggestion; never edits a policy):
python -m tools.calibrate --tenant merger_app --root ~/Dev/<repo> relevance   # needs LM Studio
python -m tools.calibrate --tenant merger_app entailment                      # from audit records
```

Requirements files are layered: `requirements.txt` (graph + API) ⊂
`requirements-local.txt` (adds LM Studio client + in-process reranker, pulls torch)
⊂ `requirements-dev.txt` (adds pytest + ruff). `eval/harness.py` needs only
`requirements.txt` — no model and no server.

**Imports go at the top of the module** (user's rule, 2026-11-16). A function-level
import is allowed only where module level would be genuinely circular — e.g. an
adapter importing `config`, since `config` imports `adapters`; say so in a comment.
Libraries an adapter imports are therefore required by everything that imports
`config`, so they belong in `requirements.txt`, not `-local`.

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
| `ports/interfaces.py` | The contracts (Protocols). Core codes against these shapes and reaches providers only via `config.get_*()`; nothing imports a concrete adapter. |
| `nodes/`, `app/graph.py` | The graph. Holds the safety design. |
| `adapters/` | Concrete providers, self-registering via `@adapter(kind, name)`. |
| `surfaces/` | Capabilities, self-registering via `@surface(name)`. |
| `tenants/<name>/` | One directory per codebase: `policy.json` + `prompts/<surface>/<node>.md`. |
| `registry.py` / `config.py` | The plug: decorators register, env resolves. |
| `schemas/` | `Diagnosis`, `ReviewResult`, evidence models (Pydantic). |
| `eval/` | Safety-gate regression suite, 100% bar, `pass^k`. Model-free. |
| `tests/` | Unit tests. Deliberately separate from the gate suite. |
| `adapters/platforms/` | The **only** code that knows about hardware. |
| `app/doctor.py` | Environment check. Stdlib at module level; the checks that import `config` need `requirements.txt` installed. |

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
4. **The approved action is built once and posted verbatim.** On the AI path
   `independent_review` builds it (on approve only); on the deterministic path
   `await_approval` does. `await_approval` shows the human exactly that action and
   passes it through unchanged. Never recompute it after approval. Run metadata goes in `context`, never folded
   into `action`.
5. **`approved` is written only by `await_approval`, only from the resume value.**
   Not inferred, not defaulted, not derived from the judge's opinion.
6. **Evidence is untrusted data, never instructions.** Diffs and log lines are
   redacted in `fetch_evidence` before anything is written to state (T25: the
   checkpoint keeps every super-step, so later is too late), then
   sanitised *and* fenced via `nodes/_render.py` before any model sees them —
   `config.get_input_guard()` redacts secrets/PII and escapes the fence markers, and
   its report lands in `state["_guard"]` and the audit record. Prose injection is
   *counted, not rewritten*; the real defence against it is that no model output can
   become an action without four booleans and a human.
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
(imported for the registration side effect, and listed in that package's
`__all__` -- there is no lint ignore for it). Nothing in `nodes/` changes. Same
shape for `surfaces/`.

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
a dataclass. **Every model that enters a channel must be listed in
`app.state.CHECKPOINT_TYPES`**, and every checkpointer must be built with
`app.state.checkpoint_serde()` — `app/deps.py` does this for both backends. An
unregistered type is not an error: it comes back from the checkpoint as a plain
`dict`, so a resumed run silently loses the model it was written with and fails
later as an `AttributeError` far from the cause (T15). CI runs the suite a second
time with `LANGGRAPH_STRICT_MSGPACK=true`, and `tests/test_docs_consistency.py`
plus the `schemas/` tripwire in `test_graph_resume.py` fail if a model is added
without registering it.

**Schema-constrained decoding governs structure, not content.** Passing
`json_schema` to `LLM.complete` reliably fixes field names, ordering and enum values — and
does **nothing** to stop a model looping inside a string until the token budget dies.
Measured: `frequency_penalty` does not break such a loop, and a schema `maxLength` is
ignored by LM Studio's decoder. This is why `_parse_diagnosis` must fail safe rather than
trust the schema, and why a model that cannot terminate at temperature 0 is the wrong model
rather than a tuning problem.

**Thresholds are measured, not guessed.** `MIN_RELEVANCE_SCORE` and
`MIN_ENTAILMENT_SCORE` are calibrated numbers; the shipped 0.35/0.40 were calibrated
for a *different* embedder. The real tenants now use a measured interim 0.75;
entailment is still owed its measurement -- `python -m tools.calibrate`
measures both (relevance needs the embedder; entailment reads real findings from the
audit store, so it gets better with every review run). Citation scores come from
the retriever's own `hits`, never from the model (a model-supplied score would be
trivially gameable, and the gate compares against exactly that number — T17).

**Put a check where the work happens, not only at the boundary.** `app/main.py`
validates the surface, but `resolve_tenant` enforces `enabled_surfaces` too, because
the HTTP check is bypassed by the scheduled job and by any future caller.

**CI is commented out; `python check.py` is the gate.** This project is run
locally only (`.github/workflows/ci.yml` is disabled but preserved). `check.py`
runs the same steps in the same order and exits non-zero on any real failure.
Two of them are easy to skip and are the ones that have caught real defects: the
gate suite runs **first**, model-free, so no later step can mask a weakened
gate; and pytest runs **twice**, the
second time with `LANGGRAPH_STRICT_MSGPACK=true`. Never conclude "tests pass"
from a bare `pytest`. After the gate suite, the remaining steps run in parallel
(~18 s instead of ~31 s); `--serial` restores one at a time.

**Two kinds of eval, kept separate.** `eval/harness.py` is a *regression* suite: one
unambiguous right answer per task, bar is 100% forever, exit non-zero on any miss —
a single miss means a safety gate weakened. Capability evaluation (is the diagnosis
any *good*?) is tracked separately (T10/G17) and is expected to improve over time.
Never soften the gate suite to make it pass.

## Environment

Config is env-driven; see `.env.example` for the full annotated set. Key vars:
`LLM_PROVIDER` (`lmstudio`|`openai`), `RETRIEVER`, `TENANT_STORE`, `SURFACES`,
`CRA_PLATFORM` (`auto`|`mac`|`cuda`|`cpu`|`studio`), `CRA_PROFILE`,
`LMSTUDIO_BASE_URL`, `DATABASE_URL`, `RERANKER` (`cross_encoder`|`none`; offline
callers declare `none`), `ENTAILER` (`lexical`|`cross_encoder`),
`CRA_GUARD` (`regex`|`null`), `CRA_AUDIT`, `CRA_TRACE`, `CRA_DATA_DIR`,
`CRA_LLM_RETRIES` (default 2) and `CRA_LLM_RETRY_BACKOFF` (default `5,15` seconds).

**Model calls retry only TRANSIENT failures** (`adapters/llm_lmstudio.py::with_retry`,
shared by the embedder): connection errors, timeouts, 5xx, 429, and LM Studio's
backend crash -- which it reports as **HTTP 400**, so it is matched by an allowlist of
observed crash text. Any other 400 fails on the first attempt. The OpenAI client's
own hidden retries are off (`max_retries=0`) so there is one loop, and every retry
is printed to stderr and traced as `llm.retry`, which the monitor shows on the step.

**The defaults that must stay real:** `CRA_AUDIT`, `CRA_TRACE` and `CRA_GUARD` all
default to working implementations, and the `null` variants exist for tests only. A
record, a trace or a redaction boundary you get by *remembering* is not one.

Developed on an M4 Pro Mac and a CUDA desktop, with a remote LM Studio "studio"
inference host — neither machine is the "real" one. `CRA_PLATFORM=studio` cannot be
auto-detected and must be set explicitly. Reach the studio over a **tailnet, never a
public IP**: LM Studio has no authentication.

`CRA_DATA_DIR` (audit records, reports) holds evidence-derived text. T12 is now
closed, so secrets and PII are redacted before any model call **and** before anything
reaches `var/` — but redaction is pattern-based, so treat `var/` as sensitive anyway:
do not commit it, sync it, or ship it to a log aggregator. `CRA_TRACE_PROMPTS=1` stores
prompt text and stays off by default. `CRA_GUARD=null` disables the guard and must never
be set outside a test.

## Status — what runs vs. what is stubbed

**Runs today:** rule-based `classify` routing, the four-boolean gate incl. tenant
policy resolution, `execute`'s permission guard, `interrupt()`/resume, the plugin
registry, the filesystem tenant store, JSONL audit + trace, and namespaced hybrid
retrieval (`RETRIEVER=indexed`, the default: dense cosine + BM25, RRF-ordered) over an
on-disk index that `tools/ingest.py` builds. `RETRIEVER=hybrid` is the 4-document
offline demo; tests, the gate suite and the demo declare it.

**The reranker is built and OFF by default (T13, 2026-11-17, on measurement).**
`RERANKER=cross_encoder` makes `IndexedRetriever` rescore the top 100 fused candidates
with `BAAI/bge-reranker-v2-m3` (in-process, `runtime()["reranker_device"]`, loaded from
the HF cache only -- a review never downloads it). The hit's `score` is then the RERANK
score with `score_kind="rerank"`, the cosine beside it; the gate picks `min_rerank_score`
or `min_relevance_score` by kind (`independent_review._bar_for`), so no number is held to
a bar measured for another model. **Why off:** `tools.calibrate rerank` (176 Masterclass
queries, same pools) gave same/other-file separation J 0.31 vs the cosine's 0.68, same-file
in top 3 81.8% vs 98.9%, at 13.5 s/query. The proxy favours the cosine; a human-labelled
relevance set is the test that would settle it.
`sentence-transformers` + torch are in `requirements.txt` now (imports-at-top rule):
`import config` costs ~3 s more and `check.py` takes ~40 s instead of ~18 s.

**Entailment is per claim** (2026-11-17, `independent_review.entailment_claims`): the
comment and each finding are scored separately; a finding's premise is its OWN citations
plus the diff of its OWN file (the change is the claim's subject, and new code is not in
the index yet); the lowest claim decides. `tools.calibrate ... entailment` scores recorded
reviews with the same function.

**Every run leaves an audit record, crashes included** (G9, 2026-11-17): `_traced` writes
outcome `failed`/`interrupted` with the node and guarded error, then re-raises; a CLI
review left at the gate without a terminal is `abandoned`. Policies are validated at run
start (G11, `schemas/policy.py`). `risky_areas` in a tenant policy now routes (T27,
union with the surface defaults).

**Retrieval is per changed file** (2026-11-16). `CodeReviewSurface.build_queries`
makes one query per changed file (path + that file's diff; code, then config, then
docs; data never), plus the whole-change query; `generate_diagnosis.merge_hits`
round-robins them, deduplicates, keeps each chunk's best retriever score, and caps
at 8 -- sized for the judge's 8k context, with each passage clipped to 1,500 chars
in the PROMPT only (entailment reads full text). One query for the whole change had
left 4 of 5 findings citing a different file than they were about. The retriever and
cross-encoder are shared per process (`config._shared`), and BM25's statistics are
saved beside the vectors (`bm25.npz`, keyed by the index fingerprint).

**Errors: name the family, never `except Exception`** (user, 2026-11-16).
`errors.py` holds the domain errors (`ConfigError`, `EvidenceError`, `ProviderError`,
`StorageError` -- each also a builtin, so old `except ValueError` keeps working) and
the families a boundary may catch: `WRITE_ERRORS` (disk / serialisation),
`CONFIG_ERRORS`, `PARSE_ERRORS`. Anything outside its family is a bug and
propagates; the activity log records it with its traceback. There is NO broad
catch anywhere, re-raising ones included (2026-11-17): to record a failure and let
it propagate, use try/finally with a done-flag set as the body's last step, and
read `sys.exception()` in `finally` only when the flag is unset -- unguarded, it
also returns an exception a CALLER is handling, so a success would be recorded as
a failure. The label in `PRCommentSink` (after an irreversible comment post)
catches `LABEL_ERRORS` and logs at ERROR.

**Coding standards are enforced on this repo, not just reviewed in others**
(user, 2026-11-17): no `# noqa` / `# pylint: disable` / `# type: ignore` / isort
skips and no config-level ignores -- fix the cause; no broad except; functions
within pylint's default limits (5 args, 15 locals, 12 branches, 50 statements, 6
returns -- split into helpers, never raise a limit); the HTTP API is typed, strict
(undeclared field = 400), enveloped `{success, message, data}` and under /v1.
`python -m tools.standards_audit` runs the reviewer's own rules
(tools/prepush/standards/) over the whole tree; check.py blocks on it.

**Activity log** (`app/activity.py`, port of the Masterclass `Logger` design):
`ActivityLogger(...)` spans and `log_event(...)`; secrets/PII masked on the way in;
one correlation id per run (= thread id) or request (`X-Correlation-Id`). Every
graph node, LLM call, query embedding, approval decision, API request and ingest
job is recorded in `var/activity/YYYY-MM-DD.jsonl` and shown on the monitor's
**Activity log** panel. Never log prompt or evidence text -- counts and ids only.

**Pre-push checks run on every code review** (`nodes/prepush_checks.py` ->
`tools/change_checks.py`, a facade over `tools/prepush/`: `change` reads the diff
and blobs in four batched git calls, plus `n_plus_one`, `hygiene` and `conventions`), modelled on the Masterclass `/prepush-ritual`: generic
checks always, convention checks where `code_review.prepush` in a tenant's policy
declares them. Results are `pass` / `flag` / `skip` (skip always says why). They
reach the prompt (fenced), the review (a code-generated checklist appended AFTER the
gate, never entailed), the audit record and the monitor. **They never route or
decide** -- making one a veto is a fifth boolean and must be a deliberate change.

**Local review works without GitHub:** `adapters/source_git_local.py` reads a diff from
the tenant's checkout (`staged` | `<rev>` | `a..b` | `a...b`) and `ReviewFileSink`
writes an approved review to `var/reviews/`. Selected per surface by
`CRA_SOURCE_<SURFACE>` / `CRA_SINK_<SURFACE>` (`config.source_name_for`), which
`app/review.py` defaults for you. Where a repo lives on THIS machine is
`tools/codebase.codebase_root` — shared by ingest and the source, so they never
disagree.

**Stubbed deliberately and visibly** (`NotImplementedError`, never fake success): the
four network calls in `tools/github_client.py` (the `EvidenceSource` / `ActionSink`
adapters around it are real), the S3 read in `adapters/source_s3logs.py` (inject
`reader=` and the rest of the pipeline runs — that is how the 200k-line figure was
measured), `adapters/llm_openai.py`, and `jobs/daily_log_review.py`.
**There is no GitLab adapter at all** — it is named in the docs as what the port exists
to make swappable, not as a stub. `generate_diagnosis` and `independent_review` have
offline stub paths that model a *grounded* diagnosis so the pipeline runs end to end
without a model server.

Scope cuts are owned explicitly, not hidden: `docs/ARCHITECTURE_GAPS.md` (G1–G17,
prioritized P0–P2), `TODO.md` (T1–T28, each with a "done when"), and the node
docstrings. When closing one, update the tracker in the same change.

## Current state — last updated 2026-11-17

Distilled from the session transcripts (`../chat-archive/`); `docs/DECISIONS.md` §8/§9 is
the fuller narrative and the designated handoff. **Verify before relying on any of it** —
this is a snapshot, not a guarantee.

**Repo:** branch `ID_SB_Ongoing_Development_Branch` on
`github.com/SohanBiswasSB/Code-Reviewer-Agent` (private). Re-run and confirmed on this Mac
on 2026-11-16: **ruff clean · 313 tests pass (both normally and under
`LANGGRAPH_STRICT_MSGPACK=true`) · 54/54 gates pass**, on Python 3.14.7 with langgraph
1.2.11, pydantic 2.13.5, fastapi 0.141.1. Core deps are now pinned in `requirements.txt`.

**T15 is closed, and the fix was registration — not a version pin.** An earlier note in
`docs/DECISIONS.md` diagnosed the strict-serialiser failure as a dependency regression to
"bisect and pin". It was not: langgraph *added* an explicit allowlist plus a strict mode,
and its own warning names the fix. `app.state.CHECKPOINT_TYPES` now registers all eleven
models in `schemas/`, and `app.state.checkpoint_serde()` is passed to both checkpointer
backends. **The failure mode to remember: an unregistered type does not raise, it comes
back as a plain `dict`** — so it is invisible until a resumed run touches an attribute.
Both suites are green, so CI should pass.

**Known broken / unverified** (each re-checked on 2026-09-10):

- **T1 is closed** (`390f1b0`, 2026-09-11): both LLM nodes resolve from
  `config.get_llm(role)`, proven on the real graph with nothing injected. Offline
  callers declare `LLM_PROVIDER=stub`. Pushed (it is in `origin`'s history).
- **2026-11-16: first real review of another codebase (Masterclass)** -- see
  DECISIONS §8 for what it exposed (diff budget, findings not posted, T26, forced
  offline gates, Metal OOM) and §9 for the next steps (calibration first).
- **T10: both real tenants are embedded.** Masterclass (17.6k chunks, 2026-11-16) and
  MilesOne (2026-11-17: 1,196 files, 15,306 chunks, 2,610 values redacted, 698 s --
  indexed **whole**, `Miscellaneous/` included, as the user chose; DECISIONS §8).
  `project_a`/`project_b` remain placeholders with no repo.
- The Postgres checkpointer path is unverified — there is no Postgres on this machine.
- `.env.example` was ignored by `.gitignore`'s `.env.*` while `TODO.md` called it done.
  Decided 2026-11-16: the ignore was wrong -- it holds names and defaults, no values --
  so `.gitignore` now has `!.env.example`, and the file was brought up to date.
- `min_relevance_score` was shipped **uncalibrated** (0.35 / 0.40), and the first live
  run showed why it matters: the one real citation scored **0.2 against the 0.35 bar**,
  and entailment landed at **0.24 against 0.25**. The verdict turned on two constants
  calibrated for an embedder this project no longer uses. **Since 2026-11-17 the real
  tenants (`merger_app`, `milesone`, `_template`) use 0.75**, an interim value taken
  from the measurement below; `_default`, `project_a` and `project_b` still carry the
  old numbers. `min_entailment_score` (0.25) is still unmeasured.
- **MEASURED 2026-11-16 (`tools.calibrate`, masterclass -- now `merger_app`, 305 per-file queries from 40
  commits): the 0.40 relevance bar is INERT.** EmbeddingGemma packs cosines into
  ~0.7-0.9: same-file passages p10/median/p90 = 0.780/0.823/0.869, other-file
  passages 0.727/0.770/0.803 -- so 0.40 can never fail on real retrieval (T21's
  shape). Youden cut 0.793 (J 0.66); a 0.78 cut keeps 90% of same-file evidence and
  admits 33% of other-file passages. NOT adopted: the first approved run cited a
  `models.py` chunk at 0.774, which 0.78 would escalate. "Same file" is a noisy proxy
  for relevant; the user decides the value. Entailment has 5 findings so far -- too
  few to measure; every audited review adds to it.

**The inference host is decided and implemented** (the `studio` platform + profile):

| Role | Model | LM Studio name | Quant | ~Size |
|---|---|---|---|---|
| `diagnosis` | Qwen3-Coder-30B-A3B-Instruct | `qwen3-coder-30b-a3b` | MLX 6-bit | ~25 GB |
| `judge` | Gemma-4-12B | `gemma-4-12b` | MLX 4-bit | ~7 GB |
| `utility` | Qwen3-1.7B | `qwen3-1.7b` | MLX 4-bit | ~1.1 GB |
| `embedding` | EmbeddingGemma-300m | `text-embedding-embeddinggemma-300m-qat` | QAT | 229 MB |

**`bge-m3` is gone from every profile.** It is not obtainable through LM Studio under any
name, which nobody could discover until a call reached a live server. The embedder is now
EmbeddingGemma-300m at **768 dims, measured** against `/v1/embeddings` rather than assumed.

**The `mac_max` profile changed on 2026-09-11, on measurement:** its diagnosis model is now
`mistralai/ministral-3-14b-reasoning` (9.1 GB), not `gemma-4-26b-a4b-qat`. The bigger model
**cannot emit parseable JSON against this project's own prompt** — it enters a repetition
loop inside a JSON string, 0 of 6 parses at temperatures 0–0.3. Ministral gives 3/3 at
temperature 0 with byte-identical output. The profile is now `resident` at 11.87 GB with
32k context, instead of `jit_swap` at 18.35 GB with 16k.

A **64 GB Mac Studio**, LM Studio + the MLX runtime (not vLLM — no CUDA). ≈34 GB of
weights, ~10 GB left for KV cache, everything resident at 32k context. Raise the GPU
budget with `sudo sysctl iogpu.wired_limit_mb=49152`. `lms get <name> --mlx`, then
`lms server start`. The **reranker stays in-process on the laptop** — neither LM Studio nor
MLX exposes a rerank endpoint. Reach the Studio over **Tailscale, never a public IP**: LM
Studio has no auth and the adapter's API key is a placeholder the server ignores.
T12 now redacts secrets and PII before a prompt leaves the process, which reduces but
does not remove the reason for the tailnet.

**Layer 0 is measured, not estimated:** 200,000 log lines in → **2,041 bytes** of evidence
in the checkpoint. The reduction happens in the *source*, while lines are still a stream —
there is no later point where it is still cheap.

**A recurring defect shape, now seen five times** (T5, `action_type`, log_review's missing
`normalize`, nested Pydantic reaching `build_query`, and `eval/demo_multitenant.py`): a
component fails **safe**, renders a whole surface inert, and the suite stays green because
a stub stood in for real input. When something routes to `human_review` or escalates 100%
of the time, suspect this before concluding the guardrails are working.

The fifth one adds a corollary worth applying: `demo_multitenant.py` was broken three ways
for weeks because **nothing in the suite ran it**, while `README.md` told humans to run it.
Anything a doc instructs a human to run should be exercised by CI, or the doc is an
unchecked claim — the same argument as pinning the drifted doc numbers with a test (T24).

**Git hygiene, specific to this repo:** clicking commit in VS Code's Source Control pane has
twice produced a commit whose message is the unedited git template boilerplate (`b32e8d6`,
`673447b`). Check `git log -1` after committing that way. Also: this directory has
previously acquired a fresh `git init` on `main` with zero commits — committing that shape
would produce a root commit with no ancestor in common with the remote, pushable only by
force. Fetch the real history and reset onto it instead.

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
- `TODO.md` — code-level worklist (T1–T28), each with a "done when"
- `MODEL_CARD.md` — scope limits and known failure modes
- `../chat-archive/` — readable Claude Code session transcripts, kept **outside** this
  repo (2.7 MB of JSONL per session, and a much wider sensitive surface than the
  decisions themselves). `DECISIONS.md` §8/§9 remains the canonical handoff; the
  archive is the raw source it was distilled from.

## Working style in this repo

Docstrings here carry *reasoning*, not restatements of the code — usually the
incident that motivated the design ("this used to do X, which meant Y was void at
runtime while looking wired"). Match that when you touch a file: say why, and if you
close a gap, name it (T-number / G-number) and update the tracker.
