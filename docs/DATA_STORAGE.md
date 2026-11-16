# Data storage — what is kept, where, and why

The agent takes actions. Until 2026-09-10 it kept **no record of any of them**:
`AuditSink` and `Tracer` were ports with no implementation, there was no `logging`
call anywhere in the project, and nothing wrote a file. The only persistence was
the checkpointer, which exists to make the approval gate's pause durable and was
accidentally doing the job of an audit trail — badly.

This document is the design. `docs/ARCHITECTURE_GAPS.md` G6 is the gap it closes;
`TODO.md` T23 is the tracing half.

---

## The distinction that drives everything

Two things get confused, and only one of them is a real constraint.

**LLM context is not affected by storing anything.** The model sees exactly what
`nodes/_render.py` puts in the prompt. Writing a record afterwards never touches
a context window.

**Checkpoint size is the real constraint.** LangGraph re-serialises a channel's
whole value on every super-step, so anything large sitting in `TriageState` is
written repeatedly, per run. A code diff is fine. A `LogWindow` is not — G4 puts a
day of logs at ~2M lines.

So the rule is not "store less". It is:

> **Graph state carries references and digests. Payloads live in a store.**

Storing more, outside the checkpoint, costs nothing that matters.

---

## Four kinds of data, deliberately not one store

| Kind | Volume | Lifetime | Access pattern | Status |
|---|---|---|---|---|
| **Run records** (audit, G6) | ~5–50 KB per run | forever, immutable | by tenant / date / target | **built** |
| **Log-review reports** | 1 per tenant per day | forever | by date; compared over time | **built** |
| **Traces** (spans, T23) | ~10 rows per run | weeks | debug one run by id | **built** |
| **Checkpoints** | whole state | until resumed, then N days | by `thread_id` only | **built** |

Conflating these is the mistake the current design made. They differ in every
column, most importantly in mutability: a checkpoint is *meant* to be overwritten
as a run progresses, and an audit record is meant never to be.

---

## Layer 0 — keep payloads out of the checkpoint (built, with G4)

**Measured: 200,000 log lines in, 2,041 bytes of evidence out.** That number is
the layer. LangGraph re-serialises a channel's whole value every super-step, so
without this a day of logs would be rewritten on every step of every run.

For log review, `_raw` / `evidence` should hold a reference, not the lines:

```json
{"ref": "s3://<bucket>/daily/2026-09-10/", "digest": "sha256:…",
 "candidates": [...], "sample": [...], "counts": {"error": 812, "timeout": 44}}
```

The detect -> aggregate -> sample pipeline (G4) runs in the **source**, not in
`normalize_evidence` — the lines must be reduced while they are still a stream. A
`gather()` returning 2M lines for a later node to summarise would already have
put them in memory and, the moment they landed in `_raw`, in every checkpoint of
the run. There is no later point at which the reduction is still cheap.

`adapters/detectors.py` matches an ordered signal table (first match wins, so a
line is counted once under its most specific signal rather than inflating
several). `adapters/source_s3logs.py` streams in windows, keeps running counts
over **every** line, and caps the sample that reaches the prompt. `LogWindow`
(`extra="forbid"`) then validates it, so a source that regresses to returning raw
lines fails loudly instead of quietly making every super-step rewrite a day.

Only the S3 read itself is stubbed — inject `reader=` to drive the real pipeline.

---

## Layer 1 — `AuditSink`: one immutable record per run

`@adapter("audit", "jsonl")`, written by `nodes/audit.py` at every terminal state.

**Written in the graph, not at the HTTP boundary.** The same reasoning as
`resolve_tenant`'s surface check and `execute`'s permission guard: a record that
only exists when a run came through `app/main.py` is missing for the scheduled
job and for every future caller. Put it where the work happens.

**Where the outcomes are recorded:**

| Outcome | Recorded by |
|---|---|
| `human_review` (classify routed to a human) | `audit` node |
| `escalated` (the gate refused) | `audit` node |
| `executed` (approved and written) | `audit` node |
| `denied` (a human said no) | **`await_approval`** |

That last row is an asymmetry, and it is deliberate. A denial resumes the graph
into `execute`, whose `PermissionError` is the mechanism that stops it — pinned by
`test_truthy_is_not_approval`, which exists precisely to prove the guard and not
the routing is what refuses. Routing denials to an audit node instead would
remove that defence in depth. LangGraph has no error edges (G9), so the node that
*observes* the denial has to record it, and that node is `await_approval`.

When G9 lands and the graph gains failure edges, denials and node errors should
both route to one terminal audit path and this asymmetry should go.

**The record.** `schema_version` is first because a store you cannot re-read after
a field change is not a store.

```
schema_version, run_id, thread_id, recorded_at
outcome                             # human_review | escalated | denied | executed
tenant, surface, target, route
policy_version, prompt_version      # already pinned into state by resolve_tenant
retrieved[]  {id, score}            # THE highest-value field -- see below
checks[]     {check, step, status, message, file, line}   # pre-push detectors
gates        {min_relevance_score, min_entailment_score, severity_allowlist, allowed_actions}
                                    # EFFECTIVE bars: independent_review.effective_gates()
runtime      {platform, profile, models{role: id}, entailer, model_server}
diagnosis    {summary, severity, citations[], findings_count,
              findings[] {severity, file, line, issue, suggestion, grounded, citation_ids},
              proposed_comment}
review       {groundedness_ok, risk_ok, permission_ok, entailment_ok, failures[],
              reasoning}                # judge's free text: explanation, never the verdict
approved, approver                  # identity is a token today; see Open questions
action, result                      # what was posted, verbatim, and what came back
```

`retrieved[]` carrying real scores is the field that earns the layer. **T21 — the
grounding gate being inert for weeks — would have been one line of this**, showing
every score as 1.0 on every query. The first live model runs are about to produce
the numbers that say whether the thresholds are right; without this they evaporate.

---

## Layer 2 — `Tracer`: spans (built, T23)

Per node: name, duration, error. Per retrieval: query, ids, scores. Per LLM call:
model, token counts, **prompt hash**.

**Full prompt text stays off by default**, behind `CRA_TRACE_PROMPTS=1`. T12
now redacts before the prompt is built, so a stored prompt is a sanitised one —
but the guard is pattern-based, and a durable verbatim copy of every prompt is
still a larger surface than a hash. Hash always; text only while debugging.

---

## Live status — `var/live/` (built 2026-11-16)

Traces are written when a span ENDS, so they cannot say what is running now; a
3-minute diagnosis was invisible until it finished. `app/live.py` writes the other
half: one small JSON snapshot per graph run (`run-<thread>.json`, every stage's
start, end, duration and status) and per ingest (`ingest-<tenant>.json`, phase,
done/total, rate, ETA). `python -m app.monitor` renders them at
`http://127.0.0.1:8765`, with stage estimates from the traces and finished runs
from the audit store.

- **Overwritten, not appended** — it is a view of now, not a record. The audit
  record remains the artefact of what happened.
- **Never evidence text** — ids, scores, counts, booleans, durations only. The
  page is a wider surface than a file, so the diff and snippets stay out.
- **Best-effort by contract** — a failed write costs the view, never the run.
- **The monitor binds 127.0.0.1 and has no auth.** Same trust level as `var/`.
  It shows the audit record's diagnosis text and findings, which is model output
  over redacted evidence, so the localhost-only rule matters more than it did.
- **Run history is traces joined with audit, by `run_id`.** A run that crashed
  mid-graph writes no audit record (G9), and listing audit records alone hid
  exactly the run worth looking at.

## Activity log — `var/activity/` (built 2026-11-16)

One JSON record per OPERATION -- graph node, LLM call, query embedding, approval
decision, API request, ingest job -- success or failure, with exception type,
message and full traceback on failure. A port of the Masterclass `Logger` design:
same record fields, same derived severity, same masking rules (secret keys ->
`[REDACTED]`, PII keys masked) applied on the way in, plus this service's InputGuard
over error text and tracebacks. Daily JSONL files; `CRA_ACTIVITY_LEVEL` gates the
minimum severity written. Shown, filterable by severity / operation / text /
correlation id / window, on the monitor's Activity log panel.

Distinct from the audit record (one row per RUN: what was decided) and the trace
(timing per node). Retention is not automated yet -- the same open question as
`var/` generally.

## Layer 3 — log-review reports

`@adapter("sink", "report")` writes **the analysis, not the logs**. The raw lines
already live in S3, which is the source of record; copying them into a second
store doubles the PII surface and buys nothing.

```
var/reports/<tenant>/<YYYY-MM-DD>.json    # structured: candidates, aggregates, citations
var/reports/<tenant>/<YYYY-MM-DD>.md      # rendered, for a human
```

Sampled lines are cited by S3 key + offset rather than copied. A day's report is
kilobytes, and you can still jump back to the source.

**A prerequisite had to be fixed to make this reachable at all.**
`independent_review` computed the action type as `comment` / `comment+label`,
hardcoded, while every log-review policy allows only `["report"]`. So
`permission_ok` was false for every log-review run, every one escalated, and
`ReportSink` could never be reached. The action type now comes from the Surface
(which already declares its own sink), falling back to the comment logic when a
surface does not define one — so the gate stays computable with no surface, which
`eval/harness.py` depends on.

---

## Layer 4 — checkpoint retention (built)

`jobs/prune_checkpoints.py`, dry-run by default (`--apply` to delete):
completed threads expire after N days, **suspended threads never expire** — those
are pending approvals, and dropping one silently discards a human's decision.
`MemorySaver` never evicts at all, fine for a test and wrong for a service.

The audit store is the index, because the checkpointer is keyed by `thread_id`
and offers no "list every thread" query. Deletion goes through the checkpointer's
own `delete_thread`, never raw SQL, so it works on any backend. The live state is
re-checked before every deletion: the audit record says what happened *when it was
written*, and the checkpointer is the authority on what is pending now.

**A prerequisite had to be fixed.** `AsyncPostgresSaver.from_conn_string()`
returns an async *context manager*, and `get_graph()` handed that straight to
`build_graph` — so with `DATABASE_URL` set, the graph was compiled with a context
manager as its checkpointer. It failed only in the configuration that matters,
far from the cause. `app/deps.py::open_checkpointer()` now owns the lifecycle and
a FastAPI lifespan enters it once.

---

## Backend: JSONL now, Postgres behind the same port later

```
var/audit/<tenant>/<YYYY-MM>.jsonl
var/reports/<tenant>/<YYYY-MM-DD>.{json,md}
```

Rooted at `CRA_DATA_DIR` (default `./var`). Chosen over Postgres for the first
implementation, deliberately:

- **Stdlib only**, so it works before `requirements-local.txt` — the same rule
  that keeps `doctor` and the gate suite runnable on a bare machine.
- **Append-only is closer to immutable** than a table anything can `UPDATE`.
- **It can exist before the first live model call.** Waiting for Postgres means
  the first real diagnoses go unrecorded, which is the T21 mistake a second time:
  the run that would have shown you the numbers is the one you did not record.

`@adapter("audit", "postgres")` behind the identical port when SQL is wanted —
the same swap as `tenant_fs` -> a DB store, with no node changed.

Two consequences worth taking:

- **G7 (idempotency)** becomes a lookup in this store on
  `(tenant, surface, target, content-hash)`.
- **G17 (capability eval)** gets its corpus. Diagnosis quality cannot be scored
  over runs that were never recorded.

---

## Trust boundary

**The audit store holds evidence-derived text.** T12 is now closed — the
`InputGuard` redacts secrets and PII before any model call, so the diagnosis it
quotes was produced from sanitised evidence, and `build_record` stores the
guard's **counts only**, never a redacted value. `guard.judge` holds the same counts for the judge's
DIAGNOSIS_UNDER_REVIEW block -- model output built from untrusted evidence, where
injection phrasing the model carried forward shows up. It was discarded until
2026-11-16.

**The checkpoint is redacted too (T25, closed 2026-11-17).** The guard used to run
only on the way into a prompt, so `_raw` and `evidence` persisted unredacted --
Postgres on disk in production. `fetch_evidence` now passes the whole payload
through the guard (`nodes/_render.py::guard_payload`) BEFORE the first channel
write. It has to be there and not later: a checkpointer keeps every super-step,
so redacting in `normalize_evidence`, or dropping `_raw` afterwards, would still
leave the fetch step's checkpoint holding the secret. Identifier fields (paths,
refs, URLs, commit ids) are left untouched -- routing reads paths.
`tests/test_graph_resume.py` checks the whole checkpoint HISTORY.

The accepted consequence: the approver sees `[REDACTED:<kind>:<n>]` where a
secret was in the diff. That still says a secret was committed, which is the
finding; nobody approving a review needs the value.

Redaction is also pattern-based rather than exhaustive, so `var/` is
therefore as sensitive as the source data and must not be committed, synced to a
shared drive, or shipped to a log aggregator without redaction. `var/` is in
`.gitignore` for exactly this reason.

---

## Open questions (assumed, not decided)

1. **Where `CRA_DATA_DIR` points.** Assumed local disk on the agent machine. If
   the daily log job runs as its own ECS RunTask (G16), reports need shared
   storage — an `@adapter("sink", "report_s3")` behind the same port, not a
   change to any node.
2. **Retention.** Assumed keep-forever, on the grounds that this is the record of
   every action the agent took. If anything derived from MilesOne production logs
   carries a compliance clock, reports need a TTL and this assumption is wrong.
3. **Approver identity.** `execute` receives the approver's OAuth token, and the
   record currently stores only whether a token was present — never the token
   itself. Resolving it to a username needs a `whoami` call the client does not
   yet make (G3/G13).
