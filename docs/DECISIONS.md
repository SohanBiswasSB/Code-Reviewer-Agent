# Decisions & cross-machine handoff

Development happens on **two machines** and the working notes do not travel with
them. This file is the portable memory: what was decided, **why**, and what was
discovered the hard way. `TODO.md` holds the open worklist; this holds the
context behind it.

Keep it append-only in spirit — when a decision is reversed, say so and why,
rather than editing history away.

> **Maintenance rule (standing).** This file is updated **every working session**,
> not at milestones. Chat history does not cross machines and is not durable, so
> anything decided in conversation and not written here is lost. Whenever a
> decision is made, a constraint is discovered, a number is measured, or
> something is learned the hard way, it goes in here in the same commit as the
> code it explains — not in a later "docs" commit that never happens.
>
> Section 8 is a dated log for anything that does not belong to an existing
> section. Prefer updating the relevant section and noting the date there.

---

## 1. The machines

Two development machines, both first-class — there is no primary and no fallback;
the design must run on each, and results must be comparable between them. Since
2026-09-10 there is also a third machine that runs *models only* and never the
graph (section 1a).

| | Mac | Windows |
|---|---|---|
| CPU | Apple M4 Pro, 14-core | AMD Ryzen 7 1700 (8c/16t) |
| Memory | 24 GB unified | 16 GB system RAM |
| GPU | M4 Pro integrated (20-core) | NVIDIA GTX 1060 **6 GB** (Pascal, compute 6.1) |
| LLM runtime | LM Studio **MLX** | LM Studio **llama.cpp CUDA** |
| `CRA_PLATFORM` | `mac` | `cuda` |
| conda | miniforge, env `CodeReviewerAgent` | Anaconda, env `CodeReviewerAgent` |
| Interpreter | `/opt/homebrew/Caskroom/miniforge/base/envs/CodeReviewerAgent/bin/python` | `C:\Users\sohan\anaconda3\envs\CodeReviewerAgent\python.exe` |

Python **3.14.7** on both. Recreate with:

```bash
conda create -n CodeReviewerAgent python=3.14
conda activate CodeReviewerAgent
pip install -r requirements-dev.txt
```

The 6 GB VRAM on the Windows box is the tightest constraint in the project and
is why the **default** model profile is sized the way it is. It stopped being the
binding constraint on 2026-09-10 — see the third machine below — but `portable`
remains sized for it, because a default that only one machine can hold defeats
the point of a shared profile.

### 1a. The third machine — a REMOTE inference host (added 2026-09-10)

| | Mac Studio |
|---|---|
| Memory | 64 GB unified |
| Role | Serves models over the network. Does **not** run the graph. |
| Runtime | LM Studio, MLX |
| `CRA_PLATFORM` | `studio` — **never auto-detected**, must be set explicitly |
| `CRA_PROFILE` | `studio` |
| Reached via | `LMSTUDIO_BASE_URL`, over a tailnet (**not** a public IP) |

This splits an assumption that was baked in everywhere: until now, the machine
running the agent and the machine running the models were the same computer.
`adapters/platforms/studio.py` documents which `runtime()` keys describe which
machine. The agent still runs on the M4 Pro, and the reranker with it — it is
in-process `sentence-transformers`, and no LM Studio or MLX server exposes a
rerank endpoint, so it never touches the Studio.

**vLLM was considered and rejected**: no CUDA on Apple Silicon. Staying on LM
Studio also means the existing adapter needs no change at all, and its JIT
load/auto-unload keeps all roles behind one port — vLLM serves one model per
process, which would have forced a per-role `base_url` the config has no way to
express.

---

## 2. Git

- Remote: `https://github.com/SohanBiswasSB/Code-Reviewer-Agent`
- Working branch: **`ID_SB_Ongoing_Development_Branch`** — not `main`
- The repo is **private**. It intentionally contains employer-internal detail
  (`tenants/milesone/policy.json` names a real repo; `docs/MILESONE_ADAPTATION.md`
  names real S3 buckets, ECS log streams and past incidents). Do not make it
  public without genericising that first.

**Incident worth remembering.** The branch was first created with
`git checkout -b <branch> origin/main`, which sets the branch's *upstream* to
`origin/main`. A push from the VS Code Source Control button then went to `main`
instead of the dev branch. Fixed by pointing the upstream at the right ref:

```
[branch "ID_SB_Ongoing_Development_Branch"]
    merge = refs/heads/ID_SB_Ongoing_Development_Branch   # was refs/heads/main
```

Use `git checkout -b <name> --no-track <base>` and set the upstream explicitly, or
this recurs. `main` may still need resetting to its initial commit
(`git push --force-with-lease origin <initial-sha>:main`).

`.gitattributes` pins `eol=lf`. This is load-bearing with two machines, not
cosmetic — without it each machine rewrites every file on its first commit.

---

## 3. Model decisions

### 3.1 Two axes, deliberately not one switch

SF-Crime-Prediction uses a single `SFCRIME_PLATFORM`. Here it is split, because
the two axes have opposite portability requirements:

- **`CRA_PLATFORM`** (`mac` | `cuda` | `cpu` auto-detected; `studio` set by hand)
  — *how* to run: quant format, context budget, KV dtype, residency, threads,
  reranker device. **Must** differ per machine. `studio` is the exception to
  auto-detection, because a host reached over the network cannot be detected
  locally.
- **`CRA_PROFILE`** (`portable` | `mac_resident` | `mac_max` | `studio`) —
  *which* models. **Must not** differ per machine, or Mac and Windows numbers
  stop being comparable. `portable` is the one that must hold everywhere; the
  others exist so a bigger machine can be measured *against* it, which only
  works because the comparison baseline never moves.

`adapters/platforms/` is the only code that knows about hardware. Nothing in
`nodes/` branches on it.

### 3.2 The profiles

Hot weights = diagnosis + judge + embedder, the set that must be co-resident to
avoid a model swap mid-run.

| Profile | Diagnosis | Judge | Hot weights | Runs on |
|---|---|---|---|---|
| `portable` (default) | `qwen2.5-coder-7b-instruct` | `gemma-4-e4b` | ~8.9 GB | all three |
| `mac_resident` | `gemma-4-12b` | `qwen3-8b` | ~12.9 GB | Mac, Studio |
| `mac_max` | `google/gemma-4-26b-a4b-qat` | `qwen3-4b` | ~19.3 GB | Mac (swaps), Studio |
| `studio` | `qwen3-coder-30b-a3b` @6-bit | `gemma-4-12b` | ~33.2 GB | Studio only |

Pinned across every profile, and **not** profile-scoped:

- `EMBEDDING_MODEL = bge-m3` (**1024 dims**) — replaced
  `text-embedding-nomic-embed-text-v1.5` (768) on 2026-09-10; see section 8
- `RERANKER_MODEL = BAAI/bge-reranker-v2-m3` (in-process, CPU/MPS)

### 3.3 Why these choices

**The judge is always a different model family from the diagnosis model.** Not
taste: a same-family judge exhibits self-preference bias, which would quietly
undermine the only reason `independent_review` exists.
`ModelSet.families_are_independent()` checks it structurally.

**Embedding and reranker are pinned outside the profiles** because a different
embedder is a different vector space. Varying it would break index portability
*and* silently change what every `min_relevance_score` in `tenants/*/policy.json`
means.

**`portable` is sized for 6 GB**, so both machines can hold it and the default
comparison is apples-to-apples.

**Qwen2.5-Coder-7B despite being a generation old.** Qwen3-Coder's smallest
official size is 30B-A3B — there is no 7B/8B/14B code variant — so the newest
*code-specialised* model that fits 6 GB is still the 2.5 generation. Revisit if a
smaller Qwen3-Coder ships. The alternative is a current-gen *general* model
(`qwen3-8b`) at the cost of code specialisation; worth measuring, not assuming.

**Residency is computed, not declared.** `base.plan_residency()` compares the
profile's hot weights plus a KV reserve against a per-platform memory budget
(Mac 18 of 24 GB, CUDA 5 of 6 GB VRAM). An earlier version declared
`residency: "resident"` per platform, which a 15.6 GB diagnosis model on a 24 GB
Mac proves false. `jit_swap` is **not** a failure: the graph is strictly
sequential and then waits on a human, so a few seconds of model swap is invisible.

**An oversized or unknown profile raises.** A silent downgrade would label eval
results with a profile that never ran.

**The reranker runs on CPU on the Windows box, on purpose.** LM Studio already
owns all 6 GB for the diagnosis model; co-loading a 568M cross-encoder risks an
OOM mid-generation. It reranks once per run, before any LLM call, so CPU is fine.
This is why `requirements-local.txt` documents the CPU torch wheel as the wanted
one there.

### 3.4 Downloads

Use `--mlx` on the Mac, `--gguf` on Windows. **`lms get` takes LM Studio catalog
names, not Hugging Face repo paths** — `lms get lmstudio-community/Foo-GGUF`
fails. A bare search term opens the variant picker, which is the reliable form.

```bash
# both machines -- the default profile
lms get qwen2.5-coder-7b-instruct        # diagnosis  ~4.7 GB
lms get gemma-4-e4b                      # judge      ~3.0 GB  (prefer a QAT build)
lms get qwen3-1.7b                       # utility    ~1.1 GB
lms get bge-m3                           # embedding  ~1.2 GB, 1024 dims

# the Mac Studio -- the `studio` profile, all resident
lms get qwen3-coder-30b-a3b --mlx        # diagnosis  ~25 GB at 6-bit
lms get gemma-4-12b --mlx                # judge      ~7.0 GB

# Mac only
lms get gemma-4-12b --mlx                # mac_resident diagnosis
lms get qwen3-8b --mlx                   # mac_resident judge
lms get qwen3-4b --mlx                   # mac_max judge
lms get google/gemma-4-26b-a4b-qat --mlx  # mac_max diagnosis  ~15.6 GB

# the reranker is NOT an LM Studio model -- no rerank endpoint exists
pip install -r requirements-local.txt
```

LM Studio settings that matter on the 6 GB box: context **8192**, K/V cache
quantisation **Q8_0**, flash attention on, max GPU layers that fit, temperature
**0**, and JIT load + auto-unload enabled in the Developer tab so the
diagnosis↔judge swap is automatic. Then `lms server start` on port 1234.

Models already on the Windows disk that **cannot run there**:
`google/gemma-4-26b-a4b-qat` (15.6 GB) and `qwen/qwen3.6-27b` (17.5 GB) — both
exceed 16 GB of system RAM once the OS is accounted for. They are Mac-tier.
`qwen3.6-27b` is not in any profile: it is 2 GB larger than the Gemma MoE *and*
dense, so it decodes far slower while fitting worse.

---

## 4. Other decisions from this session

**The gate suite and the unit tests are separate on purpose.** `eval/harness.py`
is a regression/safety-gate suite at a permanent 100% bar with `pass^k`
semantics, dependency-free and model-free so it gives identical results on both
machines. `pytest` covers ordinary correctness. Merging them would slowly turn
the gate bar into an average, which is exactly the conflation
`docs/BEST_PRACTICES.md` warns about. CI runs the gates **first and unpinned**, so
a broken requirements file cannot mask a weakened gate.

**Requirements are split** (core / `-local` / `-dev`), following SF-Crime.
`sentence-transformers` pulls ~2.5 GB of torch that neither the graph nor the API
needs, so it stays out of core.

**`doctor` uses stdlib only** and distinguishes FAIL (cannot run; non-zero exit
so CI can gate) from WARN (degraded but usable). A diagnostic that needs the
dependency it is diagnosing is not much of a diagnostic.

**LLMs bind their model at construction**, and `config.get_llm(role)` takes a
role. This makes it structurally impossible for a caller to reuse one model for
both diagnosis and judge.

**Evidence always goes in the user turn, never the system turn.** System content
is trusted operator instruction; an untrusted diff must never carry that
authority.

---

## 5. Discovered the hard way

- **`config.py` and `registry.py` were missing** from the original skeleton —
  only their `.pyc` survived, from a sandbox at
  `/sessions/.../mnt/outputs/Code_Reviewer_Agent/`. Both were rebuilt to the API
  their bytecode recorded. Nothing but `eval.harness` could import before that.
- **`app/state.py` says `target`; two nodes read `issue_url`.** The graph would
  have raised `KeyError` on any real run. Fixed.
- **`eval/harness.py` crashed on a `✅` under Windows cp1252** *after* printing
  all-pass, so a green run exited non-zero. **Keep the output ASCII.**
- **The IDE reports `Unable to import 'config'`** because the project root is not
  on the analysis path. `.vscode/settings.json` sets
  `python.analysis.extraPaths` to fix it; the code runs fine via `python -m`.
- LM Studio's reported model ids carry publisher and/or quantisation
  (`qwen/qwen3-8b@q4_k_m`), so `llm_lmstudio.resolves_to()` matches loosely
  rather than exactly.

---

## 6. Where things stand

Done: the plugin layer, the platform/profile layer with computed residency, the
LM Studio adapter, role-bound LLMs, `doctor`, 24 safety gates, 38 unit tests,
ruff clean, model card, `.env.example`.

**Not done, and the biggest gap: the resume path (`TODO.md` T3/T4).** Nothing
ever writes `approved`, so with the real graph **every run hits `execute`'s
`PermissionError`**. The guard is correct; the pipeline is unusable until
`interrupt()`/`Command(resume=...)` and a real checkpointer are wired.

Then T5/T6 — nodes reaching providers through their ports, and tenant prompts
actually being loaded (`TenantStore.prompt()` is currently called by nothing).

**Nothing has yet been verified against a live model.** The LM Studio request
shape and every number in `adapters/platforms/mac.py` are reasoned, not measured.
`python -m app.doctor` on the Mac is the first step to changing that.

---

## 7. Cheat sheet

```bash
conda activate CodeReviewerAgent

python -m app.doctor              # is this machine able to run the agent?
python -m eval.harness            # safety gates, 100% bar, no dependencies
pytest                            # 38 unit tests
ruff check .
python -m eval.demo_multitenant   # one diagnosis, two codebases, two outcomes

python -c "import json, config; print(json.dumps(config.registered(), indent=2))"
```

---

## 8. Session log

Dated entries for things that do not fit a section above. Newest last.

### 2026-09-09 — first working session

- Recovered the missing `config.py` / `registry.py`; made the skeleton importable.
- Built the platform/profile layer and the model set; nothing measured yet.
- Built the LM Studio adapter, role-bound LLMs, `doctor`; expanded the gate suite
  9 -> 24 and added 38 unit tests.
- Wrote `TODO.md`, `MODEL_CARD.md`, `.env.example`, this file.
- Established the maintenance rule above: keep this file current every session,
  because the chat context does not travel to the Mac.
- Still true at the end of the session: **no code path has touched a live model.**

### 2026-09-09 (later) — first end-to-end probe

Stubbed only `tools/github_client.py`'s network calls, gave the graph a real
`MemorySaver`, and invoked it. Findings, all measured:

- **The spine runs.** fetch -> normalize -> classify -> diagnose ->
  independent_review -> await_approval all execute, and the run **suspends with
  `next=('execute',)`**. The approval gate genuinely holds.
- **`execute` refuses correctly.** Resuming without setting `approved` raises
  `PermissionError`. The safety guard is real, and it is also why the pipeline is
  unusable until T3 lands: nothing ever sets that flag.
- **The gate that actually holds today is `interrupt_before=["execute"]`**, not
  `await_approval` — which runs, builds a payload, and returns without
  suspending. Confirms T4: the payload it builds is never surfaced to anyone.
- **`_policy` and `_surface` are absent from the returned state.** New finding,
  now T14: the graph never resolves the tenant, so every codebase is gated by
  `_default`'s thresholds and `classify` ignores the active surface.
  `demo_multitenant` passes `_policy` by hand, which is why this looked wired.
- **The checkpointer warns on every state type** (`IssueEvidence`, `Diagnosis`,
  `Citation`, `ReviewResult`, `Snippet`): "will be blocked in a future version".
  Now T15 — on a stricter LangGraph, suspended runs become unresumable, which
  would destroy the durability the approval gate depends on.
- The diagnosis was the hardcoded stub and the citation came from the 4-document
  demo corpus, so the `approve` verdict this produced is meaningless as evidence
  of anything except that the plumbing connects.

Conclusion: the safety machinery works; the useful work does not exist yet.

### 2026-09-09 (later still) — prompt audit

Read all eight prompts against the code that is supposed to consume them. The
prompts themselves are strong — the judge prompt in particular states outright
that a verdict "will be ignored", which is the right way to encode
judge-explains/code-decides in wording as well as in structure. Every problem
found is a mismatch between the prompts and the code:

- **T16 (safety hole).** Tenant prompt overrides do not inherit — the store falls
  back at file level, so a tenant override *replaces* `_default`. Every override
  in the repo is a short emphasis note that merely *mentions* the org contract in
  prose. Once T6 lands, overriding tenants would get a prompt with no grounding
  rule, no untrusted-input rule, no no-verdict rule and no schema, while
  `_default` — which nobody overrides — keeps all of them. Filed as G14 under P2
  hardening in ARCHITECTURE_GAPS; that is a misfiling and it should be promoted.
- **T17.** The diagnosis prompt specifies a `findings[]` array that `Diagnosis`
  has no field for, so the most useful half of the output would be dropped at
  parse. It also asks the model for `citations[].score`, which must come from the
  retriever — harmless today only because the gate looks the score up itself, but
  it invites precisely the wrong assumption.
- **T18.** The judge prompt instructs it to verify citation ids against
  RETRIEVED, but `independent_review` passes only the diagnosis. The booleans
  still do the real check, so the decision stays sound — but the reasoning shown
  to a human approver would claim a check that never happened.
- **T19.** `prompts/*.md` duplicates `tenants/_default/prompts/**` byte-for-byte,
  and the surfaces point at the flat copies even though the README calls the
  tenant copies canonical.
- **T20.** No few-shot exemplars, no structural delimiting of untrusted input, no
  version marker, no eval coverage. The first two matter more here than they
  would with a frontier model, because the default profile is a 7B diagnosis
  model and a 4B judge.

### 2026-09-09 (end of session) — RAG audit against the PDF

Measured the retriever instead of reading it, and found the most serious defect
so far.

**T21 — the grounding gate is inert against real retrieval output.**
`tools/retrieval.py` returns the RRF fused score divided by the top score, so
the best hit is always exactly 1.0. RRF is `sum(1/(60+rank))` — a function of
rank position with no similarity magnitude in it. Measured across on-topic,
off-topic and gibberish queries, the scores were **identical every time**:
`[1.0, 0.984, 0.968]`. Therefore no citation can fail `groundedness_ok` at any
threshold below 0.968, and the relevance half of the gate does nothing.

It looked correct because `eval/tasks.py` hand-supplies scores. The suite proves
the gate *logic*; it never proves the gate gets a meaningful *input*. A green
24/24 run cannot see this.

The design lesson worth keeping: the PDF's calibration method (on-topic cosine
~0.53–0.63 vs off-topic ~0.21–0.22, threshold above the noise cluster) needs a
**calibrated magnitude**. A rank has no noise cluster. **RRF is right for
ordering and wrong for gating** — the gate needs the dense cosine carried
through separately, or a cross-encoder score from the reranker.

This compounds with T10: the thresholds were already invalid for a new embedding
space; they are *also* the wrong kind of quantity.

**T22 — the third grounding check is missing.** The PDF's safety-gate design is
three checks: id is real, relevance clears a calibrated bar, and an
**entailment/faithfulness check between the claim and the cited passage's text**.
The first is implemented, the second is defective, the third does not exist.
Checks 1 and 2 can only establish that a real, on-topic snippet was cited — they
cannot catch a claim that overreaches what the passage supports, which is the
most common RAG failure. The snippet text is already sitting in `_retrieved`.

**T23 — no tracing at all.** The `Tracer` port has no implementation, and no run
records its retrieved ids, scores, citations, booleans or the prompt actually
sent. T21 would have been obvious in a trace showing every score as 1.0.

**What the RAG stack does have,** for balance: hybrid dense+BM25 retrieval fused
with real RRF (an improvement on the PDF's dense-cosine-only reference), and
`classify` is genuinely a form of Adaptive RAG — a pre-retrieval routing
decision that skips retrieval entirely for the deterministic and human_review
paths. Not built: reranking (port only), MMR, contextual retrieval, multi-query
and HyDE fusion, parent-document expansion, AST-aware chunking, and the
corrective-RAG re-retrieve loop (its prompt exists; no node, and no loop cap).

### 2026-09-09 — T21 closed

Split the retriever's single score into two fields, permanently:

- `Snippet.relevance` — calibrated similarity magnitude. The gate reads this,
  and `.score` is an alias for it so no existing caller can accidentally end up
  reading the rank instead.
- `Snippet.fusion` — the RRF value. Ordering only. Never gate on it.

`relevance_of()` scores the candidate **union**, not just the dense hits,
because a BM25-only hit would otherwise arrive with no gateable score and pass
by default.

Measured after the fix: on-topic 0.54–0.88, off-topic and gibberish 0.0. There is
now a real noise floor, which is the property a calibrated threshold needs.

Two things worth keeping from how this was verified:

1. **The regression test was checked for teeth.** Monkeypatching the old
   rank-normalised scoring back in makes the off-topic and gibberish tasks
   *approve* and the suite exit 1. A regression test that cannot fail is not a
   regression test, and asserting that explicitly is cheap.
2. **The blind spot was structural, not careless.** Every gate task
   hand-supplied its scores, which is right for pinning the gate's logic but
   means the suite could never see whether the gate's input was meaningful. The
   fix is a small class of tasks that drive the *real* retriever. Worth applying
   the same suspicion elsewhere: ask of any green suite what it is substituting
   for the real input.

Not fixed, and still tracked in T10: the demo similarity is a Jaccard proxy on a
different scale from any embedding model, so the shipped `min_relevance_score`
values are still uncalibrated. What T21 fixed is the *kind* of number the gate
receives, not its calibration.

### 2026-09-09 — T14, T4 and T3 closed: the gate actually works now

The approval gate runs end to end for the first time. Verified against the real
graph with only the GitHub network calls stubbed:

    resume=True                -> approved=True  -> posted the comment + label
    resume={"approved": True}  -> approved=True  -> posted
    resume=False               -> execute REFUSED (approved is not True)
    resume="yes please"        -> execute REFUSED
    resume=1                   -> execute REFUSED
    resume={"approved": "yes"} -> execute REFUSED
    SUSPENDED at ('await_approval',)

And over HTTP: `/run` returns `awaiting_approval` with the action, the reasoning
and the policy version; `/approve` executes or returns `refused`; `/runs/{id}`
reports what a thread is waiting on, which is how you find pending gates after a
restart.

**Design choices worth not re-litigating later.**

*The tenant is resolved by a NODE, not a `build_graph(tenant, surface)`
argument.* Running inside the graph means the resolved policy lands in the
checkpoint, so a run that pauses and resumes hours later is judged against the
policy it *started* with even if the policy file changed meanwhile. That is the
substance of G8, and compile-time binding would have discarded it. A
`_policy_version` content hash is pinned alongside and shown to the approver.

*Only serialisable values enter state.* `_surface_name` is a string; the Surface
object is resolved lazily. Putting a class instance in a channel would add
another unregistered type to T15, and a paused run that cannot be deserialised
destroys the durability the gate depends on.

*`await_approval` won the gate, not `interrupt_before`.* Both suspend; only
`interrupt()` carries a payload. A gate whose contents the approver cannot see is
a rubber stamp. `execute`'s in-node guard was NOT the mechanism removed — routing
decides whether execute is reached, the guard re-establishes the precondition
regardless of how it was reached.

*`build_graph` now refuses without a checkpointer.* `interrupt()` suspends by
persisting; with nowhere to persist, the gate silently evaporates on restart.

**Four things learned the hard way, all worth keeping.**

1. **The state schema must declare every key a node writes.** LangGraph keeps
   only the channels the schema names, and an undeclared key is *silently
   dropped* — which is why `_policy_version` and `_surface_name` came back
   `None` from a run whose `resolve_tenant` had definitely returned them. It
   fails quietly, not loudly.

2. **Enabling surface delegation exposed a lossy conversion.** `classify` built
   its evidence dict from `__dict__`, which returns stored fields only and drops
   computed properties — so `has_failing_ci` and `is_information_sparse`, the
   exact signals a surface routes on, were invisible. Fixed to include properties.

3. **Enabling delegation also exposed T5 concretely.** The real `code_review`
   surface routes on `touched_areas`, which issue-shaped evidence does not carry,
   so it returns `human_review` and never reaches the gate. It fails *safe*, but
   the code_review surface is non-functional until fetch/normalize produce
   code-review evidence. The gate tests therefore use an injected issue-shaped
   surface, and say so.

4. **A falsy resume value does not resume at all.** Measured: a resume of `{}`
   leaves the run parked at the gate rather than continuing. Safe, but a
   *different* mechanism from the PermissionError, so it is pinned separately.
   And a resume of `None` raises `UnboundLocalError: cannot access local variable
   'resume_is_map'` from `langgraph/pregel/_loop.py` — an upstream bug. The
   `/approve` endpoint coerces to a real bool, so it cannot send either.

**Also corrected one of my own tests rather than the code.** A test asserting
that project_a's 0.40 threshold escalates what `_default`'s 0.35 approves failed:
the evidence grounds at 0.5385, which clears both. The premise was wrong, not the
implementation. Forcing a divergence would have meant hunting a query landing
between the two on a four-document Jaccard corpus — a brittle test of the corpus
rather than the code. The threshold semantics stay pinned by the paired
hand-supplied gate tasks; the graph test now asserts *resolution*.

**Still not true:** durability across a real restart is untested. Only
MemorySaver has been exercised, so the Postgres path is written but unproven —
and the pause surviving a restart is the half of T3 that matters most.

Gates 27. Tests 49 to 62. ruff clean.

### 2026-09-09 — T5 closed: nodes now reach providers only through ports

No file under `nodes/` imports `tools/` any more — checked with a grep rather
than assumed. `fetch_evidence` gathers via the surface's `EvidenceSource`,
`generate_diagnosis` retrieves via the `Retriever` port using the surface's
`build_query`, and `execute` writes via the surface's `ActionSink`.
`GitHubSource.gather` and `PRCommentSink.emit` are implemented rather than
raising.

**G1 is enforced end to end now, and the proof is a refusal.** The namespace comes
from `policy.codebase.repo`, so a real run for `project_a` raises — the
four-document demo corpus cannot be partitioned. That single assertion proves
the policy resolves, the namespace reaches the retriever, and an adapter that
cannot honour isolation fails loudly instead of quietly citing across codebases.
`_default` declares no codebase, retrieves un-namespaced, and proceeds, which is
why the other graph tests use it.

**The code_review surface finally works.** It owns a `normalize`, because its
evidence *shape* differs from issue triage — that was the missing piece behind it
returning `human_review` for every change. Verified on a PR payload:

    files          ['app/migrations/0007_add_bio.py', 'app/models/user.py']
    touched_areas  ['migrations']        <- detected as risky
    route          ai_investigation
    review         escalate_to_human     <- ungroundable vs the demo corpus
    wrote          nothing

That last line is the point. A migration diff does not ground against an
auth-themed corpus, so it escalated rather than posting. My probe script assumed
approval and raised a KeyError — the script was wrong, not the pipeline.

**Decisions worth keeping.**

*Normalisation is per-shape, owned by the surface.* A GitHub issue and a code
diff are not interchangeable, and a surface routing on `touched_areas` gets
nothing from an `IssueEvidence`. Surfaces without a `normalize` fall back to the
built-in issue normaliser.

*The port's shape is the only shape a node sees.* `Retriever.retrieve` returns
dicts, so `independent_review` reads `s["id"]`/`s["score"]` and
`tools/retrieval.py`'s `Snippet` stays an implementation detail behind its
adapter. Aligning on this is what surfaced the mismatch: eleven graph tests
failed with `'dict' object has no attribute 'id'`, which was the right failure.

*`ActionSink.emit` takes an explicit `target`.* Folding the destination into the
action dict would mean mutating the approved payload, and "posted verbatim" has
to mean exactly that.

*Partial-success handling moved into the sink.* It is provider-specific: on
GitHub, commenting is open to any authenticated user while labelling needs
triage/write access, so a 403 on the label after a successful comment is routine
rather than an error. The gate suite now exercises the real `PRCommentSink` for
that case instead of a fake client.

*`touched_areas` changed meaning, deliberately.* It used to be raw paths that
`classify` filtered; it is now already filtered to risky areas by `normalize`, so
`classify` only asks whether it is empty. The same field name meaning two
different things at two points in the pipeline was the worse option. One eval
task had to be updated for this — a real contract change, not a test fudge.

**Five new gate tasks drive raw provider payloads through the surface's real
`normalize`.** The previous delegation tasks hand-built post-normalise evidence,
which is exactly how the shape mismatch stayed invisible while everything was
green. Same lesson as T21: ask what a passing suite is substituting for the real
input.

Gates 27 to 32. Tests 62 to 63. ruff clean.

### 2026-09-09 — T16, T6, T18 closed: prompts compose, and are actually sent

**T16 was the safety hole, and the demonstration is stark.** Prompt resolution
fell back at *file* level, so a tenant override replaced `_default` outright.
Restoring that old behaviour and measuring what each tenant would have sent:

    _default      2733 chars   lost guardrails: none
    milesone       518 chars   lost: grounding, untrusted, no_verdict, schema
    project_a      207 chars   lost: grounding, untrusted, no_verdict, schema
    project_b      215 chars   lost: grounding, untrusted, no_verdict, schema

A 207-character prompt against a 2733-character org contract, with **all four
guardrails gone** — for exactly the three codebases that bothered to customise
anything, while the one nobody overrides kept everything. And it read as if it
inherited, because each override says "Org contract applies (grounded-only,
untrusted-input, no-verdict, structured JSON)" in prose, referring to a file the
model never received.

Prompts now **compose**: the org contract is always included, and a tenant file
is a delta on top. `{{include _default}}` lets a tenant control placement; absent
it, the contract is prepended and the delta follows under a visible
`TENANT OVERLAY` header so a captured prompt can be attributed. This makes the
invariant in MULTI_TENANT.md structural rather than documented: **tenants tune
values, they never remove the boundary.** 24 parametrised tests assert every
guardrail clause survives for every tenant.

**T6 — prompts are resolved and PINNED by `resolve_tenant`,** not fetched by the
LLM nodes. Same reasoning as the policy: they land in the checkpoint, so a run
that pauses at the gate and resumes hours later uses the prompt text it started
with even if the tenant edited the file meanwhile. `_prompt_version` is a content
hash alongside `_policy_version`. A node with no pinned prompt **raises** rather
than falling back to something hardcoded.

**T18 — the judge now receives the retrieved snippets.** Its prompt tells it to
verify citation ids against RETRIEVED; with only the diagnosis it could not, so
any reasoning it produced about grounding was necessarily invented and then shown
to a human approver as though real. The booleans always did the true check, so
the *decision* was sound — but a justification asserting a check that never
happened is worse than silence.

**Three smaller decisions inside this work.**

*Evidence is fenced, in the user turn only.* `nodes/_render.py` wraps untrusted
material in explicit `<<<UNTRUSTED_EVIDENCE>>>` delimiters and never puts it in
the system turn. Saying "treat this as data" is not a boundary — an injected
"ignore the above" reads identically to a real instruction unless something marks
where the data starts. This is the delimiting half of T20; the redaction half is
T12 and is still open, so T20 is partial rather than done.

*A model-supplied citation score is discarded.* `_parse_diagnosis` looks every
score up in the retrieved hits. Verified: a model claiming `0.99` for a snippet
the retriever scored `0.61` gets `0.61`. The gate compares against exactly this
number, so trusting the model's would make grounding self-certified.

*A missing severity defaults to `high`, not `low`.* `risk_ok` only passes
allowlisted severities, so defaulting high means a malformed response escalates.
Defaulting low would auto-approve whatever the model failed to express clearly —
the wrong direction to fail, and an easy mistake to make while writing a parser.

Tests 63 to 96. Gates 32. ruff clean.

### 2026-09-09 — the org contract was not actually generic

Asking "what does T6 even mean" turned up a design leak worth recording. The
shared `_default` code-review prompt opened *"You are a senior reviewer of the
MilesOne Django + Celery backend"*, and its scrutiny list named Credly and
Celery; the log contract named the backend and celery-worker streams and the Zoom
sweep. Since `_default` is prepended to every tenant, **every other project was
being told it was reviewing MilesOne.**

`_default` now carries the guardrails plus universal risk categories that hold
regardless of stack — migrations, auth boundaries, credential handling,
concurrency and background work, data access and transactions, external calls,
injection, observability. The MilesOne material moved into `tenants/milesone/`,
including a new `log_review/diagnosis.md` for that service's patterns. A test
fails if project vocabulary reappears in `_default`.

Also added `tenants/_template/` to copy when onboarding, and `tenants()` now
excludes any `_`-prefixed directory rather than excluding `_default` by name, so
neither the template nor a future one can register itself as a real project.

### 2026-09-09 — T11, T15, T17, T22 closed

**T15 was the dangerous one.** LangGraph warned on every state type that
"deserializing unregistered type … will be blocked in a future version". On a
release that enforces it, **every paused run becomes unresumable** — which
destroys the single property the whole design rests on, since a durable pause is
what makes the approval gate real rather than a UI illusion. Verified rather than
assumed: the suite passes with `LANGGRAPH_STRICT_MSGPACK=true`, CI now runs it
twice (once strict) so it cannot regress quietly, and a unit test round-trips
every state type through `JsonPlusSerializer`.

**T11** — Pydantic v2. `IssueEvidence` and `CodeChange` use `extra="forbid"`: an
unexpected provider field means the payload shape changed, and finding that out
at the edge beats finding out when a node reads something no longer there.

**T17** — `Diagnosis` gains `findings[]`, and the org contract stopped asking the
model for citation *scores* (they come from the retriever; asking invites the
wrong assumption even though the parser ignores them).

**T22** — the third grounding check exists. A new `Entailer` port, defaulting to
**lexical** content-word overlap: crude, but *code* rather than a model, because
the gate's booleans must be computable with no provider reachable or the safety
suite stops being runnable everywhere. A cross-encoder NLI adapter sits behind
the same port. Both return a **score** that `independent_review` thresholds — a
port returning a boolean would be making the decision. `ReviewResult` now carries
four booleans plus a `failures` list, because a gate that only says "escalated"
makes the human redo the diagnosis to find out what tripped.

Two things this exposed, both worth keeping:

- **The eval fixtures were degenerate.** `Diagnosis("s","medium","c")` with empty
  snippet text pinned the id and score checks but made entailment impossible to
  exercise — you cannot ask whether "c" follows from "". That is the **third**
  time a degenerate fixture hid a real defect (T21's constant scores, T5's
  evidence shape). The tasks are now written as a real diagnosis and a real
  snippet would be, with an entailed claim and an overreaching one.
- **The offline stub modelled an ungrounded diagnosis.** It asserted "possible
  duplicate of a known auth regression", which entails nothing about whatever was
  retrieved, so once entailment existed every offline run escalated. It now
  derives its claim from the cited passage.

Gates 32 to 36. Tests 96 to 101.

### 2026-09-10 — a third machine: models move to a remote 64 GB Mac Studio

The 6 GB VRAM ceiling stopped being the binding constraint. Models now run on a
64 GB Mac Studio reached over the network; the graph still runs on the M4 Pro.
Added `adapters/platforms/studio.py` and a `studio` profile.

**The models, and why each.**

*Diagnosis: `qwen3-coder-30b-a3b` at MLX 6-bit (~25 GB).* This is the model
`portable` settles for a 2.5-generation Qwen *instead of* — 30B-A3B is the
smallest size Qwen3-Coder ships, and it does not fit 6 GB. MoE with ~3.3B active,
so it decodes far faster than 30B suggests. 6-bit rather than 8-bit (~32 GB)
because 32k of context is worth more than the last few points of quantisation
quality: `render_evidence` puts the whole diff *plus* five retrieved snippets in
the user turn. Revisit if real diffs turn out to fit in 16k.

*Judge: `gemma-4-12b` (~7 GB).* Different family from Qwen, which is the only
hard requirement. Deliberately not bigger: the judge writes free-text `reasoning`
that a human reads and **cannot** affect the decision — four Python booleans do
that. Deliberately not smaller either, since misleading reasoning shown to an
approver is its own failure mode.

*Utility: `qwen3-1.7b`. Reranker: unchanged, and it stays OFF the Studio* — it is
in-process `sentence-transformers`, and no LM Studio or MLX server exposes a
rerank endpoint, so it runs wherever the agent runs.

**The embedder changed, and the timing was the whole argument.**
`bge-m3` (1024 dims) replaces `text-embedding-nomic-embed-text-v1.5` (768). Doing
this later would cost a full re-index plus a re-measurement of every tenant
threshold; doing it now costs nothing, because T10 is open — there is no index and
the corpus is four documents. This is the cheapest moment the change will ever
have. `EMBEDDING_GB` went 0.1 -> 1.2, which shifts the hot-weight total of every
profile (section 3.2 updated).

It does **not** fix the calibration debt, and slightly enlarges it: the shipped
0.35/0.40 were already meaningless for a 768-dim space and are now meaningless
for a 1024-dim one. Still T10.

**Two things the existing tests got right, worth recording.**

1. **`test_registered_report_includes_hardware` failed on `768 != 1024`, exactly
   as intended.** A hardcoded literal in a test is usually a smell; here it is a
   tripwire, because changing the embedder silently re-scales every
   `min_relevance_score`. Kept it a literal and updated it to 1024, with a comment
   saying why — asserting `== base.EMBEDDING_DIMS` would be tautological and could
   never fail.

2. **`test_every_platform_supports_the_default_profile` iterated a hardcoded
   `("cuda","mac","cpu")`,** so it silently stopped covering anything added after
   it was written — `studio` registered and went unchecked. Now iterates the
   registry. Same species as the recurring lesson at the end of this file: the
   suite was substituting a fixed list for the real set.

**Design notes worth not re-litigating.**

*`CRA_PLATFORM=studio` is never auto-detected.* `detect_platform()` inspects the
local machine, and nothing local can detect a box reached over the network. Set
it explicitly or `auto` will describe the laptop.

*`Platform.runtime()` now describes two machines, and says which key is which.*
It was written when the models and the graph shared a computer. `reranker_device`
and `threads` are agent-side; `quant`, `context`, `residency`, `memory` and the
rest are inference-host-side. Splitting `Platform` into two ports is the real fix,
deferred until a second remote topology exists to generalise from — documented in
the module rather than pretended away.

*The laptop must REFUSE the `studio` profile.* `mac.profiles` does not list it, so
`CRA_PROFILE=studio` on the M4 Pro raises rather than trying to load 33 GB into 24.
Pinned by test, since setting it out of habit is the obvious mistake.

*vLLM was considered and rejected* — no CUDA on Apple Silicon. Staying on LM Studio
means the adapter needs no change at all, and its JIT load/auto-unload keeps all
roles behind one `base_url`; vLLM serves one model per process, which would have
forced a per-role endpoint the config cannot express.

**Security, stated plainly because the topology invites the mistake.** LM Studio
has no authentication and `PLACEHOLDER_KEY` is an ignored placeholder, so a public
IP here is an open inference endpoint on your own hardware. Use a tailnet. This
matters more than usual while T12 is open: nothing yet redacts PII or secrets out
of diffs and log lines before they cross the wire.

Also added the `CRA_*` settings to `.env.example`, which had none despite the
hygiene item claiming otherwise, and fixed the stale `mac_headroom` profile name in
`config.py` and `adapters/platforms/cuda.py`.

Gates 36 (unchanged — model-independent, as designed). Tests 101 to 105. ruff clean.

**Not verified:** no call has still ever reached a live model. Every number in
`studio.py` is reasoned, not measured, exactly as `mac.py`'s are. `python -m
app.doctor` against the Studio with `lms server start` is the first step.


### 2026-09-10 — the agent started keeping records (G6, Layers 1 and 3)

Until today the system took actions and kept **no record of any of them**.
`AuditSink` and `Tracer` were ports with no implementation, there was no
`logging` call anywhere in the project, and nothing wrote a file. The only
persistence was the checkpointer — which exists to make the approval gate's pause
durable, and was accidentally serving as an audit trail while being keyed by
`thread_id`, in-memory by default, and mutable.

Design is in **docs/DATA_STORAGE.md**. Built: Layer 1 (run records) and Layer 3
(log-review reports). Layer 0 (evidence by reference, with G4) and Layer 2
(tracing, T23) are specified and not built.

**The concern that started it was aimed slightly off-target, and the real answer
is worth keeping.** Storing records does not touch the LLM context window — the
model sees only what `nodes/_render.py` puts in the prompt. What *does* bloat is
the CHECKPOINT: LangGraph re-serialises a channel's whole value every super-step,
so anything large in `TriageState` is rewritten repeatedly. A diff is fine; a
`LogWindow` at ~2M lines/day is not. Hence the rule in the design doc: **graph
state carries references and digests, payloads live in a store.** Storing more,
outside the checkpoint, costs nothing that matters.

**Where records are written, and the one asymmetry.** `nodes/audit.py` sits on
every edge that reaches END normally — `human_review`, `escalate_to_human`, and a
completed `execute`. Those first two are OUTCOMES, not absences of one.

A DENIAL cannot reach that node: resuming with anything that is not `True`
continues into `execute`, whose `PermissionError` is what stops the run, and
`test_truthy_is_not_approval` exists specifically to prove the guard rather than
the routing is the mechanism. Routing denials around `execute` would delete that
defence in depth. So `await_approval` — the node that observes the denial —
records it. When G9 adds failure edges, both paths should collapse into one.

Written in the GRAPH, not in `app/main.py`, for the reason `resolve_tenant` and
`execute` give for their own checks: a record that exists only when the run came
through HTTP is missing for the scheduled job and every future caller.

**JSONL before Postgres, deliberately.** Stdlib only, so it works on a bare
machine like `doctor` and the gate suite do; append-only is closer to immutable
than a table anything can `UPDATE`; and it can exist *before* the first live
model call. Waiting for Postgres would mean the first real diagnoses go
unrecorded — the T21 mistake a second time, where the run that would have shown
you the numbers is the one you did not record. `@adapter("audit","postgres")`
behind the same port later.

The audit sink defaults to a REAL writer. Silence has to be opted into
(`CRA_AUDIT=null`) rather than being what you get by forgetting.

**A blocker found on the way in: `log_review` could never be approved.**
`independent_review` computed the action type as `comment` / `comment+label`,
hardcoded, while every log-review policy allows only `["report"]`. So
`permission_ok` was **false for every log-review run** — the surface escalated
100% of the time and its `ReportSink` was unreachable. Layer 3 was unbuildable
until this was fixed. The action type now comes from the Surface (which already
declares its own sink), falling back to the comment shape when no surface is
active so the gate stays computable from a bare state dict, which
`eval/harness.py` depends on.

Same shape as T5: it failed *safe*, and it made a whole surface inert, and
nothing was red. That is now three times a green suite has hidden a defect by
standing in for the real input.

**`ActionSink.emit` gained `context`.** Run metadata — tenant, surface, run_id —
kept separate from `action` for exactly the reason `target` already was: the
action is what a human approved and it is posted verbatim, so nothing may be
folded into it afterwards. `PRCommentSink` ignores it; `ReportSink` needs the
tenant to know where its artefact belongs.

**Reports store the ANALYSIS, not the logs.** The raw lines already live in S3,
which is the source of record; copying them into a second store doubles the PII
surface and buys nothing. `var/reports/<tenant>/<date>.{json,md}`. Sampled lines
should be cited by S3 key + offset once G4 exists.

**Measured, on a real end-to-end run:** the record's `retrieved` field came back
`[0.5385, 0.0, 0.0, 0.0]` — a genuine noise floor, the post-T21 behaviour, now
permanently captured. That field is the reason the layer earns its place: T21
would have been ONE LINE of this showing every score as 1.0.

**Discovered while verifying, and NOT fixed: T15 is only partly closed.**
Under `LANGGRAPH_STRICT_MSGPACK=true` the four Pydantic state types round-trip to
plain **dicts**, not back to their classes — `test_every_state_type_round_trips_
through_the_checkpoint_serialiser` fails on all four parameters.

**Most likely a dependency regression, not a claim that was never true.**
`requirements.txt` pins nothing (`langgraph`, by its own note "pin exact versions
before anything is deployed"), and this machine resolved **langgraph 1.2.11 /
langgraph-checkpoint 4.2.0** — newer than whatever was installed when T15 was
verified. So a serialiser change upstream almost certainly flipped it. NOT
confirmed by bisecting versions; someone should, and then pin.

CI *does* run the suite twice, strict included — the earlier note in this entry
saying otherwise was read off a stale local snapshot, not the repo. Which means
**CI on this branch is red right now**, and correctly so: it is catching exactly
what it was added to catch.

The catastrophe T15 warned about — paused runs becoming unresumable — does
**not** happen: all 116 other tests pass under strict, including every
approval, denial and execute path. The real impact is loss of type fidelity
across a resume. `nodes/_audit.py` is written to tolerate it (`ReviewResult.
decision` is a property, so it is recomputed from the booleans when `review`
comes back as a dict), and that tolerance is pinned by a test. But the claim in
section 8 above should be corrected once someone confirms it on the Windows box.

Also: `var/` added to `.gitignore` — it inherits the unredacted trust boundary
while T12 is open.

Gates 36 (unchanged — model-independent). Tests 105 to 121. ruff clean.


### 2026-09-10 — Layers 0, 2 and 4: the log surface became feasible

Design in **docs/DATA_STORAGE.md**. All four layers now exist.

**Layer 0 — the measured result is the argument.** 200,000 log lines in,
**2,041 bytes** of evidence out. LangGraph re-serialises a channel's whole value
every super-step, so without this a day of logs (G4 puts it at ~2M lines) would
be rewritten on every step of every run. That, not the LLM context window, was
always the real constraint.

**The reduction happens in the SOURCE, not in `normalize_evidence`.** The lines
have to be reduced while they are still a stream. A `gather()` returning 2M lines
for a later node to summarise would already have put them in memory and, the
moment they landed in `_raw`, in every checkpoint of the run. There is no later
point at which the reduction is still cheap. So `adapters/source_s3logs.py`
streams in windows, runs detectors over each, keeps running counts over **every**
line, and caps the sample. `LogWindow` (`extra="forbid"`) validates the result, so
a source that regresses to returning raw lines fails loudly rather than quietly
making every super-step rewrite a day.

Only the S3 read is stubbed. Inject `reader=` and the whole pipeline runs offline
— which is how the 200k-line figure above was measured rather than estimated.

**Detectors are regex and counting, deliberately.** They run over the full stream
before any model sees anything, so they must be cheap enough to touch every line;
a model-backed detector could not, which would defeat the point. Ordered table,
first match wins, so a traceback line is not also counted as a generic `ERROR` and
inflating the noisiest signal.

**`log_review` was unreachable in THREE separate ways.** Two were found today:
no `normalize`, so it fell back to the issue normaliser and raised `KeyError` on
`html_url`; and nested Pydantic models reaching `build_query`, which calls
`.get()`. The third was yesterday's `action_type`. Each failed safe and left the
surface inert with a green suite — the same shape as T5, now four times.

`evidence_as_dict` in `nodes/_surface.py` fixes the nesting generally:
`model_dump()` rather than `__dict__`, because `__dict__` leaves nested models as
models, and the port's contract with a surface is dicts all the way down.
Computed properties are added back, since `model_dump()` omits them and
`has_failing_ci` is exactly what a surface routes on.

**Layer 2 — spans wrap in `build_graph`, not inside nodes.** One insertion point,
no node learns about tracing, and a node added later is covered without anyone
remembering to instrument it. `nodes/` stays free of observability concerns the
same way it is free of provider concerns.

Two details worth keeping. `interrupt()` suspends by RAISING, so a naive tracer
marks every run that reaches the approval gate — the normal path — as a failure;
those are recorded as `suspended`. And tracing is best-effort: a span that could
take down the run it observes is worse than no span at all.

Prompt text is **off by default** (`CRA_TRACE_PROMPTS=1` to enable). T12 is open,
so nothing redacts PII or secrets from evidence; tracing every prompt verbatim
would create a durable copy of exactly what the guard is meant to remove. Hashes
always, text only while debugging.

Tracing defaults ON, unlike most debugging tooling, for one specific reason: the
first live model runs are imminent, and a trace you have to switch on after
something looks wrong is a trace you do not have for the run that went wrong.
T21 is the standing example.

**Layer 4 — and a real bug found on the way in.**
`AsyncPostgresSaver.from_conn_string()` returns an async **context manager**, and
`get_graph()` handed that object straight to `build_graph`. So with `DATABASE_URL`
set, the graph was compiled with a context manager as its checkpointer — failing
at first use, far from the cause, and **only in the configuration that matters**.
Memory-only development never touched it. `app/deps.py::open_checkpointer()` now
owns the lifecycle and a FastAPI lifespan enters it once; `get_graph()` refuses
the Postgres path rather than rebuilding the same bug. NOT verified against a real
Postgres — there is none on this machine.

`jobs/prune_checkpoints.py` is dry-run by default. **A suspended thread is a
pending human approval and never expires**; deleting one would silently discard a
decision somebody was asked to make, and they would find out by the run simply
never resuming. The audit store is the index (the checkpointer has no "list every
thread" query), but the live state is re-checked before each deletion — the record
says what happened *when it was written*, and the checkpointer is the authority on
what is pending now. Deletion goes through `delete_thread`, never raw SQL, so it
works on any backend.

Also: `thread_id` is now mirrored into state, because a record that cannot name
its thread cannot authorise deleting it. Declared in `TriageState` — an
undeclared key is silently dropped, which is the lesson from T14.

**Still not true:** no call has ever reached a live model. And the log surface
escalates on the demo corpus, correctly — four auth-themed documents cannot ground
"db_error, traceback", so `groundedness_ok` fails and it routes to a human. That
is the right behaviour and it is also T10: the corpus is a toy.

Gates 36 (unchanged — model-independent). Tests 121 to 144. ruff clean.


---

## 9. RESUME HERE (session ended 2026-09-10)

Working tree clean, all commits on `ID_SB_Ongoing_Development_Branch`.

### State

```
ruff check .   clean
pytest         144 passed
eval.harness   36 gates pass (pass^k)
pytest strict  4 FAILED  <-- LANGGRAPH_STRICT_MSGPACK, see below. CI is red on this.
app.doctor     exits 1 only because LM Studio is not running
```

All of P0 is closed. The pipeline runs end to end offline, both surfaces now
reach their sinks, every run is recorded, and every node is traced.

### What changed on 2026-09-10 (two sessions, see section 8)

1. **Models moved to a remote 64 GB Mac Studio** (section 1a). New `studio`
   platform + profile; embedder changed to `bge-m3` (1024 dims).
2. **Data storage — all four layers** (docs/DATA_STORAGE.md). Audit records,
   log-review reports, per-node traces, checkpoint retention.

### Still open, in the order I would take them

1. **Pin `requirements.txt`, then fix or re-close T15.** The strict-serialiser
   suite fails on langgraph 1.2.11; the likely cause is an upstream change since
   T15 was verified, and requirements pin nothing. Bisect, pin, then decide
   whether Pydantic state types need registering via `allowed_msgpack_modules`.
   Paused runs still resume — only type fidelity is lost — but CI is red.
2. **T12 InputGuard.** Still the biggest correctness-adjacent gap, and it now has
   a second reason: `var/` durably stores evidence-derived text, so nothing
   redacts PII before it reaches disk as well as before it reaches a provider.
3. **T10 retrieval.** Real index + namespaced ingest, then **re-measure every
   `min_relevance_score`** — they were calibrated for a 512-dim space, are now
   nominally for a 1024-dim one, and mean nothing until measured. The log surface
   escalates on the demo corpus for exactly this reason.
4. **T13 Reranker** — port defined, no adapter. Its calibrated score should then
   supersede dense relevance as the number the gate reads.
5. **T1 live** — nothing has EVER called a real model. Download the four models
   (section 3.4), `lms server start`, point `LMSTUDIO_BASE_URL` at the tailnet.
6. **G7 idempotency** — now cheap: the audit store is the index and
   `LogWindow.digest` is the content hash.

### Two things only you can answer

- **The tenant list.** `tenants/project_a` **is** MilesOne (duplicated) and
  `project_b` stands in for Masterclass. Real names + repos, languages, 2-4 risky
  areas each. `tests/test_tenants.py`, `tests/test_graph_resume.py` and
  `eval/demo_multitenant.py` reference the placeholders.
- **Retention + where `CRA_DATA_DIR` points.** docs/DATA_STORAGE.md assumes
  keep-forever on local disk. If the daily log job runs as its own RunTask (G16)
  reports need shared storage, and anything derived from MilesOne production logs
  may carry a compliance clock.

### Known-broken, deliberately left

- `eval/demo_multitenant.py` builds `Diagnosis(...)` positionally and passes
  attribute-objects where the gate subscripts dicts. It predates T11's Pydantic
  migration and raises `TypeError`. It is in the README quickstart.
- `.env.example` is matched by `.gitignore`'s `.env.*`, so it is NOT in the repo
  despite TODO.md marking it done. Decide whether to commit it with a negation
  rule or drop the hygiene claim.

### Recurring lesson, now at four

A green suite hid a defect by substituting something convenient for the real
input: T21 (hand-supplied scores hid a constant relevance), T5 (hand-built
evidence hid a shape mismatch), T22 (degenerate fixtures made a check
unexercisable), and on 2026-09-10 the log surface — unreachable **three separate
ways** while every test stayed green. When a suite passes, ask what it is
standing in for.
