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

## 1. The two machines

Both are first-class. There is no primary and no fallback; the design must run on
each, and results must be comparable between them.

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
is why the default model profile is sized the way it is.

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

- **`CRA_PLATFORM`** (`mac` | `cuda` | `cpu`, auto-detected) — *how* to run:
  quant format, context budget, KV dtype, residency, threads, reranker device.
  **Must** differ per machine.
- **`CRA_PROFILE`** (`portable` | `mac_resident` | `mac_max`) — *which* models.
  **Must not** differ per machine, or Mac and Windows numbers stop being
  comparable.

`adapters/platforms/` is the only code that knows about hardware. Nothing in
`nodes/` branches on it.

### 3.2 The profiles

| Profile | Diagnosis | Judge | Hot weights | Runs on |
|---|---|---|---|---|
| `portable` (default) | `qwen2.5-coder-7b-instruct` | `gemma-4-e4b` | ~7.8 GB | both |
| `mac_resident` | `gemma-4-12b` | `qwen3-8b` | ~11.8 GB | Mac |
| `mac_max` | `google/gemma-4-26b-a4b-qat` | `qwen3-4b` | ~18.2 GB | Mac |

Pinned across every profile, and **not** profile-scoped:

- `EMBEDDING_MODEL = text-embedding-nomic-embed-text-v1.5` (768 dims)
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
lms get text-embedding-nomic-embed-text-v1.5   # ~84 MB

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
