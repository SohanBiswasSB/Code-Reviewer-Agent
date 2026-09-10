# Code Reviewer Agent

A multi-tenant, plug-and-play **agentic code-review microservice**. One guarded agent
graph is reused across pluggable *surfaces* (PR code review, daily log review), and each
plugged-in codebase brings its own prompts and its own guardrail policy.

The design bet, in one line: **detectors and tools find, the LLM explains and prioritizes,
and code decides.** The LLM is never the source of truth and never emits the verdict.

## The graph

```
target (PR ref | issue url | YYYY-MM-DD)
  └─ fetch_evidence → normalize_evidence → classify   (RULE-BASED, zero LLM calls)
        ├─ deterministic     → await_approval → execute → END
        ├─ human_review      → END                       (no LLM touches it)
        └─ ai_investigation  → generate_diagnosis        (RAG-grounded, cited)
                             → independent_review        (SEPARATE LLM call; EXPLAINS only)
                                  │ decision = 4 Python booleans over real data:
                                  │   groundedness_ok · entailment_ok
                                  │   risk_ok · permission_ok
                                  ├─ approve           → await_approval → execute → END
                                  └─ escalate_to_human → END
```

## Non-negotiable invariants (not tenant-overridable)

1. **The judge never emits a verdict.** `independent_review` asks a second, independent LLM
   for free-text `reasoning` only; the approve/escalate decision is four booleans computed
   in Python from the retrieved scores, the claim-vs-passage entailment score, the severity
   allowlist, and the action allowlist.
2. **`execute` refuses without approval, checked in the node body** — not just in graph
   routing. A future shortcut edge straight to `execute` still cannot bypass the gate.
3. **The approved action is posted verbatim**, never recomputed after approval.
4. **Evidence is untrusted data, never instructions** (diffs and log lines can carry
   injected text).
5. **Nothing mutates infra or the database.** The agent proposes; a human approves; writes
   run as the approver's own identity.

Tenants tune *values* (thresholds, allowlists, prompts). They cannot remove a *boundary*.

## Multi-tenant = multi-codebase

A "tenant" is one codebase. `tenants/<name>/policy.json` carries the `codebase` binding
plus thresholds; `tenants/<name>/prompts/<surface>/<node>.md` overrides prompts. Everything
falls back to `tenants/_default/` (prompts by file lookup, policy by deep merge).

The same diagnosis can be **approved for one codebase and escalated for another**, purely
from policy — no code change. Escalate-to-human is deliberately **shared**: every codebase
routes to a human on the same conditions. See [docs/MULTI_TENANT.md](docs/MULTI_TENANT.md).

## Quickstart

```bash
conda activate CodeReviewerAgent

# can this machine run the agent? platform, models, memory arithmetic, gates
python -m app.doctor

# safety-gate regression suite — needs NO dependencies at all
python -m eval.harness

# same diagnosis, two codebases, different policies
python -m eval.demo_multitenant

# what's plugged into this deployment
python -c "import json, config; print(json.dumps(config.registered(), indent=2))"
```

Everything beyond that needs `pip install -r requirements-dev.txt`.

### Working on two machines

This is developed on both an M4 Pro Mac and a CUDA desktop, and neither is the
"real" one. `CRA_PLATFORM` (auto-detected) decides *how* to run; `CRA_PROFILE`
decides *which* models — and the default profile is sized to fit the smaller
machine so results stay comparable. Interpreter paths, model download commands,
per-machine LM Studio settings and every decision behind this live in
**[docs/DECISIONS.md](docs/DECISIONS.md)**. Start there on a fresh clone.

```bash
python -m app.doctor          # tells you what this machine is missing
```

## Layout

| Path | Role |
|---|---|
| `ports/interfaces.py` | The contracts. Core imports **only** these, never a provider. |
| `nodes/`, `app/graph.py` | The core graph. Holds the safety design. |
| `adapters/` | Concrete providers, self-registering via `@adapter(kind, name)`. |
| `surfaces/` | Capabilities (`code_review`, `log_review`) via `@surface(name)`. |
| `tenants/` | One directory per codebase: policy + prompt overrides. |
| `registry.py`, `config.py` | The plug: decorators register, env resolves. |
| `eval/` | Safety-gate suite, 100% bar, `pass^k` semantics. Dependency-free. |
| `tests/` | Unit tests. Deliberately separate from the gate suite. |
| `app/doctor.py` | Environment check. Stdlib only, so it runs before setup does. |
| `adapters/platforms/` | The only code that knows about hardware. |
| `docs/` | Architecture, the honest gap map, roadmap, standards. |

## Status — what runs vs. what is stubbed

**Runs today:** the rule-based `classify` routing, the four-boolean gate (including
tenant-policy resolution), the `execute` permission guard, `interrupt()`/resume on a real
checkpointer with `POST /run` + `POST /approve` driving it, one immutable audit record per
run, per-node tracing, checkpoint retention, the plugin registry, the filesystem tenant
store, and hybrid retrieval (dense + BM25 fused with RRF) over a small in-memory corpus.

**Stubbed, deliberately and visibly** (`NotImplementedError`, not fake success): the GitHub
client's network calls in `tools/github_client.py` (the `EvidenceSource` and `ActionSink`
adapters around it are real), the S3 read in the log source (inject `reader=` and the rest
of the pipeline runs), the OpenAI client, the vector index and `tools/ingest.py`, and
`jobs/daily_log_review.py`. There is **no GitLab adapter at all** — it is named in the
architecture docs as the thing the port exists to make swappable, not as a stub.

Scope cuts are represented in the schema and documented in the node docstrings, in
[docs/ROADMAP.md](docs/ROADMAP.md), and in [docs/ARCHITECTURE_GAPS.md](docs/ARCHITECTURE_GAPS.md)
(G1–G17, prioritized P0–P2). A precisely-owned gap reads better than a vague claim.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — as-built system view (single source of truth)
- [docs/ARCHITECTURE_GAPS.md](docs/ARCHITECTURE_GAPS.md) — honest audit, G1–G17
- [docs/ROADMAP.md](docs/ROADMAP.md) — prioritized work + accuracy track
- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md) — accuracy-first engineering standards
- [docs/MULTI_TENANT.md](docs/MULTI_TENANT.md) — how a codebase plugs in
- [docs/MILESONE_ADAPTATION.md](docs/MILESONE_ADAPTATION.md) — the two concrete surfaces
- [docs/DECISIONS.md](docs/DECISIONS.md) — **cross-machine handoff**: what was decided and why, machine specs, model downloads, things discovered the hard way
- [TODO.md](TODO.md) — the code-level worklist (T1–T13), with "done when" for each
- [MODEL_CARD.md](MODEL_CARD.md) — scope limits and known failure modes
