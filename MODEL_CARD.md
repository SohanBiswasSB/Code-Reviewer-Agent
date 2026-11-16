# Model card — Code Reviewer Agent

Scope limits, known biases, and what this system is explicitly not for. Written
in the same spirit as `SF-Crime-Prediction/docs/MODEL_CARD.md`: the limits are
part of the deliverable, not a disclaimer bolted on afterwards.

## What it does

Given a pull-request reference (or a date, for the log surface), it gathers
evidence, classifies it with **deterministic rules**, and on the ambiguous path
asks an LLM for a citation-grounded diagnosis. A **second, independent** LLM then
writes a critique. The approve/escalate decision is **four Python booleans over
real data** — never the LLM's own verdict. Anything actionable pauses for a human.

## Hard limits, enforced in code rather than promised in prose

| Limit | Where it is enforced |
|---|---|
| Never pushes, commits, merges or force-pushes. Comments and labels only. | `execute()` writes only through the surface's `ActionSink`, and the only sinks that exist post a comment/label or write a report file |
| Never runs without explicit human approval | `execute()` raises `PermissionError` unless `state["approved"] is True`, checked in the node body — not merely in graph routing |
| Posts exactly what the human approved | `proposed_action` is posted verbatim, never recomputed after approval |
| Never mutates infrastructure or a database | No adapter has a write path to either |
| Writes are attributed to the approver | `execute()` uses the approving visitor's own OAuth token; a shared token is a fallback only |
| No LLM ever emits the decision | `independent_review` uses the model's text as documentation; `ReviewResult.decision` is computed |

The `execute` guard and the four booleans are covered by 54 regression tasks at
a permanent 100% bar (`python -m eval.harness`). Six of those are the
**malformed-response** gates: a model reply that cannot be parsed must escalate,
never crash the node — the graph has no failure edges, so an exception there
kills the run and writes no audit record at all.

The four booleans are `groundedness_ok` (the cited id was really retrieved, and
its calibrated relevance clears the tenant's bar), `entailment_ok` (the claim
actually follows from the passage it cites — T22), `risk_ok` (severity is in the
tenant's allowlist) and `permission_ok` (the action type is in the tenant's
allowlist). `ReviewResult.decision` is `approve` only if all four hold.

## Known biases and failure modes

**Small local models escalate more.** A 4B judge and a 7B diagnosis model are
markedly worse at citation discipline than a frontier model. The gate catches
that — `groundedness_ok` fails and the run escalates — so the system stays
*safe*. But the escalation rate rises, and a gate that escalates everything
proves nothing. Escalation rate must be monitored as a quality signal, not
treated as success.

**The relevance thresholds are currently uncalibrated.** `min_relevance_score`
values of 0.35/0.40 in `tenants/*/policy.json` were calibrated for an embedder
this project no longer uses, and the embedder has changed **three times** since
(`text-embedding-3-small` truncated, then nomic, then bge-m3, now
EmbeddingGemma-300m). Each is a different vector space whose cosine scores do not
distribute the same way, so those numbers are not merely stale — they never
described the space the gate now measures in.

This is not hypothetical. The first live run scored its one real citation at
**0.2 against a 0.35 threshold**, and entailment at **0.24 against 0.25** — so the
verdict turned on two uncalibrated constants, one of them by a hundredth. **Until re-measured (TODO T10), they do not mean what they
say**, and `app.doctor` says so on every run.

**The judge shares a training distribution even when it shares no weights.**
Using a different model family (Gemma judging Qwen) mitigates self-preference
bias; it does not eliminate correlated blind spots between models trained on
overlapping public corpora. Independence here is structural, not absolute.

**`classify` is rule-based, therefore deliberately blunt.** It routes on touched
paths and CI status. It will mis-route a change whose risk is not visible in its
file paths. That is the accepted cost of a cheap, fully testable gate in front of
expensive work; the mis-routes land in `human_review`, which is the safe direction.

**Retrieval currently grounds against a four-document in-memory corpus.** Any
"grounded" claim from the system today is grounded against test fixtures. Not a
production property yet (TODO T10).

**Evidence is untrusted input, and only part of that is solvable.** Diffs and
log lines can carry injected instructions. The `InputGuard` (T12) now runs over
every fenced block before any model call: it replaces secrets and PII with stable
placeholders, and it escapes the fence markers so evidence cannot close its own
delimiter and continue as operator text.

**The guard protects the prompt AND the checkpoint** (T25, closed 2026-11-17).
Evidence is redacted in `fetch_evidence`, before anything is written to state, so
a secret in a diff reaches neither the provider, nor the audit record, nor any
checkpoint in the run's history. The approver therefore sees a placeholder where
the secret was. Redaction is still pattern-based, so a secret of an unknown shape
can get through; treat checkpoint storage as sensitive anyway.

What it does **not** do is neutralise natural-language injection, and it does not
claim to. Instruction-like phrasing is **counted and reported** into the audit
record, not rewritten — rewriting would alter the evidence a human is approving a
comment about and would still miss any paraphrase. So the residual risk is a
persuasive injected instruction that the diagnosis model obeys. The design's
answer to that is structural rather than textual: the model's output cannot
become an action on its own, because four Python booleans and a human approval
stand between them. Treat the redaction as a real boundary and the prose
detection as a **signal to monitor**, not a defence.

## Intended use

Assisting human reviewers on repositories the operator controls, with every
action reviewed before it happens. Suitable for surfacing candidate issues and
drafting grounded review comments.

## Out of scope

- **Not** an autonomous code-fixing agent. It proposes text; humans act.
- **Not** a gate that should block a merge on its own judgement.
- **Not** an evaluator of people. It reviews diffs, never contributors, and its
  output must not feed performance assessment.
- **Not** validated on any repository other than the tenants configured here.

## Provenance

Architecturally modelled on ResolveFlow (a LangGraph GitHub-issue-triage agent),
generalised to multiple codebases and surfaces. The safety design — reasoning and
execution as separate code paths, judge-explains/code-decides, a checkpointed
human gate, one guarded write path — is inherited deliberately and is not
tenant-overridable.
