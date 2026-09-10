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

The `execute` guard and the four booleans are covered by 36 regression tasks at
a permanent 100% bar (`python -m eval.harness`).

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
values of 0.35/0.40 in `tenants/*/policy.json` were measured against
`text-embedding-3-small` truncated to 512 dimensions. The system now uses
**`bge-m3` at 1024 dimensions**, whose cosine scores do not distribute the same
way — and it has changed embedder twice since those numbers were set, so they are
doubly meaningless. **Until re-measured (TODO T10), they do not mean what they
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

**Evidence is untrusted input.** Diffs and log lines can carry injected
instructions. Prompt wording alone is not a boundary — and the prompts themselves
say so. The `InputGuard` redaction and delimiting stage is **not yet built**
(TODO T12), so today this invariant is enforced by convention, which is exactly
the weakness the architecture was designed to avoid. Treat it as open.

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
