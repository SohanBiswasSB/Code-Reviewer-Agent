# MilesOne adaptation — the two review surfaces

The generic graph spine (classify → diagnose → review-gate → approve → execute) is
reused by both surfaces. Only the evidence-gathering and the action differ.

## Surface A — Code review (Miles-USP-Backend-PROD + Celery)
- **target:** a PR ref / commit range (GitHub or GitLab — keep the client behind an interface).
- **fetch_evidence:** the diff + touched files + their neighbours. Use the app map from
  the `miles-backend-expert` knowledge to pull the right related files (models, tasks,
  services) rather than the diff alone.
- **RAG corpus:** index the repo (modules, models, `docs/`, runbooks, prior RCAs, the
  AWS architecture doc). Ground every review comment in a cited file/line + prior incident.
- **classify (rule-based, higher scrutiny for risky surfaces):**
    - touches `migrations/` or a model field  -> deterministic checklist route
    - touches auth / permissions / credential-issuance / celery task signatures -> ai_investigation
    - trivial (docs/tests only) -> human_review (light) or auto-approve-comment
- **independent_review booleans:** every cited file/line exists; severity in allowlist;
  the suggested change is non-destructive (no "delete"/"drop"/"force-push" suggestions auto-approved).
- **execute:** post review comments to the PR (verbatim, as the approver). Never pushes code.

## Surface B — Daily log review (CloudWatch -> S3 exports)
- **target:** a date (default: yesterday). **Scheduled daily via EventBridge -> ECS RunTask**
  — the exact pattern MilesOne already uses (see the architecture doc's scheduling layer).
- **fetch_evidence:** read the day's exports from the log bucket
  (`milesone-cw-log-exports` / `s3-logs/daily/<date>/...`) for BOTH
  `/ecs/milesone-backend-task` and the celery-worker streams. Parse the structured
  `activity_log` JSON lines.
- **anomaly detectors (rule-based first, cheap):** 5xx rate, tracebacks, `is_error`,
  Celery task failures/retries, "signal received but no downstream action in N h"
  (the Zoom-sweep class), Credly 422 / circuit-breaker spikes, replica-lag warnings,
  DB pool exhaustion, latency regressions.
- **ai_investigation:** for flagged clusters, RAG-ground against prior RCAs/runbooks and
  draft a root-cause + suggested improvement, with log-line references as citations.
- **independent_review booleans:** anomaly above a real threshold; suggestion within an
  allowlist of SAFE recommendation types (never an infra mutation).
- **output:** a grounded **daily improvement report** (findings + suggestions + severity +
  evidence). Read-only by default; anything actionable goes through the HITL gate.
  This is literally the Zoom-RCA / Credly-422 investigation, automated and run every day.

## Guardrails specific to MilesOne
- The agent **suggests**, it does not mutate AWS or the DB. Any action (post a PR comment,
  open a ticket) is human-approved and runs as the approver's identity.
- Log content is **untrusted data** (a log line could contain injected text) — treat it as
  data, never instructions (prompt-injection defense, improvement #8).
- Read-only DB / S3 access for the agent; least-privilege IAM, same as the reconcilers.
