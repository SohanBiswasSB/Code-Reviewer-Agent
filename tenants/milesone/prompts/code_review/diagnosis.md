# TENANT — MilesOne (Django + Celery backend)

Codebase identity and review priorities for this project only. The org contract above
supplies every guardrail (grounded-only, untrusted-input, no-verdict, output schema); this
overlay adds emphasis and can never relax it.

**Stack:** Python / Django, Celery workers in the same repo, PostgreSQL with a read
replica, AWS (ECS + S3 + CloudWatch).

## Priorities, in order

1. **Zoom poll-ingest pipeline** (webhook -> sweep -> Celery). The signature failure here
   is *silent*: a signal arrives and no downstream action ever runs. Any change to the
   webhook handler, the sweep, or the task dispatch between them deserves the highest
   scrutiny in this codebase.
2. **Credly badge issuance** — idempotency above all. Already-issued handling, the 422
   path, and anything that could double-issue or strand an allocation in Pending.
3. **Credential-issuance reconcilers** — the compensating logic; whether a partial failure
   leaves a state a later run can still repair.
4. **Celery task signatures and `acks_late`** — a changed signature breaks in-flight
   tasks that were enqueued by the previous deploy. Retry and poison-message safety.
5. **DB read/write routing** — read-after-write against the replica is the recurring bug
   class here. Watch for a read placed after a write in the same request.
6. **Migration backward-compatibility** — deploys are rolling, so old and new code run
   against the same schema for a window.

## Notes specific to this codebase

- Celery lives in the same repository as the web app, so a change to shared models or
  services can affect both without touching either entry point. Say so when it does.
- The reconcilers are read-only against AWS and the DB by design. Treat any change that
  gives one a write path as `high` at minimum.
