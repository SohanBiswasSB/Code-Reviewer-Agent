# TENANT — MilesOne daily log review

Log sources for this project: the backend service and the Celery worker streams, exported
from CloudWatch to S3 as structured `activity_log` JSON lines.

## Known failure patterns, in order of how often they have actually bitten

1. **"Signal received but no downstream action in N hours."** A Zoom webhook arrives and
   the Celery ingest/fetch is never dispatched. This is the signature silent failure in
   this service: nothing errors, so nothing alerts. Treat any gap between a received
   signal and its expected downstream task as a finding even when no error accompanies it.
2. **A Celery task registered at boot but never received or run** — the worker is up and
   subscribed, and the queue is simply not reaching it.
3. **Credly 422 bursts and circuit-breaker opens**, plus allocations stuck in Pending
   behind them.
4. **Replica lag warnings** and **DB connection-pool exhaustion** — usually together, and
   usually downstream of a slow query rather than of load.
5. **5xx bursts**, tracebacks, and `is_error` clusters in the structured log.
6. **Celery retry storms** — the same task failing repeatedly on the same input.

## Notes

- The two streams must be read together. A backend-only or worker-only view hides the
  handoff, and the handoff is where pattern 1 lives.
- The agent has read-only access to the log bucket and the database by design. No finding
  may suggest otherwise.
