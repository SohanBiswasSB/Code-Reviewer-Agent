# TENANT — <Project name>

Codebase identity and review priorities for this project only. The org contract in
`tenants/_default/` supplies every guardrail — grounded-only, untrusted-input,
no-verdict, output schema. This overlay adds emphasis and can never relax it, so keep it
short: only what a reviewer could not work out from the diff itself.

**Stack:** <language / framework / datastore / anything unusual>

## Priorities, in order

1. <The subsystem where bugs here have actually hurt. Name the failure MODE, not just
   the module -- "a signal arrives and no downstream job runs" beats "the webhook".>
2. <Next.>
3. <Next.>

## Notes specific to this codebase

- <Anything non-obvious: a shared module that affects two entry points, a component that
  is read-only by design, a deploy pattern that makes migrations tricky.>
