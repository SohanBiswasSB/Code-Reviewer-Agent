"""Layer 4: evict checkpoints for runs that are finished. Never for pending ones.

    python -m jobs.prune_checkpoints --keep-days 30          # dry run (default)
    python -m jobs.prune_checkpoints --keep-days 30 --apply

## The rule that matters

**A suspended thread is a pending human approval. It never expires.** Deleting
one silently discards a decision somebody was asked to make, and they would find
out by the run simply never resuming. So expiry applies only to threads whose
audit record says they reached a terminal outcome, and even then the live state
is re-checked before anything is deleted.

That double check is not redundancy for its own sake. The audit record says what
happened *at the time it was written*; a thread could have been resumed since. The
checkpointer is the authority on what is pending, so it gets the final say — the
same "check it where the work happens" reasoning as `execute`'s permission guard.

## Why the audit store is the index

LangGraph's checkpointer is keyed by `thread_id` and offers no "list every thread"
query, which is exactly the gap docs/DATA_STORAGE.md describes. The audit store
(Layer 1) already records one row per terminal run WITH its thread id, so it is
the natural index. Deletion still goes through the checkpointer's own
`delete_thread`, never raw SQL, so this works on any backend.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from adapters.audit_jsonl import iter_records
from app.activity import ActivityLogger, capture_library_logs, log_event
from app.deps import open_checkpointer
from app.graph import build_graph

# Outcomes that mean the run is over. A denial counts: the human answered. A
# crash or an interrupt (G9) counts too -- nothing will resume that thread -- and
# "abandoned" is a CLI run whose in-memory checkpoint died with its process. The
# live-state re-check below still refuses to evict anything paused at the gate.
TERMINAL_OUTCOMES = frozenset({"executed", "denied", "escalated", "human_review", "ended",
                               "failed", "interrupted", "abandoned"})


def _parse_ts(value: str | None) -> datetime | None:
    """An ISO timestamp as an aware datetime (naive read as UTC); None if unparseable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def expired_threads(records, keep_days: int, now: datetime | None = None) -> list[dict]:
    """Threads whose audit record is terminal AND older than the retention window.

    Pure, so the policy can be tested without a database. A record with no
    `thread_id` or an unparseable timestamp is SKIPPED — "I cannot tell how old
    this is" must never authorise a deletion.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=keep_days)
    newest: dict[str, dict] = {}
    for record in records:
        thread_id = record.get("thread_id")
        recorded_at = _parse_ts(record.get("recorded_at"))
        if not thread_id or recorded_at is None:
            continue
        # A thread can hold several records (a denial, then a later attempt).
        # The NEWEST one decides: an old terminal record must not expire a thread
        # that has been active since.
        seen = newest.get(thread_id)
        if seen is None or recorded_at > seen["_at"]:
            newest[thread_id] = {"_at": recorded_at, **record}

    return [
        {"thread_id": tid, "outcome": r.get("outcome"), "tenant": r.get("tenant"),
         "recorded_at": r["_at"].isoformat()}
        for tid, r in sorted(newest.items())
        if r.get("outcome") in TERMINAL_OUTCOMES and r["_at"] < cutoff
    ]


async def prune(keep_days: int, apply: bool) -> dict:
    """Evict (or, without `apply`, list) expired threads, re-checking each one live.

    Candidates come from the audit store; the checkpointer then has the final
    say. A thread still parked at the gate is KEPT and reported, a thread already
    gone is reported as absent, and every deletion -- real or dry-run -- is logged
    as an activity event. Returns the counts and rows `main` prints.
    """

    candidates = expired_threads(iter_records(), keep_days)
    deleted, kept_pending, missing = [], [], []

    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        for row in candidates:
            cfg = {"configurable": {"thread_id": row["thread_id"]}}
            snapshot = await graph.aget_state(cfg)
            if snapshot.next:
                # Still parked at the approval gate despite a terminal record.
                kept_pending.append(row)
                continue
            if snapshot.created_at is None:
                missing.append(row)       # already gone; nothing to do
                continue
            if apply:
                await saver.adelete_thread(row["thread_id"])
            deleted.append(row)
            # A deletion is irreversible: each one is on record, with the run's
            # outcome, so "where did thread X go" always has an answer.
            log_event("job", "prune.deleted" if apply else "prune.would_delete",
                      tenant=row.get("tenant"), correlation_id=row["thread_id"],
                      metadata={"thread_id": row["thread_id"], "outcome": row.get("outcome"),
                                "keep_days": keep_days})

    return {"keep_days": keep_days, "applied": apply,
            "considered": len(candidates), "deleted": deleted,
            "kept_because_pending": kept_pending, "already_absent": missing}


def main() -> int:
    """CLI entry point; see the module docstring. Always returns 0 -- KEPT rows are not errors."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep-days", type=int, default=30)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it this is a dry run")
    args = ap.parse_args()

    capture_library_logs()
    with ActivityLogger("job", "prune_checkpoints",
                        metadata={"keep_days": args.keep_days, "apply": args.apply}) as alog:
        report = asyncio.run(prune(args.keep_days, args.apply))
        alog.add_metadata("deleted", len(report["deleted"]))
        alog.add_metadata("kept_pending", len(report["kept_because_pending"]))
    verb = "DELETED" if report["applied"] else "would delete"
    print(f"{verb} {len(report['deleted'])} thread(s) "
          f"older than {report['keep_days']}d "
          f"({report['considered']} considered)")
    for row in report["kept_because_pending"]:
        print(f"  KEPT (still awaiting approval): {row['thread_id']} [{row['tenant']}]")
    for row in report["deleted"]:
        print(f"  {verb}: {row['thread_id']} [{row['tenant']}] {row['outcome']}")
    if not report["applied"] and report["deleted"]:
        print("\ndry run — pass --apply to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
