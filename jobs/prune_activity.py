"""Retention for the activity log: delete day files older than the window.

    python -m jobs.prune_activity --keep-days 90          # dry run (default)
    python -m jobs.prune_activity --keep-days 90 --apply

`app/activity.py` writes one file per day, `var/activity/YYYY-MM-DD.jsonl`, and
until this job nothing ever removed one. It is the operational log -- spans,
retries, errors with tracebacks -- and the AUDIT store (one immutable record per
run, G6) is a separate thing this job never touches. Losing an old activity day
loses timing and debugging detail; it never loses what a run decided.

Dry run by default, like `jobs.prune_checkpoints`: deleting is the one thing
here that cannot be undone, so it has to be asked for. The cut is by the DATE IN
THE FILE NAME, not mtime -- a file copied or touched later must not look new.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta

from app.activity import ActivityLogger, capture_library_logs, log_dir

DAY_FILE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.jsonl$")
DEFAULT_KEEP_DAYS = 90


def expired(keep_days: int, today: date | None = None) -> list:
    """Day files strictly older than `keep_days` before today, oldest first.
    Anything not named like a day file is left alone."""
    if keep_days < 1:
        raise ValueError("--keep-days must be at least 1: today's file is being written")
    cutoff = (today or date.today()) - timedelta(days=keep_days)
    out = []
    for path in sorted(log_dir().glob("*.jsonl")):
        m = DAY_FILE.match(path.name)
        if m and date(int(m[1]), int(m[2]), int(m[3])) < cutoff:
            out.append(path)
    return out


def prune(keep_days: int, apply: bool, today: date | None = None) -> list:
    """Delete (or, without `apply`, list) the expired day files. Returns them."""
    files = expired(keep_days, today)
    with ActivityLogger("job", "prune_activity",
                        metadata={"keep_days": keep_days, "apply": apply}) as alog:
        if apply:
            for path in files:
                path.unlink()
        alog.add_metadata("files", len(files))
    return files


def main() -> int:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="Delete activity-log days past retention.")
    ap.add_argument("--keep-days", type=int, default=DEFAULT_KEEP_DAYS)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()
    capture_library_logs()
    files = prune(args.keep_days, args.apply)
    verb = "deleted" if args.apply else "would delete"
    for path in files:
        print(f"  {verb}: {path.name}")
    print(f"{verb} {len(files)} day file(s) older than {args.keep_days} days"
          + ("" if args.apply else ". Re-run with --apply to delete."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
