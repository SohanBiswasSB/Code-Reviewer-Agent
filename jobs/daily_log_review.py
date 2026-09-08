"""Scheduled entrypoint: run the log_review surface for a date. EventBridge -> ECS RunTask.

    python -m jobs.daily_log_review --date 2026-09-01
"""
from __future__ import annotations
import argparse, datetime as dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(dt.date.today() - dt.timedelta(days=1)))
    args = ap.parse_args()
    # import config; build graph for the 'log_review' surface; run over args.date; emit report.
    raise NotImplementedError(f"run log_review for {args.date} via config-resolved plugins")


if __name__ == "__main__":
    main()
