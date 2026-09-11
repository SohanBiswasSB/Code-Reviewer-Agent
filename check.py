"""Run locally exactly what CI used to run, in the same order.

    python check.py            # stop at the first failing step
    python check.py --all      # run every step, then summarise
    python check.py --list     # show the steps without running them

CI is commented out (see .github/workflows/ci.yml) because this project is being
run locally only. That makes this file the whole safety net, so it is written to
be the thing you actually type rather than four commands you have to remember.

Three properties inherited from the CI it replaces, each for a reason:

**The safety gates run FIRST and with no dependencies installed.** A broken
requirements file must never be able to mask a weakened gate -- so the gate suite
runs before `pip install` is even a consideration, which it can, because
`eval/harness.py` is deliberately dependency-free and model-free.

**pytest runs TWICE, the second time with LANGGRAPH_STRICT_MSGPACK=true.** That
is the check that caught T15: an unregistered state type does not raise, it comes
back from a checkpoint as a plain `dict`, so a paused run silently loses the
model it was written with. Only the strict run makes that visible.

**`doctor` never fails the check.** It reports FAIL whenever LM Studio is not
running, which is the normal state on a laptop. Running it anyway is still worth
it -- it prints the platform, profile and memory arithmetic, which is the context
you want when something else in this list fails.

Python rather than a shell script on purpose: this project is developed on a Mac
AND a Windows box (docs/DECISIONS.md section 1), and `check.sh` would only ever
work on one of them. Output is deliberately ASCII -- a tick character in
`eval/harness.py` once crashed a passing run under Windows cp1252, which is
exactly the class of thing a cross-machine helper must not reintroduce.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# (label, argv, env_overrides, fails_the_check)
#
# `argv` uses sys.executable rather than "python", so the check runs under the
# interpreter that invoked it instead of whatever is first on PATH -- the conda
# env here is easy to shadow.
STEPS: tuple[tuple[str, list[str], dict[str, str], bool], ...] = (
    ("safety gates (no deps, 100% bar)",
     [sys.executable, "-m", "eval.harness"], {"CRA_PLATFORM": "cpu"}, True),
    ("lint (ruff)",
     [sys.executable, "-m", "ruff", "check", "."], {}, True),
    ("unit tests",
     [sys.executable, "-m", "pytest", "-q"], {}, True),
    ("unit tests (strict checkpoint serialiser)",
     [sys.executable, "-m", "pytest", "-q"],
     {"LANGGRAPH_STRICT_MSGPACK": "true"}, True),
    ("multi-tenant demo",
     [sys.executable, "-m", "eval.demo_multitenant"], {"CRA_PLATFORM": "cpu"}, True),
    ("doctor (report only)",
     [sys.executable, "-m", "app.doctor"], {"CRA_PLATFORM": "cpu"}, False),
)

OK, FAIL, SKIP = "OK  ", "FAIL", "NOTE"


def run_step(label, argv, env_overrides, quiet):
    env = {**os.environ, **env_overrides}
    started = time.monotonic()
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if proc.returncode != 0 or not quiet:
        tail = (proc.stdout or "") + (proc.stderr or "")
        if tail.strip():
            print(tail.rstrip())
    return proc.returncode, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the local equivalent of CI.")
    ap.add_argument("--all", action="store_true",
                    help="run every step even after one fails, then summarise")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show output from passing steps too")
    args = ap.parse_args()

    if args.list:
        for label, argv, env_overrides, blocking in STEPS:
            flag = "" if blocking else "   (never fails the check)"
            prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
            print(f"  {label}{flag}\n      {prefix} {' '.join(argv[1:])}".rstrip())
        return 0

    results = []
    for label, argv, env_overrides, blocking in STEPS:
        print(f"\n=== {label} ===")
        code, elapsed = run_step(label, argv, env_overrides, quiet=not args.verbose)
        if code == 0:
            status = OK
        elif blocking:
            status = FAIL
        else:
            # doctor's non-zero exit is the expected state without LM Studio.
            status = SKIP
        results.append((status, label, elapsed))
        print(f"[{status}] {label}  ({elapsed:.1f}s)")
        if status == FAIL and not args.all:
            print("\nstopping at the first failure. Re-run with --all to see the rest.")
            break

    print("\n" + "-" * 68)
    for status, label, elapsed in results:
        print(f"[{status}] {label:46} {elapsed:5.1f}s")
    failed = [label for status, label, _ in results if status == FAIL]
    notes = [label for status, label, _ in results if status == SKIP]
    print("-" * 68)
    if notes:
        print(f"NOTE: {len(notes)} non-blocking step(s) reported problems: "
              f"{', '.join(notes)}")
        print("      `doctor` fails whenever LM Studio is not running. Expected "
              "on a laptop.")
    if failed:
        print(f"\nCHECK FAILED: {len(failed)} step(s) -> {', '.join(failed)}")
        return 1
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
