"""Run locally exactly what CI used to run: the same steps, reported in the same order.

    python check.py            # gate suite first (stop if it fails), then the rest in parallel
    python check.py --all      # carry on even past a failing gate suite, then summarise
    python check.py --serial   # after the gates, one step at a time, stopping at a failure
    python check.py --list     # show the steps without running them

CI is commented out (see .github/workflows/ci.yml) because this project is being
run locally only. That makes this file the whole safety net, so it is written to
be the thing you actually type rather than four commands you have to remember.

Three properties inherited from the CI it replaces, each for a reason:

**The safety gates run FIRST, and need no model and no server.** A broken test
or lint step must never be able to mask a weakened gate, so the gates go before
anything else. They need `requirements.txt` -- `eval/harness.py` imports
`config`, which imports every adapter, and adapters import their libraries at
module level (the project rule: top-level imports unless genuinely circular).
An earlier claim that the suite was "dependency-free" was already untrue -- it
loaded pydantic -- and was corrected on 2026-11-16.

**pytest runs TWICE, the second time with LANGGRAPH_STRICT_MSGPACK=true.** That
is the check that caught T15: an unregistered state type does not raise, it comes
back from a checkpoint as a plain `dict`, so a paused run silently loses the
model it was written with. Only the strict run makes that visible.

**`doctor` never fails the check.** It reports FAIL whenever LM Studio is not
running, which is the normal state on a laptop. Running it anyway is still worth
it -- it prints the platform, profile and memory arithmetic, which is the context
you want when something else in this list fails.

**After the gates, the rest run IN PARALLEL** (2026-11-16): the two pytest
passes were ~13 s each back to back, ~28 s of a ~30 s check. The gates still run
alone and first, and a gate failure still stops everything; each pytest pass
gets its own `--basetemp`, so the two never share a temp directory. `--serial`
restores one-at-a-time, for when interleaved timing is what you are debugging.

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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# (label, argv, env_overrides, fails_the_check)
#
# `argv` uses sys.executable rather than "python", so the check runs under the
# interpreter that invoked it instead of whatever is first on PATH -- the conda
# env here is easy to shadow.
# Forced on the steps that must be model-free, whatever the calling shell exports.
# The gate suite and the demo only `setdefault` these; a shell configured to RUN
# the agent (LLM_PROVIDER=lmstudio) would otherwise turn "model-free" into "calls
# whatever is loaded" -- found in doctor on 2026-11-16, same hole here.
# CRA_PROFILE is pinned WITH the platform: a shell exporting CRA_PROFILE=mac_max
# against the pinned `cpu` platform is an unsupported pair, and the platform
# layer rightly raises rather than silently downgrading.
OFFLINE = {"LLM_PROVIDER": "stub", "RETRIEVER": "hybrid", "RERANKER": "none",
           "CRA_ACTIVITY": "null",
           "CRA_PLATFORM": "cpu", "CRA_PROFILE": "portable"}

STEPS: tuple[tuple[str, list[str], dict[str, str], bool], ...] = (
    ("safety gates (no model, 100% bar)",
     [sys.executable, "-m", "eval.harness"], OFFLINE, True),
    ("lint (ruff)",
     [sys.executable, "-m", "ruff", "check", "."], {}, True),
    # The reviewer's own standards over this whole repo (tools/standards_audit.py):
    # no suppressions, no broad except, function size limits, the API contract.
    # Blocking, so the reviewer can never enforce a rule it does not meet itself.
    ("coding standards (the reviewer's own rules)",
     [sys.executable, "-m", "tools.standards_audit", "--summary"], {}, True),
    # `-p no:cacheprovider`: the two passes run concurrently and would otherwise
    # both write .pytest_cache. Each gets its own --basetemp in `_argv`.
    ("unit tests",
     [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], {}, True),
    ("unit tests (strict checkpoint serialiser)",
     [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
     {"LANGGRAPH_STRICT_MSGPACK": "true"}, True),
    ("multi-tenant demo",
     [sys.executable, "-m", "eval.demo_multitenant"], OFFLINE, True),
    ("doctor (report only)",
     [sys.executable, "-m", "app.doctor"],
     {"CRA_PLATFORM": "cpu", "CRA_PROFILE": "portable"}, False),
)

OK, FAIL, SKIP = "OK  ", "FAIL", "NOTE"


def _argv(argv: list[str]) -> list[str]:
    """A pytest step gets a private --basetemp, so concurrent passes never share
    (and never delete) each other's temp directories."""
    if argv[1:3] == ["-m", "pytest"]:
        return [*argv, "--basetemp", tempfile.mkdtemp(prefix="cra-check-")]
    return argv


def run_step(argv, env_overrides, quiet):
    """Run one step as a subprocess. Returns (exit code, seconds, output).

    A subprocess rather than an in-process call, so each step sees exactly the
    environment its overrides describe -- `LANGGRAPH_STRICT_MSGPACK` is read when
    langgraph is imported, so setting it inside an already-running interpreter
    would test nothing.

    Output is RETURNED, not printed, so parallel steps cannot interleave theirs;
    the caller prints it only on failure unless `quiet` is off -- a passing
    step's output is noise, a failing one's is the only thing worth reading.
    """
    env = {**os.environ, **env_overrides}
    started = time.monotonic()
    # check=False explicitly: a failing step is a RESULT to report, not an error
    # to raise -- `--all` has to carry on past it and summarise.
    proc = subprocess.run(_argv(argv), env=env, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, elapsed, out if (proc.returncode != 0 or not quiet) else ""


def _status(code: int, blocking: bool) -> str:
    if code == 0:
        return OK
    # doctor's non-zero exit is the expected state without LM Studio.
    return FAIL if blocking else SKIP


def _list() -> int:
    """Print each step's label, env overrides and command, for `--list`; returns 0.

    The interpreter path is dropped from the command so the line reads as
    something you could type, and non-blocking steps (doctor) are marked as such.
    """
    for label, argv, env_overrides, blocking in STEPS:
        flag = "" if blocking else "   (never fails the check)"
        prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        print(f"  {label}{flag}\n      {prefix} {' '.join(argv[1:])}".rstrip())
    return 0


def _report(results: list, step, outcome) -> str:
    """Print one finished step (its output only if it should be seen) and record it."""
    label, blocking = step[0], step[3]
    code, elapsed, out = outcome
    status = _status(code, blocking)
    print(f"\n=== {label} ===")
    if out.strip():
        print(out.rstrip())
    print(f"[{status}] {label}  ({elapsed:.1f}s)")
    results.append((status, label, elapsed))
    return status


def _run(args) -> list:
    """The gate suite FIRST and ALONE -- nothing may mask a weakened gate -- then
    the rest, in parallel unless --serial. Returns [(status, label, seconds)]."""
    results: list = []
    quiet = not args.verbose
    gate, rest = STEPS[0], STEPS[1:]
    if _report(results, gate, run_step(gate[1], gate[2], quiet)) == FAIL and not args.all:
        print("\nstopping: the safety gates failed. Re-run with --all to see the rest.")
        return results
    if args.serial:
        for step in rest:
            if _report(results, step, run_step(step[1], step[2], quiet)) == FAIL \
                    and not args.all:
                print("\nstopping at the first failure. Re-run with --all to see the rest.")
                break
        return results
    with ThreadPoolExecutor(max_workers=len(rest)) as pool:
        futures = [pool.submit(run_step, s[1], s[2], quiet) for s in rest]
        for step, fut in zip(rest, futures, strict=True):    # reported in STEPS order
            _report(results, step, fut.result())
    return results


def _summary(results: list, wall: float) -> int:
    """Print the per-step table and the verdict; return the process exit code.

    Only a blocking FAIL makes it 1. A NOTE (doctor without LM Studio) is printed
    with the reason it is expected, so the laptop's normal state does not read as
    a broken check.
    """
    print(f"\n(wall clock {wall:.1f}s)")
    print("-" * 68)
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


def main() -> int:
    """Run STEPS; exit 1 if any blocking step fails, else 0.

    Stops after a failing gate suite unless --all, because every later step runs
    against a tree already known to be unsafe. Non-blocking steps (doctor) are
    reported as NOTE and never change the exit code -- see STEPS for why.
    """
    ap = argparse.ArgumentParser(description="Run the local equivalent of CI.")
    ap.add_argument("--all", action="store_true",
                    help="run every step even after one fails, then summarise")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show output from passing steps too")
    ap.add_argument("--serial", action="store_true",
                    help="run the steps after the gate suite one at a time")
    args = ap.parse_args()
    if args.list:
        return _list()
    started = time.monotonic()
    results = _run(args)
    return _summary(results, time.monotonic() - started)


if __name__ == "__main__":
    sys.exit(main())
