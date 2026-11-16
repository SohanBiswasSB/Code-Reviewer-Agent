"""The change checks themselves: pure functions over a change and its repo.

See nodes/prepush_checks.py for what these are for and how results are used.

Modelled on the Masterclass `/prepush-ritual` (steps 0-5). Two tiers:

  · GENERIC -- always on, for any codebase: N+1 loops, new warning-silencers,
    line length + ASCII on added lines, missing handler docstrings, and an import
    of a file git does not track.
  · CONVENTIONS -- on only when the tenant's `code_review.prepush` config names
    them: per-app doc files, Postman collection + environments, tracker row,
    commit message, read-only and never-commit paths. A codebase without those
    conventions is not flagged for lacking them.

What the ritual checks that a reviewer CANNOT (manage.py check, pylint/pyright
regressions, the live query-count probe, Postman regeneration, artifact capture)
is reported as `skip` with the reason -- so "not checked" is never read as "fine".

Each check is `fn(change, config) -> list[dict]`. Stdlib only.
"""
from __future__ import annotations

import ast  # noqa: F401 -- re-exported: tests reach the parser through this module
import pathlib
import subprocess  # noqa: F401 -- re-exported: tests count git calls through it

from errors import PARSE_ERRORS
from tools.prepush.change import (  # noqa: F401 -- the public surface
    FLAG,
    MAX_REPORTED,
    PASS,
    SKIP,
    Change,
    added_lines,
    result,
    split_diff,
)
from tools.prepush.conventions import (  # noqa: F401
    _author,
    check_commit_message,
    check_docs,
    check_not_checkable,
    check_paths,
    check_postman,
    check_tracker,
)
from tools.prepush.hygiene import (  # noqa: F401
    check_handler_docstrings,
    check_line_hygiene,
    check_suppressions,
    check_untracked_imports,
)
from tools.prepush.n_plus_one import _LoopVisitor, _Module, check_n_plus_one  # noqa: F401

CHECKS = [check_n_plus_one, check_suppressions, check_line_hygiene, check_handler_docstrings,
          check_untracked_imports, check_paths, check_docs, check_postman, check_tracker,
          check_commit_message, check_not_checkable]


def run_all(evidence: dict, root: pathlib.Path, config: dict) -> list[dict]:
    """Every registered check, in order. A check that raises becomes a SKIP."""
    change = Change(root=pathlib.Path(root), files=list(evidence.get("files") or []),
                    revision=dict(evidence.get("revision") or {}))
    out: list[dict] = []
    for check in CHECKS:
        try:
            out.extend(check(change, config or {}))
        except PARSE_ERRORS as e:
            # The repo could not be read or parsed for this check: SKIP it, say
            # why, and run the rest. A bug in a check (TypeError, AttributeError)
            # is NOT in PARSE_ERRORS -- it propagates, so it gets fixed rather
            # than turning into a permanent "skip" nobody reads.
            out.append(result(check.__name__.removeprefix("check_"), -1, SKIP,
                              f"could not run: {type(e).__name__}: {e}"))
    return out
