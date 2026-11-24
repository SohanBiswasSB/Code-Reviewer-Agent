"""Hold this repository to the standards its reviewer enforces on others.

    python -m tools.standards_audit            # every violation, then a count
    python -m tools.standards_audit --summary  # counts per rule only

The pre-push checks flag what a CHANGE adds. This runs the same rules over the
whole tree with nothing to compare against, so every violation counts. check.py
runs it as a gate: a reviewer that let itself off would be enforcing a standard
it does not meet.

Scope: every tracked-or-new .py and .html file except var/ (runtime data) and
caches. Tests are included -- the user's rule was "all code".
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections import Counter

from tools.prepush.hygiene import pragma_counts
from tools.prepush.standards import api, general, web
from tools.prepush.standards.engine import Finding
from tools.prepush.standards.packs import PACKS

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {"var", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
# This codebase is a FastAPI service; its own config, like any tenant's.
CONFIG = {"frameworks": ["fastapi"], "unversioned_ok": ["/", "/healthz"]}


def files(root: pathlib.Path = ROOT, suffixes: tuple[str, ...] = (".py", ".html")):
    """The files the audit reads, in a stable order."""
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix in suffixes and not SKIP_DIRS & set(rel.parts):
            yield rel.as_posix(), p


def _python(path: str, text: str) -> list[tuple[str, Finding]]:
    """(rule, finding) for one Python file."""
    tree = ast.parse(text)
    rules = (*general.PYTHON_RULES, *api.API_RULES,
             *(r for pid in CONFIG["frameworks"] for c in PACKS[pid].concerns for r in c.rules))
    out = [(rule.__name__, f) for rule in rules for f in rule(tree, path, CONFIG)]
    out += [("size_limits", Finding(key, line, f"{key.split(':')[0]} has {value} "
                                    f"{key.rsplit(':', 1)[1]} (limit "
                                    f"{general.LIMITS[key.rsplit(':', 1)[1]]})"))
            for key, (value, line) in general.oversized(tree, general.LIMITS).items()]
    out += [("suppressions", Finding(f"pragma:{k}", 0, f"{n} `{k}` pragma(s)"))
            for k, n in pragma_counts(text).items()]
    return out


def audit(root: pathlib.Path = ROOT) -> list[tuple[str, str, Finding]]:
    """(path, rule, finding) for every violation in the tree."""
    out = []
    for path, p in files(root):
        text = p.read_text(encoding="utf-8")
        if path.endswith(".py"):
            out += [(path, rule, f) for rule, f in _python(path, text)]
        else:
            out += [(path, "css_empty_rules", Finding(sel, line, f"empty ruleset `{sel} {{}}`"))
                    for sel, line in web.empty_rules(path, text)]
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the violations (or their counts); exit 1 if there are any."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--summary", action="store_true", help="counts per rule only")
    args = ap.parse_args(argv)
    found = audit()
    if not args.summary:
        for path, rule, f in found:
            print(f"{path}:{f.line}: [{rule}] {f.message}")
    for rule, n in Counter(rule for _, rule, _ in found).most_common():
        print(f"{n:5d}  {rule}")
    print(f"{len(found)} violation(s)" if found else "no violations: the standards hold")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
