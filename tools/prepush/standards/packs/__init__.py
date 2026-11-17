"""Framework packs: the rules that only make sense for one framework.

The core standards (general.py, api.py, web.py) are framework-neutral. A pack
adds what one framework needs on top. Adding a framework is adding a module here
with a `PACK` and listing it in PACKS; nothing else changes.

## Which packs run

A codebase is never flagged by a pack it does not use:

  * declared: `prepush.standards.frameworks: ["django"]` in the tenant's policy
    runs exactly those packs;
  * otherwise detected: a pack runs when a changed Python file imports one of
    its markers (`django`, `rest_framework`; `fastapi`, `starlette`).
"""
from __future__ import annotations

import ast

from tools.prepush.change import Change
from tools.prepush.standards.engine import Pack, run_rules, standards_config
from tools.prepush.standards.packs import django, fastapi

PACKS: dict[str, Pack] = {p.id: p for p in (django.PACK, fastapi.PACK)}


def imported_roots(tree: ast.Module) -> set[str]:
    """The top-level package names a module imports."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".", 1)[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".", 1)[0])
    return roots


def active_packs(change: Change, config: dict) -> list[Pack]:
    """The packs that apply to this change: declared by the codebase, else detected."""
    declared = standards_config(config).get("frameworks")
    if declared is not None:
        return [PACKS[i] for i in declared if i in PACKS]
    roots: set[str] = set()
    for path in change.py_files():
        tree = change.tree_for(path)
        if tree is not None:
            roots |= imported_roots(tree)
    return [p for p in PACKS.values() if roots & set(p.markers)]


def check_framework_packs(change: Change, config: dict) -> list[dict]:
    """Every concern of every pack that applies; nothing for packs that do not."""
    out: list[dict] = []
    for pack in active_packs(change, config):
        for concern in pack.concerns:
            out += run_rules(change, config, (f"{pack.id}:{concern.name}", 1, concern.rules),
                             (concern.what, concern.ok))
    return out
