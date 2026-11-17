"""The shared engine behind the coding-standards checks.

A RULE reads one parsed file and returns every violation in it. A CHECK runs
rules over the change's Python files and flags only what the change ADDED: a rule
runs on the new file and on its base, and a violation already in the base is not
the change's doing. That is the same contract `check_suppressions` keeps, and it
is what makes a check fair to someone who touches a file full of old debt.

A violation's `key` names WHAT and WHERE without the line number ("view.get:
Exception"), so moving code up or down a file is not mistaken for adding it.
Duplicates are counted: a second `except Exception` in the same function is new
even though the first one was already there.

The same rules also run over this repository itself, with an empty base
(`all_findings`), which is how the reviewer is held to what it enforces.
"""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from tools.prepush.change import FLAG, SKIP, Change, _skip_path, report, result


@dataclass(frozen=True)
class Finding:
    """One violation: a line-free `key`, where it is, and what to do about it."""

    key: str
    line: int
    message: str


# A rule: (parsed module, file path, the codebase's `standards` config) -> findings.
Rule = Callable[[ast.Module, str, dict], list[Finding]]


@dataclass(frozen=True)
class Concern:
    """One reported check inside a framework pack: its rules and its wording."""

    name: str
    rules: tuple[Rule, ...]
    what: str          # what to call the flags when there are too many to list
    ok: str            # the PASS line


@dataclass(frozen=True)
class Pack:
    """A framework's rules, kept apart so the core stays framework-neutral.

    `markers` are the top-level import names that mean "this code uses the
    framework". A pack's checks report as `<id>:<concern>` (e.g. django:models),
    so the framework appears only as a pack id, never in the core's names.
    """

    id: str
    title: str
    markers: tuple[str, ...]
    concerns: tuple[Concern, ...]


def standards_config(config: dict) -> dict:
    """The tenant's `prepush.standards` block ({} when it sets none)."""
    return config.get("standards") or {}


def enabled(config: dict, check: str) -> bool:
    """False when the codebase listed `check` under `standards.disable`."""
    return check not in set(standards_config(config).get("disable") or [])


def added_findings(new: list[Finding], old: list[Finding]) -> list[Finding]:
    """The findings in `new` that `old` does not account for, counted per key."""
    budget = Counter(f.key for f in old)
    out = []
    for f in new:
        if budget[f.key]:
            budget[f.key] -= 1
        else:
            out.append(f)
    return out


def scoped(tree: ast.AST) -> Iterator[tuple[ast.AST, str]]:
    """Every node with the dotted name of the def/class it sits in ('' at module level)."""
    def visit(node: ast.AST, scope: str) -> Iterator[tuple[ast.AST, str]]:
        """Depth-first over `node`'s children, extending `scope` at each def/class."""
        for child in ast.iter_child_nodes(node):
            yield child, scope
            named = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            yield from visit(child, f"{scope}.{child.name}".lstrip(".") if named else scope)
    yield from visit(tree, "")


def own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """A def's own nodes: a nested def, class or lambda is yielded but not entered,
    because its body is its own unit and is measured on its own."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def dotted(node: ast.AST) -> str:
    """`a.b.c` for a Name/Attribute chain, the callee's name for a Call, else ''."""
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Attribute):
        head = dotted(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return node.id if isinstance(node, ast.Name) else ""


def keyword(call: ast.Call, name: str):
    """The constant value of keyword argument `name`, or None when absent or computed."""
    hit = next((k.value for k in call.keywords if k.arg == name), None)
    return hit.value if isinstance(hit, ast.Constant) else None


def all_findings(tree: ast.Module, path: str, rules: Iterable[Rule], cfg: dict) -> list[Finding]:
    """Every finding of `rules` in one file: the whole-repository view, no base."""
    return [f for rule in rules for f in rule(tree, path, cfg)]


def run_rules(change: Change, config: dict, spec: tuple[str, int, tuple[Rule, ...]],
              wording: tuple[str, str]) -> list[dict]:
    """Flag what each changed Python file ADDED against `spec`'s rules.

    `spec` is (check name, ritual step, rules); `wording` is (what to call the
    flags when there are too many, the PASS line). A codebase that disabled the
    check gets one SKIP saying so, never a silent nothing.
    """
    check, step, rules = spec
    if not enabled(config, check):
        return [result(check, step, SKIP, "disabled in this codebase's standards config")]
    cfg = standards_config(config)
    flags = []
    for path in change.py_files():
        new_tree = change.tree_for(path)
        if _skip_path(path, config) or new_tree is None:
            continue
        old_tree = change.tree_for(path, "base") or ast.Module(body=[], type_ignores=[])
        for f in added_findings(all_findings(new_tree, path, rules, cfg),
                                all_findings(old_tree, path, rules, cfg)):
            flags.append(result(check, step, FLAG, f.message, path, f.line))
    return report(check, step, flags, wording[0], wording[1])
