"""Language-level standards, for any Python codebase.

  * broad_except  -- `except Exception`, `except BaseException` and a bare
    `except:` catch the failure they were written for AND every bug nearby. Catch
    the specific classes. To record a failure and let it propagate, use
    try/finally: set a flag as the body's last step, and in `finally` read
    `sys.exception()` only when the flag is unset. Unguarded, `sys.exception()`
    also returns an exception a CALLER is handling, so a success would be
    recorded as that caller's failure.
  * size_limits   -- SOLID's single-responsibility rule, measured the way pylint
    measures it (its default limits): arguments, locals, branches, statements
    and returns per function. The fix is a helper, not a higher limit.
  * match_capture -- `case STATUS_OK:` does not compare against STATUS_OK. A bare
    name in a case pattern CAPTURES whatever is matched, so that case always wins.
"""
from __future__ import annotations

import ast

from tools.prepush.change import FLAG, SKIP, Change, _skip_path, report, result
from tools.prepush.standards.engine import Finding, dotted, enabled, own_nodes, run_rules, scoped

BROAD = {"Exception", "BaseException"}

# pylint's defaults (max-args, max-locals, max-branches, max-statements,
# max-returns). A codebase may tighten them under `standards.limits`.
LIMITS = {"args": 5, "locals": 15, "branches": 12, "statements": 50, "returns": 6}


# -- broad except ----------------------------------------------------------------
def broad_excepts(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """Every handler that catches Exception, BaseException, or everything."""
    out = []
    for node, scope in scoped(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            caught = ["everything (bare except)"]
        else:
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            caught = [n for n in (dotted(t) for t in types) if n.split(".")[-1] in BROAD]
        for name in caught:
            out.append(Finding(
                f"{scope or '<module>'}:{name}", node.lineno,
                f"`except {name}` in {scope or 'module scope'}: catch the specific classes. "
                "To record a failure and re-raise, use try/finally: set a done-flag at "
                "the end of the body and read sys.exception() only when it is unset."))
    return out


def check_broad_except(change: Change, config: dict) -> list[dict]:
    """No NEW handler that catches Exception, BaseException or everything."""
    return run_rules(change, config, ("broad_except", 1, (broad_excepts,)),
                     ("broad excepts", "no new broad exception handlers"))


# -- size limits (SRP) -----------------------------------------------------------
def _arg_count(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Named parameters, not counting `self`/`cls` or *args/**kwargs."""
    names = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
    return len(names) - (1 if names[:1] in (["self"], ["cls"]) else 0)


def _branches(node: ast.AST) -> int:
    """How many branches one node opens, counted as pylint counts them."""
    if isinstance(node, (ast.If, ast.match_case)):
        return 1
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return 1 + bool(node.orelse)
    if isinstance(node, (ast.Try, ast.TryStar)):
        return len(node.handlers) + bool(node.orelse) + bool(node.finalbody)
    return 0


def measure(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    """The five size metrics of one function, its nested defs excluded."""
    nodes = list(own_nodes(fn))
    params = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    stored = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    nested = {n.name for n in nodes
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return {"args": _arg_count(fn), "locals": len(params | stored | nested),
            "branches": sum(_branches(n) for n in nodes),
            "statements": sum(isinstance(n, ast.stmt) for n in nodes),
            "returns": sum(isinstance(n, ast.Return) for n in nodes)}


def oversized(tree: ast.Module, limits: dict[str, int]) -> dict[str, tuple[int, int]]:
    """`function:metric` -> (value, line) for every metric over its limit."""
    out = {}
    for node, scope in scoped(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = f"{scope}.{node.name}".lstrip(".")
        for metric, value in measure(node).items():
            if value > limits[metric]:
                out[f"{name}:{metric}"] = (value, node.lineno)
    return out


def check_size_limits(change: Change, config: dict) -> list[dict]:
    """A function the change pushed over a size limit, or further over one.

    Not the generic new-key comparison: a function already over a limit that the
    change made BIGGER is flagged too, because growing debt is still new debt.
    """
    if not enabled(config, "size_limits"):
        return [result("size_limits", 1, SKIP, "disabled in this codebase's standards config")]
    limits = {**LIMITS, **((config.get("standards") or {}).get("limits") or {})}
    flags = []
    for path in change.py_files():
        new_tree = change.tree_for(path)
        if _skip_path(path, config) or new_tree is None:
            continue
        old = oversized(change.tree_for(path, "base") or ast.Module(body=[], type_ignores=[]),
                        limits)
        for key, (value, line) in oversized(new_tree, limits).items():
            if value > old.get(key, (0, 0))[0]:
                fn, metric = key.rsplit(":", 1)
                flags.append(result(
                    "size_limits", 1, FLAG,
                    f"{fn} has {value} {metric} (limit {limits[metric]}): split it, moving "
                    "logic into a helper, rather than raising the limit", path, line))
    return report("size_limits", 1, flags, "oversized functions",
                  "no function pushed over a size limit")


# -- match capture ---------------------------------------------------------------
def _module_constants(tree: ast.Module) -> set[str]:
    """Names assigned at module level: what a bare case name usually MEANT to compare."""
    return {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            if isinstance(t, ast.Name)}


def match_captures(tree: ast.Module, _path: str, _cfg: dict) -> list[Finding]:
    """`case NAME:` where NAME is a constant: a capture that reads like a comparison."""
    constants = _module_constants(tree)
    out = []
    for node, scope in scoped(tree):
        pattern = getattr(node, "pattern", None) if isinstance(node, ast.match_case) else None
        if isinstance(pattern, ast.MatchAs) and pattern.pattern is None and pattern.name and \
                (pattern.name.isupper() or pattern.name in constants):
            out.append(Finding(
                f"{scope or '<module>'}:{pattern.name}", pattern.lineno,
                f"`case {pattern.name}:` captures instead of comparing. Use a dotted "
                f"name (Status.{pattern.name}) or `case x if x == {pattern.name}`."))
    return out


def check_match_capture(change: Change, config: dict) -> list[dict]:
    """No NEW case pattern that captures a constant's name instead of comparing."""
    return run_rules(change, config, ("match_capture", 1, (match_captures,)),
                     ("capturing case patterns", "no case pattern captures a constant's name"))


# The rules the whole-repository gate runs (size has its own comparison above).
PYTHON_RULES = (broad_excepts, match_captures)
