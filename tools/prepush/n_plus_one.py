"""Step 0 -- N+1: a query, cache or HTTP call made once per row inside a loop.

A static AST analysis with its own vocabulary (ORM methods, loop shapes, local
helpers followed through the module). Informs the review; never decides it.
"""
from __future__ import annotations

import ast
import pathlib
import re

from tools.prepush.change import (
    FLAG,
    SKIP,
    Change,
    _skip_path,
    report,
    result,
)

QS_BUILDERS = {"objects", "_default_manager", "_base_manager", "filter", "exclude", "all",
               "select_related", "prefetch_related", "order_by", "values", "values_list",
               "annotate", "only", "defer", "using", "distinct", "aggregate"}
QS_TERMINAL = {"get", "first", "last", "exists", "count", "create", "update", "delete",
               "aggregate", "earliest", "latest", "contains", "iterator", "values",
               "values_list", "all", "filter", "exclude"}
SYNC_ORM_STRONG = {"save", "refresh_from_db", "get_or_create", "update_or_create",
                   "bulk_create", "bulk_update", "select_for_update", "in_bulk"}
ASYNC_ORM = {"aget", "afirst", "alast", "aexists", "acount", "acreate", "abulk_create",
             "abulk_update", "aget_or_create", "aupdate_or_create", "aupdate", "adelete",
             "asave", "aaggregate", "ain_bulk", "acontains", "aearliest", "alatest",
             "arefresh_from_db", "aiterator", "aexplain"}
_CACHE_BASE = {"get", "set", "delete", "add", "get_or_set", "incr", "decr", "has_key", "touch"}
CACHE_METHODS = _CACHE_BASE | {"a" + m for m in _CACHE_BASE}
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "request", "send", "head",
                "options", "stream"}
RE_CACHE = re.compile(r"(^|_)cache(s)?($|_)|redis", re.I)
RE_HTTP = re.compile(r"(client|session|httpx|requests|http)", re.I)
# Word-bounded: unanchored, "tries" matched inside "entries", so
# `for i in range(len(entries))` -- a per-row loop -- was read as a retry loop.
RE_RETRY = re.compile(r"(?<![a-z])(attempts?|retr\w*|tries|try|poll\w*|backoff)(?![a-z])"
                      r"|_max\b|\bmax_", re.I)
RE_CHUNK = re.compile(r"(chunk|batch|page|slice|window)", re.I)
# Containers a function builds for itself: `cache = {}` is a MEMO, usually the
# fix for an N+1, and `cache.get(k)` on it is not a cache-server round trip.
LOCAL_CONTAINERS = {"dict", "defaultdict", "OrderedDict", "Counter", "set", "list"}
RE_CONST = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
RE_EXEMPT = re.compile(r"#\s*O\(1\)-exempt:(.*)$")
SEVERITY = {"db": "HIGH", "thread-hop": "MEDIUM", "cache": "MEDIUM", "http": "MEDIUM"}
RANK = {"db": 4, "thread-hop": 3, "http": 2, "cache": 1}


def chain_names(node) -> list[str]:
    """Root name then each attribute, walking through calls and subscripts."""
    names: list[str] = []
    while True:
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Name):
            names.append(node.id)
            break
        else:
            break
    return list(reversed(names))


def _is_thread_hop(func: ast.expr) -> bool:
    """True for `sync_to_async(f)(...)`: the call hops to a worker thread per row."""
    return isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and \
        func.func.id in ("sync_to_async", "database_sync_to_async")


def _is_db_method(meth: str, names: list[str], recv: str) -> bool:
    """True if `<chain>.meth(...)` reaches the database by the ritual's ORM rules.

    An async ORM method always does; a strong sync one does unless it is a
    form's or serializer's `save` (no query of its own); a queryset terminal or
    builder does only when a queryset builder appears earlier in the chain.
    """
    builders = [n for n in names[1:] if n in QS_BUILDERS or n.endswith("_set")]
    return meth in ASYNC_ORM \
        or (meth in SYNC_ORM_STRONG and not (meth == "save" and recv in ("form", "serializer"))) \
        or bool(builders and meth in (QS_TERMINAL | QS_BUILDERS))


def _method_kind(func: ast.Attribute) -> str | None:
    """`cache` | `db` | `http` | None for a method call, checked in that order."""
    meth, names = func.attr, chain_names(func.value)
    recv = names[-1] if names else ""
    if meth in CACHE_METHODS and RE_CACHE.search(recv):
        return "cache"
    if _is_db_method(meth, names, recv):
        return "db"
    # `request.session`, `request.headers`...: Django's request object, an
    # in-memory store -- not an HTTP client, whatever the attribute is called.
    if meth in HTTP_METHODS and RE_HTTP.search(recv) and names[:1] != ["request"]:
        return "http"
    return None


def classify_call(call: ast.Call) -> str | None:
    """`db` | `thread-hop` | `cache` | `http` | None, by the ritual's rules, in order."""
    func = call.func
    if _is_thread_hop(func):
        return "thread-hop"
    if isinstance(func, ast.Attribute):
        return _method_kind(func)
    return None


def _target_names(node) -> str:
    return " ".join(n.id for n in ast.walk(node) if isinstance(n, ast.Name))


def _bounded(it) -> bool:
    """True if the iterable's size is fixed in the source, not by the data.

    A literal collection, a CONSTANT_NAME (or `x.CONSTANT`), `range` over literals,
    and `enumerate` / `.items()` / `.keys()` / `.values()` over any of those. A
    round trip inside such a loop runs a fixed number of times, so it is not N+1.
    """
    if isinstance(it, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return True
    if isinstance(it, (ast.Name, ast.Attribute)):
        return bool(RE_CONST.match(it.id if isinstance(it, ast.Name) else it.attr))
    if isinstance(it, ast.Call):
        return _bounded_call(it)
    return False


def _bounded_call(it: ast.Call) -> bool:
    """`_bounded` for a call: `range` over literals, or a wrapper over a bounded value.

    `enumerate(x)` and `x.items()` / `.keys()` / `.values()` are as bounded as
    `x`; any other call's size is decided by the data.
    """
    f = it.func
    if isinstance(f, ast.Name) and f.id == "range":
        return all(isinstance(a, ast.Constant) for a in it.args)
    if isinstance(f, ast.Name) and f.id == "enumerate":
        return bool(it.args) and _bounded(it.args[0])
    if isinstance(f, ast.Attribute) and f.attr in ("items", "keys", "values"):
        return _bounded(f.value)
    return False


def loop_shape(it, target) -> str:
    """`bounded` | `chunked` | `n` -- only `n` loops can be N+1."""
    if _bounded(it):
        return "bounded"
    words = f"{_target_names(target)} {ast.unparse(it)}"
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" \
            and RE_RETRY.search(words):
        return "bounded"
    # Chunked is decided by the LOOP VARIABLE: `for batch in batched(rows)` walks
    # chunks, but `for row in page_rows` walks rows -- matching the iterable's
    # name too exempted that per-row loop.
    if RE_CHUNK.search(_target_names(target)):
        return "chunked"
    return "n"


def _local_defs(tree: ast.Module) -> dict[str, ast.AST]:
    """Module functions by name, and methods as `Class.method`."""
    defs: dict[str, ast.AST] = {}
    fns = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, fns):
            defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            defs.update({f"{node.name}.{i.name}": i for i in node.body if isinstance(i, fns)})
    return defs


def _strongest(kinds) -> str | None:
    """The highest-ranked round-trip kind among `kinds` (None entries ignored)."""
    present = [k for k in kinds if k]
    return max(present, key=RANK.get) if present else None


class _Module:
    """Per-module index: which local functions do a round trip (and of what kind)."""

    def __init__(self, tree: ast.Module):
        """Classify every local def by its own calls, then by what it calls."""
        self.defs = _local_defs(tree)
        self.kind: dict[str, str | None] = {}
        callees: dict[str, set[str]] = {}
        for key, fn in self.defs.items():
            cls = key.split(".")[0] if "." in key else None
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
            self.kind[key] = _strongest(classify_call(c) for c in calls)
            callees[key] = {r for c in calls if (r := self.resolve(c, cls))}
        self._propagate(callees)

    def _propagate(self, callees: dict[str, set[str]]) -> None:
        """A def is as strong a round trip as the strongest local def it calls.

        Iterated to a FIXED POINT over the call edges, built once. The old loop
        re-walked every function's AST up to three times and stopped there, so a
        query four helpers deep was silently missed.
        """
        changed = True
        while changed:
            changed = False
            for key, targets in callees.items():
                best = _strongest([self.kind[key], *(self.kind.get(t) for t in targets)])
                if best != self.kind[key]:
                    self.kind[key], changed = best, True

    def resolve(self, call: ast.Call, cls: str | None) -> str | None:
        """The local def a call reaches: `name()` or `self.m()` / `cls.m()`."""
        f = call.func
        if isinstance(f, ast.Name) and f.id in self.defs:
            return f.id
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and \
                f.value.id in ("self", "cls") and cls and f"{cls}.{f.attr}" in self.defs:
            return f"{cls}.{f.attr}"
        return None


class _LoopVisitor(ast.NodeVisitor):
    """Finds round trips inside `n`-shaped loops, following module-local helpers."""

    def __init__(self, index: _Module):
        """Empty loop stack; `found` collects (line, loop line, kind, how) per hit."""
        self.index, self.loops, self.cls = index, [], None
        self.local_containers: set[str] = set()
        self.found: list[tuple[int, int, str, str]] = []   # line, loop line, kind, how

    def _in_loop(self, node, it, target, inside):
        """Visit `inside` with this loop pushed, shaped by its iterable and target.

        The iterable is visited BEFORE the push: it is evaluated once, not per row.
        """
        self.visit(it)
        self.loops.append((node.lineno, loop_shape(it, target)))
        for n in inside:
            self.visit(n)
        self.loops.pop()

    def _visit_for(self, node):
        """A `for` (or `async for`) loop: shaped by what it iterates."""
        self._in_loop(node, node.iter, node.target, [node.target, *node.body, *node.orelse])

    def _visit_while(self, node):
        """A `while` loop: `bounded` if its test reads like a retry/poll, else `n`."""
        shape = "bounded" if RE_RETRY.search(ast.unparse(node.test)) else "n"
        self.loops.append((node.lineno, shape))
        for n in [node.test, *node.body, *node.orelse]:
            self.visit(n)
        self.loops.pop()

    def _visit_comprehension(self, node):
        """A comprehension is a loop too, shaped by its FIRST generator.

        The first iterable is visited outside the loop (evaluated once); later
        iterables, the conditions and the element run per row, so inside it.
        """
        gens = node.generators
        self.visit(gens[0].iter)
        self.loops.append((node.lineno, loop_shape(gens[0].iter, gens[0].target)))
        for i, g in enumerate(gens):
            if i:
                self.visit(g.iter)
            for cond in g.ifs:
                self.visit(cond)
        for part in ("elt", "key", "value"):
            if hasattr(node, part):
                self.visit(getattr(node, part))
        self.loops.pop()

    def _visit_class(self, node):
        """Track the enclosing class, so `self.m()` resolves to `Class.m`."""
        outer, self.cls = self.cls, node.name
        self.generic_visit(node)
        self.cls = outer

    def _visit_function(self, node):
        """A def starts with an empty loop stack and its own local containers."""
        saved, self.loops = self.loops, []        # a def inside a loop is not "in loop"
        saved_local, self.local_containers = self.local_containers, _containers(node)
        self.generic_visit(node)
        self.loops, self.local_containers = saved, saved_local

    def _visit_call(self, node):
        """Record a round trip made directly, or via a local helper, in a per-row loop."""
        # ANY per-row loop on the stack, not just the innermost: a query inside
        # a bounded inner loop still runs once per row of the outer one.
        loop = next((lp for lp in reversed(self.loops) if lp[1] == "n"), None)
        if loop:
            kind, how = classify_call(node), "directly"
            if kind == "cache" and (chain_names(node.func) or [""])[0] in self.local_containers:
                kind = None                       # a memo dict, not a cache server
            if kind is None:
                callee = self.index.resolve(node, self.cls)
                kind = self.index.kind.get(callee or "")
                how = f"via {callee}()" if kind else how
            if kind:
                self.found.append((node.lineno, loop[0], kind, how))
        self.generic_visit(node)

    # ast.NodeVisitor dispatches on "visit_" + the node's class name, so these
    # names are dictated by that protocol; the logic lives in the snake_case methods.
    visit_For = visit_AsyncFor = _visit_for
    visit_While = _visit_while
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension
    visit_ClassDef = _visit_class
    visit_FunctionDef = visit_AsyncFunctionDef = _visit_function
    visit_Call = _visit_call


def _containers(fn) -> set[str]:
    """Names a function assigns a literal dict/set/list (or dict()-like call) to."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            v = n.value
            built = isinstance(v, (ast.Dict, ast.Set, ast.List, ast.DictComp, ast.SetComp,
                                   ast.ListComp)) or (
                isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id in LOCAL_CONTAINERS)
            if built:
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        out.add(t.id)
    return out


def path_context(path: str) -> str:
    """`request` (runs per HTTP call), `offline` (scripts, tasks) or `shared`.

    The ritual's own rule, so a reviewer can weigh a finding: an N+1 in a view
    multiplies with traffic, one in a one-off backfill multiplies once.
    """
    p = pathlib.PurePosixPath(path)
    name = p.name
    if (p.parts and p.parts[0] == "Miscellaneous") or "management" in p.parts or \
            name in ("tasks.py", "signals.py") or \
            any(w in name for w in ("migration", "backfill", "reconcil")):
        return "offline"
    if name in ("views.py", "serializers.py", "admin.py", "urls.py") or name.startswith("views_"):
        return "request"
    return "shared"


def _shown_fields(cls: ast.ClassDef) -> set[str]:
    """Every string named in the class's `list_display` / `readonly_fields` / `fields`.

    An admin method is a per-row hook only when one of these lists names it.
    """
    shown: set[str] = set()
    for item in cls.body:
        if isinstance(item, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("list_display", "readonly_fields",
                                                     "fields") for t in item.targets):
            shown |= {c.value for c in ast.walk(item.value)
                      if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return shown


def _db_call_lines(fn: ast.AST, cls_name: str, index: _Module) -> set[int]:
    """The lines inside a hook that make it a db round trip, directly or via a helper.

    A db call ADDED inside an existing hook is new even though the `def` line
    is not, so these lines anchor the finding as well as the `def`.
    """
    return {c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
            and (classify_call(c) == "db"
                 or index.kind.get(index.resolve(c, cls_name) or "") == "db")}


def _class_hooks(cls: ast.ClassDef, index: _Module) -> list[tuple[int, str, str, set]]:
    """`_per_row_hooks` for one class: none unless it is a serializer or an admin."""
    bases = [ast.unparse(b) for b in cls.bases]
    is_ser = any(b.split(".")[-1].endswith("Serializer") for b in bases)
    is_admin = any("Admin" in b or "Inline" in b for b in bases)
    if not (is_ser or is_admin):
        return []
    shown = _shown_fields(cls)
    where = "serializer" if is_ser else "admin list"
    out = []
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        hook = (is_ser and (fn.name.startswith("get_") or fn.name == "to_representation")) \
            or (is_admin and fn.name in shown)
        if hook and index.kind.get(f"{cls.name}.{fn.name}") == "db":
            out.append((fn.lineno, f"{cls.name}.{fn.name}", where,
                        _db_call_lines(fn, cls.name, index)))
    return out


def _per_row_hooks(tree: ast.Module, index: _Module) -> list[tuple[int, str, str, set]]:
    """Serializer `get_*`/`to_representation` and admin `list_display` callables
    that do a db round trip: they run once per row with no visible loop."""
    out = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        out += _class_hooks(cls, index)
    return out


def _file_hits(tree: ast.Module) -> list[tuple[int, int, str, str, set]]:
    """Every round trip per row in one module, new or not.

    Each hit is (line, loop line, kind, message, the lines whose being NEW makes
    the finding new): loop hits from the visitor, then per-row framework hooks.
    """
    index = _Module(tree)
    visitor = _LoopVisitor(index)
    visitor.visit(tree)
    hits = [(ln, lp, k, f"{k} round trip {how} inside a loop (loop at line {lp})", {ln, lp})
            for ln, lp, k, how in visitor.found]
    hits += [(ln, ln, "db", f"{name} does a db round trip once per {where} row",
              {ln, *calls}) for ln, name, where, calls in _per_row_hooks(tree, index)]
    return hits


def _new_hits(hits, new: set[int]):
    """Yield (line, loop line, kind, message) for hits the change added, once each.

    Only NEW findings count, and a (line, kind) pair is reported once even when
    the loop visitor and a hook both found it.
    """
    seen = set()
    for ln, lp, kind, what, anchors in hits:
        if (ln, kind) in seen or not anchors & new:
            continue
        seen.add((ln, kind))
        yield ln, lp, kind, what


def _exempt_marker(lines: list[str], ln: int, lp: int) -> re.Match | None:
    """The `# O(1)-exempt:` marker on the call's line, else on its loop's line."""
    return next((m for i in (ln, lp)
                 if 0 < i <= len(lines) and (m := RE_EXEMPT.search(lines[i - 1]))), None)


def _scan_file(change: Change, path: str, tree: ast.Module) -> tuple[list[dict], int]:
    """(flags, exempted count) for one parsed file of the change.

    A marker with a reason exempts the finding; one without a reason is
    reported, and the message says the reason is missing.
    """
    lines, new = (change.content_for(path) or "").splitlines(), change.added_numbers(path)
    flags, exempt = [], 0
    for ln, lp, kind, what in _new_hits(_file_hits(tree), new):
        marker = _exempt_marker(lines, ln, lp)
        if marker and marker.group(1).strip():
            exempt += 1
            continue
        why = " (O(1)-exempt marker has no reason)" if marker else ""
        flags.append(result("n_plus_one", 0, FLAG,
                            f"{SEVERITY[kind]} [{path_context(path)}]: {what}{why} -- "
                            f"fix it, or mark `# O(1)-exempt: <the bound>`", path, ln))
    return flags, exempt


def check_n_plus_one(change: Change, config: dict) -> list[dict]:
    """Step 0: NEW HIGH/MEDIUM round trips per row, unless `# O(1)-exempt: <reason>`."""
    if not change.readable:
        return [result("n_plus_one", 0, SKIP, "no readable revision to scan")]
    flags, scanned, exempt = [], 0, 0
    for path in change.py_files():
        tree = None if _skip_path(path, config) else change.tree_for(path)
        if tree is None:
            continue
        scanned += 1
        file_flags, file_exempt = _scan_file(change, path, tree)
        flags += file_flags
        exempt += file_exempt
    return report("n_plus_one", 0, flags, "N+1 findings",
                  f"no new per-row round trips in {scanned} file(s)"
                  + (f" ({exempt} exempted with a reason)" if exempt else ""))
