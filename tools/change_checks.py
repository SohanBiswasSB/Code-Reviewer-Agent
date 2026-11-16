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

import ast
import csv
import fnmatch
import io
import json
import pathlib
import re
import subprocess
import tokenize
from dataclasses import dataclass, field

PASS, FLAG, SKIP = "pass", "flag", "skip"
MAX_REPORTED = 8          # per check, so one bad file cannot drown the review


def result(check: str, step: int, status: str, message: str,
           file: str | None = None, line: int | None = None) -> dict:
    """One check outcome. Plain dict: it travels through the checkpoint."""
    out = {"check": check, "step": step, "status": status, "message": message}
    if file:
        out["file"] = file
    if line:
        out["line"] = line
    return out


def _git(root: pathlib.Path, *args: str) -> str | None:
    """git output, or None on failure -- a check that cannot read must SKIP."""
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=False)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(diff: str) -> list[tuple[int, str]]:
    """(new line number, text) for every `+` line of one file's unified diff."""
    out, lineno = [], 0
    for raw in diff.splitlines():
        m = _HUNK.match(raw)
        if m:
            lineno = int(m.group(1))
            continue
        if raw.startswith(("+++", "---")):
            continue
        if raw.startswith("+"):
            out.append((lineno, raw[1:]))
            lineno += 1
        elif not raw.startswith("-"):
            lineno += 1
    return out


def removed_lines(diff: str) -> list[str]:
    """Text of every `-` line of one file's unified diff."""
    return [raw[1:] for raw in diff.splitlines()
            if raw.startswith("-") and not raw.startswith("---")]


@dataclass
class Change:
    """The change under review, read lazily and in full from git.

    `files` is the complete changed-file list. Diffs and contents come from git,
    never from the packed evidence diff, which may have been cut to a budget -- a
    file must never pass a check because it fell past that cut.
    """
    root: pathlib.Path
    files: list[str]
    revision: dict
    _cache: dict = field(default_factory=dict)

    @property
    def readable(self) -> bool:
        """True when the repo and revision are available to read from."""
        return bool(self.revision.get("diff_args")) and (self.root / ".git").exists()

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def diff_for(self, path: str) -> str:
        """This file's full unified diff ('' if unreadable)."""
        args = self.revision.get("diff_args") or []
        return self._memo(("diff", path),
                          lambda: (_git(self.root, *args, "--", path) or "") if args else "")

    def content_for(self, path: str) -> str | None:
        """The file's NEW content at the reviewed revision; None if deleted/unreadable."""
        ref = self.revision.get("new_ref")
        return self._memo(("new", path),
                          lambda: _git(self.root, "show", f"{ref}{path}") if ref else None)

    def base_for(self, path: str) -> str:
        """The file BEFORE the change ('' for a new file): the baseline to compare."""
        ref = self.revision.get("base_ref")
        return self._memo(("base", path),
                          lambda: (_git(self.root, "show", f"{ref}{path}") or "") if ref else "")

    def added(self, path: str) -> list[tuple[int, str]]:
        """Added lines of one file, with their new line numbers."""
        return self._memo(("added", path), lambda: added_lines(self.diff_for(path)))

    def added_numbers(self, path: str) -> set[int]:
        """Just the line numbers the change added or modified."""
        return {n for n, _ in self.added(path)}

    def tracked(self, path: str) -> bool:
        """Is `path` in git at the reviewed revision (the index, for staged)?"""
        ref = self.revision.get("new_ref") or ":"
        if ref == ":":
            return _git(self.root, "ls-files", "--error-unmatch", "--", path) is not None
        return _git(self.root, "cat-file", "-e", f"{ref}{path}") is not None

    def py_files(self) -> list[str]:
        """Changed Python files that still exist after the change."""
        return [f for f in self.files if f.endswith(".py") and self.content_for(f) is not None]


def _cfg(config: dict, key: str) -> dict:
    """A convention block from the tenant's `prepush` config ({} when absent)."""
    return config.get(key) or {}


def _skip_path(path: str, config: dict) -> bool:
    parts = set(pathlib.PurePosixPath(path).parts)
    return bool(parts & set(config.get("skip_dirs", [".claude", "migrations", "Parked"])))


def _cap(check: str, step: int, items: list[dict], what: str) -> list[dict]:
    """Report at most MAX_REPORTED flags, plus one line saying how many more."""
    if len(items) <= MAX_REPORTED:
        return items
    more = len(items) - MAX_REPORTED
    return items[:MAX_REPORTED] + [result(check, step, FLAG, f"...and {more} more {what}")]


# ════════════════════════════════════════════════════════════════════════════
# Step 0 -- N+1: a query, cache or HTTP call made once per row inside a loop
# ════════════════════════════════════════════════════════════════════════════
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
RE_RETRY = re.compile(r"(attempt|retr|tries|try|poll|backoff|_max|max_)", re.I)
RE_CHUNK = re.compile(r"(chunk|batch|page|slice|window)", re.I)
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


def classify_call(call: ast.Call) -> str | None:
    """`db` | `thread-hop` | `cache` | `http` | None, by the ritual's rules, in order."""
    func = call.func
    if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and \
            func.func.id in ("sync_to_async", "database_sync_to_async"):
        return "thread-hop"
    if not isinstance(func, ast.Attribute):
        return None
    meth, names = func.attr, chain_names(func.value)
    recv = names[-1] if names else ""
    builders = [n for n in names[1:] if n in QS_BUILDERS or n.endswith("_set")]
    if meth in CACHE_METHODS and RE_CACHE.search(recv):
        return "cache"
    if meth in ASYNC_ORM:
        return "db"
    if meth in SYNC_ORM_STRONG and not (meth == "save" and recv in ("form", "serializer")):
        return "db"
    if builders and meth in (QS_TERMINAL | QS_BUILDERS):
        return "db"
    if meth in HTTP_METHODS and RE_HTTP.search(recv):
        return "http"
    return None


def _target_names(node) -> str:
    return " ".join(n.id for n in ast.walk(node) if isinstance(n, ast.Name))


def _bounded(it) -> bool:
    if isinstance(it, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return True
    if isinstance(it, ast.Name) and RE_CONST.match(it.id):
        return True
    if isinstance(it, ast.Attribute) and RE_CONST.match(it.attr):
        return True
    if isinstance(it, ast.Call):
        f = it.func
        if isinstance(f, ast.Name) and f.id == "range" and \
                all(isinstance(a, ast.Constant) for a in it.args):
            return True
        if isinstance(f, ast.Name) and f.id == "enumerate" and it.args:
            return _bounded(it.args[0])
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
    if RE_CHUNK.search(words):
        return "chunked"
    return "n"


class _Module:
    """Per-module index: which local functions do a round trip (and of what kind)."""

    def __init__(self, tree: ast.Module):
        self.defs: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.defs[f"{node.name}.{item.name}"] = item
        self.kind: dict[str, str | None] = {}
        for key, fn in self.defs.items():
            kinds = [k for c in ast.walk(fn) if isinstance(c, ast.Call)
                     for k in [classify_call(c)] if k]
            self.kind[key] = max(kinds, key=RANK.get) if kinds else None
        for _ in range(3):                      # propagate through local calls
            changed = False
            for key, fn in self.defs.items():
                cls = key.split(".")[0] if "." in key else None
                for c in ast.walk(fn):
                    if isinstance(c, ast.Call):
                        k = self.kind.get(self.resolve(c, cls) or "")
                        if k and RANK[k] > RANK.get(self.kind[key] or "", 0):
                            self.kind[key], changed = k, True
            if not changed:
                break

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
        self.index, self.loops, self.cls = index, [], None
        self.found: list[tuple[int, int, str, str]] = []   # line, loop line, kind, how

    def _in_loop(self, node, it, target, inside):
        self.visit(it)
        self.loops.append((node.lineno, loop_shape(it, target)))
        for n in inside:
            self.visit(n)
        self.loops.pop()

    def visit_For(self, node):  # noqa: N802 -- ast.NodeVisitor naming
        self._in_loop(node, node.iter, node.target, [node.target, *node.body, *node.orelse])

    visit_AsyncFor = visit_For  # noqa: N815

    def visit_While(self, node):  # noqa: N802
        shape = "bounded" if RE_RETRY.search(ast.unparse(node.test)) else "n"
        self.loops.append((node.lineno, shape))
        for n in [node.test, *node.body, *node.orelse]:
            self.visit(n)
        self.loops.pop()

    def _comp(self, node):
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

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp  # noqa: N815

    def visit_ClassDef(self, node):  # noqa: N802
        outer, self.cls = self.cls, node.name
        self.generic_visit(node)
        self.cls = outer

    def visit_FunctionDef(self, node):  # noqa: N802
        saved, self.loops = self.loops, []        # a def inside a loop is not "in loop"
        self.generic_visit(node)
        self.loops = saved

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node):  # noqa: N802
        if self.loops and self.loops[-1][1] == "n":
            kind, how = classify_call(node), "directly"
            if kind is None:
                callee = self.index.resolve(node, self.cls)
                kind = self.index.kind.get(callee or "")
                how = f"via {callee}()" if kind else how
            if kind:
                self.found.append((node.lineno, self.loops[-1][0], kind, how))
        self.generic_visit(node)


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


def _per_row_hooks(tree: ast.Module, index: _Module) -> list[tuple[int, str, str]]:
    """Serializer `get_*`/`to_representation` and admin `list_display` callables
    that do a db round trip: they run once per row with no visible loop."""
    out = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        bases = [ast.unparse(b) for b in cls.bases]
        is_ser = any(b.split(".")[-1].endswith("Serializer") for b in bases)
        is_admin = any("Admin" in b or "Inline" in b for b in bases)
        if not (is_ser or is_admin):
            continue
        shown: set[str] = set()
        for item in cls.body:
            if isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in ("list_display", "readonly_fields",
                                                         "fields") for t in item.targets):
                shown |= {c.value for c in ast.walk(item.value)
                          if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hook = (is_ser and (fn.name.startswith("get_") or fn.name == "to_representation")) \
                or (is_admin and fn.name in shown)
            if hook and index.kind.get(f"{cls.name}.{fn.name}") == "db":
                where = "serializer" if is_ser else "admin list"
                out.append((fn.lineno, f"{cls.name}.{fn.name}", where))
    return out


def check_n_plus_one(change: Change, config: dict) -> list[dict]:
    """Step 0: NEW HIGH/MEDIUM round trips per row, unless `# O(1)-exempt: <reason>`."""
    if not change.readable:
        return [result("n_plus_one", 0, SKIP, "no readable revision to scan")]
    flags, scanned, exempt = [], 0, 0
    for path in change.py_files():
        if _skip_path(path, config):
            continue
        src = change.content_for(path) or ""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        scanned += 1
        lines, new = src.splitlines(), change.added_numbers(path)
        index = _Module(tree)
        visitor = _LoopVisitor(index)
        visitor.visit(tree)
        hits = [(ln, lp, k, f"{k} round trip {how} inside a loop (loop at line {lp})")
                for ln, lp, k, how in visitor.found]
        hits += [(ln, ln, "db", f"{name} does a db round trip once per {where} row")
                 for ln, name, where in _per_row_hooks(tree, index)]
        seen = set()
        for ln, lp, kind, what in hits:
            if (ln, kind) in seen or not ({ln, lp} & new):
                continue                               # only NEW findings count
            seen.add((ln, kind))
            marker = next((RE_EXEMPT.search(lines[i - 1]) for i in (ln, lp)
                           if 0 < i <= len(lines) and RE_EXEMPT.search(lines[i - 1])), None)
            if marker and marker.group(1).strip():
                exempt += 1
                continue
            why = " (O(1)-exempt marker has no reason)" if marker else ""
            flags.append(result("n_plus_one", 0, FLAG,
                                f"{SEVERITY[kind]} [{path_context(path)}]: {what}{why} -- "
                                f"fix it, or mark `# O(1)-exempt: <the bound>`", path, ln))
    if flags:
        return _cap("n_plus_one", 0, flags, "N+1 findings")
    return [result("n_plus_one", 0, PASS,
                   f"no new per-row round trips in {scanned} file(s)"
                   + (f" ({exempt} exempted with a reason)" if exempt else ""))]


# ════════════════════════════════════════════════════════════════════════════
# Step 1 -- static hygiene on what the change ADDED
# ════════════════════════════════════════════════════════════════════════════
RE_PYLINT = re.compile(r"pylint:\s*disable(?:-next)?=([a-z0-9,\s-]+)", re.I)
RE_OTHER_PRAGMA = re.compile(r"#\s*(type:\s*ignore|noqa|isort:\s*skip)", re.I)


def pragma_counts(src: str) -> dict[str, int]:
    """Warning-silencers in COMMENT tokens only (a docstring quoting one is fine)."""
    counts: dict[str, int] = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return counts
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        for m in RE_PYLINT.finditer(tok.string):
            for name in (n.strip() for n in m.group(1).split(",")):
                if name:
                    counts[f"pylint:{name}"] = counts.get(f"pylint:{name}", 0) + 1
        if RE_OTHER_PRAGMA.search(tok.string):
            counts["type-ignore/noqa/isort-skip"] = counts.get("type-ignore/noqa/isort-skip", 0) + 1
    return counts


def check_suppressions(change: Change, config: dict) -> list[dict]:
    """Step 1: any NEW 'silence this warning' comment, counted against the base."""
    flags = []
    for path in change.py_files():
        new = pragma_counts(change.content_for(path) or "")
        old = pragma_counts(change.base_for(path))
        grown = {k: new[k] - old.get(k, 0) for k in new if new[k] > old.get(k, 0)}
        for key, n in grown.items():
            # Locate it: the first added line whose own comment adds to this key.
            line = next((ln for ln, text in change.added(path)
                         if "#" in text and key in pragma_counts(text)), None)
            flags.append(result("suppressions", 1, FLAG,
                                f"{n} new `{key}` pragma(s) -- fix the warning instead "
                                f"of silencing it", path, line))
    if flags:
        return _cap("suppressions", 1, flags, "new pragmas")
    return [result("suppressions", 1, PASS, "no new warning-silencing comments")]


def check_line_hygiene(change: Change, config: dict) -> list[dict]:
    """Step 1: added lines over the length limit, or with non-ASCII characters."""
    limit = int(config.get("max_line_length", 120))
    suffixes = tuple(config.get("hygiene_suffixes", [".py"]))
    long_, nonascii = [], []
    for path in change.files:
        if not path.endswith(suffixes) or _skip_path(path, config):
            continue
        for ln, text in change.added(path):
            if len(text) > limit:
                long_.append(result("line_length", 1, FLAG,
                                    f"{len(text)} chars (limit {limit})", path, ln))
            if any(ord(ch) > 126 or (ord(ch) < 32 and ch not in "\t") for ch in text):
                nonascii.append(result("ascii", 1, FLAG,
                                       "non-ASCII character (house rule: plain ASCII)", path, ln))
    out = _cap("line_length", 1, long_, "long lines") or \
        [result("line_length", 1, PASS, f"no added line over {limit} chars")]
    out += _cap("ascii", 1, nonascii, "non-ASCII lines") or \
        [result("ascii", 1, PASS, "added lines are plain ASCII")]
    return out


HANDLER_NAMES = {"get", "post", "put", "patch", "delete", "head", "options"}


def _handler_gaps(src: str) -> dict[str, int]:
    """`Class.method` -> line, for HTTP handler methods without a docstring."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    gaps = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    fn.name.lower() in HANDLER_NAMES and not ast.get_docstring(fn):
                gaps[f"{cls.name}.{fn.name}"] = fn.lineno
    return gaps


def check_handler_docstrings(change: Change, config: dict) -> list[dict]:
    """Step 1: no NEW undocumented HTTP handler (gaps already at the base are not new)."""
    flags = []
    for path in change.py_files():
        if _skip_path(path, config):
            continue
        new = _handler_gaps(change.content_for(path) or "")
        old = _handler_gaps(change.base_for(path))
        for name, line in new.items():
            if name not in old:
                flags.append(result("handler_docstrings", 1, FLAG,
                                    f"{name} has no docstring", path, line))
    return _cap("handler_docstrings", 1, flags, "undocumented handlers") or \
        [result("handler_docstrings", 1, PASS, "every new HTTP handler has a docstring")]


def _module_candidates(path: str, line: str) -> list[str]:
    """Repo paths an import line could resolve to (absolute and relative imports)."""
    line = line.strip()
    mods: list[str] = []
    m = re.match(r"from\s+(\.*)([\w.]*)\s+import\s+(.+)", line)
    if m:
        dots, mod, names = m.groups()
        base = list(pathlib.PurePosixPath(path).parent.parts)
        if dots:
            base = base[:len(base) - (len(dots) - 1)] if len(dots) > 1 else base
            stem = base + (mod.split(".") if mod else [])
        else:
            stem = mod.split(".")
        mods.append("/".join(stem))
        for name in re.split(r"[,\s()]+", names):
            if name and name not in ("as", "*") and name.isidentifier():
                mods.append("/".join([*stem, name]))
    else:
        m = re.match(r"import\s+(.+)", line)
        if m:
            for part in m.group(1).split(","):
                mods.append(part.strip().split(" ")[0].replace(".", "/"))
    out = []
    for mod in mods:
        if mod:
            out += [f"{mod}.py", f"{mod}/__init__.py"]
    return out


def check_untracked_imports(change: Change, config: dict) -> list[dict]:
    """Global: an added import that resolves to a file present but NOT in git.

    The commit would import a module the repository does not contain, so it works
    on the author's machine and fails everywhere else.
    """
    if not change.readable:
        return [result("untracked_import", -1, SKIP, "no readable revision")]
    flags = []
    for path in change.py_files():
        if path.startswith(".claude/"):
            continue
        for ln, text in change.added(path):
            if not re.match(r"\s*(import|from)\s", text):
                continue
            for cand in _module_candidates(path, text):
                if (change.root / cand).is_file() and not change.tracked(cand):
                    flags.append(result("untracked_import", -1, FLAG,
                                        f"imports `{cand}`, which is not in git", path, ln))
                    break
    return _cap("untracked_import", -1, flags, "untracked imports") or \
        [result("untracked_import", -1, PASS, "every added import resolves to a tracked file")]


# ════════════════════════════════════════════════════════════════════════════
# Conventions -- only when the tenant's `prepush` config declares them
# ════════════════════════════════════════════════════════════════════════════
def _match(path: str, patterns: list[str]) -> bool:
    name = pathlib.PurePosixPath(path).name
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(path, p) or
               (p.endswith("/") and f"/{p}" in f"/{path}") for p in patterns)


def check_paths(change: Change, config: dict) -> list[dict]:
    """Never-commit and read-only paths: the change must not touch them at all."""
    out = []
    for key, check, why in (("never_commit", "never_commit", "must never be committed"),
                            ("read_only", "read_only", "is read-only history")):
        patterns = config.get(key) or []
        if not patterns:
            continue
        hits = [f for f in change.files if any(f.startswith(p) or fnmatch.fnmatchcase(f, p)
                                               for p in patterns)]
        out += [result(check, 3 if key == "never_commit" else 1, FLAG, f"this path {why}", f)
                for f in hits] or [
            result(check, 1, PASS, f"no {key.replace('_', '-')} path touched")]
    return out


def check_docs(change: Change, config: dict) -> list[dict]:
    """Step 1: each app's docs updated where the change reaches, with a change-log entry."""
    docs = _cfg(config, "docs")
    if not docs:
        return []
    folder = docs.get("folder", "Code Documentation")
    allowed = set(docs.get("files", []))
    changed = set(change.files)
    flags, needed = [], {}
    for path in change.files:
        parts = pathlib.PurePosixPath(path).parts
        if len(parts) < 2 or folder in parts or _skip_path(path, config):
            continue
        app_docs = pathlib.PurePosixPath(parts[0], folder)
        if not (change.root / app_docs).is_dir():
            continue
        rest = "/".join(parts[1:])
        for rule in docs.get("rules", []):
            if _match(rest, rule["when"]):
                needed.setdefault(str(app_docs / rule["doc"]), []).append(path)
    for doc, because in sorted(needed.items()):
        if doc not in changed:
            flags.append(result("docs_updated", 1, FLAG,
                                f"not updated, but {because[0]}"
                                + (f" (+{len(because) - 1} more)" if len(because) > 1 else "")
                                + " changed", doc))
    heading = re.compile(docs.get("changelog_heading", r"^###\s*\[\d{4}-\d{2}-\d{2}\]\s*-\s*\S"))
    for path in change.files:
        if folder in pathlib.PurePosixPath(path).parts and path.endswith(".md") and \
                docs.get("changelog_heading") is not None and \
                not any(heading.search(t) for _, t in change.added(path)):
            flags.append(result("docs_changelog", 1, FLAG,
                                "doc changed without a dated change-log entry", path))
    if allowed:
        for app_docs in {pathlib.PurePosixPath(*pathlib.PurePosixPath(p).parts[:2])
                         for p in change.files
                         if len(pathlib.PurePosixPath(p).parts) > 2
                         and pathlib.PurePosixPath(p).parts[1] == folder}:
            listing = _git(change.root, "ls-tree", "--name-only",
                           (change.revision.get("new_ref") or "HEAD:").rstrip(":") or "HEAD",
                           f"{app_docs}/") if change.revision.get("new_ref") != ":" else \
                _git(change.root, "ls-files", "--", f"{app_docs}/")
            names = {pathlib.PurePosixPath(n).relative_to(app_docs).parts[0]
                     for n in (listing or "").split("\n") if n.strip()}
            extra = sorted(names - allowed - set(docs.get("allowed_dirs", [])))
            if extra:
                flags.append(result("docs_four_files", 1, FLAG,
                                    f"unexpected file(s) {extra}; allowed: {sorted(allowed)}",
                                    str(app_docs)))
    return flags or [result("docs_updated", 1, PASS, "docs updated wherever the change reaches")]


def check_postman(change: Change, config: dict) -> list[dict]:
    """Step 2: an API change carries a regenerated collection; secrets stay blank."""
    pm = _cfg(config, "postman")
    if not pm:
        return []
    out = []
    collection = pm.get("collection")
    api_change = [f for f in change.files if _match(f, pm.get("when", []))]
    if collection and api_change and collection not in change.files:
        out.append(result("postman_collection", 2, FLAG,
                          f"API code changed ({api_change[0]}"
                          + (f" +{len(api_change) - 1}" if len(api_change) > 1 else "")
                          + ") but the Postman collection was not regenerated", collection))
    inputs = pm.get("generator_inputs", [])
    if collection in change.files and inputs and not any(f in change.files for f in inputs):
        out.append(result("postman_hand_edit", 2, FLAG,
                          "collection changed with no generator change -- never hand-edit it",
                          collection))
    if collection in change.files:
        out += _postman_structure(change, collection, pm)
    for env in [f for f in change.files if fnmatch.fnmatchcase(f, pm.get("environments", ""))]:
        try:
            data = json.loads(change.content_for(env) or "{}")
        except ValueError:
            out.append(result("postman_env", 2, FLAG, "environment file is not valid JSON", env))
            continue
        leaked = [v.get("key") for v in data.get("values", [])
                  if v.get("type") == "secret" and str(v.get("value") or "").strip()]
        if leaked:
            out.append(result("postman_env_secret", 2, FLAG,
                              f"secret value(s) not blank: {leaked}", env))
    return out or [result("postman", 2, PASS, "no API change, or collection regenerated cleanly")]


def _postman_structure(change: Change, collection: str, pm: dict) -> list[dict]:
    try:
        data = json.loads(change.content_for(collection) or "{}")
    except ValueError:
        return [result("postman_structure", 2, FLAG, "collection is not valid JSON", collection)]
    no_fail_ok = set(pm.get("no_failure_example_ok", []))
    bad = []

    def walk(items):
        for it in items or []:
            if "item" in it:
                walk(it["item"])
                continue
            req, name = it.get("request") or {}, it.get("name", "?")
            responses = it.get("response") or []
            if not responses:
                bad.append(f"{name}: no example response")
            if not str(req.get("description") or "").strip():
                bad.append(f"{name}: no description")
            fails = any(int(r.get("code") or 0) >= 300 for r in responses)
            if name not in no_fail_ok and not fails:
                bad.append(f"{name}: no failure example")
    walk(data.get("item"))
    return [result("postman_structure", 2, FLAG, msg, collection) for msg in bad[:MAX_REPORTED]] + (
        [result("postman_structure", 2, FLAG, f"...and {len(bad) - MAX_REPORTED} more", collection)]
        if len(bad) > MAX_REPORTED else [])


def _author(change: Change) -> str:
    commit = change.revision.get("commit")
    out = _git(change.root, "log", "-1", "--format=%an", commit) if commit else \
        _git(change.root, "config", "user.name")
    return (out or "").strip()


def check_tracker(change: Change, config: dict) -> list[dict]:
    """Step 4: the author's tracker gains the same new row in .md and .csv."""
    tr = _cfg(config, "tracker")
    if not tr:
        return []
    who = _author(change)
    owner = (tr.get("owners") or {}).get(who)
    if not owner:
        return [result("tracker", 4, FLAG, f"git author {who!r} has no tracker in the policy "
                                           f"(known: {sorted(tr.get('owners') or {})})")]
    base = f"{tr['dir'].rstrip('/')}/{owner['base']}"
    md, csv_path, prefix = f"{base}.md", f"{base}.csv", owner["prefix"]
    out = [result("tracker_row", 4, FLAG, "tracker not updated in this change", f)
           for f in (md, csv_path) if f not in change.files]
    if csv_path in change.files:
        out += _tracker_csv(change, csv_path, md, prefix, who, tr)
    return out or [result("tracker", 4, PASS, f"{who}'s tracker gained a valid row")]


def _tracker_csv(change, csv_path, md, prefix, who, tr) -> list[dict]:
    text = change.content_for(csv_path) or ""
    out = []
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as e:
        return [result("tracker_csv", 4, FLAG, f"CSV does not parse: {e}", csv_path)]
    header = tr.get("header")
    if header and (not rows or rows[0] != header):
        out.append(result("tracker_csv", 4, FLAG, "header differs from the expected 12 columns",
                          csv_path, 1))
    width = len(header) if header else len(rows[0]) if rows else 0
    for i, row in enumerate(rows[1:], start=2):
        if row and len(row) != width:
            out.append(result("tracker_csv", 4, FLAG,
                              f"row has {len(row)} fields, expected {width}", csv_path, i))
    if any(ord(ch) > 126 for ch in text):
        out.append(result("tracker_csv", 4, FLAG, "non-ASCII character in the tracker", csv_path))
    key_re = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    old = [int(m.group(1)) for r in csv.reader(io.StringIO(change.base_for(csv_path)))
           if r for m in [key_re.match(r[0])] if m]
    new_keys = [r[0] for _, line in change.added(csv_path)
                for r in csv.reader([line]) if r and key_re.match(r[0])]
    if not new_keys:
        out.append(result("tracker_row", 4, FLAG, f"no new {prefix}N row added", csv_path))
        return out
    expected = f"{prefix}{max(old, default=0) + 1}"
    if new_keys[0] != expected:
        out.append(result("tracker_key", 4, FLAG,
                          f"new key {new_keys[0]} is not next in series ({expected})", csv_path))
    if md in change.files and not any(new_keys[0] in t for _, t in change.added(md)):
        out.append(result("tracker_md", 4, FLAG,
                          f"{new_keys[0]} is in the CSV but not the .md", md))
    added_text = " ".join(t for _, t in change.added(csv_path))
    for block in tr.get("required_blocks", []):
        if block not in added_text:
            out.append(result("tracker_blocks", 4, FLAG, f"new row lacks `{block}`", csv_path))
    if re.search(r"claude|co-authored", added_text, re.I):
        out.append(result("tracker_attribution", 4, FLAG, "AI attribution in the row", csv_path))
    return out


def check_commit_message(change: Change, config: dict) -> list[dict]:
    """Step 5: the commit message -- the real one for a commit, the drafted file for staged."""
    cm = _cfg(config, "commit_message")
    if not cm:
        return []
    commit = change.revision.get("commit")
    if commit:
        text = _git(change.root, "log", "-1", "--format=%B", commit) or ""
        where = f"commit {commit}"
    else:
        path = change.root / cm.get("file", "commit_message.txt")
        if not path.is_file():
            return [result("commit_message", 5, FLAG,
                           f"{cm.get('file', 'commit_message.txt')} does not exist -- "
                           f"the ritual writes it before a push")]
        text, where = path.read_text(encoding="utf-8", errors="replace"), cm.get("file")
    out, limit = [], int(cm.get("max_line_length", 120))
    lines = text.splitlines()
    subject = re.compile(cm.get("subject", r"^\w+\(\S+\): .+"))
    if not lines or not subject.match(lines[0]):
        out.append(result("commit_subject", 5, FLAG,
                          f"subject does not match `{subject.pattern}`", where, 1))
    out += [result("commit_line_length", 5, FLAG, f"{len(t)} chars (limit {limit})", where, i)
            for i, t in enumerate(lines, 1) if len(t) > limit][:MAX_REPORTED]
    if any(ord(ch) > 126 for ch in text):
        out.append(result("commit_ascii", 5, FLAG, "non-ASCII character", where))
    if re.search(r"claude|co-authored", text, re.I):
        out.append(result("commit_attribution", 5, FLAG, "AI attribution in the message", where))
    for block in cm.get("required_blocks", ["VERIFIED."]):
        if not re.search(rf"(^|\n){re.escape(block)}", text):
            out.append(result("commit_blocks", 5, FLAG, f"no `{block}` paragraph", where))
    return out or [result("commit_message", 5, PASS, f"{where} follows the house format")]


def check_not_checkable(change: Change, config: dict) -> list[dict]:
    """What the ritual runs that a reviewer cannot: said out loud, never implied."""
    return [result(name, step, SKIP, why) for name, step, why in config.get("not_checkable", [])]


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
        except Exception as e:  # noqa: BLE001 -- one broken check must not hide the rest
            out.append(result(check.__name__.removeprefix("check_"), -1, SKIP,
                              f"check crashed: {type(e).__name__}: {e}"))
    return out
