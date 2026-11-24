"""Step 1 + global -- hygiene on what the change ADDED, for any codebase:
new warning-silencers, line length / ASCII, handler docstrings, and imports of
files git does not track.
"""
from __future__ import annotations

import ast
import io
import pathlib
import re
import tokenize

from tools.prepush.change import (
    FLAG,
    SKIP,
    Change,
    _skip_path,
    non_ascii,
    report,
    result,
)

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
        if _skip_path(path, config):       # e.g. generated migrations carry their own noqa
            continue
        new = pragma_counts(change.content_for(path) or "")
        old = pragma_counts(change.base_for(path))
        grown = {k: n - old.get(k, 0) for k, n in new.items() if n > old.get(k, 0)}
        for key, n in grown.items():
            # Locate it: the first added line whose own comment adds to this key.
            line = next((ln for ln, text in change.added(path)
                         if "#" in text and key in pragma_counts(text)), None)
            flags.append(result("suppressions", 1, FLAG,
                                f"{n} new `{key}` pragma(s) -- fix the warning instead "
                                f"of silencing it", path, line))
    return report("suppressions", 1, flags, "new pragmas", "no new warning-silencing comments")


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
            if non_ascii(text):
                nonascii.append(result("ascii", 1, FLAG,
                                       "non-ASCII character (house rule: plain ASCII)", path, ln))
    return (report("line_length", 1, long_, "long lines", f"no added line over {limit} chars")
            + report("ascii", 1, nonascii, "non-ASCII lines", "added lines are plain ASCII"))


HANDLER_NAMES = {"get", "post", "put", "patch", "delete", "head", "options"}


def _handler_gaps(tree: ast.Module | None) -> dict[str, int]:
    """`Class.method` -> line, for HTTP handler methods without a docstring."""
    if tree is None:
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
        new = _handler_gaps(change.tree_for(path))
        old = _handler_gaps(change.tree_for(path, "base"))
        for name, line in new.items():
            if name not in old:
                flags.append(result("handler_docstrings", 1, FLAG,
                                    f"{name} has no docstring", path, line))
    return report("handler_docstrings", 1, flags, "undocumented handlers",
                  "every new HTTP handler has a docstring")


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


def check_untracked_imports(change: Change, _config: dict) -> list[dict]:
    """Global: an added import that resolves to a file present but NOT in git.

    The commit would import a module the repository does not contain, so it works
    on the author's machine and fails everywhere else. `_config` is unread: the
    check is the same for every codebase, but `run_all` calls every check as
    `check(change, config)`, positionally.
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
    return report("untracked_import", -1, flags, "untracked imports",
                  "every added import resolves to a tracked file")
