"""The change under review and the plumbing every check shares.

Git reads are batched here (one diff, one `cat-file --batch` per side, one
tracked-file listing), so no check ever spawns a process per file. Stdlib only.
"""
from __future__ import annotations

import ast
import pathlib
import re
import subprocess
from dataclasses import dataclass, field

PASS, FLAG, SKIP = "pass", "flag", "skip"
MAX_REPORTED = 8          # per check, so one bad file cannot drown the review


def result(check: str, step: int, status: str, message: str, *where: str | int | None) -> dict:
    """One check outcome. Plain dict: it travels through the checkpoint.

    `where` is the optional `(file, line)` the outcome points at, passed
    positionally as every check always has; a variadic tail keeps the fixed
    signature inside the five-argument limit without touching ~50 callers.
    """
    if len(where) > 2:
        raise TypeError(f"result() takes at most 6 positional arguments ({4 + len(where)} given)")
    file, line = (*where, None, None)[:2]
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


def split_diff(text: str) -> dict[str, str]:
    """A multi-file unified diff -> {path: that file's chunk}, each chunk exactly
    what `git diff ... -- <path>` prints for the file alone.

    The path comes from `+++ b/` (or `--- a/` for a deletion), never from the
    `diff --git` header, which is ambiguous for a path containing " b/". A chunk
    with neither (a binary or mode-only change) is skipped, and the caller falls
    back to asking git for that one file.
    """
    out: dict[str, str] = {}
    for chunk in re.split(r"(?m)^(?=diff --git )", text):
        if not chunk.startswith("diff --git "):
            continue
        path = None
        for line in chunk.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:]
                break
            if line.startswith("--- a/"):
                path = line[6:]
            elif line.startswith("@@"):
                break
        if path is not None:
            out[path] = chunk
    return out


def _cat_files(root: pathlib.Path, specs: list[str]) -> dict[str, str | None]:
    """Many blobs (`<ref><path>`) through ONE `git cat-file --batch` process.
    {spec: text, or None when the object does not exist}; {} if git failed."""
    try:
        proc = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                              input=b"".join(s.encode() + b"\n" for s in specs),
                              capture_output=True, check=False)
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    out, buf, pos = {}, proc.stdout, 0
    for spec in specs:
        nl = buf.find(b"\n", pos)
        if nl < 0:
            break
        header = buf[pos:nl].split()
        pos = nl + 1
        if len(header) < 3 or header[-1] == b"missing":   # "<spec> missing"
            out[spec] = None
            continue
        size = int(header[2])
        out[spec] = buf[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1                                    # the blob's trailing LF
    return out


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
        """`fn()` once per `key` for this change; later calls return the cached value."""
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # Reads are BATCHED: the first request for any file fetches every file. Per
    # file, this was three git processes (diff, new blob, base blob) plus one per
    # import candidate in `tracked` -- measured on a 15-file change, ~200 git
    # calls and ~1.4 s, against 4 calls now. Results are byte-identical to the
    # per-file calls; a file the batch could not split still falls back to one.
    def _diffs(self) -> dict[str, str]:
        """Every changed file's diff from ONE git call, as {path: chunk}; {} if unreadable."""
        args =self.revision.get("diff_args") or []
        # --no-renames: a per-path diff can never pair a rename, a multi-path one
        # can, so without it the batched chunk would differ from the per-file one.
        return self._memo("diffs", lambda: split_diff(
            _git(self.root, *args, "--no-renames", "--", *self.files) or "")
            if args and self.files else {})

    def _blobs(self, side: str) -> dict[str, str | None]:
        """Every changed file's content on one side (`new` or base) from ONE
        `cat-file --batch`, as {path: text, or None if absent on that side}.

        A path containing a newline is left out, since it cannot be written to the
        batch's line-based input; `content_for` / `base_for` fall back for it.
        """
        ref =self.revision.get("new_ref" if side == "new" else "base_ref")
        safe = [f for f in self.files if "\n" not in f]
        return self._memo(("blobs", side), lambda: {
            spec[len(ref):]: text
            for spec, text in _cat_files(self.root, [f"{ref}{f}" for f in safe]).items()}
            if ref and safe else {})

    def diff_for(self, path: str) -> str:
        """This file's full unified diff ('' if unreadable)."""
        args = self.revision.get("diff_args") or []
        hit = self._diffs().get(path)
        return hit if hit is not None else self._memo(
            ("diff", path), lambda: (_git(self.root, *args, "--", path) or "") if args else "")

    def content_for(self, path: str) -> str | None:
        """The file's NEW content at the reviewed revision; None if deleted/unreadable."""
        blobs = self._blobs("new")
        if path in blobs:
            return blobs[path]
        ref = self.revision.get("new_ref")
        return self._memo(("new", path),
                          lambda: _git(self.root, "show", f"{ref}{path}") if ref else None)

    def base_for(self, path: str) -> str:
        """The file BEFORE the change ('' for a new file): the baseline to compare."""
        blobs = self._blobs("base")
        if path in blobs:
            return blobs[path] or ""
        ref = self.revision.get("base_ref")
        return self._memo(("base", path),
                          lambda: (_git(self.root, "show", f"{ref}{path}") or "") if ref else "")

    def tree_for(self, path: str, side: str = "new") -> ast.Module | None:
        """The parsed module (None if it does not parse). Parsed once per side:
        the N+1 scan and the docstring check used to parse each file separately."""
        def parse():
            """The module for `side`, or None on a SyntaxError."""
            try:
                return ast.parse(self.content_for(path) or "" if side == "new"
                                 else self.base_for(path))
            except SyntaxError:
                return None
        return self._memo(("tree", side, path), parse)

    def added(self, path: str) -> list[tuple[int, str]]:
        """Added lines of one file, with their new line numbers."""
        return self._memo(("added", path), lambda: added_lines(self.diff_for(path)))

    def added_numbers(self, path: str) -> set[int]:
        """Just the line numbers the change added or modified."""
        return {n for n, _ in self.added(path)}

    def tracked_files(self) -> frozenset[str]:
        """Every path git holds at the reviewed revision (the index, for staged).

        ONE listing, then set lookups. `tracked` used to run a git process per
        import candidate, uncached -- 140 calls and ~1 s on one change, 70% of
        the whole pre-push step.
        """
        ref = self.revision.get("new_ref") or ":"
        args = ("ls-files", "-z") if ref == ":" else \
            ("ls-tree", "-r", "-z", "--name-only", ref.rstrip(":"))
        return self._memo("tracked", lambda: frozenset(
            p for p in (_git(self.root, *args) or "").split("\0") if p))

    def tracked(self, path: str) -> bool:
        """Is `path` in git at the reviewed revision (the index, for staged)?"""
        return path in self.tracked_files()

    def py_files(self) -> list[str]:
        """Changed Python files that still exist after the change."""
        return [f for f in self.files if f.endswith(".py") and self.content_for(f) is not None]


def _cfg(config: dict, key: str) -> dict:
    """A convention block from the tenant's `prepush` config ({} when absent)."""
    return config.get(key) or {}


def _skip_path(path: str, config: dict) -> bool:
    """True if any directory in `path` is in the tenant's `skip_dirs` (default:
    `.claude`, `migrations`, `Parked`)."""
    parts = set(pathlib.PurePosixPath(path).parts)
    return bool(parts & set(config.get("skip_dirs", [".claude", "migrations", "Parked"])))


def _cap(check: str, step: int, items: list[dict], what: str) -> list[dict]:
    """Report at most MAX_REPORTED flags, plus one line saying how many more."""
    if len(items) <= MAX_REPORTED:
        return items
    more = len(items) - MAX_REPORTED
    return items[:MAX_REPORTED] + [result(check, step, FLAG, f"...and {more} more {what}")]


def report(check: str, step: int, flags: list[dict], what: str, ok: str) -> list[dict]:
    """How a check ends: its flags, capped, or one PASS line saying what was fine."""
    return _cap(check, step, flags, what) or [result(check, step, PASS, ok)]


# House text rules shared by every check that reads prose or code. One copy each:
# there were three diverging non-ASCII tests (one also caught control characters,
# two did not) and two copies of the attribution pattern.
RE_AI_ATTRIBUTION = re.compile(r"claude|co-authored", re.I)


def non_ascii(text: str) -> bool:
    """A character outside printable ASCII. Tab and line breaks are allowed."""
    return any(ord(ch) > 126 or (ord(ch) < 32 and ch not in "\t\n\r") for ch in text)
