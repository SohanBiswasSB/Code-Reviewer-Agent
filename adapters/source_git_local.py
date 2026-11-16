"""Evidence from a LOCAL git checkout: review a diff without a PR or a network.

    CRA_SOURCE_CODE_REVIEW=git_local     (app/review.py sets this for you)

Written because the `github` source's network calls are still stubs
(tools/github_client.py), so the code_review surface could not see ANY change --
and because the change most worth reviewing is often one that is not a PR yet:
staged work, a local commit, a branch before it is pushed.

## Targets

    staged          `git diff --cached` -- what the next commit would contain
    <rev>           one commit, e.g. HEAD, HEAD~2, ec0d533
    <a>..<b>        the difference between two revisions, e.g. main..HEAD
    <a>...<b>       what <b> changed since it branched from <a> -- a PR's view

## Returns the GitHub source's shape

`{html_url, title, body, diff, files: [{filename}], check_runs}` -- exactly what
`CodeReviewSurface.normalize` already reads, so the surface, `classify`, the
gate and the prompts are untouched. A second evidence shape would mean a second
`normalize` and a second set of ways for routing to go quietly inert, which is
how log_review ended up unreachable three separate ways.

`check_runs` is always empty: a local checkout has no CI. That makes the
`deterministic` route unreachable from this source, which is honest -- that
route answers a red build, and there is none to answer.

## The diff budget

Code first, then config, then prose; data files are named but never shown.
See `pack_diff` for why path order was the wrong default.

## Untrusted input, twice over

The TARGET comes from a caller, so it is refused if it looks like an option
(`--output=...` would otherwise reach git as a flag) and is always passed as a
list argument, never through a shell. The DIFF is evidence like any other and
is sanitised and fenced by `nodes/_render.py` before a model sees it.
"""
from __future__ import annotations

import pathlib
import subprocess

from registry import adapter
from tools.codebase import codebase_root

# The diff BUDGET. ~48k chars is ~12k tokens: room for the prompt and retrieved
# code inside a 32k context. What does not fit is NAMED, never silently dropped --
# a review that approves a change it never saw has to say so on its face.
MAX_DIFF_CHARS = 48_000

# How the budget is spent: code first, then config, then prose; data never.
#
# It used to be one `git diff` cut at 48k, i.e. spent in path order. On the first
# real Masterclass run that meant ~40k chars of staged CSV exports were shown
# while the riskiest file in the change -- a script that splits shared question
# banks in the database -- fell past the cut and was never seen. Path order is
# alphabetical, and alphabetical is not a priority.
DATA_SUFFIXES = {".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".jsonl", ".ndjson",
                 ".sqlite", ".db", ".pkl", ".npy", ".npz"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}
CONFIG_SUFFIXES = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json", ".lock", ".env"}


def _rank(path: str) -> int:
    """0 code · 1 config · 2 docs · 3 data (content never shown)."""
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    if suffix in DATA_SUFFIXES:
        return 3
    if suffix in DOC_SUFFIXES:
        return 2
    if suffix in CONFIG_SUFFIXES:
        return 1
    return 0


def pack_diff(files: list[str], diff_for, budget: int = MAX_DIFF_CHARS) -> tuple[str, str]:
    """Spend the budget by rank; return (diff, note). `note` is "" when all fitted.

    `diff_for(path)` returns one file's diff, so a file is shown whole or not at
    all -- except the FIRST file, which is cut rather than dropped when it alone
    exceeds the budget, so a review is never handed nothing. Data files are listed
    by name only: their content is records, not code, and is the part of a change
    most likely to carry someone's personal details into a prompt.
    """
    order = sorted(files, key=lambda p: (_rank(p), files.index(p)))
    parts: list[str] = []
    used = 0
    cut, unseen, data = [], [], []
    for path in order:
        if _rank(path) == 3:
            data.append(path)
            continue
        d = diff_for(path)
        if used + len(d) <= budget:
            parts.append(d)
            used += len(d)
        elif not parts:
            parts.append(d[:budget])
            used = budget
            cut.append(path)
        else:
            unseen.append(path)

    notes = []
    if cut or unseen:
        notes.append(f"[DIFF TRUNCATED at the {budget:,}-char budget. The review "
                     f"covers only what is shown.]")
    if cut:
        notes.append(f"[CUT PART-WAY: {', '.join(cut)}]")
    if unseen:
        notes.append(f"[NOT SHOWN: {', '.join(unseen)}]")
    if data:
        notes.append(f"[DATA FILES, content never shown: {', '.join(data)}]")
    return "".join(parts), "\n".join(notes)


class GitTargetError(ValueError):
    """The target is not something this source will hand to git."""


def _git(root, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=False)
    if proc.returncode != 0:
        raise GitTargetError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _verify(root, rev: str) -> str:
    """Resolve a revision to a commit, refusing anything shaped like an option."""
    if not rev or rev.startswith("-"):
        raise GitTargetError(f"refusing revision {rev!r}: empty or option-shaped")
    return _git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()


@adapter("source", "git_local")
class GitLocalSource:
    """A diff from the tenant's checkout, in the shape the code_review surface reads."""

    def gather(self, target: str, **opts) -> dict:
        """Evidence for `target` (staged | <rev> | <a>..<b>) in the tenant's repo.

        The repo is the tenant's `codebase` binding resolved for this machine --
        the same `tools.codebase.codebase_root` ingest uses, so the diff and the index
        always describe the same checkout.
        """
        codebase = opts.get("codebase") or {}
        root = codebase_root(codebase)
        if not str(root) or not (root / ".git").exists():
            raise GitTargetError(
                f"no git checkout at {str(root)!r}. Set CRA_CODEBASE_ROOT (or "
                f"--root) to where this tenant's repo lives on this machine.")
        target = (target or "").strip()

        if target == "staged":
            diff_args = ("diff", "--cached")
            files = _git(root, "diff", "--cached", "--name-only").split("\n")
            new_ref = ":"                       # the index: `git show :path`
            base_ref = "HEAD:"
            title = f"Staged changes in {codebase.get('name') or root.name}"
            body = _git(root, "diff", "--cached", "--stat")
            ref = "staged"
        elif ".." in target:
            # `a...b` = what b changed since it branched from a (a PR's view);
            # `a..b` = the straight difference between the two trees.
            sep = "..." if "..." in target else ".."
            a, b = target.split(sep, 1)
            a, b = _verify(root, a), _verify(root, b)
            span = f"{a}{sep}{b}"
            new_ref = f"{b}:"
            # Three dots compare against where b branched from a, so the base is
            # the merge base, not a itself.
            base = _git(root, "merge-base", a, b).strip() if sep == "..." else a
            base_ref = f"{base}:"
            diff_args = ("diff", span)
            files = _git(root, "diff", "--name-only", span).split("\n")
            log = _git(root, "log", "--format=%h %s", f"{a}..{b}")
            title = f"{target} ({len(log.splitlines())} commit(s))"
            body = log
            ref = f"{a[:12]}..{b[:12]}"
        else:
            sha = _verify(root, target)
            diff_args = ("show", "--format=", sha)
            new_ref = f"{sha}:"
            base_ref = f"{sha}^:"
            files = _git(root, "show", "--format=", "--name-only", sha).split("\n")
            title = _git(root, "log", "-1", "--format=%s", sha).strip()
            body = _git(root, "log", "-1", "--format=%b", sha).strip()
            ref = sha[:12]

        files = [f for f in (x.strip() for x in files) if f]
        if not files:
            raise GitTargetError(f"target {target!r} contains no changes to review")

        # One file at a time, so the budget can be spent by priority. `--` ends
        # option parsing: a path is a path even if it starts with a dash.
        diff, note = pack_diff(files, lambda p: _git(root, *diff_args, "--", p))
        if note:
            diff = f"{diff}\n\n{note}"
            body = f"{body}\n\n{note}".strip()

        return {
            "html_url": f"git-local://{codebase.get('repo') or root.name}@{ref}",
            "title": title,
            "body": body,
            "diff": diff,
            "files": [{"filename": f} for f in files],
            "check_runs": [],
            # Where checks read the FULL change from (nodes/prepush_checks.py):
            # the packed `diff` above is cut to a budget, and a check must not
            # pass a file merely because it fell past the cut.
            "revision": {"diff_args": list(diff_args), "new_ref": new_ref,
                         "base_ref": base_ref,
                         "commit": ref if ref != "staged" and ".." not in ref else None},
        }
