"""Conventions -- only when a tenant's `prepush` config declares them: never-commit
and read-only paths, per-app docs, Postman, the tracker row, the commit message.
A codebase without those conventions is never flagged for lacking them.
"""
from __future__ import annotations

import csv
import fnmatch
import io
import json
import pathlib
import re

from tools.prepush.change import (
    FLAG,
    MAX_REPORTED,
    PASS,
    RE_AI_ATTRIBUTION,
    SKIP,
    Change,
    _cap,
    _cfg,
    _git,
    _skip_path,
    non_ascii,
    result,
)

COMMIT_MESSAGE_FILE = "commit_message.txt"   # where the ritual drafts it, for staged work


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
    """Step 1: each app's docs updated where the change reaches, a dated change-log
    entry on every doc touched, and nothing but the allowed files in a docs folder."""
    docs = _cfg(config, "docs")
    if not docs:
        return []
    folder = docs.get("folder", "Code Documentation")
    flags = (_docs_needed(change, config, docs, folder)
             + _docs_changelog(change, docs, folder)
             + _docs_allowed_files(change, docs, folder))
    return flags or [result("docs_updated", 1, PASS, "docs updated wherever the change reaches")]


def _docs_needed(change: Change, config: dict, docs: dict, folder: str) -> list[dict]:
    """A doc the tenant's rules tie to a changed file, but which did not change."""
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
    return flags


def _docs_changelog(change: Change, docs: dict, folder: str) -> list[dict]:
    """A doc that changed without a dated change-log heading among its added lines."""
    heading = re.compile(docs.get("changelog_heading", r"^###\s*\[\d{4}-\d{2}-\d{2}\]\s*-\s*\S"))
    flags = []
    for path in change.files:
        if folder in pathlib.PurePosixPath(path).parts and path.endswith(".md") and \
                docs.get("changelog_heading") is not None and \
                not any(heading.search(t) for _, t in change.added(path)):
            flags.append(result("docs_changelog", 1, FLAG,
                                "doc changed without a dated change-log entry", path))
    return flags


def _docs_allowed_files(change: Change, docs: dict, folder: str) -> list[dict]:
    """A docs folder the change touched that now holds files outside the allowed set."""
    allowed = set(docs.get("files", []))
    flags = []
    if allowed:
        for app_docs in {pathlib.PurePosixPath(*pathlib.PurePosixPath(p).parts[:2])
                         for p in change.files
                         if len(pathlib.PurePosixPath(p).parts) > 2
                         and pathlib.PurePosixPath(p).parts[1] == folder}:
            prefix = f"{app_docs}/"
            names = {pathlib.PurePosixPath(n[len(prefix):]).parts[0]
                     for n in change.tracked_files() if n.startswith(prefix)}
            extra = sorted(names - allowed - set(docs.get("allowed_dirs", [])))
            if extra:
                flags.append(result("docs_four_files", 1, FLAG,
                                    f"unexpected file(s) {extra}; allowed: {sorted(allowed)}",
                                    str(app_docs)))
    return flags


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
    if collection in change.files and _looks_hand_edited(change, collection, pm, api_change):
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


def _looks_hand_edited(change: Change, collection: str, pm: dict, api_change: list[str]) -> bool:
    """True only when nothing that FEEDS the generator changed.

    The collection is generated from the URLconf, the views' docstrings and the
    sample files, so a regenerated collection legitimately changes whenever any of
    those do. The first version knew only the generator script itself, and flagged
    every honest regeneration on the Merger repo (2026-11-24): 079f52f, whose only
    collection change was the "GENERATED ... on <date>" stamp, and d454df7, whose
    example bodies changed because their `API Samples/*.json` changed in the same
    commit. A change that touches only the stamp is a no-op regeneration.
    `generator_inputs` entries are `_match` patterns (a trailing `/` is a folder).
    """
    inputs = pm.get("generator_inputs", [])
    if not inputs or api_change or any(_match(f, inputs) for f in change.files):
        return False
    stamp = pm.get("generated_stamp")
    added = [text for _, text in change.added(collection)]
    return not (stamp and added and all(re.search(stamp, t) for t in added))


def _request_exempt(name: str, exempt: set[str]) -> bool:
    """A request is exempt when its name IS a listed entry or ends with one.

    The collection names a request "<SURFACE> <METHOD> <route-name>" ("UNIVERSAL
    GET faqs-v1"), while the policy lists route names ("faqs-v1"), as the repo's
    own ritual does. Exact match only meant the list never matched anything, and
    six requests the repo deliberately exempts were flagged on every review that
    touched the collection (2026-11-24).
    """
    return name in exempt or any(name.endswith(f" {e}") for e in exempt)


def _postman_structure(change: Change, collection: str, pm: dict) -> list[dict]:
    """Every request in the changed collection has an example response, a
    description, and a failure example (status >= 300) unless the tenant's
    `no_failure_example_ok` names it. Flags capped at MAX_REPORTED."""
    try:
        data = json.loads(change.content_for(collection) or "{}")
    except ValueError:
        return [result("postman_structure", 2, FLAG, "collection is not valid JSON", collection)]
    no_fail_ok = set(pm.get("no_failure_example_ok", []))
    bad = []

    def walk(items):
        """Recurse through folders (`item` lists), recording each request's gaps."""
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
            if not fails and not _request_exempt(name, no_fail_ok):
                bad.append(f"{name}: no failure example")
    walk(data.get("item"))
    return _cap("postman_structure", 2,
                [result("postman_structure", 2, FLAG, msg, collection) for msg in bad],
                "collection problems")


def _head(change: Change) -> str | None:
    """The commit whose author and message describe the change: the commit itself,
    or the head of a range. None only for staged work, which has no commit yet.
    A range used to fall through to the REVIEWER's `git config user.name` and the
    working-tree commit_message.txt -- checking someone else's change against you.
    """
    return change.revision.get("head") or change.revision.get("commit")


def commit_author(change: Change) -> str:
    """The change's git author: the head commit's, or for staged work (no commit
    yet) whoever `git config user.name` says is committing."""
    head =_head(change)
    out = _git(change.root, "log", "-1", "--format=%an", head) if head else \
        _git(change.root, "config", "user.name")
    return (out or "").strip()


def check_tracker(change: Change, config: dict) -> list[dict]:
    """Step 4: the author's tracker gains the same new row in .md and .csv."""
    tr = _cfg(config, "tracker")
    if not tr:
        return []
    who = commit_author(change)
    owner = (tr.get("owners") or {}).get(who)
    if not owner:
        return [result("tracker", 4, FLAG, f"git author {who!r} has no tracker in the policy "
                                           f"(known: {sorted(tr.get('owners') or {})})")]
    base = f"{tr['dir'].rstrip('/')}/{owner['base']}"
    md, csv_path, prefix = f"{base}.md", f"{base}.csv", owner["prefix"]
    out = [result("tracker_row", 4, FLAG, "tracker not updated in this change", f)
           for f in (md, csv_path) if f not in change.files]
    if csv_path in change.files:
        out += _tracker_csv_shape(change, csv_path, tr)
        out += _tracker_new_row(change, csv_path, md, prefix, tr)
    return out or [result("tracker", 4, PASS, f"{who}'s tracker gained a valid row")]


def _tracker_csv_shape(change: Change, csv_path: str, tr: dict) -> list[dict]:
    """The whole tracker file: it parses, the header and every row's width match,
    and it is plain ASCII."""
    text = change.content_for(csv_path) or ""
    out = []
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as e:
        return [result("tracker_csv", 4, FLAG, f"CSV does not parse: {e}", csv_path)]
    header = tr.get("header")
    if header and (not rows or rows[0] != header):
        out.append(result("tracker_csv", 4, FLAG,
                          f"header differs from the expected {len(header)} columns", csv_path, 1))
    width = len(header) if header else len(rows[0]) if rows else 0
    for i, row in enumerate(rows[1:], start=2):
        if row and len(row) != width:
            out.append(result("tracker_csv", 4, FLAG,
                              f"row has {len(row)} fields, expected {width}", csv_path, i))
    if non_ascii(text):
        out.append(result("tracker_csv", 4, FLAG, "non-ASCII character in the tracker", csv_path))
    return out


def _tracker_keys(change: Change, csv_path: str, prefix: str) -> tuple[list[int], list[str]]:
    """(the numbers of the base revision's `<prefix>N` keys, the keys the change ADDED).

    The base side gives where the series stands; the added side, what the
    change claims is next.
    """
    key_re = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    old = [int(m.group(1)) for r in csv.reader(io.StringIO(change.base_for(csv_path)))
           if r for m in [key_re.match(r[0])] if m]
    new_keys = [r[0] for _, line in change.added(csv_path)
                for r in csv.reader([line]) if r and key_re.match(r[0])]
    return old, new_keys


def _tracker_row_content(change: Change, csv_path: str, tr: dict) -> list[dict]:
    """What the added row says: every required block present, no AI attribution."""
    out = []
    added_text = " ".join(t for _, t in change.added(csv_path))
    for block in tr.get("required_blocks", []):
        if block not in added_text:
            out.append(result("tracker_blocks", 4, FLAG, f"new row lacks `{block}`", csv_path))
    if RE_AI_ATTRIBUTION.search(added_text):
        out.append(result("tracker_attribution", 4, FLAG, "AI attribution in the row", csv_path))
    return out


def _tracker_new_row(change: Change, csv_path: str, md: str, prefix: str,
                     tr: dict) -> list[dict]:
    """The row this change ADDED: next key in series, mirrored in the .md, carrying
    the required blocks, and no AI attribution."""
    out = []
    old, new_keys = _tracker_keys(change, csv_path, prefix)
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
    return out + _tracker_row_content(change, csv_path, tr)


def check_commit_message(change: Change, config: dict) -> list[dict]:
    """Step 5: the commit message -- the real one for a commit, the drafted file for staged."""
    cm = _cfg(config, "commit_message")
    if not cm:
        return []
    commit = _head(change)
    if commit:
        text = _git(change.root, "log", "-1", "--format=%B", commit) or ""
        where = f"commit {commit[:12]}"
    else:
        name = cm.get("file", COMMIT_MESSAGE_FILE)
        path = change.root / name
        if not path.is_file():
            return [result("commit_message", 5, FLAG,
                           f"{name} does not exist -- "
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
    if non_ascii(text):
        out.append(result("commit_ascii", 5, FLAG, "non-ASCII character", where))
    if RE_AI_ATTRIBUTION.search(text):
        out.append(result("commit_attribution", 5, FLAG, "AI attribution in the message", where))
    for block in cm.get("required_blocks", ["VERIFIED."]):
        if not re.search(rf"(^|\n){re.escape(block)}", text):
            out.append(result("commit_blocks", 5, FLAG, f"no `{block}` paragraph", where))
    return out or [result("commit_message", 5, PASS, f"{where} follows the house format")]


def check_not_checkable(_change: Change, config: dict) -> list[dict]:
    """What the ritual runs that a reviewer cannot: said out loud, never implied."""
    return [result(name, step, SKIP, why) for name, step, why in config.get("not_checkable", [])]
