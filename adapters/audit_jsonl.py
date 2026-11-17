"""Append-only JSONL audit records — one immutable line per run (G6).

Stdlib only, on purpose. The same rule that keeps `doctor` and the gate suite
runnable on a bare machine applies here: an audit trail that needs a database
driver installed is an audit trail that is missing exactly when someone is
debugging a broken environment.

Append-only is also the closest thing to immutable that a filesystem offers. A
row in a table can be `UPDATE`d; a line already written to a JSONL file can only
be corrupted, which is louder.

Swap for `@adapter("audit", "postgres")` behind the identical port when SQL is
wanted. No node changes — see docs/DATA_STORAGE.md.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime

from datadir import data_dir as _data_dir
from registry import adapter

SCHEMA_VERSION = 1

# One process may run several graph threads concurrently. A single line written
# with one write() call is atomic enough on POSIX for typical record sizes, but
# "typical" is not a guarantee once findings[] grows, so serialise the append.
_LOCK = threading.Lock()


def data_dir() -> str:
    """`CRA_DATA_DIR` as a string, resolved at call time (see datadir.py)."""
    return str(_data_dir())


def _default(obj):
    """Make Pydantic models, sets and stray objects JSON-serialisable.

    Falls back to `repr` rather than raising. An audit sink that refuses to write
    because one field was an unexpected type would lose the whole record — and
    the record is most valuable precisely when something unexpected happened.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return repr(obj)


@adapter("audit", "jsonl")
class JsonlAuditSink:
    """var/audit/<tenant>/<YYYY-MM>.jsonl, one JSON object per line."""

    def __init__(self, root: str | None = None):
        """`root` overrides `CRA_DATA_DIR` (tests point it at a temp dir)."""
        self.root = root or data_dir()

    def path_for(self, tenant: str, when: datetime | None = None) -> str:
        """The month file a record for `tenant` lands in; `when` defaults to now (UTC)."""
        when = when or datetime.now(UTC)
        # Sanitised: a tenant name reaches this from a request body, and a
        # tenant called "../../etc" must not choose where the file lands.
        safe = "".join(c for c in (tenant or "_unknown") if c.isalnum() or c in "-_.")
        safe = safe.lstrip(".") or "_unknown"
        return os.path.join(self.root, "audit", safe, f"{when:%Y-%m}.jsonl")

    def record(self, run: dict) -> None:
        """Append one record for a finished run (the `AuditSink` port, G6).

        Stamps `schema_version` and `recorded_at` ahead of the caller's keys (which
        win on a clash), serialises odd values via `_default`, and appends one line
        under `_LOCK`. Never rewrites an earlier line. Write failures (a full disk,
        an unwritable directory) are not caught here; they reach the caller.
        """
        run = {"schema_version": SCHEMA_VERSION,
               "recorded_at": datetime.now(UTC).isoformat(),
               **run}
        path = self.path_for(run.get("tenant", "_unknown"))
        line = json.dumps(run, default=_default, ensure_ascii=False) + "\n"
        with _LOCK:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)


@adapter("audit", "null")
class NullAuditSink:
    """Records nothing. For tests that are not about auditing.

    Registered rather than improvised so that turning auditing off is an explicit
    CRA_AUDIT=null, visible in `config.registered()`, instead of a silent no-op
    somebody has to discover by reading code.
    """

    def record(self, run: dict) -> None:
        """Discard the record."""
        return None


def iter_records(root: str | None = None):
    """Every audit record on disk, oldest file first.

    Reading is deliberately NOT on the `AuditSink` port. The port's job is to
    make one immutable record per run; querying the store is a different concern
    with a different backend answer (a Postgres adapter would answer this with
    SQL, not by streaming files). Keeping it out of the port stops every future
    sink from having to implement a query language.

    A corrupt line is SKIPPED rather than fatal: a truncated final line from a
    killed process must not make the whole history unreadable.
    """
    root = root or data_dir()
    audit_root = os.path.join(root, "audit")
    if not os.path.isdir(audit_root):
        return
    for dirpath, _, filenames in sorted(os.walk(audit_root)):
        for name in sorted(filenames):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
