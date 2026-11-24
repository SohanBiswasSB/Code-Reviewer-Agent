"""What the reviewer told you last time, so the next run can say what changed.

The precheck (`app/precheck.py`) is a check the user runs by hand before a push:
"these are the things to fix" -> fix them -> run again -> "fixed, clear to push".
Without memory every run is a fresh wall of text, and the user cannot see which
lines are new, which are the same ones as an hour ago, and which they already
fixed. This module is that memory: one JSON file per tenant under
`<CRA_DATA_DIR>/ledger/`, as sensitive as the rest of `var/` (finding messages
quote file paths and identifiers from the reviewed repo).

## Identity: check + file + message, never the line number

A finding is the same finding when the same check flags the same file for the
same reason. The line is NOT part of the identity, because an edit above it moves
it and the user would see "fixed" plus "new" for one unchanged problem. Digits
are dropped from the message for the same reason: "has 18 locals (limit 15)" is
still that finding after a partial fix takes it to 16.

## Only computed checks decide "clear to push"

The pre-push checks are computed by code, so "absent now" really means fixed. The
model's findings are regenerated on every run, with different wording, so their
absence proves nothing; they are reported as advice and never counted towards
the verdict. They can still be DISMISSED (matched by file plus word overlap), so a
false positive the user has already judged is not shown to them again.

## Status of a tracked finding

    open         flagged on the latest run
    fixed        was open; its file is still in the change and it is no longer flagged
    left_change  was open; its file (or commit) is no longer part of the change
    dismissed    the user judged it a false positive or accepted it, with a reason;
                 never shown as something to fix again

A dismissal is permanent until the user lifts it (`precheck undismiss`). Tying it
to the file's content would re-raise it on every edit of a file under active
development, which is exactly the noise the user asked to be spared.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from datadir import data_dir

OPEN, FIXED, LEFT, DISMISSED = "open", "fixed", "left_change", "dismissed"
AI_MATCH = 0.6           # word-set overlap at which two model findings are "the same"
_WORD = re.compile(r"[a-z_][a-z0-9_]{2,}")


def ledger_path(tenant: str) -> Path:
    """`<CRA_DATA_DIR>/ledger/<tenant>.json`."""
    return data_dir() / "ledger" / f"{tenant}.json"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    """Lower-case, digits dropped, whitespace collapsed: stable across small edits."""
    return " ".join(re.sub(r"\d+", "#", (text or "").lower()).split())


def check_id(flag: dict) -> str:
    """Stable 8-hex id for a computed check flag: check + file + message, no line."""
    key = f"check|{flag.get('check', '')}|{flag.get('file') or ''}|{_norm(flag.get('message', ''))}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def ai_id(finding: dict) -> str:
    """An id for one model finding, so it can be named on the command line."""
    key = f"ai|{finding.get('file') or ''}|{_norm(finding.get('issue', ''))}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def same_ai_finding(a: dict, b: dict) -> bool:
    """Two model findings match when they are about the same file and say much the same."""
    if (a.get("file") or "") != (b.get("file") or ""):
        return False
    wa, wb = _words(a.get("issue", "")), _words(b.get("issue", ""))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= AI_MATCH


@dataclass
class Report:
    """What one precheck run found, compared with the ledger."""
    new: list[dict] = field(default_factory=list)
    still_open: list[dict] = field(default_factory=list)
    fixed: list[dict] = field(default_factory=list)
    left_change: list[dict] = field(default_factory=list)
    dismissed: list[dict] = field(default_factory=list)
    ai: list[dict] = field(default_factory=list)
    ai_dismissed: list[dict] = field(default_factory=list)

    @property
    def to_fix(self) -> list[dict]:
        """Every computed flag the user still has to deal with."""
        return self.new + self.still_open


class Ledger:
    """One tenant's tracked findings. Load, `reconcile` a run, `save`."""

    def __init__(self, tenant: str, data: dict | None = None):
        self.tenant = tenant
        self.data = data or {"version": 1, "findings": {}, "ai_dismissed": [], "last_run": None}

    @classmethod
    def load(cls, tenant: str) -> Ledger:
        """The tenant's ledger, or an empty one on the first run."""
        path = ledger_path(tenant)
        if not path.exists():
            return cls(tenant)
        return cls(tenant, json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> Path:
        """Write atomically: a crash mid-write must not lose what the user dismissed."""
        path = ledger_path(self.tenant)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ledger-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path

    @property
    def findings(self) -> dict[str, dict]:
        """id -> tracked computed-check finding."""
        return self.data["findings"]

    def reconcile(self, flags: list[dict], ai: list[dict], scope: list[str]) -> Report:
        """Compare this run's flags and model findings with what was tracked.

        `scope` is every path (and `commit <sha>` pseudo-path) in the change now,
        which is how a vanished flag is told apart: fixed in place, or simply no
        longer part of what is being pushed.
        """
        report, now = Report(), _now()
        for flag in flags:
            self._track(check_id(flag), flag, now, report)
        self._close_vanished({check_id(f) for f in flags}, set(scope), now, report)
        for finding in ai:
            row = {**finding, "id": ai_id(finding)}
            hit = any(same_ai_finding(row, d) for d in self.data["ai_dismissed"])
            (report.ai_dismissed if hit else report.ai).append(row)
        return report

    def _close_vanished(self, seen: set[str], in_scope: set[str], now: str,
                        report: Report) -> None:
        """Every open finding not flagged this run: fixed in place, or left the change."""
        for fid, entry in self.findings.items():
            if fid in seen or entry["status"] != OPEN:
                continue
            gone_from_scope = entry.get("file") and entry["file"] not in in_scope
            entry["status"] = LEFT if gone_from_scope else FIXED
            entry["closed_at"] = now
            (report.left_change if gone_from_scope else report.fixed).append(entry)

    def _track(self, fid: str, flag: dict, now: str, report: Report) -> None:
        entry = self.findings.get(fid)
        row = {"id": fid, "check": flag.get("check", ""), "file": flag.get("file") or "",
               "line": flag.get("line") or 0, "message": flag.get("message", "")}
        if entry is None:
            self.findings[fid] = {**row, "status": OPEN, "first_seen": now,
                                  "last_seen": now, "runs": 1}
            report.new.append(self.findings[fid])
            return
        entry.update(row, last_seen=now, runs=entry.get("runs", 0) + 1)
        if entry["status"] == DISMISSED:
            report.dismissed.append(entry)
        elif entry["status"] == OPEN:
            report.still_open.append(entry)
        else:                                 # it was fixed or left, and is back
            entry.update(status=OPEN, reopened_at=now)
            entry.pop("closed_at", None)
            report.new.append(entry)

    def record_run(self, target: str, scope: list[str], report: Report) -> None:
        """Remember the latest run, so `precheck list` can say when it was."""
        self.data["last_run"] = {"at": _now(), "target": target, "files": len(scope),
                                 "to_fix": len(report.to_fix), "fixed": len(report.fixed)}
        # Kept so a model finding can be dismissed by the id this run printed.
        self.data["last_ai"] = [{k: f.get(k) for k in ("id", "file", "line", "issue")}
                                for f in report.ai]

    def dismiss(self, fid: str, reason: str) -> dict:
        """Stop showing a finding: a computed flag by id, or a model finding from the
        latest run by the id that run printed. A reason is required: a dismissal
        nobody can explain later is a suppression, which the user's standards
        forbid in code."""
        if not reason.strip():
            raise ValueError("a dismissal needs a reason")
        if fid in self.findings:
            self.findings[fid].update(status=DISMISSED, dismissed={"reason": reason, "at": _now()})
            return self.findings[fid]
        for finding in self.data.get("last_ai") or []:
            if finding.get("id") == fid:
                row = {"id": fid, "file": finding.get("file") or "",
                       "issue": finding.get("issue", ""), "reason": reason, "at": _now()}
                self.data["ai_dismissed"].append(row)
                return row
        raise KeyError(f"no finding {fid!r} in the ledger or the latest run's model findings")

    def undismiss(self, fid: str) -> None:
        """Lift a dismissal; the next run decides whether the finding is still there."""
        if fid in self.findings and self.findings[fid]["status"] == DISMISSED:
            self.findings[fid]["status"] = OPEN
            self.findings[fid].pop("dismissed", None)
            return
        kept = [d for d in self.data["ai_dismissed"] if d["id"] != fid]
        if len(kept) == len(self.data["ai_dismissed"]):
            raise KeyError(f"{fid!r} is not dismissed")
        self.data["ai_dismissed"] = kept
