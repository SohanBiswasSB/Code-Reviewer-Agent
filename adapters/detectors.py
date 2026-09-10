"""Rule-based log detectors — the cheap half of G4's detect -> aggregate -> sample.

**Detectors run before any model, and they are the reason the log surface is
feasible at all.** A day is ~2M lines; no context window holds that, and no
sampling strategy is trustworthy without first knowing what is anomalous. So
code finds the candidates and the LLM only ever explains a bounded set of them —
the same "detectors and tools find, the LLM explains" bet the whole project rests
on, applied to logs.

Deliberately regex and counting, not a model. A detector that needed a GPU could
not run over the full stream, which would defeat the point.
"""
from __future__ import annotations

import re

from registry import adapter

# (signal, severity, pattern). Ordered: the FIRST match wins, so a line is
# counted once under its most specific signal rather than inflating several.
SIGNALS: tuple[tuple[str, str, str], ...] = (
    ("traceback", "high", r"Traceback \(most recent call last\)|^\s*File \".+\", line \d+"),
    ("oom", "critical", r"\bOutOfMemory|MemoryError|Killed process|oom[-_ ]?killer\b"),
    ("db_deadlock", "high", r"\bdeadlock detected|Lock wait timeout exceeded\b"),
    ("db_error", "high", r"\bOperationalError|IntegrityError|could not connect to server\b"),
    ("timeout", "medium", r"\bTimeoutError|timed out|ReadTimeout|ETIMEDOUT\b"),
    ("http_5xx", "high", r"\b(50[0234])\s|status[=:]\s*50[0234]\b"),
    ("task_retry", "medium", r"\bRetry\b.*\battempt\b|celery.*\bretrying\b"),
    ("auth_failure", "high", r"\b401 Unauthorized|403 Forbidden|InvalidToken\b"),
    ("error", "medium", r"\bERROR\b|\bCRITICAL\b"),
)


@adapter("detector", "signals")
class SignalDetector:
    """Match each line against an ordered signal table.

    `window` is a batch — {"service": str, "ref": str, "start": int,
    "lines": [str]} — not the whole day, so the caller can stream. Returns one
    entry per matching LINE; aggregation across windows is the caller's job
    (see adapters/source_s3logs.py), because only the caller knows the caps.
    """

    def __init__(self, signals=None):
        self.signals = [(name, sev, re.compile(pat, re.IGNORECASE))
                        for name, sev, pat in (signals or SIGNALS)]

    def detect(self, window: dict) -> list[dict]:
        service = window.get("service", "")
        ref = window.get("ref", "")
        start = int(window.get("start", 0))
        out: list[dict] = []
        for offset, line in enumerate(window.get("lines", [])):
            for name, severity, pattern in self.signals:
                if pattern.search(line):
                    out.append({
                        "signal": name,
                        "severity": severity,
                        "service": service,
                        "ref": ref,
                        "line_no": start + offset,
                        "text": line,
                    })
                    break          # first match wins; see SIGNALS
        return out
