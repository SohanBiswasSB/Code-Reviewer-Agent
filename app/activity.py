"""Activity log: one structured record per operation, kept, searchable, shown.

A port of the DESIGN of `Logger/activity_logger.py` in the Merged Masterclass
backend -- the same record schema, the same severity rule, the same two masking
rules and the same contract -- onto this service's storage (JSONL, like the audit
and trace stores) and its admin page (`python -m app.monitor`, "Activity log").

    with ActivityLogger("external_api", "chat.diagnosis", target_service="lmstudio",
                        metadata={"model": m}) as alog:
        ...
        alog.add_metadata("completion_tokens", n)

    log_event("approval", "approval.denied", severity=WARNING, metadata={...})

## The contract, kept from the original

* **Log, never suppress, never break.** `__exit__` records an escaping exception
  (type, message, full traceback) and returns False -- the exception continues.
  A failed WRITE falls back to one line on stderr; it never raises into the run.
* **Mask on the way in.** Secrets (key contains password/token/secret/...) become
  `[REDACTED]`; personal data is masked by exact key (`email` -> `s***@d.com`,
  `phone` -> `***1234`, `first_name` -> `S***`). Nothing unmasked is ever stored,
  so nothing needs masking on display. Error text and tracebacks additionally pass
  through this project's InputGuard, because an exception message can quote a diff.
* **Severity is derived, never passed** (for spans): ERROR if it failed, WARNING
  if slow, INFO on success, WARNING otherwise.
* **One correlation id per run**, carried in a ContextVar (the Masterclass app uses
  request attributes; the original's own notes recommend a ContextVar for an async
  stack). The API reads and echoes `X-Correlation-Id`; the CLI uses the run's
  thread id, so a run's activity rows and its audit record join on one key.

## How it differs from the trace and the audit record

The audit record is ONE immutable row per run: what was decided. A trace span is
timing per node. An activity row is per OPERATION, success or failure, with the
error and traceback when there is one -- "what happened, and what went wrong".

## Storage and retention

One file per UTC day, `<CRA_DATA_DIR>/activity/YYYY-MM-DD.jsonl`. Nothing here
deletes a day: `python -m jobs.prune_activity` does, dry run by default, and it
never touches the audit store.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime, timedelta

from datadir import data_dir
from errors import WRITE_ERRORS

SERVICE_NAME = "code-reviewer-agent"
DEBUG, INFO, WARNING, ERROR = "DEBUG", "INFO", "WARNING", "ERROR"
_LEVELS = {DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40}
LOG_TYPE = "activity_log"
# Over these, a span is "slow" and logged at WARNING (ms, by operation type).
SLOW_MS = {"request": 1000, "external_api": 120_000, "node": 300_000}
FAILURE_SOURCES = {"external", "database", "validation", "internal", "storage", "config"}

_LOCK = threading.Lock()
_CORRELATION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cra_correlation_id", default=None)

# ── masking, ported rule-for-rule ────────────────────────────────────────────
SENSITIVE_KEYS = ("password", "token", "secret", "api_key", "apikey", "authorization",
                  "access_token", "refresh_token", "private_key", "credit_card", "cvv",
                  "ssn", "otp", "pin")
_EMAIL_KEYS = {"email", "email_norm", "user_email", "actor_email", "sso_email",
               "recipient_email", "to_email", "from_email", "author_email", "committer_email"}
_PHONE_KEYS = {"phone", "mobile", "user_mobile", "user_phone", "phone_number",
               "contact_number"}
_NAME_KEYS = {"first_name", "last_name", "full_name", "user_name", "actor_name",
              "recipient_name", "author_name", "committer_name"}


def is_secret_key(key: str) -> bool:
    """Substring match, case-insensitive, with `-` read as `_`.

    The hyphen normalisation is an addition to the original rule: `api_key` does
    not occur in `X-Api-Key`, so an HTTP header slipped straight past it.
    """
    k = str(key).lower().replace("-", "_")
    return any(s in k for s in SENSITIVE_KEYS)


def mask_pii(key: str, value):
    """Exact-key PII masking for non-empty strings; anything else unchanged."""
    if not isinstance(value, str) or not value:
        return value
    k = str(key).lower()
    if k in _EMAIL_KEYS:
        local, _, domain = value.partition("@")
        return f"{local[0]}***@{domain}" if local and domain else "***"
    if k in _PHONE_KEYS:
        digits = "".join(c for c in value if c.isdigit())
        return f"***{digits[-4:]}" if len(digits) >= 4 else "***"
    if k in _NAME_KEYS:
        return f"{value[0]}***"
    return value


def sanitize(value, key: str | None = None):
    """Recursive: secret keys redacted, containers walked, PII keys masked."""
    # Numbers and booleans are never secrets: the substring rule otherwise hides
    # `completion_tokens: 30` -- a token COUNT -- behind [REDACTED]. A secret is a
    # string (or a container of them), and those are redacted whole.
    if key is not None and is_secret_key(key) and not isinstance(value, (bool, int, float)):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: sanitize(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, key) for v in value]
    return mask_pii(key, value) if key is not None else value


def _redact_text(text: str | None) -> str | None:
    """Error messages and tracebacks through the InputGuard's secret/PII patterns.

    Function-level import: `adapters` imports the LLM adapter, which imports this
    module, so importing the guard at module level here would be circular.
    """
    if not text:
        return text
    try:
        from adapters.input_guard import RegexInputGuard
        cleaned, _ = RegexInputGuard().sanitize(text)
        return cleaned
    except (ImportError, *WRITE_ERRORS):
        return text


# ── correlation ──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def correlation(correlation_id: str | None = None):
    """Bind a correlation id for everything logged inside the block."""
    cid = correlation_id or str(uuid.uuid4())
    token = _CORRELATION.set(cid)
    try:
        yield cid
    finally:
        _CORRELATION.reset(token)


def current_correlation() -> str | None:
    """The bound correlation id, if any."""
    return _CORRELATION.get()


# ── storage ──────────────────────────────────────────────────────────────────
def log_dir() -> pathlib.Path:
    """`<CRA_DATA_DIR>/activity` -- evidence-derived, as sensitive as `var/` itself."""
    return data_dir() / "activity"


def _threshold() -> int:
    return _LEVELS.get(os.getenv("CRA_ACTIVITY_LEVEL", INFO).upper(), 20)


def emit(record: dict) -> None:
    """Append one record. None-valued keys dropped. Never raises.

    A write that fails (full disk, unserialisable value) falls back to one line on
    stderr, so the event is not lost silently and the run is not broken by it.
    """
    # CRA_ACTIVITY=null records nothing -- for the gate suite, whose fault-injection
    # tasks (a refused label, a failed post) would otherwise appear as REAL errors
    # in the admin panel. Same pattern, and same "tests only" rule, as CRA_AUDIT=null.
    if os.getenv("CRA_ACTIVITY", "jsonl") == "null":
        return
    if _LEVELS.get(record.get("severity", INFO), 20) < _threshold():
        return
    record = {k: v for k, v in record.items() if v is not None}
    try:
        line = json.dumps(record, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        line = json.dumps({"severity": record.get("severity"), "message": str(record)[:2000]})
    path = log_dir() / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except WRITE_ERRORS as e:
        print(f"[activity-log write failed: {type(e).__name__}: {e}] {line[:500]}",
              file=sys.stderr, flush=True)


# ── the span ─────────────────────────────────────────────────────────────────
class ActivityLogger:
    """One operation, timed, with its outcome. Sync context manager or explicit API."""

    def __init__(self, operation_type: str = "other", operation_name: str = "", *,
                 target_service: str | None = None, correlation_id: str | None = None,
                 parent_log_id: str | None = None, tenant: str | None = None,
                 run_id: str | None = None, metadata: dict | None = None):
        """Describe the operation; nothing is timed or written until `start`/`complete`.

        The correlation id falls back to the one bound by `correlation()`, then to
        a fresh uuid, so a span logged outside any request or run is still
        findable on its own. `metadata` is masked here, on the way in.
        """
        self.log_id = str(uuid.uuid4())
        self.operation_type, self.operation_name = operation_type, operation_name
        self.target_service, self.parent_log_id = target_service, parent_log_id
        self.correlation_id = correlation_id or current_correlation() or str(uuid.uuid4())
        self.tenant, self.run_id = tenant, run_id
        self.metadata: dict = sanitize(dict(metadata or {}))
        self.status, self.is_error = "started", False
        self.error: dict = {}
        self.response_status_code: int | None = None
        self._start_wall: datetime | None = None
        self._start = 0.0
        self._done = False

    def start(self) -> ActivityLogger:
        """Begin timing. Called by `__enter__`."""
        self._start_wall, self._start = datetime.now(UTC), time.perf_counter()
        return self

    def add_metadata(self, key: str, value) -> None:
        """Attach one fact, masked on the way in; floats rounded to 2 dp."""
        if isinstance(value, float):
            value = round(value, 2)
        self.metadata[key] = sanitize(value, key)

    def set_response(self, status_code: int | None = None) -> None:
        """Record a status code (HTTP, or an exit code for a job)."""
        self.response_status_code = status_code

    def record_error(self, exception: BaseException | None = None, message: str | None = None,
                     failure_source: str = "internal") -> None:
        """Mark this operation failed, with the exception's type, message and traceback."""
        self.is_error, self.status = True, "failure"
        tb = None
        if exception is not None:
            tb = "".join(traceback.format_exception(type(exception), exception,
                                                    exception.__traceback__))
        self.error = {
            "error_message": _redact_text(message or (str(exception) if exception else None)),
            "exception_type": type(exception).__name__ if exception else None,
            "traceback": _redact_text(tb),
            "failure_source": failure_source,
            "error_category": failure_source if failure_source in FAILURE_SOURCES
            else "internal",
        }

    def severity(self, duration_ms: float) -> str:
        """ERROR if failed; WARNING if slow; INFO on success; WARNING otherwise."""
        if self.is_error:
            return ERROR
        slow = SLOW_MS.get(self.operation_type)
        if slow is not None and duration_ms > slow:
            self.metadata["slow_operation"] = True
            return WARNING
        return INFO if self.status == "success" else WARNING

    def complete(self, status: str | None = None) -> None:
        """Finish and write the record. Idempotent."""
        if self._done:
            return
        self._done = True
        if self._start_wall is None:
            self.start()
        if status:
            self.status = status
        elif self.status == "started":
            self.status = "success"
        duration = round((time.perf_counter() - self._start) * 1000, 2)
        sev = self.severity(duration)
        emit({
            "severity": sev,
            "timestamp": datetime.now(UTC).isoformat(),
            "message": f"{self.operation_type}: {self.operation_name} - {self.status}",
            "log_type": LOG_TYPE,
            "log_id": self.log_id,
            "correlation_id": self.correlation_id,
            "parent_log_id": self.parent_log_id,
            "service_name": SERVICE_NAME,
            "service_env": os.getenv("CRA_ENV"),
            "start_time": self._start_wall.isoformat(),
            "end_time": datetime.now(UTC).isoformat(),
            "duration_ms": duration,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "status": self.status,
            "target_service": self.target_service,
            "tenant": self.tenant,
            "run_id": self.run_id,
            "response_status_code": self.response_status_code,
            "is_error": self.is_error,
            **self.error,
            "metadata": self.metadata or None,
        })

    def __enter__(self) -> ActivityLogger:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Record an escaping exception (unless already recorded), write, and re-raise it."""
        if exc is not None and not self.is_error:
            self.record_error(exc)
        self.complete()
        return False                       # never suppress


def activity_log(operation_type: str, operation_name: str | None = None, **span_kw):
    """Decorator: run the function inside a span; record and RE-RAISE on error."""
    def wrap(fn):
        """Bind the decorator to `fn`; the span is named after it unless one was given."""
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            """One span per call (`functools.wraps` replaces this with `fn`'s docstring)."""
            with ActivityLogger(operation_type, operation_name or fn.__qualname__, **span_kw):
                return fn(*args, **kwargs)
        return inner
    return wrap


def log_event(operation_type: str, operation_name: str, *, severity: str = INFO,
              message: str | None = None, exception: BaseException | None = None,
              failure_source: str = "internal", metadata: dict | None = None,
              tenant: str | None = None, run_id: str | None = None,
              correlation_id: str | None = None) -> str:
    """A point-in-time event (no duration). Returns its log_id."""
    span = ActivityLogger(operation_type, operation_name, tenant=tenant, run_id=run_id,
                          correlation_id=correlation_id, metadata=metadata).start()
    if message:
        span.add_metadata("message", message)
    if severity == ERROR:
        span.record_error(exception, message, failure_source)
    elif exception is not None:
        # A WARNING that carries an exception (a retry about to happen) keeps the
        # exception's type and message, but is not a failure: the caller is about
        # to try again. If it finally fails, THAT record is the ERROR.
        span.add_metadata("exception_type", type(exception).__name__)
        span.add_metadata("exception_message", _redact_text(str(exception)))
    status = {ERROR: "failure", WARNING: "warning"}.get(severity, "success")
    span.complete(status)
    return span.log_id


# ── reading, for the admin page ──────────────────────────────────────────────
def query(*, hours: float = 24, severity: str | None = None, operation_type: str | None = None,
          q: str | None = None, correlation_id: str | None = None,
          limit: int = 300) -> dict:
    """Newest-first records in the window, filtered; plus facet counts for the UI.

    Default window is the last 24 hours, as in the Masterclass admin -- an
    unbounded error list is one nobody reads.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)
    days = {(since + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(int(hours // 24) + 2)}
    rows: list[dict] = []
    for day in sorted(days):
        path = log_dir() / f"{day}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("timestamp", "") >= since.isoformat():
                rows.append(r)
    facets = {"severity": {}, "operation_type": {}}
    for r in rows:
        for f in facets:
            facets[f][r.get(f, "?")] = facets[f].get(r.get(f, "?"), 0) + 1
    needle = (q or "").lower()
    keep = [r for r in rows
            if (not severity or r.get("severity") == severity)
            and (not operation_type or r.get("operation_type") == operation_type)
            and (not correlation_id or r.get("correlation_id") == correlation_id)
            and (not needle or needle in json.dumps(r, default=str).lower())]
    keep.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return {"total": len(rows), "matched": len(keep), "rows": keep[:limit],
            "facets": facets, "hours": hours}


# ── third-party loggers -> the activity log ─────────────────────────────────
_LOG_LEVEL = {logging.CRITICAL: ERROR, logging.ERROR: ERROR, logging.WARNING: WARNING,
              logging.INFO: INFO, logging.DEBUG: DEBUG}


class ActivityHandler(logging.Handler):
    """Routes stdlib `logging` records from LIBRARIES into the activity log.

    This service never logs through `logging` itself; this exists for what its
    dependencies emit -- uvicorn, openai, httpx, langgraph. Without it their
    warnings and errors went to stderr only, outside the one log an operator
    reads. `logging.Handler.handleError` already guarantees a failing emit never
    raises into the caller.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """One library record -> one `library` event, severity mapped from its level.

        `failure_source` is `external` because these records come from dependencies,
        not from this service's own code; an attached exception travels with it.
        """
        exc = record.exc_info[1] if record.exc_info else None
        log_event("library", record.name, severity=_LOG_LEVEL.get(record.levelno, WARNING),
                  message=record.getMessage(), exception=exc, failure_source="external",
                  metadata={"logger": record.name, "where": f"{record.module}:{record.lineno}"})


def capture_library_logs(level: int = logging.WARNING) -> None:
    """Install the bridge once, at an ENTRY POINT (API, CLI, jobs) -- never on import.

    Installing on import would also capture inside the test runner. WARNING and
    above only: library INFO is chatter. uvicorn's per-request access log is
    silenced because the request middleware (app/main.py) already records every
    request -- with its correlation id, which the access log does not have.
    """
    root = logging.getLogger()
    if not any(isinstance(h, ActivityHandler) for h in root.handlers):
        handler = ActivityHandler(level)
        root.addHandler(handler)
        for name in ("uvicorn", "uvicorn.error"):
            logging.getLogger(name).addHandler(handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True
