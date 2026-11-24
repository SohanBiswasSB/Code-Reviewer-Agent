"""`tools.ingest.run` marks the live record only when the job itself failed.

The failure is recorded from `finally`, and `sys.exception()` there ALSO returns
an exception the caller happens to be handling -- so a successful ingest run
inside an unrelated `except` block must not be recorded as an error.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pytest

from tools import ingest


@dataclass
class FakeLive:
    """Stands in for IngestProgress: remembers every `end` call."""

    tenant: str
    namespace: str
    ends: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def end(self, status, **extra):
        """Record the status and extras that IngestProgress.end would write."""
        self.ends.append((status, extra))


@pytest.fixture(name="live")
def fixture_live(monkeypatch):
    """Fake the live record and the target; the holder exposes the FakeLive the run made."""
    holder = {}

    def make(tenant, namespace):
        holder["live"] = FakeLive(tenant, namespace)
        return holder["live"]

    monkeypatch.setattr(ingest, "IngestProgress", make)
    monkeypatch.setattr(ingest, "_resolve_target",
                        lambda job: ingest.IngestTarget("org/x", pathlib.Path("."), []))
    return holder


def _logged_build_raising(exc):
    def fake(live, job, target):
        raise exc
    return fake


def test_success_inside_an_unrelated_except_records_nothing(monkeypatch, live):
    """A successful run inside the caller's except block ends no live record as an error."""
    monkeypatch.setattr(ingest, "_logged_build", lambda live, job, target: 0)
    try:
        raise KeyError("unrelated, handled by the caller")
    except KeyError:
        assert ingest.run(ingest.IngestJob("t")) == 0
    assert live["live"].ends == []


def test_error_is_recorded_and_reraised(monkeypatch, live):
    """A failing build ends the live record as error, with the exception text, and re-raises."""
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == [("error", {"error": "ValueError: boom"})]


def test_interrupt_is_recorded_as_interrupted(monkeypatch, live):
    """Ctrl-C is recorded as interrupted, distinct from an error, and still propagates."""
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == [("interrupted", {})]


def test_system_exit_is_not_recorded(monkeypatch, live):
    """SystemExit is a deliberate exit, not a failure, so nothing is recorded."""
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(SystemExit("nothing")))
    with pytest.raises(SystemExit):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == []


def test_build_keeps_its_keyword_call_shape(monkeypatch):
    """build() maps its arguments onto IngestJob in order and rejects unknown keywords."""
    seen = []
    monkeypatch.setattr(ingest, "run", lambda job: seen.append(job) or 0)
    assert ingest.build("t", None, True, None, surface="s", root_override="/r") == 0
    assert seen == [ingest.IngestJob("t", None, True, None, "s", "/r")]
    with pytest.raises(TypeError):
        ingest.build("t", None, True, None, bogus=1)
