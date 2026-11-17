"""`tools.ingest.run` marks the live record only when the job itself failed.

The failure is recorded from `finally`, and `sys.exception()` there ALSO returns
an exception the caller happens to be handling -- so a successful ingest run
inside an unrelated `except` block must not be recorded as an error.
"""
from __future__ import annotations

import pathlib

import pytest

from tools import ingest


class FakeLive:
    """Stands in for IngestProgress: remembers every `end` call."""

    def __init__(self, tenant, namespace):
        self.ends = []
        self.data = {}

    def end(self, status, **extra):
        self.ends.append((status, extra))


@pytest.fixture
def live(monkeypatch):
    holder = {}

    def make(tenant, namespace):
        holder["live"] = FakeLive(tenant, namespace)
        return holder["live"]

    monkeypatch.setattr(ingest, "IngestProgress", make)
    monkeypatch.setattr(ingest, "_resolve_target",
                        lambda job: ingest._Target("org/x", pathlib.Path("."), []))
    return holder


def _logged_build_raising(exc):
    def fake(live, job, target):
        raise exc
    return fake


def test_success_inside_an_unrelated_except_records_nothing(monkeypatch, live):
    monkeypatch.setattr(ingest, "_logged_build", lambda live, job, target: 0)
    try:
        raise KeyError("unrelated, handled by the caller")
    except KeyError:
        assert ingest.run(ingest.IngestJob("t")) == 0
    assert live["live"].ends == []


def test_error_is_recorded_and_reraised(monkeypatch, live):
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == [("error", {"error": "ValueError: boom"})]


def test_interrupt_is_recorded_as_interrupted(monkeypatch, live):
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == [("interrupted", {})]


def test_system_exit_is_not_recorded(monkeypatch, live):
    monkeypatch.setattr(ingest, "_logged_build", _logged_build_raising(SystemExit("nothing")))
    with pytest.raises(SystemExit):
        ingest.run(ingest.IngestJob("t"))
    assert live["live"].ends == []


def test_build_keeps_its_keyword_call_shape(monkeypatch):
    seen = []
    monkeypatch.setattr(ingest, "run", lambda job: seen.append(job) or 0)
    assert ingest.build("t", None, True, None, surface="s", root_override="/r") == 0
    assert seen == [ingest.IngestJob("t", None, True, None, "s", "/r")]
    with pytest.raises(TypeError):
        ingest.build("t", None, True, None, bogus=1)
