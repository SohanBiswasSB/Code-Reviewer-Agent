"""Pins two handlers that stopped catching `BaseException` / `Exception`.

`JsonlTracer.span` now classifies with try/finally + `sys.exception()`. Inside a
`finally`, `sys.exception()` also returns an exception the CALLER is handling, so
a span that succeeded inside an unrelated `except` block would be recorded as an
error unless the span tracks completion itself. The first tests pin that.

`PRCommentSink` now catches the `LABEL_ERRORS` family after the comment posts,
not everything: an expected label failure is still a partial success, while a
bug in the label path propagates.
"""
import json

import pytest

from adapters.action_sinks import PRCommentSink
from adapters.tracer_jsonl import JsonlTracer
from tools import github_client


def _rows(tracer):
    with open(tracer.path_for(), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_a_span_that_succeeds_inside_an_unrelated_except_is_ok(tmp_path):
    tracer = JsonlTracer(root=str(tmp_path))
    try:
        raise KeyError("the caller's own, already handled")
    except KeyError:
        with tracer.span("node.classify"):
            pass
    (row,) = _rows(tracer)
    assert row["status"] == "ok" and "error" not in row


def test_a_keyboard_interrupt_is_still_a_span_and_still_raises(tmp_path):
    tracer = JsonlTracer(root=str(tmp_path))
    with pytest.raises(KeyboardInterrupt), tracer.span("node.diagnose"):
        raise KeyboardInterrupt("ctrl-c")
    (row,) = _rows(tracer)
    assert row["status"] == "error" and row["error"].startswith("KeyboardInterrupt")


@pytest.fixture
def posted(monkeypatch):
    monkeypatch.setattr(github_client, "post_comment",
                        lambda url, body, token=None: {"body": body})


@pytest.mark.parametrize("exc", [PermissionError("403"), RuntimeError("403"),
                                 NotImplementedError("stub"), ValueError("bad key")])
def test_an_expected_label_failure_is_a_partial_success(posted, monkeypatch, exc):
    def fail(url, label, token=None):
        raise exc
    monkeypatch.setattr(github_client, "add_label", fail)
    out = PRCommentSink().emit({"comment": "posted", "label": "bug"}, target="u")
    assert out["comment"] == {"body": "posted"} and "skipped" in out["label"]


def test_a_bug_in_the_label_path_propagates(posted, monkeypatch):
    def broken(url, label, token=None):
        raise TypeError("a bug, not a provider failure")
    monkeypatch.setattr(github_client, "add_label", broken)
    with pytest.raises(TypeError):
        PRCommentSink().emit({"comment": "posted", "label": "bug"}, target="u")
