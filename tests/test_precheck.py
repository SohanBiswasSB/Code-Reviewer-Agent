"""app/precheck.py: the manual pre-push check -- what to fix, what is fixed, the verdict."""
from __future__ import annotations

import argparse

import pytest

from app import precheck
from app.findings_ledger import Ledger
from errors import ProviderError
from schemas.diagnosis import Diagnosis, Finding

SIZE = {"check": "size_limits", "status": "flag", "file": "a.py", "line": 9,
        "message": "f has 18 locals (limit 15)"}
SKIP = {"check": "manage_py_check", "status": "skip", "message": "needs a database"}


def state_with(checks, route="human_review", diagnosis=None):
    """A finished graph state, as precheck reads it."""
    return {"evidence": {"files": [{"filename": "a.py"}], "revision": {"head": "abc" * 5}},
            "_checks": checks, "route": route, "diagnosis": diagnosis}


@pytest.fixture(name="args")
def fixture_args(tmp_path, monkeypatch):
    """CLI args for a run against an explicit target, with data in a temp dir."""
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(precheck, "_has_changes", lambda target: True)
    return argparse.Namespace(tenant="t", root=str(tmp_path), base=None, target="x...HEAD",
                              dismiss=None, reason=None, undismiss=None, list=False)


def test_a_flag_is_to_fix_then_fixed_then_clear(args, monkeypatch, capsys):
    """The user's loop: told what to fix, fixes it, is told it is fixed and clear."""
    runs = iter([state_with([SIZE, SKIP]), state_with([SKIP])])
    monkeypatch.setattr(precheck, "_run_graph", lambda *a: (next(runs), None))
    assert precheck.run(args) == 1
    first = capsys.readouterr().out
    assert "TO FIX (1)" in first and "size_limits" in first and "VERDICT: 1 to fix" in first
    assert "NOT CHECKED here (1): manage_py_check" in first
    assert precheck.run(args) == 0
    second = capsys.readouterr().out
    assert "FIXED SINCE LAST RUN (1)" in second and "VERDICT: clear to push" in second


def test_the_root_is_remembered(args, monkeypatch):
    """Given once, the repo's path need not be typed again."""
    monkeypatch.setattr(precheck, "_run_graph", lambda *a: (state_with([]), None))
    precheck.run(args)
    assert Ledger.load("t").data["root"] == args.root


def test_model_findings_are_shown_but_never_block(args, monkeypatch, capsys):
    """Advice only: a clean check run is clear even with model findings."""
    diag = Diagnosis(findings=[Finding(file="a.py", line=3, issue="cache key ignores tenant")])
    monkeypatch.setattr(precheck, "_run_graph",
                        lambda *a: (state_with([], "ai_investigation", diag), None))
    assert precheck.run(args) == 0
    out = capsys.readouterr().out
    assert "cache key ignores tenant" in out and "clear to push" in out


def test_a_model_server_outage_still_reports_the_checks(args, monkeypatch, capsys):
    """The checks ran before any model call; an outage must not hide them."""

    class DownGraph:
        """A graph whose model step fails after the checks ran."""

        def invoke(self, *_a, **_k):
            """Fail as an exhausted gateway retry does."""
            raise ProviderError("completion: model server unavailable (HTTP 502)")

        def get_state(self, _cfg):
            """The checkpoint before the failing node."""
            return argparse.Namespace(values=state_with([SIZE], "ai_investigation"))

    monkeypatch.setattr(precheck, "build_graph", lambda **_k: DownGraph())
    assert precheck.run(args) == 1
    out = capsys.readouterr().out
    assert "AI review stopped: completion: model server unavailable" in out
    assert "size_limits" in out


def test_dismiss_from_the_command_line(args, monkeypatch, capsys):
    """Dismissed once, it is listed as dismissed and the verdict is clear."""
    monkeypatch.setattr(precheck, "_run_graph", lambda *a: (state_with([SIZE]), None))
    precheck.run(args)
    fid = next(iter(Ledger.load("t").findings))
    precheck.dismiss_cmd(argparse.Namespace(tenant="t", dismiss=fid, reason="accepted"))
    capsys.readouterr()
    assert precheck.run(args) == 0
    assert "DISMISSED, not counted (1)" in capsys.readouterr().out


def test_without_a_base_or_target_it_says_what_to_pass(args, monkeypatch):
    """A tenant with no precheck_base is told how to proceed, not guessed for."""

    no_base = argparse.Namespace(policy=lambda *_a: {})
    monkeypatch.setattr(precheck.config, "get_tenant_store", lambda: no_base)
    args.target = None
    with pytest.raises(SystemExit, match="precheck_base"):
        precheck.run(args)


def test_a_fully_merged_branch_is_nothing_to_check(args, monkeypatch, capsys):
    """No changes against the base is an answer, not a traceback."""
    monkeypatch.setattr(precheck, "_has_changes", lambda target: False)
    monkeypatch.setattr(precheck, "_run_graph", lambda *a: pytest.fail("must not run"))
    assert precheck.run(args) == 0
    assert "Nothing to check" in capsys.readouterr().out
    assert Ledger.load("t").data["root"] == args.root
