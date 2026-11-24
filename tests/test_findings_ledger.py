"""app/findings_ledger.py: what the precheck told you last time, and what changed since."""
from __future__ import annotations

import pytest

from app.findings_ledger import DISMISSED, FIXED, LEFT, OPEN, Ledger, check_id, same_ai_finding


@pytest.fixture(name="ledger")
def fixture_ledger(tmp_path, monkeypatch):
    """An empty ledger stored under a throwaway data dir."""
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path))
    return Ledger("t")


def flag(check="size_limits", file="a.py", line=10, message="f has 18 locals (limit 15)"):
    """One computed check flag, shaped as tools/change_checks.py returns it."""
    return {"check": check, "file": file, "line": line, "message": message, "status": "flag"}


def test_the_id_ignores_the_line_and_the_numbers_in_the_message():
    """An edit above a finding, or a partial fix, must not turn it into a new one."""
    assert check_id(flag(line=10)) == check_id(flag(line=40, message="f has 16 locals (limit 15)"))
    assert check_id(flag()) != check_id(flag(file="b.py"))
    assert check_id(flag()) != check_id(flag(check="line_length"))


def test_first_run_is_new_second_is_still_open(ledger):
    """The same flag on the next run is reported as seen before, not as new."""
    first = ledger.reconcile([flag()], [], ["a.py"])
    assert len(first.new) == 1 and not first.still_open
    second = ledger.reconcile([flag()], [], ["a.py"])
    assert not second.new and second.still_open[0]["runs"] == 2


def test_a_flag_gone_from_a_file_still_in_the_change_is_fixed(ledger):
    """Absent while its file is still being pushed means the user fixed it."""
    ledger.reconcile([flag()], [], ["a.py"])
    report = ledger.reconcile([], [], ["a.py"])
    assert [f["id"] for f in report.fixed] == [check_id(flag())]
    assert ledger.findings[check_id(flag())]["status"] == FIXED and not report.to_fix


def test_a_flag_whose_file_left_the_change_is_not_called_fixed(ledger):
    """Merged upstream or reverted: it is no longer in the push, which is not a fix."""
    ledger.reconcile([flag()], [], ["a.py"])
    report = ledger.reconcile([], [], ["other.py"])
    assert report.left_change and not report.fixed
    assert ledger.findings[check_id(flag())]["status"] == LEFT


def test_a_fixed_flag_that_comes_back_is_new_again(ledger):
    """A regression is something to fix, not history."""
    ledger.reconcile([flag()], [], ["a.py"])
    ledger.reconcile([], [], ["a.py"])
    report = ledger.reconcile([flag()], [], ["a.py"])
    assert report.new and ledger.findings[check_id(flag())]["status"] == OPEN


def test_a_dismissed_flag_is_never_counted_again(ledger):
    """The user judged it; it is listed as dismissed and does not block."""
    ledger.reconcile([flag()], [], ["a.py"])
    ledger.dismiss(check_id(flag()), "the renderer adds the envelope")
    report = ledger.reconcile([flag()], [], ["a.py"])
    assert not report.to_fix and report.dismissed[0]["status"] == DISMISSED


def test_a_dismissal_needs_a_reason(ledger):
    """An unexplained dismissal is a suppression."""
    ledger.reconcile([flag()], [], ["a.py"])
    with pytest.raises(ValueError):
        ledger.dismiss(check_id(flag()), "  ")


def test_undismiss_puts_it_back(ledger):
    """Lifting a dismissal lets the next run judge the finding again."""
    ledger.reconcile([flag()], [], ["a.py"])
    ledger.dismiss(check_id(flag()), "why")
    ledger.undismiss(check_id(flag()))
    assert ledger.reconcile([flag()], [], ["a.py"]).still_open


def test_a_flag_without_a_file_is_fixed_when_it_disappears(ledger):
    """'git author has no tracker' is about the change, not a path."""
    ledger.reconcile([flag(check="tracker", file="", line=0, message="no tracker")], [], [])
    assert ledger.reconcile([], [], ["a.py"]).fixed


def test_model_findings_are_advice_and_a_dismissed_one_matches_reworded(ledger):
    """Model findings never block; a dismissed one stays hidden though reworded."""
    ai = [{"file": "a.py", "line": 3, "issue": "the empty cell handling is missing in parse rows"}]
    report = ledger.reconcile([], ai, ["a.py"])
    assert report.ai and not report.to_fix
    ledger.record_run("x...HEAD", ["a.py"], report)
    ledger.dismiss(report.ai[0]["id"], "handled at csv_objectives.py:98")
    again = [{"file": "a.py", "line": 5, "issue": "empty cell handling is missing in parse rows"}]
    later = ledger.reconcile([], again, ["a.py"])
    assert not later.ai and later.ai_dismissed


def test_unrelated_model_findings_do_not_match():
    """Different file, or a different claim, is a different finding."""
    a = {"file": "a.py", "issue": "the cache key ignores the tenant"}
    assert not same_ai_finding(a, {**a, "file": "b.py"})
    assert not same_ai_finding(a, {"file": "a.py", "issue": "missing docstring on handler"})


def test_the_ledger_survives_a_save_and_load(ledger):
    """Dismissals are the user's decisions: they must persist."""
    ledger.reconcile([flag()], [], ["a.py"])
    ledger.dismiss(check_id(flag()), "why")
    ledger.data["root"] = "/repo"
    ledger.save()
    back = Ledger.load("t")
    assert back.findings[check_id(flag())]["status"] == DISMISSED and back.data["root"] == "/repo"


def test_an_unknown_id_is_an_error(ledger):
    """A typo in an id must not silently dismiss nothing."""
    with pytest.raises(KeyError):
        ledger.dismiss("deadbeef", "why")
    with pytest.raises(KeyError):
        ledger.undismiss("deadbeef")
