"""tools/calibrate.py: the cut it suggests must separate what it is told to."""
from __future__ import annotations

import json
import sys

import pytest

from tools import calibrate
from tools.calibrate import suggest


def test_a_clean_separation_is_found():
    out = suggest(pos=[0.70, 0.72, 0.75, 0.80, 0.81], neg=[0.40, 0.45, 0.50, 0.52])
    assert 0.52 < out["youden_cut"] <= 0.70 and out["youden_j"] == 1.0
    assert out["negatives_passing_that_cut"] == 0.0


def test_too_little_data_says_so_rather_than_guessing():
    assert "note" in suggest(pos=[0.5], neg=[])


def test_a_failed_measurement_is_recorded_and_still_raises(tmp_path, monkeypatch):
    """The live job must say WHY it stopped -- without catching the exception:
    main() uses try/finally + sys.exception(), never `except Exception`."""
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path))

    def broken(args, progress):
        raise LookupError("no such tenant")
    monkeypatch.setattr(calibrate, "_measure", broken)
    monkeypatch.setattr(sys, "argv", ["calibrate", "relevance", "--tenant", "nope"])
    with pytest.raises(LookupError):
        calibrate.main()
    job = json.loads(next((tmp_path / "live").glob("job-*.json")).read_text())
    assert job["status"] == "error" and "LookupError: no such tenant" in job["error"]


def test_a_success_inside_a_callers_except_block_is_still_finished(tmp_path, monkeypatch):
    """Inside `finally`, sys.exception() also returns an exception the CALLER is
    handling. Read unguarded, a measurement that succeeded there was ended as
    "error" with the caller's unrelated exception as its reason."""
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(calibrate, "_measure", lambda args, progress: {"youden_cut": 0.7})
    monkeypatch.setattr(sys, "argv", ["calibrate", "relevance", "--tenant", "t"])
    try:
        raise ValueError("the caller's own problem")
    except ValueError:
        assert calibrate.main() == 0
    job = json.loads(next((tmp_path / "live").glob("job-*.json")).read_text())
    assert job["status"] == "finished" and "error" not in job
