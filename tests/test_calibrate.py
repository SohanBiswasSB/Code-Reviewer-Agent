"""tools/calibrate.py: the cut it suggests must separate what it is told to."""
from __future__ import annotations

from tools.calibrate import suggest


def test_a_clean_separation_is_found():
    out = suggest(pos=[0.70, 0.72, 0.75, 0.80, 0.81], neg=[0.40, 0.45, 0.50, 0.52])
    assert 0.52 < out["youden_cut"] <= 0.70 and out["youden_j"] == 1.0
    assert out["negatives_passing_that_cut"] == 0.0


def test_too_little_data_says_so_rather_than_guessing():
    assert "note" in suggest(pos=[0.5], neg=[])
