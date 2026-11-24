"""The multi-tenant demo is a claim README.md tells humans to run. Check it here
too, not only in check.py, so a plain `pytest` cannot go green while it lies."""
from __future__ import annotations

from eval import demo_multitenant


def test_the_multitenant_demo_shows_its_own_point(monkeypatch):
    """Exit 0 only if one diagnosis approves when grounded and escalates when not."""
    monkeypatch.setenv("RETRIEVER", "hybrid")
    assert demo_multitenant.main() == 0
