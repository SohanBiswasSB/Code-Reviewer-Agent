"""The multi-tenant demo is a claim README.md tells humans to run. Check it here
too, not only in check.py, so a plain `pytest` cannot go green while it lies."""
from __future__ import annotations


def test_the_multitenant_demo_shows_its_own_point(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    from eval import demo_multitenant
    assert demo_multitenant.main() == 0
