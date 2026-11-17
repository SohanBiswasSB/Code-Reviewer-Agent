"""Session-wide test setup: isolation from ./var, offline models, real surfaces.

The project root reaches sys.path through pytest.ini's `pythonpath = .`, so the
imports below are ordinary top-of-file imports.
"""
import json
import os
import time
from pathlib import Path

import pytest

import config
import registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _surfaces_registered():
    """The real surfaces are registered before any test overrides one.

    Importing `config` imports surfaces/, whose decorators register the real
    classes. If that first happened lazily INSIDE a run, it would re-register the
    real CodeReviewSurface over a test's stub, and the test would exercise the
    real surface while appearing to use the stub (test_graph_resume.py). Every
    test module used to guard this with its own silenced `import config`; doing it
    once here, and asserting it, makes the guarantee explicit.
    """
    missing = set(config.enabled_surfaces()) - set(registry.list_registered()["surfaces"])
    assert not missing, f"enabled surfaces never registered: {sorted(missing)}"


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Never let a test write audit records or reports into the real ./var.

    Autouse and unconditional on purpose. The audit sink defaults to a REAL
    writer (silence has to be opted into, see config.get_audit_sink), so any test
    that drives the graph would otherwise leave records in the working tree — and
    the first person to notice would be whoever later tried to analyse them.
    """
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path / "var"))


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    """No test may reach a real model server. Autouse, for the same reason as above.

    Since 2026-09-11 the LLM nodes resolve their model from `config.get_llm(role)`
    like every other provider, so omitting an `llm=` argument no longer means
    "use a stub" -- it means "use whatever LLM_PROVIDER points at". On a laptop
    with LM Studio running, the graph tests happily started making live calls:
    slow, non-deterministic, and dependent on which models happen to be loaded.

    A test that needs the real adapter (to assert its error messages, or its
    per-role temperature) sets LLM_PROVIDER itself; monkeypatch makes that win.
    """
    monkeypatch.setenv("LLM_PROVIDER", "stub")


@pytest.fixture(autouse=True)
def _offline_retriever(monkeypatch):
    """The demo retriever, declared -- for the same reason as `_offline_llm`.

    Since T10 the default retriever is the real namespaced index, which embeds
    every query through LM Studio. A test that wants the indexed retriever builds
    one with a fake embedder and a temporary index (tests/test_index.py).
    """
    monkeypatch.setenv("RETRIEVER", "hybrid")
    # And no cross-encoder: an IndexedRetriever built in a test without an
    # explicit `reranker=` would otherwise load a 2.3 GB model (T13).
    monkeypatch.setenv("RERANKER", "none")


LIVE_MARKERS = ("needs_server", "needs_model")


def pytest_collection_modifyitems(config, items):
    """Skip live-model tests unless `-m` asks for them by name.

    pytest.ini has always said needs_server tests are "deselected by default",
    and nothing did the deselecting -- an unchecked claim. Now it is checked:
    `pytest -m needs_model` runs them, a bare `pytest` skips them with a reason.
    """
    selected = config.getoption("-m") or ""
    for item in items:
        for marker in LIVE_MARKERS:
            if marker in item.keywords and marker not in selected:
                item.add_marker(pytest.mark.skip(
                    reason=f"{marker}: run with  pytest -m {marker}"))


# ── T28: an intermittent failure must leave its NAME behind ─────────────────
FAILURE_LOG = ROOT / "var" / "test_failures.jsonl"


def pytest_runtest_logreport(report):
    """Append every failed test to var/test_failures.jsonl, with the load average.

    T28 is a test that failed twice under load and passed on rerun -- and both
    times its name scrolled away. A terminal is not a record. `var/` is never
    committed, and this writes test ids and assertion text only.
    """
    if not report.failed:
        return
    try:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAILURE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "test": report.nodeid,
                "phase": report.when, "load": os.getloadavg(),
                "strict_msgpack": os.environ.get("LANGGRAPH_STRICT_MSGPACK"),
                "error": str(report.longrepr)[-2000:]}) + "\n")
    except OSError:
        pass    # the failure itself is still reported by pytest; this is a copy
