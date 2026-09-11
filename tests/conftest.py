"""Put the project root on sys.path so `import config` works from tests/."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
