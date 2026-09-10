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
