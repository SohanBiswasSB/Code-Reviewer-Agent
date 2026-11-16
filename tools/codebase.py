"""Where a tenant's code lives on this machine. Stdlib only, imported by both sides.

A module of its own, rather than a function in `config`, because its two callers
sit on opposite sides of an import cycle: `tools/ingest.py` (which builds the
index) and `adapters/source_git_local.py` (which reads the diff). `config`
imports every adapter at module level, so an adapter importing `config` back is
circular. Here, both import it at the top.
"""
from __future__ import annotations

import os
import pathlib


def codebase_root(codebase: dict, override: str | None = None) -> pathlib.Path:
    """Where a tenant's repo lives on THIS machine.

    `codebase.root` is committed and shared, so it cannot be right on every
    machine -- milesone's is a container path, and the same repo sits under ~/Dev
    on the Mac. Precedence: an explicit override, then `CRA_CODEBASE_ROOT`, then
    the policy. One function, so ingest (which builds the index) and the
    git_local source (which reads the diff) can never disagree about which
    checkout they are looking at.
    """
    raw = override or os.getenv("CRA_CODEBASE_ROOT") or (codebase or {}).get("root") or ""
    return pathlib.Path(raw).expanduser()
