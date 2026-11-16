"""Where the agent keeps what it writes: `CRA_DATA_DIR`, default `var/`.

One definition. The default was spelled out in eight modules, and one of them
said `./var` instead of `var` -- harmless today, and exactly the kind of drift
that later points the index and the audit store at two different directories.
Stdlib only and import-free, so any module (adapters included) can use it
without a circular import.
"""
from __future__ import annotations

import os
import pathlib

DEFAULT_DATA_DIR = "var"


def data_dir() -> pathlib.Path:
    """`$CRA_DATA_DIR`, else `var/` -- read at call time, so tests can point it."""
    return pathlib.Path(os.getenv("CRA_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()
