"""What kind of file a path is, and so how much of it a review should spend on.

Shared by the git source (which packs the diff into a budget by this rank) and
the code-review surface (which builds one retrieval query per changed file in the
same order). One definition, so "what matters most in this change" cannot mean
two different things in two places.
"""
from __future__ import annotations

import pathlib

DATA_SUFFIXES = {".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".jsonl", ".ndjson",
                 ".sqlite", ".db", ".pkl", ".npy", ".npz"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}
CONFIG_SUFFIXES = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json", ".lock", ".env"}
CODE, CONFIG, DOCS, DATA = 0, 1, 2, 3


def file_rank(path: str) -> int:
    """0 code · 1 config · 2 docs · 3 data (content never shown to a model)."""
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    if suffix in DATA_SUFFIXES:
        return DATA
    if suffix in DOC_SUFFIXES:
        return DOCS
    if suffix in CONFIG_SUFFIXES:
        return CONFIG
    return CODE
