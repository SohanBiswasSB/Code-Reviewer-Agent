"""The reviewer's exceptions, and the error FAMILIES each kind of boundary expects.

## Why named families instead of `except Exception`

A broad `except Exception` catches the failure it was written for AND every bug
that happens to raise nearby -- a `KeyError` from a typo, an `AttributeError` from
a refactor. Those then vanish into a "best-effort" path and the suite stays green.
That is the shape this project keeps finding (T5, T21, the inert log surface).

So every handler names what it expects. The tuples below are the vocabulary:
a boundary that writes to disk expects `WRITE_ERRORS`, one that resolves config
expects `CONFIG_ERRORS`, and so on. Anything outside the family is a bug, and a
bug propagates -- the activity log (app/activity.py) records it with its
traceback, so propagating is loud, not lost.

## Why each class also inherits a builtin

`EvidenceError` is a `ValueError`, `StorageError` an `OSError`, and so on. Callers
written before this module (`except ValueError`, `pytest.raises(ValueError)`)
keep working, and new callers can be precise.
"""
from __future__ import annotations

import csv
import subprocess


class ReviewerError(Exception):
    """Base for every error this service raises on purpose."""


class ConfigError(ReviewerError, ValueError):
    """Configuration that cannot work: a missing namespace, an unknown profile,
    a policy value of the wrong shape. Fix the config; retrying will not help."""


class EvidenceError(ReviewerError, ValueError):
    """The change or log window could not be read (bad git target, no checkout)."""


class ProviderError(ReviewerError, ConnectionError):
    """A model or embedding server failed, answered wrongly, or is unreachable."""


class StorageError(ReviewerError, OSError):
    """A durable store (index, audit, trace) is unreadable, corrupt or mismatched."""


class IndexMismatchError(StorageError, ValueError):
    """An index was built by a different embedder than the one about to query it."""


# ── error families, by boundary ──────────────────────────────────────────────
# Writing a JSON record to disk: the disk fails (OSError) or a value will not
# serialise (TypeError / ValueError).
WRITE_ERRORS: tuple[type[BaseException], ...] = (OSError, TypeError, ValueError)

# Resolving configuration: a registry miss (KeyError -> LookupError), a bad or
# unsupported value (ValueError), a missing or unreadable file (OSError), or a
# platform module that will not import.
CONFIG_ERRORS: tuple[type[BaseException], ...] = (LookupError, ValueError, OSError,
                                                   ImportError)

# Reading and parsing a file from a repo: I/O, decoding, Python that does not
# parse, CSV/JSON that does not parse, a git subprocess failing, or source nested
# deeply enough to exhaust the AST recursion limit.
PARSE_ERRORS: tuple[type[BaseException], ...] = (OSError, UnicodeError, ValueError,
                                                  SyntaxError, csv.Error, LookupError,
                                                  subprocess.SubprocessError, RecursionError)
