"""Import surfaces so they self-register.

Each module's `@surface` decorator runs at import; `__all__` declares the modules
as this package's exports so the imports are deliberate, not unused.
"""
from . import code_review, log_review

__all__ = ["code_review", "log_review"]
