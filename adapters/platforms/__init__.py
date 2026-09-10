"""Import every platform so it self-registers. Add a target (rocm, xpu) by
dropping a module beside these and importing it here — no other file changes.
"""
from . import (  # noqa: F401  (self-register)
    base,  # noqa: F401  (shared model profiles)
    cpu,
    cuda,
    mac,
    studio,
)
