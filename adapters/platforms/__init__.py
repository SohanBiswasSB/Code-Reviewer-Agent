"""Import every platform so it self-registers. Add a target (rocm, xpu) by
dropping a module beside these and importing it here -- no other file changes.

`base` holds the shared model profiles; the others register a platform with
`@adapter` when imported. `__all__` declares them as this package's exports, so
the imports read as what they are -- deliberate -- rather than unused names.
"""
from . import (
    base,
    cpu,
    cuda,
    mac,
    studio,
)

__all__ = ["base", "cpu", "cuda", "mac", "studio"]
