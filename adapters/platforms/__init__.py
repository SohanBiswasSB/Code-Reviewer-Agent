"""Import every platform so it self-registers. Add a target (rocm, xpu) by
dropping a module beside these and importing it here — no other file changes.
"""
from . import base                       # noqa: F401  (shared model profiles)
from . import cuda, mac, cpu             # noqa: F401  (self-register)
