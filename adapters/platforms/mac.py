"""Apple Silicon. Calibrated for an M4 Pro (14-core) with 24 GB unified memory.

NOT YET EXERCISED on the Mac — the numbers below come from the M4 Pro's known
limits, not from measurement. Same honesty rule as docs/ROADMAP.md: represent
designed-but-unproven things as exactly that. `doctor` prints the arithmetic so
the first real run either confirms it or shows exactly which figure was wrong.

Decisions specific to this machine:
  · MEMORY_BUDGET_GB = 18.0 of the 24. LM Studio's own guardrail lands near
    there, and macOS plus the app need real headroom. Weights are only part of
    the cost — KV cache at 32k is reserved separately (KV_RESERVE_GB).
  · MLX quants, not GGUF. Noticeably better prefill and decode on Apple Silicon.
    MLX does not exist on the CUDA box, which is precisely why the quant format
    lives here in runtime() and not in the shared model profile.
  · Residency is COMPUTED, not declared. `mac_resident` (~12 GB hot) stays
    loaded; `mac_max` (~18 GB hot) cannot co-reside with its judge and resolves
    to jit_swap. Declaring "resident" for both was the bug this replaces.
  · Context drops for mac_max — a 15.6 GB model plus a 32k KV cache would blow
    the budget the weights already nearly fill.
  · torch on `mps` for the reranker: unified memory means no VRAM contention
    with LM Studio at this size. Watch the libomp collision though — torch and
    another OpenMP-shipping wheel in one process abort on macOS.
"""
from __future__ import annotations

import os

from adapters.platforms import base
from registry import adapter

MEMORY_BUDGET_GB = 18.0     # of 24 GB unified; leaves macOS + LM Studio headroom
KV_RESERVE_GB = 3.0         # ~32k context, f16
KV_RESERVE_GB_LARGE = 2.0   # reduced-context profiles


@adapter("platform", "mac")
class MacPlatform:
    name = "mac"
    profiles = ("portable", "mac_resident", "mac_max")

    def _context_for(self, profile: str) -> tuple[int, float]:
        """Large-weight profiles trade context for the memory the weights need."""
        if profile == "mac_max":
            return 16384, KV_RESERVE_GB_LARGE
        return 32768, KV_RESERVE_GB

    def runtime(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        models = base.resolve(profile, self.profiles)
        context, kv_gb = self._context_for(profile)
        residency, accounting = base.plan_residency(models, MEMORY_BUDGET_GB, kv_gb)
        cpus = os.cpu_count() or 14
        return {
            "reranker_device": "mps",     # unified memory: no VRAM contention
            "llm_backend": "mlx",         # LM Studio MLX runtime, not llama.cpp
            "quant": "4bit",              # MLX naming, not GGUF's Q4_K_M
            "lms_format_flag": "--mlx",   # what `lms get` needs on this machine
            "context": context,
            "kv_cache_dtype": "f16",
            "residency": residency,       # computed, see `memory` below
            "gpu_layers": -1,
            "threads": max(1, cpus - 4),  # leave the efficiency cores alone
            "flash_attention": True,
            "memory": accounting,
        }

    def models(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        return base.resolve(profile, self.profiles).as_dict()
