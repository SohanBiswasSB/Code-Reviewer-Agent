"""Apple Silicon. Calibrated for an M4 Pro (14-core) with 24 GB unified memory.

The budget numbers below come from the M4 Pro's known limits, not from
measurement. Same honesty rule as docs/ROADMAP.md: represent
designed-but-unproven things as exactly that. `doctor` prints the arithmetic so
a real run either confirms it or shows exactly which figure was wrong. Real
reviews have since run on this Mac (docs/DECISIONS.md §8), and one hit a Metal
OOM with both LLMs loaded at LM Studio's `parallel 4` -- a load setting this
arithmetic does not model; it was reloaded at `parallel 1`.

Decisions specific to this machine:
  · MEMORY_BUDGET_GB = 18.0 of the 24. LM Studio's own guardrail lands near
    there, and macOS plus the app need real headroom. Weights are only part of
    the cost — KV cache at 32k is reserved separately (KV_RESERVE_GB).
  · MLX quants, not GGUF. Noticeably better prefill and decode on Apple Silicon.
    MLX does not exist on the CUDA box, which is precisely why the quant format
    lives here in runtime() and not in the shared model profile.
  · Residency is COMPUTED, not declared. Declaring "resident" for every profile
    was the bug this replaces. Today all three profiles fit: `mac_resident`
    (~12 GB hot) and `mac_max` (~11.9 GB hot, since its diagnosis model changed
    to ministral-3-14b on 2026-09-11) both resolve to resident.
  · Context drops only when the weights need it (`_context_for`): 32k when the
    hot weights plus the 32k KV reserve fit the budget, else 16k. It used to
    drop for mac_max by name, back when that profile carried a 15.6 GB model;
    no current profile triggers it.
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
class MacPlatform(base.PlatformBase):
    """Apple Silicon running LM Studio's MLX runtime locally, beside the agent."""

    name = "mac"
    profiles = ("portable", "mac_resident", "mac_max")

    def _context_for(self, profile: str) -> tuple[int, float]:
        """Trade context for weights only when the weights actually need it.

        COMPUTED from the profile's hot weights, not matched on its name. It used
        to read `if profile == "mac_max": return 16384`, which is the same
        "declared rather than computed" mistake `plan_residency` exists to avoid
        (docs/DECISIONS.md section 3.3) -- and it broke the moment mac_max's
        diagnosis model changed: at 11.87 GB hot the profile is now LIGHTER than
        mac_resident, and was still being handed a halved context because of its
        name.

        A profile that fits the full KV reserve gets the full context; one that
        does not trades context for weights. Rename a profile, or change a model
        in it, and this stays correct on its own.
        """
        models = base.resolve(profile, self.profiles)
        if models.hot_gb + KV_RESERVE_GB <= MEMORY_BUDGET_GB:
            return 32768, KV_RESERVE_GB
        return 16384, KV_RESERVE_GB_LARGE

    def runtime(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        """The `Platform` port's HOW-to-run half for `profile` on this Mac.

        MLX 4-bit, f16 KV, context and residency both COMPUTED from the profile's
        hot weights against the 18 GB budget, and the reranker on mps. Raises (via
        `base.resolve`) for a profile this Mac does not offer (`studio`).
        """
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
