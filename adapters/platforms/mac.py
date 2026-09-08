"""Apple Silicon. Calibrated against an M4 Pro (14-core) with 24 GB unified memory.

NOT YET EXERCISED — written from the M4 Pro's known limits, but every number
here is unverified until it runs on the Mac. Treat as a starting point, not a
measurement. (Same honesty rule as docs/ROADMAP.md: represent designed-but-
unproven things as exactly that.)

Notes specific to this machine:
  · LM Studio's default guardrail allows roughly 18 GB of the 24 for models, so
    `portable` (~7 GB) leaves both models resident with room to spare, and
    `mac_headroom` (~15 GB) fits but with little margin for a long KV cache.
  · Prefer MLX quants over GGUF here — noticeably better prefill and decode on
    Apple Silicon. MLX does not exist on the CUDA box, which is precisely why
    the quant format lives in runtime() and not in the shared model profile.
  · Qwen3-Coder-30B-A3B is deliberately absent: ~17 GB alone leaves no room for
    a resident judge at 24 GB.
  · torch on `mps` for the reranker. Watch the libomp collision — torch and any
    other OpenMP-shipping wheel in one process aborts on macOS; that is what a
    `doctor --fix` step is for.
"""
from __future__ import annotations
import os
from registry import adapter
from adapters.platforms import base


@adapter("platform", "mac")
class MacPlatform:
    name = "mac"
    profiles = ("portable", "mac_headroom")

    def runtime(self) -> dict:
        cpus = os.cpu_count() or 10
        return {
            "reranker_device": "mps",     # unified memory: no VRAM contention
            "llm_backend": "mlx",         # LM Studio MLX runtime, not llama.cpp
            "quant": "4bit",              # MLX naming, not GGUF's Q4_K_M
            "context": 32768,             # unified memory affords a real context
            "kv_cache_dtype": "f16",
            "residency": "resident",      # both models stay loaded; no swap cost
            "gpu_layers": -1,
            "threads": max(1, cpus - 4),  # leave the efficiency cores alone
            "flash_attention": True,
        }

    def models(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        return base.resolve(profile, self.profiles).as_dict()
