"""A REMOTE Apple Silicon inference host — the 64 GB Mac Studio.

Selected explicitly with `CRA_PLATFORM=studio`. `detect_platform()` will never
return it, and that is not an oversight: the models run on a different machine
from the graph, and nothing local can detect a box reached over the network.

## This object describes TWO machines, and says which key belongs to which

`runtime()` was designed when the models ran on the same machine as the agent, so
its keys quietly assume one computer. With a remote host they split, and glossing
over that would put numbers in `doctor` that describe the wrong hardware:

  · INFERENCE HOST (this Studio) — llm_backend, quant, lms_format_flag, context,
    kv_cache_dtype, residency, memory, gpu_layers
  · AGENT HOST (whatever runs the graph) — reranker_device, threads

The reranker is the honest example. It is in-process `sentence-transformers`, not
an LM Studio call — no LM Studio or MLX server exposes a rerank endpoint — so it
runs wherever the agent runs and never touches this box. `mps` below assumes the
agent also runs on Apple Silicon; change it per agent machine when that stops
being true.

Splitting `Platform` into two ports is the real fix. Not worth doing until a
second remote topology exists to generalise from, so the split is documented here
rather than pretended away.

## The memory budget

48 GB of the 64. macOS caps GPU-wired memory near 75% of unified memory by
default; raise it with `sudo sysctl iogpu.wired_limit_mb=49152` and leave the
remainder for macOS, LM Studio itself and the page cache.

As with mac.py, these numbers are REASONED, not measured. `doctor` prints the
arithmetic so the first real run either confirms them or shows exactly which
figure was wrong.

## Reaching it

LM Studio has no authentication and the adapter's API key is an ignored
placeholder, so a public IP here is an open inference endpoint on your hardware.
Put it on a tailnet and point LMSTUDIO_BASE_URL at that address. This matters
more than usual while T12 (InputGuard) is open: nothing yet redacts PII or
secrets out of diffs and log lines before they cross the wire.
"""
from __future__ import annotations

import os

from adapters.platforms import base
from registry import adapter

MEMORY_BUDGET_GB = 48.0     # of 64 GB unified, after the wired-limit bump
KV_RESERVE_GB = 6.0         # 32k context on a 30B-class MoE, f16
KV_RESERVE_GB_SMALL = 3.0   # the 7-12B profiles need far less


@adapter("platform", "studio")
class StudioPlatform:
    name = "studio"
    # 64 GB holds every profile, including the two sized for the 24 GB laptop.
    # Kept available so `studio` can be A/B'd against them on one machine —
    # which is the only way to find out whether the bigger model actually helps.
    profiles = ("portable", "mac_resident", "mac_max", "studio")

    def _kv_for(self, profile: str) -> float:
        return KV_RESERVE_GB if profile == "studio" else KV_RESERVE_GB_SMALL

    def _quant_for(self, profile: str) -> str:
        """MLX naming. Only STUDIO's weights_gb were computed at 6-bit."""
        return "6bit" if profile == "studio" else "4bit"

    def runtime(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        models = base.resolve(profile, self.profiles)
        kv_gb = self._kv_for(profile)
        residency, accounting = base.plan_residency(models, MEMORY_BUDGET_GB, kv_gb)
        cpus = os.cpu_count() or 8
        return {
            # ── agent-side: never runs on the inference host ─────────────────
            "reranker_device": "mps",
            "threads": max(1, cpus - 4),
            # ── inference-host-side ──────────────────────────────────────────
            "llm_backend": "mlx",            # LM Studio MLX runtime, not llama.cpp
            "quant": self._quant_for(profile),
            "lms_format_flag": "--mlx",
            "context": 32768,                # the budget affords it at every profile
            "kv_cache_dtype": "f16",
            "residency": residency,          # computed, see `memory`
            "gpu_layers": -1,
            "flash_attention": True,
            "memory": accounting,
        }

    def models(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        return base.resolve(profile, self.profiles).as_dict()
