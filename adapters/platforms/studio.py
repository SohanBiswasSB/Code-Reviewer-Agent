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
even with T12 closed: the InputGuard's redaction is pattern-based, so treat
anything crossing the wire as sensitive.
"""
from __future__ import annotations

import os

from adapters.platforms import base
from registry import adapter

MEMORY_BUDGET_GB = 48.0     # of 64 GB unified, after the wired-limit bump
KV_RESERVE_GB = 6.0         # 32k context on a 30B-class MoE, f16
KV_RESERVE_GB_SMALL = 3.0   # the 7-12B profiles need far less
LARGE_MODEL_GB = 15.0       # diagnosis weights at/above this get the large reserve


@adapter("platform", "studio")
class StudioPlatform(base.PlatformBase):
    """The remote 64 GB Mac Studio; offers every profile. Never auto-detected."""

    name = "studio"
    # 64 GB holds every profile, including the two sized for the 24 GB laptop.
    # Kept available so `studio` can be A/B'd against them on one machine —
    # which is the only way to find out whether the bigger model actually helps.
    profiles = ("portable", "mac_resident", "mac_max", "studio")

    def _kv_for(self, models: base.ModelSet) -> float:
        """KV reserve from the diagnosis model's SIZE, never the profile's name --
        the mistake mac.py's `_context_for` documents fixing. A 30B-class model
        needs the large reserve whatever profile it sits in."""
        big = models.weights_gb.get("diagnosis", 0.0) >= LARGE_MODEL_GB
        return KV_RESERVE_GB if big else KV_RESERVE_GB_SMALL

    def runtime(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        """The `Platform` port's HOW-to-run half for `profile`, split across two hosts.

        Agent-side keys (`reranker_device`, `threads`) describe the machine running
        the graph; the rest describe the Studio (module docstring). 32k context at
        every profile, quant read from the profile's `quant`, KV reserve sized from
        the diagnosis model's weights, residency computed against 48 GB.
        """
        models = base.resolve(profile, self.profiles)
        kv_gb = self._kv_for(models)
        residency, accounting = base.plan_residency(models, MEMORY_BUDGET_GB, kv_gb)
        cpus = os.cpu_count() or 8
        return {
            # ── agent-side: never runs on the inference host ─────────────────
            "reranker_device": "mps",
            "threads": max(1, cpus - 4),
            # ── inference-host-side ──────────────────────────────────────────
            "llm_backend": "mlx",            # LM Studio MLX runtime, not llama.cpp
            "quant": models.quant,           # what weights_gb was computed at
            "lms_format_flag": "--mlx",
            "context": 32768,                # the budget affords it at every profile
            "kv_cache_dtype": "f16",
            "residency": residency,          # computed, see `memory`
            "gpu_layers": -1,
            "flash_attention": True,
            "memory": accounting,
        }

