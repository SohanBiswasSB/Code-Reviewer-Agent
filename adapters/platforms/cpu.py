"""CPU-only. What CI uses, and the honest fallback when no GPU is present.

Slow by definition — this exists so the pipeline RUNS everywhere (so the graph,
the gate and the adapters are all exercisable in CI), not so it runs fast.
"""
from __future__ import annotations

import os

from adapters.platforms import base
from registry import adapter

RAM_BUDGET_GB = 8.0
KV_RESERVE_GB = 0.5


@adapter("platform", "cpu")
class CpuPlatform:
    name = "cpu"
    profiles = ("portable",)

    def runtime(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        models = base.resolve(profile, self.profiles)
        residency, accounting = base.plan_residency(models, RAM_BUDGET_GB, KV_RESERVE_GB)
        cpus = os.cpu_count() or 4
        return {
            "reranker_device": "cpu",
            "llm_backend": "cpu",
            "quant": "Q4_K_M",
            "lms_format_flag": "--gguf",
            "context": 4096,              # KV cache is RAM here; keep it small
            "kv_cache_dtype": "f16",
            "residency": residency,
            "gpu_layers": 0,
            "threads": max(1, cpus // 2),
            "flash_attention": False,
            "memory": accounting,
        }

    def models(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        return base.resolve(profile, self.profiles).as_dict()
