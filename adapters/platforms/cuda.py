"""NVIDIA / CUDA desktop. The machine this is calibrated against: GTX 1060 6 GB
(Pascal, compute 6.1), 16 GB system RAM, Ryzen 7 1700.

Three hardware facts drive every number below:
  · 6 GB VRAM cannot hold two 7B-class models at once (~9 GB of weights). But
    the graph is strictly sequential — generate_diagnosis fully completes before
    independent_review starts — so residency is "jit_swap": LM Studio loads one,
    unloads it, loads the other. The few seconds cost nothing behind a gate that
    then waits on a human.
  · Pascal has no usable FP16 compute path, so Q4_K_M/Q5_K_M quantised kernels
    are the fast route; f16 KV cache is also wasteful here, hence q8_0.
  · 8k context, not 32k. KV cache competes with weights for the same 6 GB, and
    retrieval is meant to keep context tight anyway (docs/BEST_PRACTICES.md §2.2).
"""
from __future__ import annotations
import os
from registry import adapter
from adapters.platforms import base


@adapter("platform", "cuda")
class CudaPlatform:
    name = "cuda"
    # 6 GB cannot hold mac_headroom's ~15 GB of weights. Not offered.
    profiles = ("portable",)

    def runtime(self) -> dict:
        cpus = os.cpu_count() or 8
        return {
            # NOT "cuda". LM Studio already owns all 6 GB for the diagnosis
            # model, so loading the 568M cross-encoder beside it risks an OOM
            # mid-generation. Reranking a top-100 set on CPU costs a second or
            # two, once per run, before any LLM call. Deliberate, not a TODO.
            "reranker_device": "cpu",
            "llm_backend": "cuda",        # LM Studio: llama.cpp CUDA runtime
            "quant": "Q4_K_M",            # GGUF; Pascal has no FP16 fast path
            "context": 8192,              # KV cache shares the 6 GB with weights
            "kv_cache_dtype": "q8_0",     # ~halves KV cost vs f16
            "residency": "jit_swap",      # cannot hold diagnosis + judge together
            "gpu_layers": -1,             # offload all layers that fit
            "threads": max(1, cpus // 2), # physical cores; SMT does not help llama.cpp
            "flash_attention": True,
        }

    def models(self, profile: str = base.DEFAULT_PROFILE) -> dict:
        return base.resolve(profile, self.profiles).as_dict()
