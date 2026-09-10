"""Model profiles — the part that is deliberately NOT hardware-specific.

A profile names one model per role. The same profile on two machines means the
same model ids, so a diagnosis produced on the Mac is comparable with one
produced on the CUDA desktop. Only the *runtime* (quant format, context, KV
dtype, residency) differs per platform; see adapters/platforms/<name>.py.

Model ids are LM Studio catalog names, NOT Hugging Face repo paths. LM Studio
resolves the format per machine (`--mlx` on Apple Silicon, `--gguf` on CUDA), so
one id covers both — which is exactly why the id can live in the shared profile
and the format cannot.

Two roles must never vary by machine, for correctness rather than taste:
  · EMBEDDING — a different embedder is a different vector space, so the index
    stops being portable AND every min_relevance_score in tenants/*/policy.json
    silently changes meaning.
  · RERANKER  — runs in-process (sentence-transformers), not via LM Studio, so
    it is CPU-identical everywhere by default. Keep it that way.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Pinned across every profile and every platform. Changing either invalidates
# the vector index and de-calibrates every tenant threshold. See docs/ROADMAP.md.
#
# BGE-M3 replaces nomic-embed-text-v1.5 (768 dims), and is changed NOW on purpose:
# T10 is still open, so there is no index to re-embed and the corpus is four
# documents. This is the cheapest moment the change will ever have. Once ingest
# exists it costs a full re-index PLUS a re-measurement of every tenant's
# min_relevance_score -- which is owed anyway, since the shipped 0.35/0.40 were
# calibrated for text-embedding-3-small@512 and mean nothing in any other space.
#
# NOT yet verified against `lms ls` on the Studio; the other ids here were.
# `llm_lmstudio.resolves_to()` matches loosely, so the catalog's probable
# "text-embedding-bge-m3" still resolves from this shorter id.
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIMS = 1024        # a DIFFERENT vector space from the 768-dim nomic model
EMBEDDING_GB = 1.2           # 568M params at f16; nomic was ~0.1 GB
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"    # in-process cross-encoder, never an LM Studio call


@dataclass(frozen=True)
class ModelSet:
    """One model per role, plus the weight cost that decides residency.

    `judge` MUST be a different family from `diagnosis`. That is not a style
    preference: a judge from the same family as the model it grades exhibits
    self-preference bias (docs/BEST_PRACTICES.md §7), which would quietly
    undermine independent_review's reason for existing.

    `weights_gb` is the ~4-bit on-disk size per role. It exists so a platform can
    COMPUTE whether both models fit in memory rather than declaring a residency
    it cannot honour — the bug that a 24 GB Mac holding a 15.6 GB diagnosis model
    makes obvious.
    """
    diagnosis: str
    judge: str
    utility: str
    weights_gb: dict[str, float] = field(default_factory=dict)
    embedding: str = EMBEDDING_MODEL
    reranker: str = RERANKER_MODEL

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("weights_gb")
        return d

    @property
    def hot_gb(self) -> float:
        """Weights that must be co-resident for a run to avoid a model swap.

        diagnosis + judge only. `utility` is small and called at most once per
        run, the embedder is ~100 MB, and the reranker is a separate in-process
        cross-encoder that never competes for LM Studio's budget.
        """
        return (self.weights_gb.get("diagnosis", 0.0)
                + self.weights_gb.get("judge", 0.0)
                + EMBEDDING_GB)

    def families_are_independent(self) -> bool:
        """Cheap structural check that the judge is not from the diagnosis family."""
        def fam(model_id: str) -> str:
            tail = model_id.split("/")[-1]
            return tail.split("-")[0].split("_")[0].lower()
        return fam(self.diagnosis) != fam(self.judge)


# ── the profiles ─────────────────────────────────────────────────────────────
# THE DEFAULT. Sized to fit the SMALLEST supported machine (6 GB VRAM) so every
# platform can hold it and cross-machine results stay comparable.
#
# Qwen3-Coder's smallest official size is 30B-A3B, so the newest CODE-specialised
# model that fits 6 GB is still the 2.5 generation. Deliberate, not stale.
PORTABLE = ModelSet(
    diagnosis="qwen2.5-coder-7b-instruct",
    judge="gemma-4-e4b",                      # Google family -> independent of Qwen
    utility="qwen3-1.7b",
    weights_gb={"diagnosis": 4.7, "judge": 3.0, "utility": 1.1},
)

# The M4 Pro 24 GB daily driver: both models stay RESIDENT, so no swap latency,
# and there is room for a real 32k context. Current-generation on both sides.
MAC_RESIDENT = ModelSet(
    diagnosis="gemma-4-12b",                  # multimodal, runs well on modest memory
    judge="qwen3-8b",                         # Qwen family -> independent of Gemma
    utility="qwen3-4b",
    weights_gb={"diagnosis": 7.0, "judge": 4.7, "utility": 2.5},
)

# Maximum quality the 24 GB Mac can reach. 26B-A4B is MoE with ~3.8B active, so
# it decodes far faster than its size suggests, and the QAT build holds up at
# 4-bit better than a post-hoc quantisation. Too large to co-reside with a judge,
# so this profile computes to jit_swap — accepted knowingly for quality.
#
# Alternative if code specialisation matters more than raw size:
# qwen3-coder-30b-a3b (~17 GB) as diagnosis with gemma-4-e4b as judge, but it
# needs the context dropped to ~16k to stay inside the memory budget.
MAC_MAX = ModelSet(
    diagnosis="google/gemma-4-26b-a4b-qat",
    judge="qwen3-4b",                         # small on purpose: it only writes reasoning text
    utility="qwen3-1.7b",
    weights_gb={"diagnosis": 15.6, "judge": 2.5, "utility": 1.1},
)

# The 64 GB Mac Studio used as a REMOTE inference host (adapters/platforms/studio.py).
# Everything stays resident: ~33 GB of hot weights against a 48 GB budget leaves
# room for a real 32k KV cache, so there is no diagnosis->judge swap at all.
#
# Qwen3-Coder-30B-A3B is the point of the profile. It is the model PORTABLE
# settles for a 2.5-generation Qwen *instead of*, purely because 30B-A3B is the
# smallest size Qwen3-Coder ships and it cannot fit 6 GB of VRAM. MoE with ~3.3B
# active, so it decodes far faster than 30B suggests.
#
# 6-bit rather than 8-bit (~32 GB) deliberately: 32k of context is worth more
# here than the last few points of quantisation quality, because render_evidence
# puts the whole diff PLUS five retrieved snippets in the user turn. Revisit if
# real diffs turn out to fit comfortably in 16k.
STUDIO = ModelSet(
    diagnosis="qwen3-coder-30b-a3b",
    judge="gemma-4-12b",                      # Google family -> independent of Qwen
    utility="qwen3-1.7b",
    weights_gb={"diagnosis": 25.0, "judge": 7.0, "utility": 1.1},
)

PROFILES: dict[str, ModelSet] = {
    "portable": PORTABLE,
    "mac_resident": MAC_RESIDENT,
    "mac_max": MAC_MAX,
    "studio": STUDIO,
}

DEFAULT_PROFILE = "portable"


def resolve(profile: str, supported: tuple[str, ...]) -> ModelSet:
    """Look up a profile, refusing anything this platform cannot actually hold.

    Deliberately raises instead of falling back to `portable`: a silent
    downgrade would produce eval numbers labelled with a profile that never ran.
    """
    if profile not in PROFILES:
        raise KeyError(f"no model profile '{profile}'. have: {sorted(PROFILES)}")
    if profile not in supported:
        raise ValueError(
            f"profile '{profile}' is not supported on this platform "
            f"(supports: {list(supported)}). Refusing to silently downgrade."
        )
    return PROFILES[profile]


def plan_residency(models: ModelSet, budget_gb: float, kv_gb: float) -> tuple[str, dict]:
    """Decide residency from real numbers instead of declaring it per platform.

    Returns ("resident" | "jit_swap", accounting) so `doctor` can show the sum
    that produced the verdict rather than an unexplained string.

    jit_swap is not a failure: the graph is strictly sequential — diagnosis
    completes before the judge starts — and the run then waits on a human at the
    approval gate, so a few seconds of model swap costs nothing that matters.
    """
    needed = models.hot_gb + kv_gb
    fits = needed <= budget_gb
    return ("resident" if fits else "jit_swap"), {
        "hot_weights_gb": round(models.hot_gb, 2),
        "kv_reserve_gb": round(kv_gb, 2),
        "needed_gb": round(needed, 2),
        "budget_gb": round(budget_gb, 2),
        "fits_together": fits,
    }
