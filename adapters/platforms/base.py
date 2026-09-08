"""Model profiles — the part that is deliberately NOT hardware-specific.

A profile names one model per role. The same profile on two machines means the
same model ids, so a diagnosis produced on the CUDA desktop is comparable with
one produced on the Mac. Only the *runtime* (context, KV dtype, residency)
differs per platform; see adapters/platforms/<name>.py.

Two roles must never vary by machine, for correctness rather than taste:
  · EMBEDDING — a different embedder is a different vector space, so the index
    stops being portable AND every min_relevance_score in tenants/*/policy.json
    silently changes meaning.
  · RERANKER  — runs in-process (sentence-transformers), not via LM Studio, so
    it is CPU-identical everywhere by default. Keep it that way.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

# Pinned across every profile and every platform. Changing either invalidates
# the vector index and de-calibrates every tenant threshold. See docs/ROADMAP.md.
EMBEDDING_MODEL = "nomic-embed-text-v1.5"     # 768 dims, 8k ctx (NOT 512 like text-embedding-3-small)
EMBEDDING_DIMS = 768
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"    # in-process cross-encoder, never an LM Studio call


@dataclass(frozen=True)
class ModelSet:
    """One model per role. `judge` MUST be a different family from `diagnosis`.

    That is not a style preference: a judge from the same family as the model it
    is grading exhibits self-preference bias (docs/BEST_PRACTICES.md §7), which
    would quietly undermine independent_review's whole purpose.
    """
    diagnosis: str
    judge: str
    utility: str
    embedding: str = EMBEDDING_MODEL
    reranker: str = RERANKER_MODEL

    def as_dict(self) -> dict:
        return asdict(self)


# The default. Sized to run on the SMALLEST supported machine (6 GB VRAM), so
# every platform can hold it and results are comparable by default.
PORTABLE = ModelSet(
    diagnosis="qwen2.5-coder-7b-instruct",    # code-specialised; reads diffs
    judge="gemma-3-4b-it",                    # Google family -> independent of Qwen
    utility="qwen3-1.7b",                     # query reformulation / HyDE
)

# NOT "the better profile" — a second data point, for measuring how much of the
# diagnosis quality ceiling is model-limited vs prompt-limited (gap G17).
# Needs ~15 GB of model weights resident; only the 24 GB Mac can hold it.
MAC_HEADROOM = ModelSet(
    diagnosis="qwen3-14b",
    judge="gemma-3-12b-it",
    utility="qwen3-1.7b",
)

PROFILES: dict[str, ModelSet] = {
    "portable": PORTABLE,
    "mac_headroom": MAC_HEADROOM,
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
