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
  · RERANKER  — the same model id everywhere, for the same reason: a
    `min_rerank_score` is measured against one model's scale. It runs
    in-process (sentence-transformers), not via LM Studio, and only its DEVICE
    varies -- `runtime()["reranker_device"]`, mps on mac/studio, cpu on
    cuda/cpu. It is OFF by default (T13, on measurement; see
    adapters/rerank_cross_encoder.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Pinned across every profile and every platform. Changing either invalidates
# the vector index and de-calibrates every tenant threshold. See docs/ROADMAP.md.
#
# EmbeddingGemma-300m (QAT) replaces bge-m3, 2026-09-11, for a blunt reason:
# **bge-m3 is not obtainable through LM Studio.** `lms get` finds no such
# artifact under `bge-m3`, `text-embedding-bge-m3`, `bge-m3-gguf`, `BAAI/bge-m3`
# or `lmstudio-community/bge-m3`. The previous comment here guessed that the
# catalog's "probable text-embedding-bge-m3" would resolve via loose matching;
# the guess was wrong, and it was never verified because no call had ever reached
# a live server. An unobtainable pin is not a pin.
#
# The DIMS ARE MEASURED, not assumed -- POST /v1/embeddings against the running
# server returned a 768-vector for this exact id. That matters because the number
# below is what `config.registered()` advertises and what a future index is built
# against; the last two values here were both reasoned rather than measured.
#
# Changed now for the same reason bge-m3 was: T10 is open, so there is no index to
# re-embed and the corpus is four documents. This is the cheapest moment such a
# change will ever have. The shipped 0.35/0.40 min_relevance_score values were
# calibrated for text-embedding-3-small@512 and mean nothing in any other space.
# Since measured on this embedder (tools.calibrate, 2026-11-16): the real tenants
# (masterclass, milesone, _template) now carry an interim 0.75; _default,
# project_a and project_b still carry the old numbers.
#
# The id is FULLY QUALIFIED on purpose. `llm_lmstudio.resolves_to()` would match
# the short form loosely, but T26 is exactly a case of loose matching binding a
# role to the wrong model -- so do not depend on it for a pinned constant.
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
EMBEDDING_DIMS = 768         # MEASURED against the live server, not assumed
EMBEDDING_GB = 0.25          # 300M at QAT 4-bit, 229 MB on disk; bge-m3 assumed 1.2
# The cosine below which a relevance bar can never fail for THIS embedder:
# `tools.calibrate relevance` (2026-11-16, Masterclass, 305 per-file queries) put
# even OTHER-file passages at p10 0.727, so a bar under ~0.70 passes ~all of them.
# Tied to EMBEDDING_MODEL: a new embedder means re-measuring this too.
MEASURED_COSINE_FLOOR = 0.70
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
    # The quantisation `weights_gb` was computed at. A property of the MODELS, so
    # a platform reads it rather than guessing from the profile's name.
    quant: str = "4bit"

    def as_dict(self) -> dict:
        """role -> model id. Sizing and quantisation are not roles."""
        d = asdict(self)
        d.pop("weights_gb")
        d.pop("quant")
        return d

    @property
    def hot_gb(self) -> float:
        """Weights that must be co-resident for a run to avoid a model swap.

        diagnosis + judge + the embedder (EMBEDDING_GB, ~0.25 GB): the embedder is
        called for the review's queries in the same run, so it shares LM Studio's
        budget. `utility` is left out (small, called at most once per run), and so
        is the reranker, a separate in-process cross-encoder that never competes
        for LM Studio's budget.
        """
        return (self.weights_gb.get("diagnosis", 0.0)
                + self.weights_gb.get("judge", 0.0)
                + EMBEDDING_GB)

    def families_are_independent(self) -> bool:
        """Cheap structural check that the judge is not from the diagnosis family."""
        def fam(model_id: str) -> str:
            """First `-`/`_` token of the id's last path part: `qwen/qwen3-4b` -> qwen3."""
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

# The best diagnosis model this 24 GB Mac can actually USE, which is not the same
# thing as the largest one it can hold. Changed 2026-09-11 on measurement.
#
# This profile used google/gemma-4-26b-a4b-qat (15.6 GB, MoE, ~3.8B active),
# chosen for "maximum quality" on reasoning alone. The first live run showed it
# **cannot produce parseable output against this project's own prompt**: with the
# real 3757-char diagnosis prompt it enters a degenerate repetition loop INSIDE a
# JSON string ("client-side side-effect of client-side side-effect of ...") and
# burns the entire 2047-token budget without closing the quote.
#
#   gemma-4-26b-a4b-qat  temp 0 / 0.1 / 0.2 / 0.3 -> 0 of 6 parse (finish=length)
#                        temp 0.5 / 0.7           -> parses, but needs sampling
#                                                    noise to stop babbling
#   ministral-3-14b-reas temp 0                   -> 3 of 3 parse, BYTE-IDENTICAL
#                                                    across runs, 308 tokens each
#
# Constrained decoding does not save the bigger model: the schema governs
# structure (field names, ordering, the severity enum were all correct) and says
# nothing about what goes inside a string. `frequency_penalty` and a schema
# `maxLength` were both measured and both failed.
#
# Ministral wins on every axis that matters here, not just parseability:
#   · 9.1 GB, so diagnosis+judge+embedder = 11.87 GB -> RESIDENT, not jit_swap
#   · therefore 32k context instead of the 16k a swapping profile has to accept
#   · "mistral" is a genuinely different family from the qwen3-4b judge, rather
#     than a string-comparison coincidence (see the note on qwen3.8-27b below)
#   · it populates findings[], which no live model had ever done before
#
# The cost is 26B -> 14B on the diagnosis role. A model that cannot emit valid
# JSON at temperature 0 is not delivering 26B of quality to this pipeline.
#
# NOT qwen/qwen3.8-27b, which is also on disk: `families_are_independent()` splits
# on the first "-", so it reads as family "qwen3.8" against the judge's "qwen3"
# and PASSES the independence check while both are Qwen. That is the same weakness
# as T26 and must be fixed before that model is safe to select here.
MAC_MAX = ModelSet(
    diagnosis="mistralai/ministral-3-14b-reasoning",
    judge="qwen3-4b",                         # small on purpose: it only writes reasoning text
    utility="qwen3-1.7b",
    weights_gb={"diagnosis": 9.12, "judge": 2.5, "utility": 1.1},
)

# The 64 GB Mac Studio used as a REMOTE inference host (adapters/platforms/studio.py).
# Everything stays resident: ~32 GB of hot weights against a 48 GB budget leaves
# room for a real 32k KV cache, so there is no diagnosis->judge swap at all.
#
# Qwen3-Coder-30B-A3B is the point of the profile. It is the model PORTABLE
# settles for a 2.5-generation Qwen *instead of*, purely because 30B-A3B is the
# smallest size Qwen3-Coder ships and it cannot fit 6 GB of VRAM. MoE with ~3.3B
# active, so it decodes far faster than 30B suggests.
#
# 6-bit rather than 8-bit (~32 GB) deliberately: 32k of context is worth more
# here than the last few points of quantisation quality, because render_evidence
# puts the whole diff PLUS up to eight retrieved snippets (generate_diagnosis
# MAX_HITS) in the user turn. Revisit if real diffs turn out to fit comfortably
# in 16k.
STUDIO = ModelSet(
    diagnosis="qwen3-coder-30b-a3b",
    judge="gemma-4-12b",                      # Google family -> independent of Qwen
    utility="qwen3-1.7b",
    weights_gb={"diagnosis": 25.0, "judge": 7.0, "utility": 1.1},
    quant="6bit",
)

PROFILES: dict[str, ModelSet] = {
    "portable": PORTABLE,
    "mac_resident": MAC_RESIDENT,
    "mac_max": MAC_MAX,
    "studio": STUDIO,
}

DEFAULT_PROFILE = "portable"


class PlatformBase:
    """What every platform shares. `models()` was the same three lines in all four."""

    profiles: tuple[str, ...] = ()

    def models(self, profile: str = DEFAULT_PROFILE) -> dict:
        """role -> model id for `profile`; raises if this platform cannot hold it."""
        return resolve(profile, self.profiles).as_dict()


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
