"""In-process cross-encoder reranking, behind the `Reranker` port (T13).

## Why this exists: the embedder cannot tell relevant from irrelevant

Measured on the Masterclass index (2026-11-16, `tools.calibrate`): EmbeddingGemma
packs every cosine into ~0.7-0.9. Passages from the changed file scored p10/median
0.780/0.823; passages from OTHER files 0.727/0.770. The two bands overlap so much
that no cosine threshold separates them -- the 0.40 bar never failed, and the best
statistical cut (0.793) would still have escalated a known-good review.

A bi-encoder embeds the query and the passage SEPARATELY and compares the vectors
afterwards, so it can only say "these are about similar things". A cross-encoder
reads the query and the passage TOGETHER, attending across both, and is trained to
output how relevant one is to the other. Its scores spread out, which is the
property a threshold needs.

## What it returns, and the contract with the gate

`rerank` returns copies of the candidates, best first, each with a `rerank` score
in 0..1 (sigmoid of the model's logit). The retriever then makes that the gate's
`score` and marks it `score_kind = "rerank"`, keeping the cosine beside it. The
gate thresholds a rerank score against `min_rerank_score`, never against
`min_relevance_score`: a number from a different model is a different scale, and
comparing it to the cosine bar would repeat T21 (a gate reading a number that
looks like relevance and means something else).

It scores; it never decides. Same rule as the entailer.

## Measured, and therefore OFF by default (2026-11-17)

The argument above was the hypothesis. `tools.calibrate rerank` tested it on 176
Masterclass queries, scoring the same 100-candidate pools with both instruments:

                                         cosine    this reranker
    same- vs other-file (Youden J)        0.68         0.31
    same file in the top 3                98.9%        81.8%
    seconds per query                     ~0           13.5

The proxy ("a passage from the changed file is relevant") favours the cosine --
the query is that file's own path and diff -- and the reranker may be rating
genuinely related OTHER files highly. But no measurement shows it earning its
cost, so `config.get_reranker` returns None unless RERANKER=cross_encoder. The
next honest test is a human-labelled set of (query, passage, relevant?) pairs.

## Where it runs

In this process, on `runtime()["reranker_device"]` (mps on a Mac, cpu elsewhere).
LM Studio and MLX expose no rerank endpoint. The model loads from the local
Hugging Face cache ONLY (`local_files_only`): a review must never start a 2.3 GB
download as a side effect. Fetch it once with

    python -c "from huggingface_hub import snapshot_download as s; s('BAAI/bge-reranker-v2-m3')"
"""
from __future__ import annotations

import os

import torch
from sentence_transformers import CrossEncoder

from adapters.platforms.base import RERANKER_MODEL
from errors import ProviderError
from registry import adapter

# Tokens per (query, passage) pair. bge-reranker-v2-m3 accepts 8192, but cost is
# quadratic in length and a review reranks ~100 pairs per changed file. The query
# (path + a 1,500-char diff) and a chunk both fit in 512 for most files; the
# longer side is truncated first.
DEFAULT_MAX_LENGTH = 512
BATCH_SIZE = 16


def _device() -> str:
    """The platform's reranker device. `config` is imported here because it is
    CIRCULAR at module level: config imports `adapters`, which imports this."""
    import config
    return config.get_platform().runtime(config.get_profile()).get("reranker_device", "cpu")


def passage_text(c: dict) -> str:
    """What the cross-encoder reads for a chunk: where it lives, then the code.

    The path and symbol name are prefixed for the same reason ingest prefixes them
    before embedding (tools/chunk_text.py): a function body alone often does not
    say what file or class it belongs to, and the query starts with a path.
    """
    where = " :: ".join(p for p in (c.get("path"), c.get("name")) if p)
    return f"{where}\n{c.get('text', '')}" if where else c.get("text", "")


@adapter("reranker", "cross_encoder")
class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 by default (pinned in adapters/platforms/base.py)."""

    def __init__(self, model: str | None = None, device: str | None = None,
                 max_length: int | None = None):
        """Configure only; the ~2.3 GB model loads on the first `scores()` call.

        `model` defaults to RERANKER_MODEL (env, else the pin), `device` to the
        platform's `reranker_device`, `max_length` to RERANK_MAX_LENGTH (512).
        """
        self.model = model or os.getenv("RERANKER_MODEL", RERANKER_MODEL)
        self._device = device
        self.max_length = max_length or int(os.getenv("RERANK_MAX_LENGTH",
                                                      str(DEFAULT_MAX_LENGTH)))
        self._encoder = None

    def _load(self) -> CrossEncoder:
        """The cross-encoder, loaded once from the local HF cache only.

        Raises ProviderError with the fetch command when the model is not cached,
        never downloads it: a review must not start a 2.3 GB download.
        """
        if self._encoder is None:
            try:
                self._encoder = CrossEncoder(self.model, device=self._device or _device(),
                                             max_length=self.max_length,
                                             local_files_only=True)
            except OSError as e:
                # Not in the local cache. Said plainly, with the fix, rather than
                # left as a Hugging Face stack trace.
                raise ProviderError(
                    f"reranker model {self.model!r} is not in the local Hugging Face "
                    f"cache, and a review never downloads one. Fetch it once: python "
                    f"-c \"from huggingface_hub import snapshot_download as s; "
                    f"s('{self.model}')\", or set RERANKER=none.") from e
        return self._encoder

    def scores(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, 0..1. Exposed for calibration."""
        if not passages:
            return []
        out = self._load().predict([(query, p) for p in passages], batch_size=BATCH_SIZE,
                                   activation_fn=torch.nn.Sigmoid(),
                                   show_progress_bar=False)
        return [round(float(s), 4) for s in out]

    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
        """The candidates, best first, each a COPY carrying its `rerank` score.

        The `Reranker` port: at most `top_n` of `candidates`, input dicts never
        mutated. It returns a score and an order, never a decision; turning
        `rerank` into the gate's `score` is the retriever's job.
        """
        scored = [{**c, "rerank": s} for c, s in
                  zip(candidates, self.scores(query, [passage_text(c) for c in candidates]),
                      strict=True)]
        scored.sort(key=lambda c: c["rerank"], reverse=True)
        return scored[:top_n]
