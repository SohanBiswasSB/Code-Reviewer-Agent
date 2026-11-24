"""The reranker served by the model gateway (`POST {LMSTUDIO_BASE_URL}/rerank`).

Same model as `rerank_cross_encoder` -- BAAI/bge-reranker-v2-m3 -- run on the
devops DGX host instead of this machine. Selected with `RERANKER=gateway`.

## Why a second adapter for the same model

Two measured reasons (2026-11-17):

  · **Cost.** In-process on this Mac it took 13.5 s per query (100 pairs on
    mps) and competed with LM Studio for unified memory -- the Metal OOM that
    night. The gateway answered 3 pairs in 0.07 s.
  · **It reads the whole passage.** The local adapter truncates every pair to
    512 tokens (`RERANK_MAX_LENGTH`); the gateway does not -- a 3,270-token pair
    was scored whole. On a passage whose answer sat at the END, the gateway
    scored 0.51 and the local adapter 0.0009. Code chunks run to ~2k tokens, so
    the local measurement (separation J 0.31 vs the cosine's 0.68) was partly a
    measurement of the truncation. Short pairs agreed: 0.9605 vs 0.9716.

## The score

The gateway returns the model's raw logit as `relevance_score` (-4.8, -11.0...).
It is put through the same sigmoid the local adapter applies, so both adapters
emit 0..1 on one scale and `min_rerank_score` means the same thing for either.
Results are mapped back by `index`, never by position -- a server may return them
sorted by score.

It scores; it never decides. The gate thresholds it (`independent_review._bar_for`).
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

from adapters.llm_lmstudio import auth_headers, base_url, with_retry
from adapters.platforms.base import RERANKER_MODEL
from adapters.rerank_cross_encoder import rank_by
from errors import ConfigError, ProviderAuthError, ProviderError
from registry import adapter

# Documents per request. One review query sends RERANK_CANDIDATES (100); a cap
# keeps a single request bounded if that is ever raised.
BATCH = 128
TIMEOUT_S = 120.0


def _sigmoid(x: float) -> float:
    # Split by sign so a large negative logit cannot overflow exp().
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@adapter("reranker", "gateway")
class GatewayReranker:
    """bge-reranker-v2-m3 behind the gateway's `/rerank`. Same contract as the local one."""

    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or os.getenv("RERANKER_MODEL", RERANKER_MODEL)
        self._url = url

    @property
    def url(self) -> str:
        """`<LMSTUDIO_BASE_URL>/rerank`, resolved per call so a repointed env is honoured."""
        return f"{self._url or base_url()}/rerank"

    def _post(self, query: str, documents: list[str]) -> list[float]:
        body = json.dumps({"model": self.model, "query": query,
                           "documents": documents}).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={**auth_headers(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Before URLError, which it subclasses: the server WAS reached.
            if e.code in (401, 403):
                raise ProviderAuthError(
                    f"the gateway at {self.url} refused the API key (HTTP {e.code}). "
                    f"Set LMSTUDIO_API_KEY to a valid key.") from None
            if e.code >= 500 or e.code == 429:
                # A ConnectionError, so `with_retry` treats it as transient.
                raise ProviderError(f"the gateway at {self.url} answered HTTP {e.code}") from None
            raise ConfigError(
                f"the gateway at {self.url} rejected the rerank request (HTTP {e.code}: "
                f"{e.reason}). Check RERANKER_MODEL={self.model!r} is served.") from None
        except (urllib.error.URLError, OSError) as e:
            raise ProviderError(f"the gateway at {self.url} is not reachable "
                                f"({getattr(e, 'reason', e)})") from None
        try:
            by_index = {int(r["index"]): float(r["relevance_score"])
                        for r in payload["results"]}
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"the gateway's rerank answer has an unexpected shape "
                                f"({type(e).__name__}: {e})") from None
        if set(by_index) != set(range(len(documents))):
            # A score per document or none: a missing one must not default to
            # anything, or a passage would reach the gate with an invented number.
            raise ProviderError(f"the gateway scored {len(by_index)} of {len(documents)} "
                                f"documents")
        return [round(_sigmoid(by_index[i]), 4) for i in range(len(documents))]

    def scores(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, 0..1. Exposed for calibration."""
        out: list[float] = []
        for start in range(0, len(passages), BATCH):
            chunk = passages[start:start + BATCH]
            out.extend(with_retry(lambda c=chunk: self._post(query, c),
                                  what="rerank", model=self.model))
        return out

    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
        """The candidates, best first, each a COPY carrying its `rerank` score."""
        return rank_by(self.scores, query, candidates, top_n)
