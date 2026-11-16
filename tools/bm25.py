"""Okapi BM25. The real thing, replacing a word-overlap count.

`tools/retrieval.py::_bm25_search` scored a document by `len(query_terms &
doc_terms)` -- no IDF, no term frequency saturation, no length normalisation.
That is not BM25 and does not behave like it:

  · **No IDF** means a match on "the" counted exactly as much as a match on
    "oauth". In a code corpus, where `self`, `def` and `return` are everywhere,
    common terms dominate the ranking outright.
  · **No saturation** means a document repeating a term ten times scored ten
    times higher, so the longest file usually wins.
  · **No length normalisation** compounds it.

Pure stdlib on purpose: a sparse retriever is simple enough that pulling
`rank_bm25` in would cost more than it saves.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Split on non-alphanumerics, then also split snake_case and camelCase, because
# an identifier like `get_user_token` should match a query for "user token".
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

K1 = 1.5   # term-frequency saturation; 1.2-2.0 is the usual range
B = 0.75   # length normalisation; 0 = none, 1 = full


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, with identifiers split into their parts.

    Code-aware rather than prose-aware: `getUserToken` and `get_user_token` both
    become ["get", "user", "token"], so a natural-language query can reach an
    identifier. Without this a query for "user token" misses every call site.
    """
    out: list[str] = []
    for raw in _WORD.findall(text):
        for part in _CAMEL.sub(" ", raw).split():
            out.append(part.lower())
    return out


class BM25:
    """Okapi BM25 over a fixed set of documents.

    Built once per namespace at index load and kept in memory: the corpus is
    static between ingests, so recomputing IDF per query would be waste.
    """

    def __init__(self, corpus: list[list[str]], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.n = len(corpus)
        self.lengths = [len(d) for d in corpus]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in corpus]

        df: Counter = Counter()
        for doc in corpus:
            df.update(set(doc))
        # Lucene's IDF variant: always positive, so a term appearing in every
        # document contributes ~0 rather than going negative and actively
        # penalising documents that contain it.
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for `query`, in corpus order."""
        terms = tokenize(query)
        out = [0.0] * self.n
        if not self.n:
            return out
        for i in range(self.n):
            freq, length, total = self.freqs[i], self.lengths[i], 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                total += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            out[i] = total
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """Best k as (corpus index, score). Zero scores are dropped, not returned:
        a document sharing no term with the query is not a weak match, it is none."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda p: p[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0.0]
