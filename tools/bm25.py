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

## Stored as an inverted index, and saved beside the vectors

The first version kept a `Counter` per document and scored a query by looping
over EVERY document in Python. Measured on the 17.6k-chunk Masterclass index:
~1.36 s to build per process (tokenising 3.65M tokens), ~10 ms per query, and
~105 MB of token lists nobody read again. Now:

  · postings in flat numpy arrays (term -> the documents that contain it), so a
    query touches only the documents sharing a term with it;
  · `save` / `load` to `bm25.npz` next to the vectors, tied to the index's
    FINGERPRINT -- a rebuilt index never reuses statistics from the old one.
    `tools.ingest` writes it; the retriever builds and saves it once if missing.

The scores are the same BM25 as before, to float32 precision.
"""
from __future__ import annotations

import pathlib
import re
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from errors import WRITE_ERRORS
from tools.chunk_text import bm25_text

# Split on non-alphanumerics, then also split snake_case and camelCase, because
# an identifier like `get_user_token` should match a query for "user token".
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

K1 = 1.5   # term-frequency saturation; 1.2-2.0 is the usual range
B = 0.75   # length normalisation; 0 = none, 1 = full
FILE = "bm25.npz"
FORMAT = 1  # bump when the saved layout changes, so an old file is rebuilt


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, with identifiers split into their parts.

    Code-aware rather than prose-aware: `getUserToken` and `get_user_token` both
    become ["get", "user", "token"], so a natural-language query can reach an
    identifier. Without this a query for "user token" misses every call site.
    """
    out: list[str] = []
    for raw in _WORD.findall(text):
        # A token with no capital has no camelCase boundary to split: skip the
        # regex. Same output, and most of a code corpus is lowercase.
        if raw.islower() or raw.isdigit():
            out.append(raw.lower())
            continue
        for part in _CAMEL.sub(" ", raw).split():
            out.append(part.lower())
    return out


def _count_postings(corpus: list[list[str]]) -> tuple[dict, np.ndarray]:
    """({term: [(doc, term frequency), ...]}, length per doc), in corpus order.

    The counting pass of a build, kept apart from the flattening into arrays.
    """
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths = np.zeros(len(corpus), dtype=np.float32)
    for doc, tokens in enumerate(corpus):
        lengths[doc] = len(tokens)
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).append((doc, tf))
    return postings, lengths


@dataclass(frozen=True, eq=False)
class _Postings:
    """The inverted index's arrays, as built from a corpus or read from `bm25.npz`.

    Term `i`'s postings are `docs[offsets[i]:offsets[i + 1]]` with the matching
    `tfs`; `lengths` is each document's token count. One value, so the two ways
    of obtaining the arrays hand `BM25._init` exactly the same shape.
    """

    terms: list[str]
    offsets: np.ndarray
    docs: np.ndarray
    tfs: np.ndarray
    lengths: np.ndarray

    @classmethod
    def from_corpus(cls, corpus: list[list[str]]) -> _Postings:
        """Count every document's terms, then flatten into term-sorted arrays."""
        postings, lengths = _count_postings(corpus)
        terms = sorted(postings)
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(postings[t]) for t in terms])
        flat = [p for t in terms for p in postings[t]]
        docs = np.fromiter((d for d, _ in flat), dtype=np.int32, count=len(flat))
        tfs = np.fromiter((f for _, f in flat), dtype=np.float32, count=len(flat))
        return cls(terms, offsets, docs, tfs, lengths)

    @classmethod
    def from_saved(cls, saved) -> _Postings:
        """The arrays of an open `bm25.npz` (`np.load`'s NpzFile).

        Each array goes through `np.asarray`, which returns NpzFile's ndarray
        as it is: it states the type, which pylint cannot infer from NpzFile's
        `__getitem__` (it read `terms` as the `np.array` FUNCTION). A missing
        key is a KeyError, which `BM25.load` treats as an unusable file.
        """
        return cls(np.asarray(saved["terms"]).tolist(), np.asarray(saved["offsets"]),
                   np.asarray(saved["docs"]), np.asarray(saved["tfs"]),
                   np.asarray(saved["lengths"]))


class BM25:
    """Okapi BM25 over a fixed set of documents, as an inverted index.

    State: the `postings` arrays as built or loaded, the parameters `k1` and
    `b`, and what is derived from them once -- `vocab` (term -> row), `idf`,
    the document count `n` and the mean length `avg_len`.
    """

    def __init__(self, corpus: list[list[str]], k1: float = K1, b: float = B):
        """Build from one token list per document (corpus order = row order)."""
        self._init(_Postings.from_corpus(corpus), k1, b)

    @classmethod
    def for_chunks(cls, metas: list[dict]) -> BM25:
        """BM25 over an index's chunks, in ROW order -- what ingest saves and the
        retriever loads, built the one way so the two can never disagree."""
        return cls([tokenize(bm25_text(m)) for m in metas])

    def _init(self, arrays: _Postings, k1: float, b: float) -> None:
        """Set every attribute from the postings arrays, and derive IDF.

        The one place an instance is filled in, shared by `__init__` (built from
        a corpus) and `load` (read from `bm25.npz`), so a loaded index cannot
        differ from a freshly built one in anything but how it got its arrays.
        """
        self.k1, self.b = float(k1), float(b)
        self.postings = replace(arrays, terms=list(arrays.terms))
        self.vocab = {t: i for i, t in enumerate(self.postings.terms)}
        self.n = len(self.postings.lengths)
        self.avg_len = float(self.postings.lengths.mean()) if self.n else 0.0
        df = np.diff(self.postings.offsets).astype(np.float64)
        # Lucene's IDF variant: always positive, so a term appearing in every
        # document contributes ~0 rather than going negative and actively
        # penalising documents that contain it.
        self.idf = np.log1p((self.n - df + 0.5) / (df + 0.5)).astype(np.float32)

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for `query`, in corpus order."""
        return self._scores(query).tolist()

    def _scores(self, query: str) -> np.ndarray:
        """`scores` as a float32 array, touching only each query term's postings."""
        out =np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        arrays = self.postings
        norm = self.k1 * (1 - self.b + self.b * arrays.lengths / (self.avg_len or 1))
        for term in tokenize(query):
            t = self.vocab.get(term)
            if t is None:
                continue
            lo, hi = arrays.offsets[t], arrays.offsets[t + 1]
            docs, tf = arrays.docs[lo:hi], arrays.tfs[lo:hi]
            # A document appears once per term's postings, so `+=` on the fancy
            # index is safe; repeated query terms add, as in the loop version.
            out[docs] += self.idf[t] * (tf * (self.k1 + 1)) / (tf + norm[docs])
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """Best k as (corpus index, score). Zero scores are dropped, not returned:
        a document sharing no term with the query is not a weak match, it is none."""
        scores = self._scores(query)
        hit = np.flatnonzero(scores > 0)
        if len(hit) == 0:
            return []
        # Only the documents that share a term are sorted, by (score desc, row
        # asc) -- the same tie order the old stable sort over all rows gave.
        best = hit[np.lexsort((hit, -scores[hit]))[:k]]
        return [(int(i), float(scores[i])) for i in best]

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: pathlib.Path, fingerprint: str) -> bool:
        """Write beside the index, atomically. Best effort: False if it could not
        be written -- the statistics are a cache, rebuildable from the chunks."""
        try:
            tmp = path.with_suffix(".tmp.npz")
            arrays = self.postings
            np.savez(tmp, format=np.int64(FORMAT), fingerprint=np.str_(fingerprint),
                     terms=np.array(arrays.terms, dtype=np.str_), offsets=arrays.offsets,
                     docs=arrays.docs, tfs=arrays.tfs, lengths=arrays.lengths,
                     params=np.array([self.k1, self.b], dtype=np.float64))
            tmp.replace(path)
            return True
        except WRITE_ERRORS:
            return False

    @classmethod
    def load(cls, path: pathlib.Path, fingerprint: str) -> BM25 | None:
        """The saved statistics, or None if missing, unreadable, of an old
        format, or built from a DIFFERENT index (fingerprint mismatch)."""
        try:
            with np.load(path, allow_pickle=False) as z:
                if int(z["format"]) != FORMAT or str(z["fingerprint"]) != fingerprint:
                    return None
                self = cls.__new__(cls)
                k1, b = (float(x) for x in np.asarray(z["params"]))
                self._init(_Postings.from_saved(z), k1, b)
                return self
        except (OSError, ValueError, KeyError):
            return None
