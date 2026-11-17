"""Cross-encoder entailment, behind the same `Entailer` port.

Not the default: it needs a model download, and the gate must stay computable
offline. (`sentence-transformers` itself is no longer the obstacle: it is in
requirements.txt and imported at module level, like the reranker.) Unlike the
reranker it does not pass `local_files_only`, so the first `score()` fetches the
model if it is not cached. Select it with ENTAILER=cross_encoder once the model
is present.

The caller scores PER CLAIM (`independent_review.entailment_claims`): each
finding against its own citations plus its own file's diff, the comment against
every cited passage plus the whole diff. So `passage` here can be long -- far
past the model's input limit (512 tokens for nli-deberta-v3-base). Handed over
whole, sentence-transformers truncates the pair longest-side-first, so premise
text past the cut was never read: a finding supported only by its file's diff,
which comes AFTER the citations, scored as unsupported.

So a long premise is scored in overlapping token windows (`_windows`), each
paired with the full claim, and the claim's score is the BEST window: support
anywhere in the premise is support. A claim that needs two windows at once to
be supported scores low -- the safe direction (invariant 12). Past MAX_WINDOWS
the tail is not read, which also fails closed.

Same principle as the lexical one -- it returns a SCORE and `independent_review`
thresholds it. The model never decides.
"""
from __future__ import annotations

import os

import numpy as np
from sentence_transformers import CrossEncoder

from registry import adapter

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_MAX_LENGTH = 512     # used when neither the model nor its tokenizer says
WINDOW_OVERLAP = 64          # tokens shared by neighbours, so no sentence is split blind
MAX_WINDOWS = 32             # ~14k premise tokens; beyond it, unread = unsupported
PAIR_OVERHEAD = 4            # [CLS] premise [SEP] claim [SEP], plus slack


@adapter("entailer", "cross_encoder")
class CrossEncoderEntailer:
    """NLI cross-encoder (`cross-encoder/nli-deberta-v3-base` unless ENTAILER_MODEL)."""

    def __init__(self, model: str | None = None):
        """Records the model name only; loading waits for the first `score()`."""
        self.model_name = model or os.getenv("ENTAILER_MODEL", DEFAULT_MODEL)
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, claim: str, passage: str) -> float:
        """P(entailment) for passage -> claim: the best over the premise's windows.

        Note the argument order: the PASSAGE is the premise and the CLAIM is the
        hypothesis. Reversing them scores whether the passage follows from the
        claim, which is a different and useless question here.
        """
        if not claim or not passage:
            return 0.0
        model = self._load()
        windows = _windows(model, claim, passage)
        logits = np.asarray(model.predict([(w, claim) for w in windows]))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        # nli-deberta label order: contradiction, entailment, neutral
        return round(float(probs[:, 1].max()), 4)


def _max_length(model) -> int:
    """The pair's token limit: the CrossEncoder's own, else its tokenizer's.

    A tokenizer with no limit reports a huge sentinel (~1e30), so anything above
    a plausible size falls back to DEFAULT_MAX_LENGTH."""
    for n in (getattr(model, "max_length", None),
              getattr(model.tokenizer, "model_max_length", None)):
        if isinstance(n, int) and 0 < n <= 32_768:
            return n
    return DEFAULT_MAX_LENGTH


def _windows(model, claim: str, passage: str) -> list[str]:
    """The premise cut into overlapping spans that fit beside the claim.

    Cut on the tokenizer's own offsets, so a window is exactly what the model
    would read and the text is sliced from the original (no decode round trip).
    A premise that already fits is returned whole, as one window.
    """
    tok = model.tokenizer
    claim_tokens = len(tok(claim, add_special_tokens=False)["input_ids"])
    budget = max(_max_length(model) - claim_tokens - PAIR_OVERHEAD, 2 * WINDOW_OVERLAP)
    offsets = tok(passage, add_special_tokens=False,
                  return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= budget:
        return [passage]
    out, step = [], budget - WINDOW_OVERLAP
    for start in range(0, len(offsets), step):
        end = min(start + budget, len(offsets))
        out.append(passage[offsets[start][0]:offsets[end - 1][1]])
        if end == len(offsets) or len(out) == MAX_WINDOWS:
            break
    return out
