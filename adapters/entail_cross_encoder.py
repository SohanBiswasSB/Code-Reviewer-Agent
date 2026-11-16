"""Cross-encoder entailment, behind the same `Entailer` port.

Not the default: it needs `sentence-transformers` and a model download, and the
gate must stay computable offline. Register it by setting ENTAILER=cross_encoder
once the model is present.

Same principle as the lexical one -- it returns a SCORE and `independent_review`
thresholds it. The model never decides.
"""
from __future__ import annotations

import os

import numpy as np
from sentence_transformers import CrossEncoder

from registry import adapter

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


@adapter("entailer", "cross_encoder")
class CrossEncoderEntailer:
    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("ENTAILER_MODEL", DEFAULT_MODEL)
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, claim: str, passage: str) -> float:
        """P(entailment) for passage -> claim.

        Note the argument order: the PASSAGE is the premise and the CLAIM is the
        hypothesis. Reversing them scores whether the passage follows from the
        claim, which is a different and useless question here.
        """
        if not claim or not passage:
            return 0.0
        logits = self._load().predict([(passage, claim)])[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        # nli-deberta label order: contradiction, entailment, neutral
        return round(float(probs[1]), 4)
