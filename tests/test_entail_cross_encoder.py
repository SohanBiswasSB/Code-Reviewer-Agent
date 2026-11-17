"""The cross-encoder entailer reads the WHOLE premise, not the first 512 tokens.

Per-claim entailment puts a finding's own file diff AFTER its citations, so the
supporting text is usually at the end -- exactly where truncation cut it. These
tests use a fake model (whitespace tokenizer, a "SUPPORTED" marker) so they need
no download and run offline.
"""
from __future__ import annotations

import re

import numpy as np

from adapters import entail_cross_encoder as ece
from adapters.entail_cross_encoder import CrossEncoderEntailer


class _Tok:
    model_max_length = 10**30            # what a limitless tokenizer reports

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class _Model:
    """Entails iff the window contains SUPPORTED; refuses an over-long window."""
    def __init__(self, max_length):
        self.max_length, self.tokenizer, self.calls = max_length, _Tok(), []

    def predict(self, pairs):
        self.calls.append(pairs)
        rows = []
        for premise, claim in pairs:
            assert len(_Tok()(premise)["input_ids"]) + len(_Tok()(claim)["input_ids"]) \
                + ece.PAIR_OVERHEAD <= self.max_length, "window would be truncated"
            rows.append([0.0, 5.0, 0.0] if "SUPPORTED" in premise else [5.0, 0.0, 0.0])
        return np.array(rows)


def _entailer(max_length=200):
    e = CrossEncoderEntailer()
    e._model = _Model(max_length)
    return e


def test_support_at_the_end_of_a_long_premise_is_read():
    premise = " ".join(["filler"] * 2_000) + " SUPPORTED"
    assert _entailer().score("the claim", premise) > 0.9


def test_a_short_premise_is_one_call_with_the_text_unchanged():
    e = _entailer()
    assert e.score("the claim", "short SUPPORTED premise") > 0.9
    assert e._model.calls == [[("short SUPPORTED premise", "the claim")]]


def test_no_support_anywhere_stays_low():
    assert _entailer().score("the claim", " ".join(["filler"] * 2_000)) < 0.1


def test_windows_overlap_and_cover_every_token():
    premise = " ".join(f"w{i}" for i in range(1_000))
    wins = ece._windows(_Model(200), "c", premise)
    assert len(wins) > 1 and wins[0].startswith("w0 ") and wins[-1].endswith("w999")
    # neighbours share WINDOW_OVERLAP tokens, so nothing falls between them
    assert wins[0].split()[-ece.WINDOW_OVERLAP:] == wins[1].split()[:ece.WINDOW_OVERLAP]


def test_past_max_windows_the_tail_is_unread_and_fails_closed():
    premise = " ".join(["filler"] * 200_000) + " SUPPORTED"
    assert len(ece._windows(_Model(200), "c", premise)) == ece.MAX_WINDOWS
    assert _entailer().score("the claim", premise) < 0.1


def test_a_limitless_tokenizer_falls_back_to_the_default_length():
    m = _Model(None)
    assert ece._max_length(m) == ece.DEFAULT_MAX_LENGTH
