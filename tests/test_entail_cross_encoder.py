"""The cross-encoder entailer reads the WHOLE premise, not the first 512 tokens.

Per-claim entailment puts a finding's own file diff AFTER its citations, so the
supporting text is usually at the end -- exactly where truncation cut it. These
tests use a fake model (whitespace tokenizer, a "SUPPORTED" marker) so they need
no download and run offline.

The fake is installed where the adapter loads its model (`CrossEncoder`), and the
windows are read from what the fake was asked to predict on: that is exactly the
text the real model would see, so the windowing is pinned through `score()`
rather than through the private helpers that compute it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from adapters import entail_cross_encoder as ece
from adapters.entail_cross_encoder import CrossEncoderEntailer


@dataclass
class _Tok:
    """Whitespace tokenizer with the call shape the entailer uses."""

    model_max_length: int = 10**30            # what a limitless tokenizer reports

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


@dataclass
class _Model:
    """Entails iff the window contains SUPPORTED; refuses an over-long window.

    `max_length=None` is a model that declares no limit, so the over-long check
    has nothing to hold it to; the test using it asserts the window size itself.
    """

    max_length: int | None
    tokenizer: _Tok = field(default_factory=_Tok)
    calls: list = field(default_factory=list)

    def predict(self, pairs):
        """NLI logits (contradiction, entailment, neutral) per (premise, claim) pair."""
        self.calls.append(pairs)
        rows = []
        for premise, claim in pairs:
            if self.max_length is not None:
                assert len(_Tok()(premise)["input_ids"]) + len(_Tok()(claim)["input_ids"]) \
                    + ece.PAIR_OVERHEAD <= self.max_length, "window would be truncated"
            rows.append([0.0, 5.0, 0.0] if "SUPPORTED" in premise else [5.0, 0.0, 0.0])
        return np.array(rows)


def _entailer(monkeypatch, model: _Model) -> CrossEncoderEntailer:
    """A real entailer whose model load returns `model` instead of downloading one."""
    monkeypatch.setattr(ece, "CrossEncoder", lambda name: model)
    return CrossEncoderEntailer()


def _windows(model: _Model) -> list[str]:
    """The premise windows the entailer's one predict call was given."""
    assert len(model.calls) == 1, "every window goes in one batched predict"
    return [premise for premise, _ in model.calls[0]]


def test_support_at_the_end_of_a_long_premise_is_read(monkeypatch):
    """Support past the model's input limit still counts: the tail is windowed, not cut."""
    premise = " ".join(["filler"] * 2_000) + " SUPPORTED"
    assert _entailer(monkeypatch, _Model(200)).score("the claim", premise) > 0.9


def test_a_short_premise_is_one_call_with_the_text_unchanged(monkeypatch):
    """A premise that fits is scored whole, as the (premise, claim) pair, in one call."""
    model = _Model(200)
    assert _entailer(monkeypatch, model).score("the claim", "short SUPPORTED premise") > 0.9
    assert model.calls == [[("short SUPPORTED premise", "the claim")]]


def test_no_support_anywhere_stays_low(monkeypatch):
    """Windowing takes the best window, so it must not invent support that is absent."""
    assert _entailer(monkeypatch, _Model(200)).score(
        "the claim", " ".join(["filler"] * 2_000)) < 0.1


def test_windows_overlap_and_cover_every_token(monkeypatch):
    """Neighbouring windows overlap and together reach the last token."""
    model = _Model(200)
    premise = " ".join(f"w{i}" for i in range(1_000))
    _entailer(monkeypatch, model).score("c", premise)
    wins = _windows(model)
    assert len(wins) > 1 and wins[0].startswith("w0 ") and wins[-1].endswith("w999")
    # neighbours share WINDOW_OVERLAP tokens, so nothing falls between them
    assert wins[0].split()[-ece.WINDOW_OVERLAP:] == wins[1].split()[:ece.WINDOW_OVERLAP]


def test_past_max_windows_the_tail_is_unread_and_fails_closed(monkeypatch):
    """Past MAX_WINDOWS the tail is not read, so support only there scores low (invariant 12)."""
    model = _Model(200)
    premise = " ".join(["filler"] * 200_000) + " SUPPORTED"
    score = _entailer(monkeypatch, model).score("the claim", premise)
    assert len(_windows(model)) == ece.MAX_WINDOWS
    assert score < 0.1


def test_a_limitless_tokenizer_falls_back_to_the_default_length(monkeypatch):
    """A ~1e30 tokenizer limit is a sentinel, not a size: windows use DEFAULT_MAX_LENGTH.

    Pinned by the first window's size: the claim ("c", one token) plus the pair
    overhead plus the window fills exactly the default length.
    """
    model = _Model(None)
    premise = " ".join(f"w{i}" for i in range(1_000))
    _entailer(monkeypatch, model).score("c", premise)
    first = _windows(model)[0]
    assert len(first.split()) == ece.DEFAULT_MAX_LENGTH - 1 - ece.PAIR_OVERHEAD
