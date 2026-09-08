"""Deterministic entailment scoring by content-word overlap.

The default `Entailer`. Crude by design, and honest about it:

  · It is **code, not a model.** The gate's booleans must be computable and
    testable without a provider, so the DEFAULT grounding check cannot depend on
    an LLM being reachable. A cross-encoder NLI model is the better instrument
    and belongs behind this same port (see `entail_cross_encoder`), but it must
    not be the only option or the gate stops working offline.

  · It catches the failure that matters most and misses subtler ones. A claim
    asserting things the passage never mentions -- "this will corrupt the billing
    ledger" cited against a snippet about an OAuth 500 -- scores near zero. A
    claim that paraphrases its passage scores high. What it cannot detect is a
    claim that reuses the passage's vocabulary while inverting its meaning; that
    needs a real NLI model, and until one is wired this check is a floor rather
    than a ceiling.

Score = the share of the claim's content words that appear in the passage. So it
asks "is this claim talking about what the passage talks about", which is a
necessary condition for entailment, not a sufficient one.
"""
from __future__ import annotations

import re

from registry import adapter

# Words that carry no topical signal, so their presence or absence says nothing
# about whether a claim follows from a passage.
_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being am
and or but if then than so because as of in on at to for from with without
by about into over under again further once here there all any both each few
more most other some such no nor not only own same too very can will just
should now it its it's do does did doing have has had having would could may
might must shall you your we our they their he she his her them us me my
""".split())

_WORD = re.compile(r"[a-z0-9_]+")


def content_words(text: str) -> set[str]:
    """Topical words only, lowercased. Short tokens carry little signal."""
    return {
        w for w in _WORD.findall((text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    }


@adapter("entailer", "lexical")
class LexicalEntailer:
    def score(self, claim: str, passage: str) -> float:
        """Share of the claim's content words present in the passage, 0.0-1.0.

        Returns 0.0 for an empty claim: a claim with no content words cannot be
        shown to follow from anything, and the gate should treat "unverifiable"
        as failing rather than as passing. Fail-closed is the whole point of a
        grounding check.
        """
        claim_words = content_words(claim)
        if not claim_words:
            return 0.0
        passage_words = content_words(passage)
        if not passage_words:
            return 0.0
        return round(len(claim_words & passage_words) / len(claim_words), 4)
