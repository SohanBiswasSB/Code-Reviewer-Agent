"""Retrieval invariants — chiefly the one T21 broke.

The defect: `retrieve()` returned the RRF fused score divided by the top hit's,
which forces the best result to exactly 1.0 on every query. `groundedness_ok`
therefore could not fail at any threshold below ~0.97, and the relevance half of
the safety gate did nothing. It stayed invisible because every gate task
hand-supplied its scores.

These tests pin the two properties that keep it fixed: relevance must be a
calibrated magnitude with a real noise floor, and it must never be the same
number as the rank-fusion value.
"""
import pytest

from tools import retrieval

ON_TOPIC = [
    "OAuth callback returns 500 when state param missing",
    "token refresh race intermittent 401 after login",
]
OFF_TOPIC = [
    "banana bread recipe with walnuts",
    "quarterly revenue forecast spreadsheet",
    "zzzz qqqq xyzzy plugh",
]


def _best_relevance(query: str) -> float:
    hits = retrieval.retrieve(query, k=3)
    return max((h.relevance for h in hits), default=0.0)


@pytest.mark.parametrize("query", OFF_TOPIC)
def test_off_topic_queries_have_a_noise_floor(query):
    """The property a calibrated threshold needs: unrelated text scores at/near 0.

    Without this there is no noise cluster to set a threshold above, which is
    exactly why a rank-derived score cannot be gated on.
    """
    assert _best_relevance(query) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("query", ON_TOPIC)
def test_on_topic_queries_score_well_above_the_floor(query):
    """The other half of a usable threshold: real matches sit clearly above the noise."""
    assert _best_relevance(query) > 0.3


def test_on_topic_and_off_topic_are_separated():
    """The single assertion that would have caught T21."""
    worst_on = min(_best_relevance(q) for q in ON_TOPIC)
    best_off = max(_best_relevance(q) for q in OFF_TOPIC)
    assert worst_on > best_off, (
        f"on-topic ({worst_on}) must outscore off-topic ({best_off}); "
        f"if these are equal the relevance score carries no information"
    )


def test_relevance_is_not_the_rank_fusion_value():
    """Gating on a rank is the defect. Keep the two numbers distinct."""
    hits = retrieval.retrieve(ON_TOPIC[0], k=3)
    assert hits
    assert any(h.relevance != h.fusion for h in hits)


def test_relevance_varies_by_query_but_fusion_does_not():
    """Fusion depends only on rank position, so it is identical across queries.

    This is not a bug in RRF -- it is what RRF *is*. It is the reason the two
    numbers must stay separate: the invariant below is precisely why fusion
    cannot serve as a relevance signal.
    """
    top_fusions = {retrieval.retrieve(q, k=3)[0].fusion for q in ON_TOPIC + OFF_TOPIC}
    top_relevances = {retrieval.retrieve(q, k=3)[0].relevance for q in ON_TOPIC + OFF_TOPIC}
    assert len(top_fusions) == 1, "fusion should be rank-determined and query-independent"
    assert len(top_relevances) > 1, "relevance must respond to the query"


def test_fusion_still_orders_the_results():
    """The fix must not cost us the hybrid ordering RRF exists to provide."""
    hits = retrieval.retrieve(ON_TOPIC[0], k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    fusions = [h.fusion for h in hits]
    assert fusions == sorted(fusions, reverse=True)


def test_score_alias_points_at_relevance_not_fusion():
    """The gate and the eval tasks read `.score`. It must be the gateable number."""
    hit = retrieval.retrieve(ON_TOPIC[0], k=1)[0]
    assert hit.score == hit.relevance
    assert hit.score != hit.fusion


def test_bm25_only_hits_still_carry_a_relevance_score():
    """A keyword-only hit must not arrive with no gateable score and pass by
    default -- which is why relevance is computed over the candidate union."""
    for hit in retrieval.retrieve("webhook secret token configure", k=4):
        assert hit.relevance is not None
        assert 0.0 <= hit.relevance <= 1.0
