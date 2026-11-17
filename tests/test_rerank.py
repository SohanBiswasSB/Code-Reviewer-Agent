"""T13: the reranker, and the gate's bar-by-score-kind contract.

No model is loaded here. A fake reranker stands in for the cross-encoder, the way
FakeEmbedder stands in for LM Studio in test_index.py. One test, marked
`needs_model`, loads the real one and is deselected by default.
"""
import json
import urllib.error

import pytest

import config
from adapters import rerank_gateway
from adapters.rerank_cross_encoder import CrossEncoderReranker, passage_text
from adapters.rerank_gateway import GatewayReranker
from adapters.retriever_indexed import IndexedRetriever
from adapters.vector_index_local import LocalVectorIndex
from errors import ProviderAuthError, ProviderError
from nodes.independent_review import _bar_for, _groundedness, effective_gates
from tests.test_index import CORPUS, FakeEmbedder, _build


@pytest.fixture
def built(tmp_path):
    """The test_index corpus, built here rather than imported: importing another
    module's fixture only works by shadowing it, which is what a lint rule flags."""
    index = LocalVectorIndex(root=tmp_path / "index")
    _build(index, "org/backend", [dict(c) for c in CORPUS])
    return index


class FakeReranker:
    """Scores a passage by whether it contains a keyword; records what it saw."""

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.seen: list[int] = []

    def rerank(self, query, candidates, top_n=8):
        self.seen.append(len(candidates))
        scored = [{**c, "rerank": 0.97 if self.keyword in c["text"] else 0.02}
                  for c in candidates]
        scored.sort(key=lambda c: c["rerank"], reverse=True)
        return scored[:top_n]


def test_the_reranker_supplies_the_gate_score_and_says_so(built):
    rr = FakeReranker("hmac")
    r = IndexedRetriever(embedder=FakeEmbedder(), index=built, reranker=rr)
    hits = r.retrieve("oauth callback state", k=2, namespace="org/backend")
    assert rr.seen, "the reranker must be called"
    top = hits[0]
    assert top["id"] == "webhook", "the reranker's order wins over fusion's"
    assert top["score"] == 0.97 and top["score_kind"] == "rerank"
    assert 0.0 <= top["cosine"] <= 1.0, "the cosine is kept beside the rerank score"
    assert [h["rank"] for h in hits] == [1, 2]


def test_the_reranker_sees_the_wide_pool_not_just_k(built):
    rr = FakeReranker("x")
    IndexedRetriever(embedder=FakeEmbedder(), index=built, reranker=rr).retrieve(
        "oauth", k=1, namespace="org/backend")
    assert rr.seen[0] > 1, "reranking only the top k could never change which k win"


def test_without_a_reranker_the_score_is_the_cosine(built):
    hits = IndexedRetriever(embedder=FakeEmbedder(), index=built, reranker=None).retrieve(
        "oauth", k=2, namespace="org/backend")
    assert all(h["score_kind"] == "cosine" and h["score"] == h["cosine"] for h in hits)


def test_reranker_none_is_honoured(monkeypatch):
    monkeypatch.setenv("RERANKER", "none")
    assert config.get_reranker() is None


def test_the_reranker_is_off_by_default(monkeypatch):
    """OFF by default, on measurement -- see config.get_reranker."""
    monkeypatch.delenv("RERANKER", raising=False)
    assert config.get_reranker() is None


def test_the_cross_encoder_is_selectable_and_loads_nothing_until_used(monkeypatch):
    monkeypatch.setenv("RERANKER", "cross_encoder")
    r = config.get_reranker()
    assert isinstance(r, CrossEncoderReranker) and r._encoder is None


def test_a_missing_model_is_a_clear_error_not_a_download(monkeypatch):
    r = CrossEncoderReranker(model="no-such-org/no-such-reranker", device="cpu")
    with pytest.raises(ProviderError, match="never downloads"):
        r.scores("q", ["p"])


def test_the_passage_carries_its_path_and_name():
    assert passage_text({"path": "app/x.py", "name": "f", "text": "body"}) == \
        "app/x.py :: f\nbody"
    assert passage_text({"text": "body"}) == "body"


# ── the gate: one bar per KIND of score ─────────────────────────────────────
def test_each_kind_of_score_is_gated_on_its_own_bar():
    bars = effective_gates({"min_relevance_score": 0.75, "min_rerank_score": 0.5})
    assert _bar_for({"score_kind": "rerank"}, bars) == 0.5
    assert _bar_for({"score_kind": "cosine"}, bars) == 0.75
    assert _bar_for({}, bars) == 0.75, "a pre-T13 hit has no kind and keeps the cosine bar"


def test_a_rerank_score_is_never_compared_with_the_cosine_bar():
    """0.6 passes a 0.5 rerank bar and would FAIL a 0.75 cosine bar -- the gate
    must use the bar the number was measured for."""
    hits = [{"id": "a", "score": 0.6, "score_kind": "rerank"},
            {"id": "b", "score": 0.6, "score_kind": "cosine"}]
    bars = effective_gates({"min_relevance_score": 0.75, "min_rerank_score": 0.5})
    per_id = {h["id"]: _bar_for(h, bars) for h in hits}
    scores = {h["id"]: h["score"] for h in hits}
    assert _groundedness(["a"], scores, per_id) == (True, [])
    ok, failures = _groundedness(["b"], scores, per_id)
    assert not ok and "b (0.6 < 0.75)" in failures[0]


@pytest.mark.needs_model
def test_the_real_cross_encoder_separates_relevant_from_irrelevant():
    r = CrossEncoderReranker()
    q = ("Masterclass/models.py\n+class QuestionBank(models.Model):\n"
         "+    unique_together = ('course', 'name')")
    good, bad = r.scores(q, [
        "class QuestionBank(models.Model):\n    course = models.ForeignKey(Course)",
        "def send_email(user):\n    smtp.send(user.email)"])
    # Separation, not an absolute level: the same relevant passage scored 0.95 or
    # 0.25 depending on how much of the model it showed. The LEVEL is what
    # tools.calibrate measures for min_rerank_score; the ORDER is the model's job.
    assert good > 0.1 and bad < 0.01 and good > 20 * bad


# ── the gateway's reranker (RERANKER=gateway) ───────────────────────────────



class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _gateway(monkeypatch, results=None, error=None):
    sent = []

    def urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        if error:
            raise error
        return _Resp({"results": results})
    monkeypatch.setattr(rerank_gateway.urllib.request, "urlopen", urlopen)
    return sent


def test_gateway_scores_map_by_index_through_the_same_sigmoid(monkeypatch):
    """Results may come back sorted by score; mapping by position would give
    passages each other's scores."""
    sent = _gateway(monkeypatch, [{"index": 1, "relevance_score": -11.0},
                                  {"index": 0, "relevance_score": 3.0}])
    r = GatewayReranker(url="https://gw.example/v1")
    assert r.scores("q", ["good", "bad"]) == [0.9526, 0.0]
    assert sent[0]["model"] == "BAAI/bge-reranker-v2-m3" and sent[0]["documents"] == ["good", "bad"]


def test_gateway_rerank_orders_and_keeps_top_n(monkeypatch):
    _gateway(monkeypatch, [{"index": 0, "relevance_score": -5.0},
                           {"index": 1, "relevance_score": 4.0}])
    out = GatewayReranker(url="https://gw.example/v1").rerank(
        "q", [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], top_n=1)
    assert [c["id"] for c in out] == ["b"] and out[0]["rerank"] > 0.9


def test_a_missing_score_is_an_error_never_a_default(monkeypatch):
    _gateway(monkeypatch, [{"index": 0, "relevance_score": 1.0}])
    with pytest.raises(ProviderError, match="scored 1 of 2"):
        GatewayReranker(url="https://gw.example/v1").scores("q", ["a", "b"])


def test_a_refused_key_is_an_auth_error_not_retried(monkeypatch):
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    sent = _gateway(monkeypatch, error=err)
    with pytest.raises(ProviderAuthError, match="refused the API key"):
        GatewayReranker(url="https://gw.example/v1").scores("q", ["a"])
    assert len(sent) == 1, "a wrong key does not improve on a retry"


def test_reranker_gateway_is_selectable(monkeypatch):
    monkeypatch.setenv("RERANKER", "gateway")
    assert isinstance(config.get_reranker(), GatewayReranker)


@pytest.mark.needs_server
def test_the_real_gateway_separates_relevant_from_irrelevant():
    good, bad = GatewayReranker().scores(
        "Masterclass/models.py\n+class QuestionBank(models.Model):",
        ["class QuestionBank(models.Model):\n    course = models.ForeignKey(Course)",
         "def send_email(user):\n    smtp.send(user.email)"])
    assert good > 0.5 > bad
