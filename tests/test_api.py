"""The HTTP API's contract: strict typed input, exact output, one envelope, /v1.

A fake graph stands in for the real one, so these pin the API layer alone: what
it accepts, what it refuses (and with which status), and exactly which fields it
sends back.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.deps import set_graph

ENVELOPE = {"success", "message", "data"}


class FakeGraph:
    """Records what the API asked of the graph and answers with canned state."""

    def __init__(self, state=None, snapshot_next=("execute",), raise_on_invoke=None):
        self.state = state or {}
        self.snapshot_next = snapshot_next
        self.raise_on_invoke = raise_on_invoke
        self.invoked: list = []

    async def ainvoke(self, payload, config):
        """Record the call; raise or return the canned state."""
        self.invoked.append((payload, config))
        if self.raise_on_invoke:
            raise self.raise_on_invoke
        return self.state

    async def aget_state(self, _config):
        """A snapshot paused at `snapshot_next` (empty: not waiting)."""
        values = {"_policy_version": "p1", "proposed_action": {"type": "comment"}}
        return SimpleNamespace(next=self.snapshot_next, created_at="t", values=values)


@pytest.fixture
def client():
    yield TestClient(main.app)
    set_graph(None)


def use(graph: FakeGraph) -> FakeGraph:
    set_graph(graph)
    return graph


def test_every_body_is_the_envelope_even_errors(client):
    ok = client.get("/healthz").json()
    assert set(ok) == ENVELOPE and ok["success"] is True and ok["data"] == {"ok": True}
    missing = client.get("/v1/nothing-here")
    assert missing.status_code == 404 and set(missing.json()) == ENVELOPE
    assert missing.json()["success"] is False and missing.json()["data"] is None


def test_the_api_lives_under_v1_and_the_old_paths_are_gone(client):
    use(FakeGraph(state={"route": "human_review"}))
    assert client.post("/run/code_review", json={"target": "HEAD"}).status_code == 404
    assert client.post("/v1/run/code_review", json={"target": "HEAD"}).status_code == 200


def test_an_undeclared_field_is_a_400_naming_it(client):
    graph = use(FakeGraph())
    r = client.post("/v1/run/code_review", json={"target": "HEAD", "tenant_id": "x"})
    assert r.status_code == 400 and "tenant_id" in r.json()["message"]
    assert graph.invoked == [], "a rejected request must never reach the graph"


def test_a_missing_target_is_a_400(client):
    r = client.post("/v1/run/code_review", json={"tenant": "merger_app"})
    assert r.status_code == 400 and "target" in r.json()["message"]


def test_an_unserved_surface_is_a_404_in_the_envelope(client):
    r = client.post("/v1/run/no_such_surface", json={"target": "HEAD"})
    assert r.status_code == 404 and r.json()["success"] is False


def test_a_run_sends_exactly_the_agreed_fields(client):
    use(FakeGraph(state={"route": "ai_investigation", "_policy_version": "p1",
                         "review": SimpleNamespace(decision="escalate_to_human"),
                         "github_token": "SECRET", "evidence": {"files": ["a.py"]}}))
    data = client.post("/v1/run/code_review", json={"target": "HEAD"}).json()["data"]
    assert set(data) == set(main.RunResultV1.model_fields)
    assert data["status"] == "ended" and data["review"] == "escalate_to_human"
    assert "SECRET" not in str(data), "state fields outside the model must never leak out"


def test_a_paused_run_returns_what_a_human_needs(client):
    pause = SimpleNamespace(value={"proposed_action": {"type": "comment", "comment": "x"},
                                   "reasoning": "why", "policy_version": "p1"})
    use(FakeGraph(state={"route": "ai_investigation", "__interrupt__": (pause,)}))
    body = client.post("/v1/run/code_review", json={"target": "HEAD", "thread_id": "t1"}).json()
    assert body["data"]["status"] == "awaiting_approval" and body["data"]["thread_id"] == "t1"
    assert body["data"]["proposed_action"]["type"] == "comment"


@pytest.mark.parametrize("approved", ["yes", 1, "true", None])
def test_approval_takes_a_real_boolean_only(client, approved):
    """A truthy non-boolean used to be read as a refusal and resumed the run; now it
    is a 400 and the run stays paused for a real answer."""
    graph = use(FakeGraph())
    r = client.post("/v1/approve", json={"thread_id": "t1", "approved": approved})
    assert r.status_code == 400 and graph.invoked == []


def test_approving_a_run_that_is_not_waiting_is_a_409(client):
    use(FakeGraph(snapshot_next=()))
    r = client.post("/v1/approve", json={"thread_id": "t1", "approved": True})
    assert r.status_code == 409 and r.json()["success"] is False


def test_a_denial_is_reported_as_refused(client):
    use(FakeGraph(raise_on_invoke=PermissionError("execute() refused")))
    data = client.post("/v1/approve", json={"thread_id": "t1", "approved": False}).json()["data"]
    assert data["status"] == "refused" and data["approved"] is False


def test_an_approved_run_reports_what_was_written(client):
    use(FakeGraph(state={"approved": True, "result": {"review": {"markdown": "r.md"}}}))
    body = client.post("/v1/approve", json={"thread_id": "t1", "approved": True}).json()
    assert body["data"]["status"] == "executed" and body["message"] == "the review was written"


def test_run_status_and_registered_are_enveloped(client):
    use(FakeGraph())
    status = client.get("/v1/runs/t1").json()["data"]
    assert status["awaiting_approval"] is True and status["next"] == ["execute"]
    reg = client.get("/v1/registered").json()
    assert reg["success"] is True and "llm" in reg["data"]["selected"]
