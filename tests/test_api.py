"""The HTTP surface, via TestClient: serve path, state, and the SSE tail."""

import pytest
from fastapi.testclient import TestClient

from coalmine.api.server import ServingApp, create_app


@pytest.fixture
def client(tmp_path):
    serving = ServingApp(
        seed=9, n_queries=40, shadow_rate=1.0, n_judges=1, db_path=tmp_path / "api.db"
    )
    with TestClient(create_app(serving)) as c:
        yield c


def test_serve_returns_response_and_logs_events(client):
    r = client.post("/serve", json={})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"request_index", "query_id", "served_by", "response", "latency_ms"}
    # In shadow stage the challenger serves no user traffic.
    assert body["served_by"] == "champion"
    serving = client.app.state.serving
    assert serving.lifecycle.stage_name == "shadow"


def test_serve_specific_query(client):
    r = client.post("/serve", json={"query_id": "q00003"})
    assert r.status_code == 200
    assert r.json()["query_id"] == "q00003"


def test_state_shape_and_verdicts_flow(client):
    for _ in range(40):
        assert client.post("/serve", json={}).status_code == 200
    state = client.get("/state").json()
    assert state["stage"] == "shadow"
    assert state["share"] == 0.0
    assert state["state"] == "running"
    assert isinstance(state["transitions"], list)


def test_events_stream_emits_sse(client):
    for _ in range(3):
        client.post("/serve", json={})
    # Bounded stream (max_seconds) so the test consumes it to completion —
    # TestClient cannot cancel an infinite SSE generator on early close.
    response = client.get("/events/stream", params={"max_seconds": 1.2})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.text.startswith(": connected")
    assert "request_received" in response.text
    assert "champion_response" in response.text


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "coalmine" in r.text and "EventSource" in r.text
