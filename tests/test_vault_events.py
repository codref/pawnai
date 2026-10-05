"""In-process vault event ring and the long-poll endpoint."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection
from pawn_server.core import api_server
from pawn_server.core.vault_events import VaultEventBus

AUTH = {"Authorization": "Bearer tok"}


def test_buffered_event_returns_immediately() -> None:
    bus = VaultEventBus()
    seq = bus.publish(["Pawn/Notes/A.md", "Pawn/Notes/A.md"], source="chat", run_id="run-1")
    assert seq == 1
    page = asyncio.run(bus.wait(0, 5))
    assert page["seq"] == 1
    assert page["resync"] is False
    assert page["events"] == [
        {
            "seq": 1,
            "paths": ["Pawn/Notes/A.md"],
            "source": "chat",
            "run_id": "run-1",
        }
    ]
    assert bus.publish([], source="chat") == 1


def test_timeout_with_no_event() -> None:
    bus = VaultEventBus()
    page = asyncio.run(bus.wait(0, 0.05))
    assert page == {"seq": 0, "resync": False, "events": []}


def test_wait_wakes_when_published() -> None:
    bus = VaultEventBus()

    async def _run() -> dict:
        async def later() -> None:
            await asyncio.sleep(0.05)
            bus.publish(["Pawn/Today.md"], source="matrix", run_id="run-2")

        task = asyncio.create_task(later())
        page = await bus.wait(0, 1.0)
        await task
        return page

    page = asyncio.run(_run())
    assert page["events"][0]["paths"] == ["Pawn/Today.md"]
    assert page["events"][0]["source"] == "matrix"


def test_gap_behind_ring_requests_resync() -> None:
    bus = VaultEventBus(maxlen=2)
    for name in ("a.md", "b.md", "c.md", "d.md"):
        bus.publish([name], source="chat", run_id=name)
    # Ring kept seq 3 and 4. A client still at seq 1 missed seq 2.
    missed = bus.snapshot(1)
    assert missed["resync"] is True
    assert missed["events"] == []
    assert missed["seq"] == 4
    caught = bus.snapshot(2)
    assert caught["resync"] is False
    assert [event["seq"] for event in caught["events"]] == [3, 4]


def test_vault_events_endpoint(monkeypatch) -> None:
    bus = VaultEventBus()
    monkeypatch.setattr("pawn_server.core.vault_events.vault_events", bus)
    bus.publish(["Pawn/Notes/A.md"], source="queue", run_id="run-9")
    cfg = SimpleNamespace(api_token="tok", api=ApiSection(token="tok"))
    with TestClient(api_server.create_app(cfg)) as client:
        denied = client.get("/v1/vault/events")
        assert denied.status_code == 401
        resp = client.get(
            "/v1/vault/events",
            headers=AUTH,
            params={"since": 0, "timeout": 0},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["seq"] == 1
    assert body["resync"] is False
    assert body["events"][0]["paths"] == ["Pawn/Notes/A.md"]
    assert body["events"][0]["run_id"] == "run-9"
