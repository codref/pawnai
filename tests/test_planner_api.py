"""Day planner API (/v1/planner/*) and event capture kind."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection, CaptureConfig, PlannerConfig
from pawn_agent.utils.db import AgentRun, VaultTask
from pawn_core.config import VaultConfig
from pawn_core.database import Base, VaultNote
from pawn_core.vault import VaultStore, parse_frontmatter
from pawn_server.core import api_server
from pawn_server.core.planner import (
    build_day,
    enrich_event_fields_from_content,
    event_fingerprint,
    is_stub_event_title,
    parse_event_slash_arg,
    replace_prep_section,
    title_similarity,
    upsert_event,
)
from tests.test_vault_store import FakeS3Client

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'planner.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(
        engine,
        tables=[AgentRun.__table__, VaultTask.__table__, VaultNote.__table__],
    )
    engine.dispose()

    api = ApiSection(token="tok", stream_keepalive_seconds=0.05)
    cfg = SimpleNamespace(
        api=api,
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        vault=VaultConfig(agent_root="Pawn"),
        coworker=SimpleNamespace(timezone="UTC", goals_path="Goals.md", enabled=False),
        vault_watcher=SimpleNamespace(matrix_target="matrix"),
        capture=CaptureConfig(auto_enrich=False),
        planner=PlannerConfig(events_dir="{agent_root}/Events"),
        agent=SimpleNamespace(default=""),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())

    monkeypatch.setattr("pawn_server.core.planner.vault_store_from_config", lambda _c: store)
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda _c: store)
    monkeypatch.setattr("pawn_server.core.captures.vault_store_from_config", lambda _c: store)

    return SimpleNamespace(cfg=cfg, store=store, dsn=dsn)


def test_upsert_fingerprint_and_slash_parse(env: SimpleNamespace) -> None:
    first = upsert_event(
        env.cfg,
        title="Product weekly",
        start="2026-10-28T10:00",
        end="2026-10-28T10:30",
        source="manual",
        store=env.store,
    )
    assert first["path"].startswith("Pawn/Events/2026-10-28-")
    assert first["id"].startswith("evt_")
    note = env.store.read(first["path"])
    assert "pawn: event" in note or "pawn:event" in note.replace(" ", "")

    second = upsert_event(
        env.cfg,
        title="Product weekly",
        start="2026-10-28T10:00",
        source="manual",
        body="Updated body",
        store=env.store,
    )
    assert second["path"] == first["path"]
    assert second["id"] == first["id"]
    assert "Updated body" in env.store.read(second["path"])

    url_fp = event_fingerprint(source_url="https://outlook.office.com/x", title="A", start="")
    other = event_fingerprint(source_url="https://outlook.office.com/y", title="A", start="")
    assert url_fp != other

    title, start = parse_event_slash_arg("Coffee with Ada 2026-10-29T09:00")
    assert title == "Coffee with Ada"
    assert start == "2026-10-29T09:00"


def test_delete_event(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Throwaway",
        start="2026-10-28T14:00",
        store=env.store,
    )
    path = ev["path"]
    assert env.store.exists(path)
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.delete(f"/v1/planner/events/{ev['id']}", headers=AUTH)
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] is True
        again = client.delete(f"/v1/planner/events/{ev['id']}", headers=AUTH)
        assert again.status_code == 404
    assert not env.store.exists(path)


def test_replace_prep_and_link(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Sync",
        start="2026-10-28",
        store=env.store,
    )
    text = env.store.read(ev["path"])
    updated = replace_prep_section(text, "- [[Pawn/Transcripts/x]] prior decision", prep_job_id="j1")
    env.store.write(ev["path"], updated)
    meta, body = parse_frontmatter(updated)
    assert meta.get("prep_job_id") == "j1"
    assert "## Prep" in body
    assert "prior decision" in body

    with TestClient(api_server.create_app(env.cfg)) as client:
        link = client.post(
            f"/v1/planner/events/{ev['id']}/link",
            headers=AUTH,
            json={"session_id": "sess-1"},
        )
        assert link.status_code == 200, link.text
        assert link.json()["session_id"] == "sess-1"

        unlink = client.post(
            f"/v1/planner/events/{ev['id']}/link",
            headers=AUTH,
            json={"session_id": None},
        )
        assert unlink.status_code == 200
        assert not unlink.json().get("session_id")


def test_planner_day_and_capture_event(env: SimpleNamespace) -> None:
    upsert_event(
        env.cfg,
        title="Standup",
        start="2026-10-28T09:00",
        store=env.store,
    )
    fake_sessions = [
        {
            "session_id": "standup-audio",
            "title": "Standup recording",
            "created_at": "2026-10-28T09:05:00+00:00",
            "updated_at": "2026-10-28T09:05:00+00:00",
            "transcript_path": None,
            "day": "2026-10-28",
        }
    ]
    with patch(
        "pawn_server.core.planner._sessions_for_day",
        return_value=fake_sessions,
    ):
        day = build_day(env.cfg, "2026-10-28", store=env.store)
    assert day["date"] == "2026-10-28"
    assert len(day["events"]) == 1
    assert any(b["kind"] == "event" for b in day["blocks"])
    assert day["suggestions"]  # title overlap standup

    with patch(
        "pawn_server.core.planner._sessions_for_day",
        return_value=fake_sessions,
    ):
        with TestClient(api_server.create_app(env.cfg)) as client:
            resp = client.get("/v1/planner/day?date=2026-10-28", headers=AUTH)
            assert resp.status_code == 200, resp.text
            assert resp.json()["date"] == "2026-10-28"

    with TestClient(api_server.create_app(env.cfg)) as client:
        create = client.post(
            "/v1/planner/events",
            headers=AUTH,
            json={
                "title": "Outlook sync",
                "start": "2026-10-29T11:00",
                "end": "2026-10-29T11:30",
                "source": "outlook",
                "source_url": "https://outlook.office.com/calendar/item/abc",
                "attendees": ["Ada"],
            },
        )
        assert create.status_code == 200, create.text
        assert create.json()["source"] == "outlook"

        cap = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {
                    "kind": "event",
                    "title": "Captured meet",
                    "start": "2026-10-30T15:00",
                    "source_url": "https://outlook.live.com/calendar/0/item/1",
                },
                "snippets": [
                    {
                        "id": "snip-evt-1",
                        "kind": "text",
                        "text": "Agenda: roadmap",
                    }
                ],
            },
        )
        assert cap.status_code == 200, cap.text
        body = cap.json()
        assert body["mode"] == "event"
        assert body["path"].startswith("Pawn/Events/")
        note = env.store.read(body["path"])
        assert "Agenda: roadmap" in note
        assert "<!-- pawn-snippet:snip-evt-1 -->" in note

        # Second save with sticky path + image must append, not leave a stub.
        import base64

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        cap2 = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {
                    "kind": "event",
                    "title": "Captured meet",
                    "start": "2026-10-30T15:00",
                    "source_url": "https://outlook.live.com/calendar/0/item/1",
                    "path": body["path"],
                },
                "snippets": [
                    {
                        "id": "snip-evt-img",
                        "kind": "image",
                        "data_base64": base64.b64encode(png).decode("ascii"),
                        "media_type": "image/png",
                    }
                ],
            },
        )
        assert cap2.status_code == 200, cap2.text
        assert cap2.json()["path"] == body["path"]
        assert "snip-evt-img" in cap2.json()["written"]
        note2 = env.store.read(body["path"])
        assert "Agenda: roadmap" in note2
        assert "<!-- pawn-snippet:snip-evt-img -->" in note2
        assert "![[" in note2 and "Events/assets/snip-evt-img" in note2


def test_title_similarity() -> None:
    assert title_similarity("Product weekly sync", "Product weekly") > 0.3
    assert title_similarity("alpha", "omega") < 0.2


def test_enrich_event_fields_from_content() -> None:
    from pawn_server.core.planner import normalize_local_dt

    assert is_stub_event_title("Calendar | Microsoft Teams")
    assert is_stub_event_title("(2) Outlook")
    assert normalize_local_dt("10/28/2026, 11:00 AM") == "2026-10-28T11:00"
    text = """
Outlook event: Review Insights Connector
Start: 2026-10-28T11:00
End: 2026-10-28T11:30
Mehrdad Kiani (Organizer)
Davide Dal Farra (Required)
Ada Lovelace (Optional)
Location: Building 3 / Room A
Meeting: https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc
Topics: roadmap, connectors
""".strip()
    got = enrich_event_fields_from_content("Calendar | Microsoft Teams", text)
    assert got["title"] == "Review Insights Connector"
    assert got["organizer"] == "Mehrdad Kiani"
    assert "Davide Dal Farra" in got["attendees"]
    assert "Ada Lovelace" in got["attendees"]
    assert got["location"].startswith("Building 3")
    assert "teams.microsoft.com" in got["meeting_url"]
    assert "roadmap" in got["topics"]


def test_event_capture_enriches_stub_title_and_people(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        cap = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {
                    "kind": "event",
                    "title": "Calendar | Microsoft Teams",
                    "start": "2026-10-28T11:00",
                    "source_url": "https://teams.microsoft.com/calendar/0",
                },
                "snippets": [
                    {
                        "id": "snip-enrich-1",
                        "kind": "text",
                        "text": (
                            "Review Insights Connector\n"
                            "Mehrdad Kiani (Organizer)\n"
                            "Davide Dal Farra (Required)\n"
                            "10/28/2026, 11:00 AM – 11:30 AM\n"
                            "Location: Conf Room B\n"
                            "https://teams.microsoft.com/l/meetup-join/xyz\n"
                        ),
                    }
                ],
            },
        )
        assert cap.status_code == 200, cap.text
        body = cap.json()
        assert body["title"] == "Review Insights Connector"
        note = env.store.read(body["path"])
        meta, _ = parse_frontmatter(note)
        assert meta.get("title") == "Review Insights Connector"
        attendees = meta.get("attendees") or []
        assert "Mehrdad Kiani" in attendees
        assert "Davide Dal Farra" in attendees
        assert meta.get("organizer") == "Mehrdad Kiani"
        assert "Conf Room B" in str(meta.get("location") or "")
        assert "teams.microsoft.com" in str(meta.get("meeting_url") or "")


def test_prep_enqueue(env: SimpleNamespace) -> None:
    ev = upsert_event(env.cfg, title="Prep me", start="2026-10-28T12:00", store=env.store)

    async def _noop_prep(*_a, **_k):
        return "ok"

    with patch("pawn_server.core.meeting_prep.run_meeting_prep", side_effect=_noop_prep):
        with TestClient(api_server.create_app(env.cfg)) as client:
            resp = client.post(
                f"/v1/planner/events/{ev['id']}/prep",
                headers=AUTH,
            )
            assert resp.status_code == 202, resp.text
            assert resp.json()["kind"] == "meeting_prep"
