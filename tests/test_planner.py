"""Planner patch, range, session spans, plan text and the event-note cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection, CaptureConfig, PlannerConfig
from pawn_agent.utils.db import AgentRun, VaultTask, update_vault_task, upsert_vault_task
from pawn_core.config import VaultConfig
from pawn_core.database import Base, VaultNote
from pawn_core.vault import VaultStore, parse_frontmatter
from pawn_server.core import api_server
from pawn_server.core.planner import (
    build_day,
    build_range,
    clear_event_cache,
    list_events,
    replace_prep_section,
    session_span,
    update_event,
    upsert_event,
)
from pawn_server.core.vault_events import publish_vault_event
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
    cfg = SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05),
        api_token="tok",
        db_dsn=dsn,
        vault=VaultConfig(agent_root="Pawn"),
        coworker=SimpleNamespace(timezone="UTC", daily_dir="Pawn/Daily"),
        capture=CaptureConfig(auto_enrich=False),
        planner=PlannerConfig(events_dir="{agent_root}/Events"),
        agent=SimpleNamespace(default=""),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    monkeypatch.setattr("pawn_server.core.planner.vault_store_from_config", lambda _c: store)
    clear_event_cache()
    return SimpleNamespace(cfg=cfg, store=store, dsn=dsn)


def _note_with(
    store: VaultStore, path: str, extra_frontmatter: str = "", extra_body: str = ""
) -> None:
    text = store.read(path)
    if extra_frontmatter:
        text = text.replace("pawn: event\n", f"pawn: event\n{extra_frontmatter}\n", 1)
    if extra_body:
        text = text.rstrip() + "\n\n" + extra_body.strip() + "\n"
    store.write(path, text)


def test_session_span_batch_versus_chunked() -> None:
    end = datetime(2026, 11, 27, 10, 44, tzinfo=timezone.utc)
    start, finish = session_span(end, end, 42 * 60)
    assert finish is not None and start is not None
    assert datetime.fromisoformat(finish) == end
    assert datetime.fromisoformat(finish) - datetime.fromisoformat(start) == timedelta(minutes=42)

    first = end - timedelta(minutes=40)
    chunk_start, chunk_end = session_span(first, end, 42 * 60)
    assert chunk_start == first.isoformat()
    assert datetime.fromisoformat(chunk_end) - first == timedelta(minutes=42)

    assert session_span(None, end, 60) == (None, None)


def test_update_event_keeps_prep_snippets_and_renames(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Product weekly",
        start="2026-11-27T09:00",
        end="2026-11-27T09:45",
        source="manual",
        store=env.store,
    )
    snippet = (
        "<!-- pawn-snippet:abc -->\n### Snippet\n\nhello from the page\n"
        "<!-- /pawn-snippet:abc -->"
    )
    prepared = replace_prep_section(env.store.read(ev["path"]), "Bring the launch checklist")
    prepared = prepared.replace("## Prep", snippet + "\n\n## Prep", 1)
    env.store.write(ev["path"], prepared)
    _note_with(env.store, ev["path"], extra_frontmatter="room: 4")
    old_fp = ev["fingerprint"]

    renamed = update_event(
        env.cfg,
        ev["id"],
        title="Launch review",
        start="2026-11-28T09:00",
        focus=True,
        store=env.store,
    )
    assert renamed["path"] != ev["path"]
    assert "2026-11-28" in renamed["path"]
    assert not env.store.exists(ev["path"])
    note = env.store.read(renamed["path"])
    meta, body = parse_frontmatter(note)
    assert meta["room"] == 4
    assert meta["focus"] is True
    assert meta["focus_before"] == 15
    assert "Bring the launch checklist" in body
    assert "pawn-snippet:abc" in body
    assert renamed["fingerprint"] != old_fp
    assert renamed["prep"].startswith("Bring")

    edited = update_event(env.cfg, ev["id"], body="Notes after the move", store=env.store)
    meta, body = parse_frontmatter(env.store.read(edited["path"]))
    assert "Notes after the move" in body
    assert "pawn-snippet:abc" in body
    assert "Bring the launch checklist" in body
    assert meta["room"] == 4


def test_update_event_keeps_outlook_fingerprint(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Outlook sync",
        start="2026-11-27T11:00",
        source="outlook",
        source_url="https://outlook.office.com/calendar/item/abc",
        store=env.store,
    )
    patched = update_event(env.cfg, ev["id"], title="Renamed invite", store=env.store)
    assert patched["fingerprint"] == ev["fingerprint"]
    assert patched["title"] == "Renamed invite"


def test_patch_route_and_clear_session(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Sync",
        start="2026-11-27T11:00",
        session_id="weekly-sync",
        store=env.store,
    )
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.patch(
            f"/v1/planner/events/{ev['id']}",
            headers=AUTH,
            json={"session_id": None, "location": "Room 4"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert not body.get("session_id")
        assert body["location"] == "Room 4"


def test_build_range_loads_events_once(env: SimpleNamespace) -> None:
    upsert_event(env.cfg, title="Monday", start="2026-11-23T09:00", store=env.store)
    upsert_event(env.cfg, title="Wednesday", start="2026-11-25T09:00", store=env.store)
    calls = {"events": 0, "sessions": 0}
    real_list = list_events

    def _list(*args, **kwargs):
        calls["events"] += 1
        return real_list(*args, **kwargs)

    def _candidates(*_args, **_kwargs):
        calls["sessions"] += 1
        return []

    with (
        patch("pawn_server.core.planner.list_events", _list),
        patch("pawn_agent.tools.list_sessions.list_session_candidates_impl", _candidates),
    ):
        payload = build_range(env.cfg, "2026-11-23", "2026-11-25", store=env.store)
    assert calls["events"] == 1
    assert calls["sessions"] == 1
    assert [day["date"] for day in payload["days"]] == [
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
    ]
    titles = [ev["title"] for day in payload["days"] for ev in day["events"]]
    assert titles == ["Monday", "Wednesday"]

    with TestClient(api_server.create_app(env.cfg)) as client:
        too_wide = client.get(
            "/v1/planner/range",
            headers=AUTH,
            params={"from": "2026-11-01", "to": "2026-11-20"},
        )
        assert too_wide.status_code == 422


def test_event_cache_invalidates_on_vault_event(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Cached",
        start="2026-11-27T08:00",
        store=env.store,
    )
    clear_event_cache()
    reads: list[str] = []
    original = env.store.read

    def _read(key: str) -> str:
        reads.append(key)
        return original(key)

    env.store.read = _read  # type: ignore[method-assign]
    assert list_events(env.cfg, date_from="2026-11-27", date_to="2026-11-27", store=env.store)
    assert reads
    reads.clear()
    again = list_events(env.cfg, date_from="2026-11-27", date_to="2026-11-27", store=env.store)
    assert again[0]["title"] == "Cached"
    assert reads == []

    full = env.store._full_key(ev["path"])
    obj = env.store._client.objects[full]
    obj["body"] = original(ev["path"]).replace("Cached", "Refreshed").encode()
    stale = list_events(env.cfg, date_from="2026-11-27", date_to="2026-11-27", store=env.store)
    assert stale[0]["title"] == "Cached"
    publish_vault_event([ev["path"]], source="test")
    fresh = list_events(env.cfg, date_from="2026-11-27", date_to="2026-11-27", store=env.store)
    assert fresh[0]["title"] == "Refreshed"


def test_plan_section_and_prep_job_status(env: SimpleNamespace) -> None:
    ev = upsert_event(
        env.cfg,
        title="Standup",
        start="2026-11-27T09:00",
        store=env.store,
    )
    prepared = replace_prep_section(
        env.store.read(ev["path"]), "Skim the board", prep_job_id="job-1"
    )
    env.store.write(ev["path"], prepared)
    env.store.write("Pawn/Daily/2026-11-27.md", "# 2026-11-27\n\n## Plan\n\n- Ship the day view\n")
    upsert_vault_task(
        env.dsn,
        task_id="job-1",
        key="Pawn/Jobs/job-1.md",
        instruction_hash="h",
        conversation_id="c",
        status="running",
        kind="meeting_prep",
    )
    day = build_day(env.cfg, "2026-11-27", store=env.store)
    assert day["plan"] == "- Ship the day view"
    assert day["plan_path"] == "Pawn/Daily/2026-11-27.md"
    assert day["events"][0]["prep_job_status"] == "running"

    update_vault_task(env.dsn, "job-1", status="done")
    done = build_day(env.cfg, "2026-11-27", store=env.store)
    assert done["events"][0]["prep_job_status"] is None
