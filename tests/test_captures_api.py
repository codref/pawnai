"""Browser snippet capture API (/v1/captures, /v1/sessions)."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.core.session_candidates import SessionCandidate
from pawn_agent.utils.config import ApiSection
from pawn_agent.utils.db import AgentRun, VaultTask
from pawn_core.config import VaultConfig
from pawn_core.database import Base, VaultNote
from pawn_core.vault import VaultStore
from pawn_core.vault_db import upsert_vault_note
from pawn_server.core import api_server
from pawn_server.core.captures import (
    extract_annotations,
    note_has_snippet,
    replace_annotations,
)
from tests.test_vault_store import FakeS3Client

AUTH = {"Authorization": "Bearer tok"}

# 1x1 PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'captures.db'}"
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
        coworker=SimpleNamespace(timezone="UTC"),
        vault_watcher=SimpleNamespace(matrix_target="matrix"),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    store.write(
        "Projects/Roadmap.md",
        "---\npawn: editable\n---\n# Roadmap\n\nBody\n",
        skip_guards=True,
    )
    store.write("Private.md", "# Private\n\nNo edit flag\n", skip_guards=True)

    monkeypatch.setattr("pawn_server.core.captures.vault_store_from_config", lambda _c: store)
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda _c: store)

    return SimpleNamespace(cfg=cfg, store=store, dsn=dsn)


def test_new_page_and_append_order(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {
                    "kind": "new",
                    "title": "Teams agenda",
                    "source_url": "https://t.example/1",
                },
                "snippets": [
                    {
                        "id": "snip-a",
                        "kind": "text",
                        "text": "First",
                        "captured_at": "2026-10-07T10:00:00Z",
                    },
                    {
                        "id": "snip-b",
                        "kind": "text",
                        "text": "Second",
                        "captured_at": "2026-10-07T10:01:00Z",
                    },
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["written"] == ["snip-a", "snip-b"]
        path = body["path"]
        assert path.startswith("Pawn/Captures/")
        assert path.endswith(".md")

        note = env.store.read(path)
        assert "pawn: capture" in note or "pawn:capture" in note.replace(" ", "")
        assert note.index("First") < note.index("Second")
        assert note_has_snippet(note, "snip-a")
        assert note_has_snippet(note, "snip-b")

        again = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "new", "path": path, "title": "Teams agenda"},
                "snippets": [
                    {"id": "snip-c", "kind": "text", "text": "Third"},
                    {"id": "snip-a", "kind": "text", "text": "First again"},
                ],
            },
        )
        assert again.status_code == 200
        assert again.json()["written"] == ["snip-c"]
        assert again.json()["skipped"] == ["snip-a"]
        note2 = env.store.read(path)
        assert "Third" in note2
        assert note2.count("<!-- pawn-snippet:snip-a -->") == 1


def test_image_under_captures_assets(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "new", "title": "Shot"},
                "snippets": [
                    {
                        "id": "img-1",
                        "kind": "image",
                        "data_base64": base64.b64encode(_PNG).decode("ascii"),
                        "media_type": "image/png",
                        "source_url": "https://t.example/shot",
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["images"] == ["Pawn/Captures/assets/img-1.png"]
        assert env.store.exists("Pawn/Captures/assets/img-1.png")
        note = env.store.read(data["path"])
        assert "![[Pawn/Captures/assets/img-1.png]]" in note


def test_session_annotations_splice(env: SimpleNamespace) -> None:
    transcript = (
        "---\npawn: transcript\nsession_id: meet-1\n---\n"
        "# meet-1\n\n"
        "## Speakers\n\n- A\n\n"
        "## Annotations\n"
        "_(Add notes and tags here.)_\n\n"
        "## Transcript\n\nHello\n"
    )
    env.store.write("Pawn/Transcripts/2026-10-07 meet-1.md", transcript, skip_guards=True)
    upsert_vault_note(
        env.dsn,
        session_id="meet-1",
        key="Pawn/Transcripts/2026-10-07 meet-1.md",
    )

    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "session", "session_id": "meet-1"},
                "snippets": [
                    {"id": "ann-1", "kind": "text", "text": "Agenda item one"},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["mode"] == "annotations"
        assert resp.json()["path"] == "Pawn/Transcripts/2026-10-07 meet-1.md"

        note = env.store.read("Pawn/Transcripts/2026-10-07 meet-1.md")
        ann = extract_annotations(note)
        assert "Agenda item one" in ann
        assert "_(Add notes and tags here.)_" not in ann
        assert "## Transcript" in note
        assert note.index("## Annotations") < note.index("## Transcript")
        assert "Agenda item one" in note
        assert "Hello" in note


def test_session_without_transcript_creates_capture(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "session", "session_id": "orphan-9", "title": "Orphan"},
                "snippets": [{"id": "o1", "kind": "text", "text": "chat clip"}],
            },
        )
        assert resp.status_code == 200, resp.text
        path = resp.json()["path"]
        assert path.startswith("Pawn/Captures/")
        note = env.store.read(path)
        assert "session_id: orphan-9" in note
        assert "chat clip" in note


def test_write_denied_outside_pawn(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "note", "path": "Private.md"},
                "snippets": [{"id": "x1", "kind": "text", "text": "nope"}],
            },
        )
        assert resp.status_code == 403

        ok = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "note", "path": "Projects/Roadmap.md"},
                "snippets": [{"id": "x2", "kind": "text", "text": "allowed"}],
            },
        )
        assert ok.status_code == 200, ok.text
        assert "allowed" in env.store.read("Projects/Roadmap.md")


def test_delete_snippet(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        created = client.post(
            "/v1/captures",
            headers=AUTH,
            json={
                "target": {"kind": "new", "title": "Del"},
                "snippets": [
                    {
                        "id": "del-img",
                        "kind": "image",
                        "data_base64": base64.b64encode(_PNG).decode("ascii"),
                        "media_type": "image/png",
                    },
                    {"id": "del-txt", "kind": "text", "text": "keep me"},
                ],
            },
        )
        path = created.json()["path"]
        assert env.store.exists("Pawn/Captures/assets/del-img.png")

        deleted = client.delete(
            "/v1/captures/snippets/del-img",
            headers=AUTH,
            params={"path": path},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        note = env.store.read(path)
        assert not note_has_snippet(note, "del-img")
        assert note_has_snippet(note, "del-txt")
        assert not env.store.exists("Pawn/Captures/assets/del-img.png")


def test_list_captures_and_sessions(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    env.store.write(
        "Pawn/Captures/2026-10-07 Alpha.md",
        "---\npawn: capture\n---\n# Alpha\n\n## Snippets\n\n",
        skip_guards=True,
    )
    env.store.write_bytes("Pawn/Captures/assets/skip.png", b"x", content_type="image/png")

    def fake_sessions(cfg, name_filter="", limit=10):
        return [
            SessionCandidate(
                session_id="meet-1",
                title="Standup",
                updated_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
                created_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
                summary="notes",
                segments=12,
                duration_seconds=120.0,
            )
        ]

    monkeypatch.setattr(
        "pawn_server.core.captures.list_session_candidates_impl",
        fake_sessions,
    )
    upsert_vault_note(
        env.dsn,
        session_id="meet-1",
        key="Pawn/Transcripts/2026-10-07 meet-1.md",
    )

    with TestClient(api_server.create_app(env.cfg)) as client:
        caps = client.get("/v1/captures", headers=AUTH)
        assert caps.status_code == 200
        paths = [row["path"] for row in caps.json()["data"]]
        assert "Pawn/Captures/2026-10-07 Alpha.md" in paths
        assert all("/assets/" not in p for p in paths)

        sessions = client.get("/v1/sessions", headers=AUTH)
        assert sessions.status_code == 200
        row = sessions.json()["data"][0]
        assert row["session_id"] == "meet-1"
        assert row["transcript_path"] == "Pawn/Transcripts/2026-10-07 meet-1.md"
        assert row["title"] == "Standup"


def test_replace_annotations_helper() -> None:
    md = "# t\n\n## Annotations\nold\n\n## Transcript\nx\n"
    out = replace_annotations(md, "new ann\n")
    assert extract_annotations(out).strip() == "new ann"
    assert "## Transcript" in out
