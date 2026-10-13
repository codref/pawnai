"""Sessions management API (/v1/sessions*, /v1/speakers)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection
from pawn_core.config import SpeakersConfig, VaultConfig
from pawn_server.core import api_server
from pawn_server.core import sessions_api
from pawn_server.core.sessions_api import SessionsApiError, _speakers_summary

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    api = ApiSection(token="tok", stream_keepalive_seconds=0.05)
    cfg = SimpleNamespace(
        api=api,
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=f"sqlite:///{tmp_path / 'unused.db'}",
        vault=VaultConfig(agent_root="Pawn", bucket=""),
        speakers=SpeakersConfig(),
        models=SimpleNamespace(embedding_model="test-embed", hf_token=None),
    )
    return SimpleNamespace(cfg=cfg)


def test_speakers_summary_flags_anonymous() -> None:
    labels = {"SPEAKER_00": {"duration_s": 3.0}, "Davide": {"duration_s": 2.0}}
    mapping = {
        "Davide": SimpleNamespace(display_name="Davide", speaker_id="davide", local_label="Davide")
    }
    pills, unmapped, needs = _speakers_summary(labels, mapping)  # type: ignore[arg-type]
    assert needs is True
    assert unmapped == 1
    assert "SPEAKER_00" in pills
    assert "Davide" in pills


def test_list_sessions_http(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sessions_api,
        "list_sessions_page",
        lambda cfg, *, limit=50, q="", needs_speakers=False, speaker="": {
            "data": [
                {
                    "session_id": "meet-1",
                    "title": "Standup",
                    "updated_at": datetime(2026, 10, 7, tzinfo=timezone.utc).isoformat(),
                    "created_at": None,
                    "segments": 2,
                    "duration_seconds": 8.0,
                    "summary": "Talked about roadmap",
                    "transcript_path": "Pawn/Transcripts/meet-1.md",
                    "speakers": ["Davide", "SPEAKER_00"],
                    "unmapped_count": 1,
                    "needs_speakers": True,
                    "tags": ["roadmap"],
                    "topics": ["Q4 plan"],
                    "sentiment": "neutral",
                    "has_analysis": True,
                }
            ],
            "facets": {
                "speakers": [{"name": "Davide", "count": 1}],
                "needs_speakers": 1,
            },
        },
    )
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.get("/v1/sessions", headers=AUTH, params={"speaker": "Davide"})
        assert resp.status_code == 200
        body = resp.json()
        row = body["data"][0]
        assert row["session_id"] == "meet-1"
        assert row["tags"] == ["roadmap"]
        assert body["facets"]["speakers"][0]["name"] == "Davide"
        assert body["facets"]["needs_speakers"] == 1


def test_get_session_detail_http(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sessions_api,
        "get_session",
        lambda cfg, session_id: {
            "session_id": session_id,
            "title": "Standup",
            "labels": [
                {
                    "local_label": "SPEAKER_00",
                    "duration_s": 5.0,
                    "sample_text": "Hello roadmap",
                    "mapped_speaker_id": None,
                    "mapped_name": None,
                    "has_centroid": True,
                },
                {
                    "local_label": "SPEAKER_01",
                    "duration_s": 1.0,
                    "sample_text": "hi",
                    "mapped_speaker_id": None,
                    "mapped_name": None,
                    "has_centroid": False,
                },
            ],
            "gallery": [],
            "needs_speakers": True,
            "unmapped_count": 2,
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "transcript_path": None,
            "segments": 2,
            "duration_seconds": 6.0,
            "summary": None,
            "updated_at": None,
            "created_at": None,
        },
    )
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.get("/v1/sessions/meet-1", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["labels"][0]["has_centroid"] is True
        assert body["labels"][1]["has_centroid"] is False


def test_get_session_404(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_cfg, _sid):
        raise SessionsApiError("Unknown session: x", status_code=404)

    monkeypatch.setattr(sessions_api, "get_session", boom)
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.get("/v1/sessions/x", headers=AUTH)
        assert resp.status_code == 404


def test_create_speaker_and_enroll_http(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sessions_api,
        "create_speaker",
        lambda cfg, name: {
            "id": "davide",
            "display_name": name,
            "enrollments": 0,
            "created": True,
        },
    )

    def enroll(cfg, session_id, *, speaker, from_label, force=False):
        if from_label == "SPEAKER_01":
            raise SessionsApiError(
                "No session_state embedding for label 'SPEAKER_01'",
                status_code=422,
            )
        return {
            "enrollment_id": "enr-1",
            "speaker_id": "davide",
            "display_name": "Davide",
            "duration": 5.0,
            "quality_score": 1.0,
            "embedding_model": "test-embed",
            "embedding_dim": 4,
        }

    monkeypatch.setattr(sessions_api, "enroll_session_speaker", enroll)

    with TestClient(api_server.create_app(env.cfg)) as client:
        created = client.post("/v1/speakers", headers=AUTH, json={"display_name": "Davide"})
        assert created.status_code == 200
        assert created.json()["id"] == "davide"

        missing = client.post(
            "/v1/sessions/meet-1/enroll",
            headers=AUTH,
            json={"speaker": "Davide", "from_label": "SPEAKER_01"},
        )
        assert missing.status_code == 422

        ok = client.post(
            "/v1/sessions/meet-1/enroll",
            headers=AUTH,
            json={"speaker": "Davide", "from_label": "SPEAKER_00"},
        )
        assert ok.status_code == 200
        assert ok.json()["enrollment_id"] == "enr-1"


def test_reprocess_http(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    async def enqueue(cfg, sid, **kwargs):
        if not kwargs.get("confirm"):
            raise SessionsApiError("confirm must be true", status_code=422)
        mode = kwargs.get("mode", "rediarize")
        if mode not in ("rediarize", "retranscribe"):
            raise SessionsApiError("mode must be 'rediarize' or 'retranscribe'", status_code=422)
        return {
            "job_id": "job-1",
            "session_id": sid,
            "mode": mode,
            "queued": True,
            "command": "retranscribe" if mode == "retranscribe" else "rediarize",
            "audio_files": 2,
            "backend": kwargs.get("backend") if mode == "retranscribe" else None,
            "language": kwargs.get("language"),
            "result": "queued",
        }

    monkeypatch.setattr(sessions_api, "enqueue_session_reprocess", enqueue)

    with TestClient(api_server.create_app(env.cfg)) as client:
        denied = client.post(
            "/v1/sessions/meet-1/reprocess",
            headers=AUTH,
            json={"confirm": False, "mode": "rediarize"},
        )
        assert denied.status_code == 422

        bad_mode = client.post(
            "/v1/sessions/meet-1/reprocess",
            headers=AUTH,
            json={"confirm": True, "mode": "nope"},
        )
        assert bad_mode.status_code == 422

        ok = client.post(
            "/v1/sessions/meet-1/reprocess",
            headers=AUTH,
            json={
                "confirm": True,
                "mode": "retranscribe",
                "backend": "whisper",
                "language": "it",
                "threshold": 0.65,
                "cross_file_threshold": 0.5,
            },
        )
        assert ok.status_code == 202, ok.text
        body = ok.json()
        assert body["queued"] is True
        assert body["mode"] == "retranscribe"
        assert body["language"] == "it"
        assert body["job_id"] == "job-1"


def test_reidentify_and_relabel_http(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sessions_api,
        "reidentify_session_api",
        lambda cfg, sid: {
            "session_id": sid,
            "segments_updated": 3,
            "matched": {"SPEAKER_00": "Davide"},
            "unmatched": ["SPEAKER_01"],
            "summary": "ok",
            "vault_refreshed": False,
            "vault_error": None,
        },
    )
    monkeypatch.setattr(
        sessions_api,
        "relabel_session_api",
        lambda cfg, sid, *, from_label, to_label: {
            "session_id": sid,
            "from": from_label,
            "to": to_label,
            "segments_updated": 1,
            "summary": "relabeled",
            "vault_refreshed": False,
            "vault_error": None,
        },
    )
    with TestClient(api_server.create_app(env.cfg)) as client:
        rid = client.post("/v1/sessions/meet-1/reidentify", headers=AUTH)
        assert rid.status_code == 200
        assert rid.json()["matched"]["SPEAKER_00"] == "Davide"

        rel = client.post(
            "/v1/sessions/meet-1/relabel",
            headers=AUTH,
            json={"from": "SPEAKER_01", "to": "Alice"},
        )
        assert rel.status_code == 200, rel.text
        assert rel.json()["to"] == "Alice"


def test_enroll_session_speaker_missing_centroid(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    gallery = MagicMock()
    gallery.get_speaker.return_value = None
    gallery.find_speaker_by_name.return_value = None
    gallery.create_speaker.return_value = SimpleNamespace(id="davide", display_name="Davide")
    monkeypatch.setattr(
        "pawn_server.core.sessions_api.SpeakerGallery",
        lambda *a, **k: gallery,
    )
    monkeypatch.setattr("pawn_server.core.sessions_api.get_engine", lambda _dsn: MagicMock())
    monkeypatch.setattr("pawn_server.core.sessions_api.init_db", lambda _e: None)
    monkeypatch.setattr(
        "pawn_server.core.sessions_api.load_session_state",
        lambda *_a, **_k: ({}, 0.0, [], 0),
    )

    with pytest.raises(SessionsApiError, match="embedding"):
        sessions_api.enroll_session_speaker(
            env.cfg,
            "meet-1",
            speaker="Davide",
            from_label="SPEAKER_00",
        )
