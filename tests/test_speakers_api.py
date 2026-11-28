"""Speakers gallery list, rename, merge, and the People note."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from pawn_agent.utils.config import ApiSection
from pawn_agent.utils.db import VaultTask
from pawn_core.config import SpeakersConfig, VaultConfig
from pawn_core.database import (
    SessionAnalysis,
    SessionSpeakerMap,
    Speaker,
    SpeakerEnrollment,
    get_engine,
)
from pawn_core.vault import VaultNotFound
from pawn_diarize.core.speaker_gallery import SpeakerGallery
from pawn_server.core import api_server
from pawn_server.core import sessions_api

AUTH = {"Authorization": "Bearer tok"}


@compiles(JSONB, "sqlite")
def _jsonb_as_json(element, compiler, **kw):  # noqa: ANN001
    return "JSON"


class _Vault:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def exists(self, key: str) -> bool:
        return key in self.files

    def read(self, key: str) -> str:
        if key not in self.files:
            raise VaultNotFound(key)
        return self.files[key]

    def write(self, key: str, body: str, skip_guards: bool = False) -> None:
        self.files[key] = body


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=f"sqlite:///{tmp_path / 'speakers.db'}",
        vault=VaultConfig(agent_root="Pawn", bucket=""),
        speakers=SpeakersConfig(),
        coworker=SimpleNamespace(people_dir="People"),
        models=SimpleNamespace(embedding_model="test-embed", hf_token=None),
        s3=None,
    )


def _prepare(cfg: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> _Vault:
    monkeypatch.setattr("pawn_diarize.core.speaker_gallery.init_db", lambda engine: None)
    engine = get_engine(cfg.db_dsn)
    for table in (
        Speaker.__table__,
        SpeakerEnrollment.__table__,
        SessionSpeakerMap.__table__,
        SessionAnalysis.__table__,
        VaultTask.__table__,
    ):
        table.create(engine, checkfirst=True)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add_all(
            [
                Speaker(
                    id="ada",
                    display_name="Ada",
                    aliases=["A. Lovelace"],
                    notes="lead",
                    active=True,
                    created_at=now,
                    updated_at=now,
                ),
                Speaker(
                    id="bob",
                    display_name="Bob",
                    aliases=[],
                    notes=None,
                    active=True,
                    created_at=now,
                    updated_at=now,
                ),
            ]
        )
        db.add(
            SpeakerEnrollment(
                id="enr-ada",
                speaker_id="ada",
                embedding=[0.1, 0.2],
                embedding_model="test",
                embedding_dim=2,
                source_session_id="weekly",
                start_time=12.0,
                end_time=40.0,
                duration=28.0,
                quality_score=0.9,
                approved_at=now,
            )
        )
        db.add(
            SessionSpeakerMap(
                session_id="weekly",
                local_label="SPEAKER_00",
                speaker_id="ada",
                display_name="Ada",
                match_method="manual",
                updated_at=now,
            )
        )
        db.add(
            SessionSpeakerMap(
                session_id="weekly",
                local_label="SPEAKER_01",
                speaker_id="bob",
                display_name="Bob",
                match_method="manual",
                updated_at=now,
            )
        )
        db.add(
            SessionAnalysis(
                id="an-weekly",
                session_id="weekly",
                source="test",
                model="test",
                title="Product weekly",
            )
        )
        db.commit()
    vault = _Vault()
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda cfg: vault)
    monkeypatch.setattr("pawn_agent.core.people.notes.vault_store_from_config", lambda cfg: vault)
    return vault


def test_merge_moves_enrollments_and_deactivates_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg, monkeypatch)
    gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
    dest, affected = gallery.merge_speakers("ada", "bob")
    assert dest.id == "bob"
    assert affected == ["weekly"]
    assert "Ada" in dest.aliases
    assert "A. Lovelace" in dest.aliases
    assert dest.display_name == "Bob"
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        enrollment = db.get(SpeakerEnrollment, "enr-ada")
        assert enrollment is not None
        assert enrollment.speaker_id == "bob"
        mapped = db.get(SessionSpeakerMap, {"session_id": "weekly", "local_label": "SPEAKER_00"})
        assert mapped is not None
        assert mapped.speaker_id == "bob"
        assert mapped.display_name == "Bob"
        source = db.get(Speaker, "ada")
        assert source is not None
        assert source.active is False
    with pytest.raises(ValueError, match="itself"):
        gallery.merge_speakers("bob", "bob")


def test_speakers_http_list_patch_merge_and_people(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    vault = _prepare(cfg, monkeypatch)
    seen: list[str] = []

    def fake_reidentify(cfg, session_id):  # noqa: ANN001
        seen.append(session_id)
        return {"session_id": session_id}

    monkeypatch.setattr(sessions_api, "reidentify_session_api", fake_reidentify)
    with TestClient(api_server.create_app(cfg)) as client:
        listed = client.get("/v1/speakers", headers=AUTH)
        assert listed.status_code == 200, listed.text
        names = [row["display_name"] for row in listed.json()["data"]]
        assert names == ["Ada", "Bob"]
        assert listed.json()["data"][0]["enrollments"] == 1
        assert listed.json()["data"][0]["sessions"] == 1
        assert listed.json()["data"][0]["people_path"] is None

        found = client.get("/v1/speakers", headers=AUTH, params={"q": "lovelace"})
        assert [row["id"] for row in found.json()["data"]] == ["ada"]

        detail = client.get("/v1/speakers/ada", headers=AUTH)
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["enrollment_rows"][0]["source_session_id"] == "weekly"
        assert body["recent_sessions"][0]["title"] == "Product weekly"

        missing = client.get("/v1/speakers/nope", headers=AUTH)
        assert missing.status_code == 404

        renamed = client.patch(
            "/v1/speakers/ada",
            headers=AUTH,
            json={"display_name": "Ada Lovelace", "reidentify": True},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["display_name"] == "Ada Lovelace"
        assert renamed.json()["referenced_sessions"] == 1
        assert renamed.json()["job_id"]
        assert seen == ["weekly"]

        empty = client.patch("/v1/speakers/ada", headers=AUTH, json={"display_name": "  "})
        assert empty.status_code == 422

        created = client.post("/v1/speakers/ada/people", headers=AUTH)
        assert created.status_code == 200, created.text
        assert created.json()["created"] is True
        assert created.json()["path"] == "People/ada.md"
        assert "Ada Lovelace" in vault.files["People/ada.md"]
        again = client.post("/v1/speakers/ada/people", headers=AUTH)
        assert again.json()["created"] is False

        listed_again = client.get("/v1/speakers/ada", headers=AUTH)
        assert listed_again.json()["people_path"] == "People/ada.md"

        gone = client.delete("/v1/speakers/ada/enrollments/missing", headers=AUTH)
        assert gone.status_code == 404
        removed = client.delete("/v1/speakers/ada/enrollments/enr-ada", headers=AUTH)
        assert removed.status_code == 204
        assert client.get("/v1/speakers/ada", headers=AUTH).json()["enrollment_rows"] == []

        itself = client.post("/v1/speakers/bob/merge", headers=AUTH, json={"into": "bob"})
        assert itself.status_code == 422

        merged = client.post("/v1/speakers/ada/merge", headers=AUTH, json={"into": "bob"})
        assert merged.status_code == 200, merged.text
        payload = merged.json()
        assert payload["into"] == "bob"
        assert payload["affected_sessions"] == ["weekly"]
        assert "Ada Lovelace" in payload["speaker"]["aliases"]
        assert payload["job_id"] is None

        active = client.get("/v1/speakers", headers=AUTH)
        assert [row["id"] for row in active.json()["data"]] == ["bob"]
        everyone = client.get("/v1/speakers", headers=AUTH, params={"include_inactive": True})
        ids = {row["id"]: row["active"] for row in everyone.json()["data"]}
        assert ids["ada"] is False
        assert ids["bob"] is True

        off = client.delete("/v1/speakers/bob", headers=AUTH)
        assert off.status_code == 200, off.text
        assert off.json()["active"] is False
