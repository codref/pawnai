"""Phone bridge API (/v1/devices/*): register, auth, ingest, toggles, purge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection, DevicesConfig
from pawn_agent.utils.db import Device, DeviceCommand, PhoneEvent
from pawn_core.database import Base
from pawn_server.core import api_server, devices

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'devices.db'}"
    engine = sa.create_engine(dsn)
    # knowledge_chunks uses pgvector; SQLite gets a minimal stand-in table.
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE knowledge_chunks (id TEXT PRIMARY KEY, source_ref TEXT)")
        )
    Base.metadata.create_all(
        engine,
        tables=[Device.__table__, PhoneEvent.__table__, DeviceCommand.__table__],
    )
    engine.dispose()

    cfg = SimpleNamespace(
        api=ApiSection(token="tok"),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        devices=DevicesConfig(retention_days=0),
        coworker=SimpleNamespace(timezone="UTC"),
    )
    indexed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(devices, "index_events", lambda _cfg, items: indexed.extend(items) or len(items))
    return SimpleNamespace(cfg=cfg, dsn=dsn, indexed=indexed)


def _register(client: TestClient) -> tuple[str, dict]:
    res = client.post("/v1/devices/register", json={"name": "Pixel"}, headers=AUTH)
    assert res.status_code == 200, res.text
    body = res.json()
    return body["device_id"], {"Authorization": f"Bearer {body['device_token']}"}


def _sms(key: str, body: str = "ciao", ts: int = 1_764_000_000_000) -> dict:
    return {
        "kind": "sms",
        "ts": ts,
        "dedup_key": key,
        "payload": {"address": "+39333111222", "name": "Mario", "body": body, "direction": "received"},
    }


def test_register_requires_shared_token(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        assert client.post("/v1/devices/register", json={"name": "x"}).status_code == 401
        device_id, dev_auth = _register(client)
        # Device token works on its own routes, not on another device or on register.
        assert client.get(f"/v1/devices/{device_id}", headers=dev_auth).status_code == 200
        assert client.get("/v1/devices/other", headers=dev_auth).status_code == 401
        assert client.post("/v1/devices/register", json={}, headers=dev_auth).status_code == 401
        assert client.get(f"/v1/devices/{device_id}", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_ingest_only_enabled_categories_and_idempotent(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        device_id, dev_auth = _register(client)
        # Nothing is enabled yet: everything is rejected.
        res = client.post(f"/v1/devices/{device_id}/events", json={"events": [_sms("a")]}, headers=dev_auth)
        assert res.json() == {"accepted": 0, "duplicates": 0, "rejected": 1}

        res = client.put(
            f"/v1/devices/{device_id}",
            json={"enabled_categories": ["sms", "bogus", "call"], "unifiedpush_endpoint": "https://ntfy.example/up1"},
            headers=dev_auth,
        )
        assert res.status_code == 200
        assert res.json()["enabled_categories"] == ["sms", "call"]
        assert res.json()["unifiedpush"] is True

        batch = [_sms("a"), _sms("b", "dopo"), _sms("a"), {"kind": "location", "ts": 1, "dedup_key": "l"}]
        res = client.post(f"/v1/devices/{device_id}/events", json={"events": batch}, headers=dev_auth)
        # dup inside the batch + location not enabled → 2 rejected
        assert res.json() == {"accepted": 2, "duplicates": 0, "rejected": 2}
        res = client.post(f"/v1/devices/{device_id}/events", json={"events": [_sms("a")]}, headers=dev_auth)
        assert res.json() == {"accepted": 0, "duplicates": 1, "rejected": 0}

    assert len(env.indexed) == 2
    kind, ref, text = env.indexed[0]
    assert kind == "phone_sms"
    assert ref.startswith(f"phone:{device_id}:")
    assert "Mario" in text and "ciao" in text


def test_disabling_category_purges_server_copy(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        device_id, dev_auth = _register(client)
        client.put(f"/v1/devices/{device_id}", json={"enabled_categories": ["sms"]}, headers=dev_auth)
        client.post(f"/v1/devices/{device_id}/events", json={"events": [_sms("a")]}, headers=dev_auth)
        res = client.put(f"/v1/devices/{device_id}", json={"enabled_categories": []}, headers=dev_auth)
        assert res.json()["purged_categories"] == ["sms"]

    engine = sa.create_engine(env.dsn)
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM phone_events")).scalar() == 0


def test_delete_device_revokes_and_purges(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        device_id, dev_auth = _register(client)
        client.put(f"/v1/devices/{device_id}", json={"enabled_categories": ["sms"]}, headers=dev_auth)
        client.post(f"/v1/devices/{device_id}/events", json={"events": [_sms("a"), _sms("b")]}, headers=dev_auth)
        res = client.delete(f"/v1/devices/{device_id}", headers=dev_auth)
        assert res.json() == {"deleted": device_id, "events_removed": 2}
        assert client.get(f"/v1/devices/{device_id}", headers=dev_auth).status_code == 401
        assert client.get("/v1/devices", headers=AUTH).json() == {"devices": []}


def test_batch_limit(env: SimpleNamespace) -> None:
    env.cfg.devices = DevicesConfig(max_batch=2, retention_days=0)
    with TestClient(api_server.create_app(env.cfg)) as client:
        device_id, dev_auth = _register(client)
        res = client.post(
            f"/v1/devices/{device_id}/events",
            json={"events": [_sms("a"), _sms("b"), _sms("c")]},
            headers=dev_auth,
        )
        assert res.status_code == 413


def test_parse_ts_variants() -> None:
    assert devices.parse_ts(1_764_000_000_000) == devices.parse_ts(1_764_000_000)
    assert devices.parse_ts("2026-11-25T10:00:00+01:00").hour == 9
    with pytest.raises(ValueError):
        devices.parse_ts(None)
