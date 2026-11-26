"""phone_events / phone_devices read tools over stored phone-bridge data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.tools.phone_impl import parse_since, phone_devices_impl, phone_events_impl
from pawn_agent.utils.db import Device, DeviceCommand, PhoneEvent
from pawn_core.database import Base
from pawn_server.core import devices


@pytest.fixture
def cfg(tmp_path: Path) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'phone.db'}"
    engine = sa.create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE knowledge_chunks (id TEXT PRIMARY KEY, source_ref TEXT)"))
    Base.metadata.create_all(
        engine, tables=[Device.__table__, PhoneEvent.__table__, DeviceCommand.__table__]
    )
    engine.dispose()
    return SimpleNamespace(db_dsn=dsn, coworker=SimpleNamespace(timezone="Europe/Rome"))


def _seed(cfg: SimpleNamespace) -> str:
    reg = devices.register_device(cfg.db_dsn, "Pixel")
    device_id = reg["device_id"]
    devices.update_device(cfg.db_dsn, device_id, enabled_categories=["sms", "call", "status"])
    now = datetime.utcnow()
    events = [
        {
            "kind": "sms",
            "ts": (now - timedelta(hours=1)).isoformat(),
            "dedup_key": "s1",
            "payload": {"address": "+39 333 111 2222", "name": "Mario", "body": "arrivo alle 8", "direction": "received"},
        },
        {
            "kind": "call",
            "ts": (now - timedelta(hours=2)).isoformat(),
            "dedup_key": "c1",
            "payload": {"number": "+390612345678", "name": "Anna", "direction": "missed"},
        },
        {
            "kind": "status",
            "ts": (now - timedelta(minutes=5)).isoformat(),
            "dedup_key": "st1",
            "payload": {"battery_pct": 42, "charging": True, "network": "wifi"},
        },
        {
            "kind": "sms",
            "ts": (now - timedelta(days=3)).isoformat(),
            "dedup_key": "s-old",
            "payload": {"address": "+39 333 111 2222", "name": "Mario", "body": "vecchio", "direction": "received"},
        },
    ]
    devices.ingest_events(cfg.db_dsn, device_id, events)
    return device_id


def test_no_device_message(cfg: SimpleNamespace) -> None:
    assert "No phone is registered" in phone_events_impl(cfg)
    assert "No phone is registered" in phone_devices_impl(cfg)


def test_events_filters_and_format(cfg: SimpleNamespace) -> None:
    _seed(cfg)
    out = phone_events_impl(cfg)
    lines = out.splitlines()
    assert lines[0].startswith("3 phone event(s) since 24h")
    assert "status battery 42% charging, network wifi" in lines[1]
    assert "sms from Mario (+39 333 111 2222): arrivo alle 8" in lines[2]
    assert "call missed Anna" in lines[3]
    assert "vecchio" not in out

    assert "vecchio" in phone_events_impl(cfg, kind="sms", since="7d")
    by_number = phone_events_impl(cfg, contact="3331112222", since="7d")
    assert by_number.count("sms from Mario") == 2
    assert "Anna" in phone_events_impl(cfg, contact="anna")
    assert "No location events since 24h (sharing of 'location' is turned off" in phone_events_impl(
        cfg, kind="location"
    )
    assert phone_events_impl(cfg, kind="bogus").startswith("Error")


def test_devices_listing(cfg: SimpleNamespace) -> None:
    _seed(cfg)
    out = phone_devices_impl(cfg)
    assert "Pixel" in out and "shares sms, call, status" in out and "live tools no" in out


def test_parse_since() -> None:
    now = datetime(2026, 11, 25, 12, 0)
    assert parse_since("24h", now=now) == datetime(2026, 11, 24, 12, 0)
    assert parse_since("2w", now=now) == datetime(2026, 11, 11, 12, 0)
    assert parse_since("", now=now) is None
    assert parse_since("2026-11-20", now=now) == datetime(2026, 11, 20)


def test_phone_skill_registered() -> None:
    registry = build_pawn_skills()
    assert "phone" in registry.names()
    assert "phone_events" in registry.get("phone").tools
