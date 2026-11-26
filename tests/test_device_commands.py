"""Phone bridge live tools: command queue, push wake-up, phone round trip."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection, DevicesConfig
from pawn_agent.utils.db import Device, DeviceCommand, PhoneEvent
from pawn_core.database import Base
from pawn_server.core import api_server, device_commands, devices
from pawn_server.core.device_commands import CommandError, describe, run_command

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'cmd.db'}"
    engine = sa.create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE knowledge_chunks (id TEXT PRIMARY KEY, source_ref TEXT)"))
    Base.metadata.create_all(
        engine, tables=[Device.__table__, PhoneEvent.__table__, DeviceCommand.__table__]
    )
    engine.dispose()
    pushes: list[str] = []
    monkeypatch.setattr(device_commands, "send_push", lambda url, **_: pushes.append(url))
    cfg = SimpleNamespace(
        api=ApiSection(token="tok"),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        devices=DevicesConfig(retention_days=0, command_timeout_s=5),
        coworker=SimpleNamespace(timezone="UTC"),
    )
    return SimpleNamespace(cfg=cfg, dsn=dsn, pushes=pushes)


def _phone(dsn: str, *, caps=("location_now", "sms_send"), endpoint="https://ntfy.example/up1") -> tuple[str, str]:
    reg = devices.register_device(dsn, "Pixel")
    devices.update_device(dsn, reg["device_id"], capabilities=list(caps) + ["bogus"], unifiedpush_endpoint=endpoint)
    return reg["device_id"], reg["device_token"]


def _answer_in_background(dsn: str, device_id: str, status: str, result: dict) -> threading.Thread:
    def worker() -> None:
        for _ in range(100):
            cmds = device_commands.take_pending(dsn, device_id)
            if cmds:
                for cmd in cmds:
                    device_commands.post_result(dsn, device_id, cmd["id"], status=status, result=result)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=worker)
    thread.start()
    return thread


def test_validate_args() -> None:
    assert device_commands.validate_args("sms_send", {"to": " +39 1 ", "body": "hi"}) == {"to": "+39 1", "body": "hi"}
    with pytest.raises(CommandError, match="Missing"):
        device_commands.validate_args("sms_send", {"to": "1"})
    with pytest.raises(CommandError, match="Unexpected"):
        device_commands.validate_args("location_now", {"x": 1})
    with pytest.raises(CommandError, match="Unknown"):
        device_commands.validate_args("wipe_phone", {})


def test_capabilities_are_filtered(env: SimpleNamespace) -> None:
    device_id, _ = _phone(env.dsn)
    assert devices.get_device(env.dsn, device_id)["capabilities"] == ["location_now", "sms_send"]


def test_round_trip_ok(env: SimpleNamespace) -> None:
    device_id, _ = _phone(env.dsn)
    thread = _answer_in_background(env.dsn, device_id, "ok", {"lat": 45.1, "lon": 9.2, "accuracy_m": 30.4})
    view = run_command(env.dsn, "location_now", {}, timeout_s=3, poll_s=0.02)
    thread.join()
    assert env.pushes == ["https://ntfy.example/up1"]
    assert view["status"] == "ok"
    assert describe(view) == "Phone location: 45.1,9.2 ±30m"


def test_denied_action(env: SimpleNamespace) -> None:
    device_id, _ = _phone(env.dsn)
    thread = _answer_in_background(env.dsn, device_id, "denied", {})
    view = run_command(env.dsn, "sms_send", {"to": "+391", "body": "ciao"}, timeout_s=3, poll_s=0.02)
    thread.join()
    assert view["status"] == "denied"
    assert view["action"] is True
    assert "declined sms_send" in describe(view)


def test_timeout_expires_pending_and_rejects_late_result(env: SimpleNamespace) -> None:
    device_id, _ = _phone(env.dsn)
    view = run_command(env.dsn, "location_now", {}, timeout_s=0.05, poll_s=0.01)
    assert view["status"] == "expired"
    assert "did not answer" in describe(view)
    # The phone shows up later: nothing to run, and a late answer is refused.
    assert device_commands.take_pending(env.dsn, device_id) == []
    with pytest.raises(CommandError) as exc:
        device_commands.post_result(env.dsn, device_id, view["id"], status="ok")
    assert exc.value.status_code == 409


def test_delivered_but_slow_action_is_reported(env: SimpleNamespace) -> None:
    device_id, _ = _phone(env.dsn)

    def take_only() -> None:
        for _ in range(100):
            if device_commands.take_pending(env.dsn, device_id):
                return
            time.sleep(0.01)

    thread = threading.Thread(target=take_only)
    thread.start()
    view = run_command(env.dsn, "sms_send", {"to": "1", "body": "x"}, timeout_s=0.5, poll_s=0.01)
    thread.join()
    assert view["status"] == "delivered"
    assert "may still complete" in describe(view)
    # The phone may still report the outcome.
    assert device_commands.post_result(env.dsn, device_id, view["id"], status="ok")["status"] == "ok"


def test_refuses_missing_capability_or_channel(env: SimpleNamespace) -> None:
    _phone(env.dsn, caps=("location_now",), endpoint="")
    with pytest.raises(CommandError, match="does not offer sms_send|no push channel"):
        run_command(env.dsn, "sms_send", {"to": "1", "body": "x"}, timeout_s=0.1)
    with pytest.raises(CommandError, match="no push channel"):
        run_command(env.dsn, "location_now", {}, timeout_s=0.1)


def test_several_devices_need_a_name(env: SimpleNamespace) -> None:
    _phone(env.dsn)
    _phone(env.dsn)
    with pytest.raises(CommandError, match="Several phones"):
        run_command(env.dsn, "location_now", {}, timeout_s=0.1)


def test_http_phone_side(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        reg = client.post("/v1/devices/register", json={"name": "Pixel"}, headers=AUTH).json()
        device_id = reg["device_id"]
        dev = {"Authorization": f"Bearer {reg['device_token']}"}
        res = client.put(
            f"/v1/devices/{device_id}",
            json={"capabilities": ["notify"], "unifiedpush_endpoint": "https://ntfy.example/x"},
            headers=dev,
        )
        assert res.json()["capabilities"] == ["notify"]

        # Phone tokens cannot queue commands; only the shared token can.
        assert client.post(f"/v1/devices/{device_id}/commands", json={"tool": "notify"}, headers=dev).status_code == 401
        bad = client.post(f"/v1/devices/{device_id}/commands", json={"tool": "notify"}, headers=AUTH)
        assert bad.status_code == 422 and "Missing" in bad.json()["detail"]

        thread = threading.Thread(
            target=lambda: client.post(
                f"/v1/devices/{device_id}/commands",
                json={"tool": "notify", "args": {"text": "hi"}, "timeout_s": 3},
                headers=AUTH,
            )
        )
        thread.start()
        commands: list = []
        for _ in range(200):
            commands = client.get(f"/v1/devices/{device_id}/commands/pending", headers=dev).json()["commands"]
            if commands:
                break
            time.sleep(0.02)
        assert commands and commands[0]["tool"] == "notify" and commands[0]["args"] == {"text": "hi"}
        done = client.post(
            f"/v1/devices/{device_id}/commands/{commands[0]['id']}/result",
            json={"status": "ok"},
            headers=dev,
        )
        thread.join()
        assert done.json()["status"] == "ok"
        assert env.pushes == ["https://ntfy.example/x"]
