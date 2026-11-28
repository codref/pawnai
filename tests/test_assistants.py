"""Assistants: desktop devices, presence, command stream, voice threads,
hand-off, phone-approved access, portal mirror, wake store, overview."""

from __future__ import annotations

import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection, DevicesConfig
from pawn_agent.utils.db import (
    AccessRequest,
    Device,
    DeviceCommand,
    PhoneEvent,
    VoiceThread,
    WakeClip,
    WakeModel,
)
from pawn_core.database import Base
from pawn_server.core import api_server, device_commands, voice_threads
from pawn_server.core.portal_models import PortalChat, PortalChatFolder, PortalMessage

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'assistants.db'}"
    engine = sa.create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE knowledge_chunks (id TEXT PRIMARY KEY, source_ref TEXT)"))
    Base.metadata.create_all(
        engine,
        tables=[
            Device.__table__,
            PhoneEvent.__table__,
            DeviceCommand.__table__,
            VoiceThread.__table__,
            AccessRequest.__table__,
            WakeClip.__table__,
            WakeModel.__table__,
            PortalChatFolder.__table__,
            PortalChat.__table__,
            PortalMessage.__table__,
        ],
    )
    engine.dispose()
    cfg = SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        devices=DevicesConfig(retention_days=0, wake_dir=str(tmp_path / "wake")),
        coworker=SimpleNamespace(timezone="UTC"),
    )
    pushes: list[str] = []
    monkeypatch.setattr(device_commands, "send_push", lambda url, **_k: pushes.append(url) or None)

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        return AgentRunResult(run_id="run-1", response="Sono le tre. [[SCREEN]]\n- 15:00")

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    return SimpleNamespace(cfg=cfg, dsn=dsn, pushes=pushes)


def _register(client: TestClient, name: str, kind: str, caps: list[str]) -> tuple[str, dict]:
    res = client.post("/v1/devices/register", json={"name": name, "kind": kind}, headers=AUTH)
    assert res.status_code == 200, res.text
    body = res.json()
    device_id, auth = body["device_id"], {"Authorization": f"Bearer {body['device_token']}"}
    update: dict[str, Any] = {"capabilities": caps}
    if kind == "phone":
        update["unifiedpush_endpoint"] = "https://ntfy.example/up123"
    assert client.put(f"/v1/devices/{device_id}", json=update, headers=auth).status_code == 200
    return device_id, auth


def _events(text: str) -> list[tuple[str, dict]]:
    import json

    out, event = [], "message"
    for block in text.split("\n\n"):
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if data is not None:
            out.append((event, data))
        event = "message"
    return out


def _wav(n: int = 3200) -> bytes:
    pcm = b"\x00\x00" * n
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16
    ) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# ── Devices ───────────────────────────────────────────────────────────────────


def test_register_desktop_and_presence(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, auth = _register(client, "laptop", "desktop", ["speak", "timer_set", "notify"])
        bad = client.post("/v1/devices/register", json={"name": "x", "kind": "fridge"}, headers=AUTH)
        assert bad.status_code == 422
        res = client.post(
            f"/v1/devices/{laptop}/presence",
            json={"state": "active", "headset": False, "last_wake_at": 1_764_000_000},
            headers=auth,
        )
        assert res.status_code == 200, res.text
        view = res.json()
        assert view["kind"] == "desktop" and view["presence"]["state"] == "active"
        assert view["presence"]["last_wake_at"].startswith("2025-11-24")
        assert client.post(f"/v1/devices/{laptop}/presence", json={"state": "x"}, headers=auth).status_code == 422


def test_auto_picks_fresh_active_device(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, lauth = _register(client, "laptop", "desktop", ["speak"])
        phone, pauth = _register(client, "pixel", "phone", ["speak", "notify"])
        client.post(f"/v1/devices/{laptop}/presence", json={"state": "idle"}, headers=lauth)
        client.post(f"/v1/devices/{phone}/presence", json={"state": "active"}, headers=pauth)
    assert device_commands.resolve_device(env.dsn, "auto", tool="speak").name == "pixel"
    # notify only on the phone; kind=phone keeps the laptop out of phone_* tools.
    assert device_commands.resolve_device(env.dsn, "auto", tool="notify").name == "pixel"
    assert device_commands.resolve_device(env.dsn, "", kind="phone").name == "pixel"
    with pytest.raises(device_commands.CommandError, match="offers timer_set"):
        device_commands.resolve_device(env.dsn, "auto", tool="timer_set")


def test_timer_args_are_validated() -> None:
    assert device_commands.validate_args("timer_set", {"seconds": "90", "label": "pasta"}) == {
        "seconds": "90",
        "label": "pasta",
    }
    with pytest.raises(device_commands.CommandError):
        device_commands.validate_args("timer_set", {"seconds": "-3"})
    with pytest.raises(device_commands.CommandError):
        device_commands.validate_args("timer_set", {"seconds": "abc"})


def test_desktop_stream_delivers_commands(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, auth = _register(client, "laptop", "desktop", ["speak"])
        result: dict[str, Any] = {}

        def _run() -> None:
            result["view"] = device_commands.run_command(
                env.dsn, "speak", {"text": "ciao"}, device="laptop", timeout_s=5, poll_s=0.05
            )

        worker = threading.Thread(target=_run)
        worker.start()
        time.sleep(0.2)
        res = client.get(
            f"/v1/devices/{laptop}/commands/stream?poll=0.05&max_seconds=0.3", headers=auth
        )
        events = _events(res.text)
        assert events[0][0] == "ready"
        cmds = [data for ev, data in events if ev == "cmd"]
        assert cmds and cmds[0]["commands"][0]["tool"] == "speak"
        cid = cmds[0]["commands"][0]["id"]
        ok = client.post(f"/v1/devices/{laptop}/commands/{cid}/result", json={"status": "ok"}, headers=auth)
        assert ok.status_code == 200
        worker.join(5)
        assert result["view"]["status"] == "ok"
        assert "Said aloud on laptop" in device_commands.describe(result["view"])
        # No push for a desktop without UnifiedPush.
        assert env.pushes == []
        history = client.get(f"/v1/devices/{laptop}/commands", headers=AUTH).json()["commands"]
        assert history[0]["status"] == "ok"


# ── Voice threads ─────────────────────────────────────────────────────────────


def _voice(client: TestClient, conv: str, device: str | None, msg: str = "che ore sono") -> list:
    body: dict[str, Any] = {"conversation": conv, "message": msg}
    if device:
        body["device_id"] = device
    res = client.post("/v1/pawn/voice", json=body, headers=AUTH)
    assert res.status_code == 200
    return _events(res.text)


def test_voice_turn_records_thread_and_mirrors_portal(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, _ = _register(client, "laptop", "desktop", ["speak"])
        _voice(client, "voice:abc", laptop)
        _voice(client, "voice:abc", laptop, "e domani?")
        active = client.get("/v1/pawn/voice/active", headers=AUTH).json()["thread"]
        assert active["conversation"] == "voice:abc" and active["device_name"] == "laptop"
        assert active["turns"] == 2 and active["last_answer"] == "Sono le tre."
        # Excluding the holder: nothing to take over.
        res = client.get(f"/v1/pawn/voice/active?exclude_device={laptop}", headers=AUTH)
        assert res.json()["thread"] is None
    engine = sa.create_engine(env.dsn)
    with engine.connect() as conn:
        chat = conn.execute(sa.select(PortalChat)).one()
        assert chat.conversation_key == "voice:abc" and "laptop" in chat.tags
        msgs = conn.execute(sa.select(PortalMessage).order_by(PortalMessage.seq)).all()
        assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert msgs[1].content.endswith("- 15:00")
        folder = conn.execute(sa.select(PortalChatFolder)).one()
        assert folder.name == "Voice"


def test_voice_turn_without_device_tables_still_answers(env: SimpleNamespace, monkeypatch: Any) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("no table")

    monkeypatch.setattr(voice_threads, "_get_session", boom)
    with TestClient(api_server.create_app(env.cfg)) as client:
        events = dict(_voice(client, "voice:z", None))
    assert events["answer"]["content"] == "Sono le tre."


def test_claim_moves_thread_and_drops_access(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, _ = _register(client, "laptop", "desktop", ["speak"])
        phone, _ = _register(client, "pixel", "phone", ["speak"])
        client.post(
            "/v1/pawn/voice",
            json={"conversation": "voice:h", "message": "ciao", "device_id": laptop, "access": "full"},
            headers=AUTH,
        )
        thread = client.get(f"/v1/pawn/voice/active?exclude_device={phone}", headers=AUTH).json()["thread"]
        assert thread["access"] == "full"
        res = client.post("/v1/pawn/voice/voice:h/claim", json={"device_id": phone}, headers=AUTH)
        assert res.status_code == 200, res.text
        claimed = res.json()
        assert claimed["device_name"] == "pixel" and claimed["from_device"] == "laptop"
        assert claimed["access"] == "chat" and claimed["last_answer"] == "Sono le tre."
        assert client.post("/v1/pawn/voice/voice:nope/claim", json={"device_id": phone}, headers=AUTH).status_code == 404
        # Ending the conversation makes it unclaimable.
        client.post(
            "/v1/pawn/voice", json={"conversation": "voice:h", "message": "/reset"}, headers=AUTH
        )


def test_access_request_round_trip(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        laptop, _ = _register(client, "laptop", "desktop", ["speak"])
        res = client.post("/v1/pawn/voice/access-requests", json={"device_id": laptop}, headers=AUTH)
        assert res.status_code == 409  # no phone can approve yet
        phone, pauth = _register(client, "pixel", "phone", ["access_approve"])
        res = client.post(
            "/v1/pawn/voice/access-requests",
            json={"device_id": laptop, "conversation": "voice:a", "minutes": 99},
            headers=AUTH,
        )
        assert res.status_code == 202, res.text
        req = res.json()
        assert req["status"] == "pending" and req["approver_name"] == "pixel"
        assert req["minutes"] == 30  # capped by access_max_minutes
        assert env.pushes == ["https://ntfy.example/up123"]
        pending = client.get(f"/v1/devices/{phone}/commands/pending", headers=pauth).json()["commands"]
        assert pending[0]["tool"] == "access_approve" and pending[0]["action"] is True
        assert pending[0]["args"]["requester"] == "laptop"
        client.post(
            f"/v1/devices/{phone}/commands/{pending[0]['id']}/result", json={"status": "ok"}, headers=pauth
        )
        done = client.get(f"/v1/pawn/voice/access-requests/{req['id']}", headers=AUTH).json()
        assert done["status"] == "approved" and done["access_until"]

        # A declined request.
        req2 = client.post("/v1/pawn/voice/access-requests", json={"device_id": laptop}, headers=AUTH).json()
        cid = client.get(f"/v1/devices/{phone}/commands/pending", headers=pauth).json()["commands"][0]["id"]
        client.post(f"/v1/devices/{phone}/commands/{cid}/result", json={"status": "denied"}, headers=pauth)
        assert client.get(f"/v1/pawn/voice/access-requests/{req2['id']}", headers=AUTH).json()["status"] == "denied"


# ── Wake store ────────────────────────────────────────────────────────────────


def test_wake_clips_and_models(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        phone, pauth = _register(client, "pixel", "phone", [])
        res = client.post(
            "/v1/assistants/wake/clips",
            data={"label": "confirmed", "device_id": phone, "score": "0.81", "ts": "1764000000000"},
            files={"file": ("a.wav", _wav(), "audio/wav")},
            headers=pauth,
        )
        assert res.status_code == 200, res.text
        clip = res.json()
        assert clip["label"] == "confirmed" and clip["device_id"] == phone
        bad = client.post(
            "/v1/assistants/wake/clips",
            data={"label": "nope", "device_id": phone},
            files={"file": ("a.wav", _wav(), "audio/wav")},
            headers=pauth,
        )
        assert bad.status_code == 422
        not_wav = client.post(
            "/v1/assistants/wake/clips",
            data={"label": "barge", "device_id": phone},
            files={"file": ("a.wav", b"hello", "audio/wav")},
            headers=pauth,
        )
        assert not_wav.status_code == 422
        wrong = client.post(
            "/v1/assistants/wake/clips",
            data={"label": "barge", "device_id": phone},
            files={"file": ("a.wav", _wav(), "audio/wav")},
            headers={"Authorization": "Bearer nope"},
        )
        assert wrong.status_code == 401
        # Listing needs the shared token.
        assert client.get("/v1/assistants/wake/clips", headers=pauth).status_code == 401
        listed = client.get("/v1/assistants/wake/clips?label=confirmed", headers=AUTH).json()["clips"]
        assert [c["id"] for c in listed] == [clip["id"]]
        audio = client.get(f"/v1/assistants/wake/clips/{clip['id']}", headers=AUTH)
        assert audio.content[:4] == b"RIFF"

        assert client.get("/v1/assistants/wake/models/current", headers=AUTH).json()["model"] is None
        pub = client.post(
            "/v1/assistants/wake/models",
            data={"version": "v2", "notes": "phone clips", "thresholds": '{"wake": 0.55}'},
            files={"file": ("hey_pawn.onnx", b"onnx-bytes", "application/octet-stream")},
            headers=AUTH,
        )
        assert pub.status_code == 200, pub.text
        assert pub.json()["thresholds"] == {"wake": 0.55}
        current = client.get(
            f"/v1/assistants/wake/models/current?device_id={phone}", headers=pauth
        ).json()["model"]
        assert current["version"] == "v2" and len(current["sha256"]) == 64
        blob = client.get(f"/v1/assistants/wake/models/v2?device_id={phone}", headers=pauth)
        assert blob.content == b"onnx-bytes"
        dup = client.post(
            "/v1/assistants/wake/models",
            data={"version": "v2"},
            files={"file": ("m.onnx", b"x", "application/octet-stream")},
            headers=AUTH,
        )
        assert dup.status_code == 409

        overview = client.get("/v1/portal/assistants", headers=AUTH).json()
        assert overview["wake"]["model"]["version"] == "v2"
        assert overview["wake"]["clips"] == {"confirmed": 1}
        row = overview["assistants"][0]
        assert row["name"] == "pixel" and row["wake_clips"] == {"confirmed": 1}
        assert "speak" in overview["tools"]

        assert client.delete(f"/v1/assistants/wake/clips/{clip['id']}", headers=AUTH).status_code == 200
        assert client.get(f"/v1/assistants/wake/clips/{clip['id']}", headers=AUTH).status_code == 404


def test_delete_device_keeps_clips_and_threads(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        phone, pauth = _register(client, "pixel", "phone", [])
        client.post(
            "/v1/assistants/wake/clips",
            data={"label": "barge", "device_id": phone},
            files={"file": ("a.wav", _wav(), "audio/wav")},
            headers=pauth,
        )
        _voice(client, "voice:d", phone)
        assert client.delete(f"/v1/devices/{phone}", headers=pauth).status_code == 200
        clips = client.get("/v1/assistants/wake/clips", headers=AUTH).json()["clips"]
        assert clips and clips[0]["device_id"] is None
        thread = client.get("/v1/pawn/voice/active", headers=AUTH).json()["thread"]
        assert thread["conversation"] == "voice:d" and thread["device_id"] is None


