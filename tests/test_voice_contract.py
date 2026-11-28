"""The voice endpoint still produces the golden transcripts in contracts/voice/.

Both assistant clients replay those files in their own tests, so a change here
that breaks this test breaks the clients too: update contracts/, run
scripts/sync-contracts.sh, and fix the clients in the same change.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection, DevicesConfig
from pawn_server.core import api_server

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
AUTH = {"Authorization": "Bearer tok"}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Events in order; comments (keep-alives) dropped."""
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event, data = "message", []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        if data:
            out.append((event, json.loads("\n".join(data))))
    return out


def golden(name: str) -> list[tuple[str, dict]]:
    return parse_sse((CONTRACTS / "voice" / name).read_text(encoding="utf-8"))


@pytest.fixture
def cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=30),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=f"sqlite:///{tmp_path / 'contract.db'}",
        devices=DevicesConfig(mirror_voice_to_portal=False),
    )


def _post(cfg: Any, message: str) -> list[tuple[str, dict]]:
    with TestClient(api_server.create_app(cfg)) as client:
        res = client.post(
            "/v1/pawn/voice",
            headers=AUTH,
            json={"conversation": "voice:contract", "message": message, "access": "chat", "lang": "it"},
        )
    assert res.status_code == 200
    return parse_sse(res.text)


def test_turn_matches_golden(cfg: SimpleNamespace, monkeypatch: Any) -> None:
    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        kwargs["on_progress"]("tool", {"gen_ai.tool.name": "planner_day"})
        return AgentRunResult(
            run_id="run-1",
            response=(
                "Domani alle nove hai la riunione con Sean. [[SCREEN]]\n"
                "- 09:00 Sean\n- 14:00 Dentista [[END]]"
            ),
        )

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    assert _post(cfg, "cosa ho domani?") == golden("turn.sse")


def test_reset_matches_golden(cfg: SimpleNamespace) -> None:
    assert _post(cfg, "/reset") == golden("reset.sse")


def test_error_matches_golden(cfg: SimpleNamespace, monkeypatch: Any) -> None:
    async def bad_turn(**_kwargs: Any) -> AgentRunResult:
        raise RuntimeError("model down")

    monkeypatch.setattr(api_server, "run_agent_turn", bad_turn)
    assert _post(cfg, "ciao") == golden("error.sse")


def test_cases_file_names_every_golden() -> None:
    cases = json.loads((CONTRACTS / "voice" / "cases.json").read_text(encoding="utf-8"))
    named = {c["file"] for c in cases["cases"]} | {cases["device_stream"]["file"]}
    on_disk = {p.name for p in (CONTRACTS / "voice").glob("*.sse")}
    assert named == on_disk


def test_device_stream_golden_uses_known_tools() -> None:
    from pawn_server.core.device_commands import TOOLS

    events = golden("device_stream.sse")
    assert events[0][0] == "ready"
    for cmd in events[1][1]["commands"]:
        assert cmd["tool"] in TOOLS
