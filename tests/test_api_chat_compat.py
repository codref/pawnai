"""OpenAI-compat surface of /v1/chat/completions used by obsidian-copilot."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection
from pawn_server.core import api_server


def _cfg(**api_kwargs: Any) -> SimpleNamespace:
    api = ApiSection(token="tok", stream_keepalive_seconds=0.05, **api_kwargs)
    return SimpleNamespace(
        api=api,
        api_token=api.token,
        api_model_idle_timeout_minutes=10.0,
        db_dsn="sqlite://",
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake_run_agent_turn(**kwargs: Any) -> AgentRunResult:
        seen.append(kwargs)
        cb = kwargs.get("on_progress")
        if cb is not None:
            cb("tool", {"gen_ai.tool.name": "sessions_list"})
        return AgentRunResult(run_id="r1", response="hello there")

    monkeypatch.setattr(api_server, "run_agent_turn", fake_run_agent_turn)
    return seen


def _client(**api_kwargs: Any) -> TestClient:
    return TestClient(api_server.create_app(_cfg(**api_kwargs)))


AUTH = {"Authorization": "Bearer tok"}


def test_models_lists_pawn_agent() -> None:
    resp = _client().get("/v1/models", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "pawn-agent"


def test_models_requires_token() -> None:
    assert _client().get("/v1/models").status_code == 401


def test_list_content_parts_are_flattened(calls: list[dict[str, Any]]) -> None:
    body = {
        "model": "pawn-agent",
        "messages": [
            {"role": "system", "content": "be nice"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "part two"},
                ],
            },
        ],
    }
    resp = _client().post("/v1/chat/completions", json=body, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello there"
    assert calls[0]["prompt"] == "part one\npart two"


def test_system_prompt_included_when_enabled(calls: list[dict[str, Any]]) -> None:
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
    }
    _client(include_system_prompt=True).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "be nice" in calls[0]["prompt"]
    assert calls[0]["prompt"].endswith("hi")


def test_session_header_used_when_no_user(calls: list[dict[str, Any]]) -> None:
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    headers = {**AUTH, "X-Pawn-Conversation": "chat:abc"}
    _client().post("/v1/chat/completions", json=body, headers=headers)
    assert calls[0]["session_id"] == "chat:abc"


def test_user_field_wins_over_header(calls: list[dict[str, Any]]) -> None:
    body = {"model": "x", "user": "note:A.md", "messages": [{"role": "user", "content": "hi"}]}
    headers = {**AUTH, "X-Pawn-Conversation": "chat:abc"}
    _client().post("/v1/chat/completions", json=body, headers=headers)
    assert calls[0]["session_id"] == "note:A.md"


def _sse_payloads(text: str) -> list[Any]:
    out: list[Any] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def test_streaming_emits_progress_then_answer(calls: list[dict[str, Any]]) -> None:
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    resp = _client().post("/v1/chat/completions", json=body, headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    payloads = _sse_payloads(resp.text)
    assert payloads[-1] == "[DONE]"
    chunks = [p for p in payloads if isinstance(p, dict)]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks)
    assert "sessions_list" in reasoning
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "hello there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert calls[0]["on_progress"] is not None


def test_streaming_progress_can_be_disabled(calls: list[dict[str, Any]]) -> None:
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    resp = _client(stream_progress=False).post("/v1/chat/completions", json=body, headers=AUTH)
    assert "reasoning_content" not in resp.text


def test_cors_preflight_allows_obsidian() -> None:
    resp = _client().options(
        "/v1/chat/completions",
        headers={
            "Origin": "app://obsidian.md",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "app://obsidian.md"


def test_cors_preflight_allows_thunderbird_extension() -> None:
    """Companion-window fetch sends Origin: moz-extension://<uuid>."""
    origin = "moz-extension://aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    resp = _client().options(
        "/v1/captures",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
