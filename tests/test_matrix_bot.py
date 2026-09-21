"""Unit tests for Matrix bot helpers and config."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from pawn_agent.utils.config import MatrixBotConfig, load_config
from pawn_server.core.matrix_bot import (
    MatrixTurnProgress,
    build_edit_content,
    build_reaction_content,
    build_text_content,
    chunk_text,
    conversation_id,
    extract_prompt,
    format_progress_status,
    is_direct_room,
    matrix_reply_body,
    next_sync_backoff,
    normalize_body,
    run_sync_with_reconnect,
    verification_allowed,
)


def test_conversation_id() -> None:
    assert conversation_id("!abc:example.com") == "matrix:!abc:example.com"


def test_is_direct_room() -> None:
    assert is_direct_room(1) is True
    assert is_direct_room(2) is True
    assert is_direct_room(3) is False


def test_normalize_body_strips_reply_fallback() -> None:
    raw = "> <@alice:hs> hi\n\nplease summarize"
    assert normalize_body(raw) == "please summarize"


def test_normalize_body_edit() -> None:
    assert (
        normalize_body("* old", is_edit=True, new_body="fresh text")
        == "fresh text"
    )
    assert normalize_body("* patched", is_edit=True) == "patched"


def test_extract_prompt_prefix_and_dm() -> None:
    assert (
        extract_prompt("!pawn hello", command_prefix="!pawn", is_dm=False)
        == "hello"
    )
    assert (
        extract_prompt("hello", command_prefix="!pawn", is_dm=True) == "hello"
    )
    assert extract_prompt("hello", command_prefix="!pawn", is_dm=False) is None
    assert extract_prompt("!pawn", command_prefix="!pawn", is_dm=False) is None


def test_chunk_text() -> None:
    assert chunk_text("short") == ["short"]
    parts = chunk_text("abcdefghij", limit=4)
    assert parts == ["abcd", "efgh", "ij"]


def test_verification_allowed() -> None:
    bot = "@bot:hs"
    assert verification_allowed(bot, bot_user_id=bot, inviters=["@a:hs"]) is True
    assert verification_allowed("@a:hs", bot_user_id=bot, inviters=["@a:hs"]) is True
    assert verification_allowed("@x:hs", bot_user_id=bot, inviters=["@a:hs"]) is False
    assert verification_allowed("@x:hs", bot_user_id=bot, inviters=[]) is True


def test_matrix_reply_body_strips_tool_trail() -> None:
    raw = (
        "[tool] sessions_list → Found 10 session(s): daniel | segments: 327\n"
        "[tool] session_analyze → ok\n\n"
        "Here are your sessions."
    )
    assert matrix_reply_body(raw) == "Here are your sessions."
    assert matrix_reply_body("Just an answer.") == "Just an answer."


def test_format_progress_status() -> None:
    assert format_progress_status("turn.start") == "Working…"
    assert (
        format_progress_status(
            "control", {"sallm.control.skill": "sessions"}
        )
        == "Skill: `sessions`…"
    )
    assert (
        format_progress_status("tool", {"gen_ai.tool.name": "sessions_list"})
        == "Ran `sessions_list`…"
    )
    assert format_progress_status("llm", tools_ran=False) is None
    assert format_progress_status("llm", tools_ran=True) == "Thinking…"
    assert format_progress_status("nudge") is None


def test_build_edit_and_reaction_content() -> None:
    edit = build_edit_content("hello **x**", "$evt")
    assert edit["m.relates_to"] == {"rel_type": "m.replace", "event_id": "$evt"}
    assert edit["m.new_content"]["body"] == "hello **x**"
    assert edit["body"].startswith("* ")
    assert "formatted_body" in edit["m.new_content"]

    react = build_reaction_content("$user", "⏳")
    assert react["m.relates_to"] == {
        "rel_type": "m.annotation",
        "event_id": "$user",
        "key": "⏳",
    }

    text = build_text_content("hi")
    assert text["msgtype"] == "m.text"
    assert text["body"] == "hi"


def test_next_sync_backoff() -> None:
    assert next_sync_backoff(1.0) == 2.0
    assert next_sync_backoff(40.0, max_s=60.0) == 60.0


def test_run_sync_with_reconnect_sets_online_and_retries() -> None:
    client = MagicMock()
    client.set_presence = AsyncMock()
    client.add_response_callback = MagicMock()
    client.sync_forever = AsyncMock(side_effect=[RuntimeError("boom"), None])

    asyncio.run(
        run_sync_with_reconnect(client, timeout_ms=1000, max_backoff_s=0.01)
    )

    assert client.set_presence.await_count >= 2
    client.sync_forever.assert_awaited()
    kwargs = client.sync_forever.await_args.kwargs
    assert kwargs.get("set_presence") == "online"
    assert kwargs.get("full_state") is True


def test_matrix_bot_config_defaults() -> None:
    mb = MatrixBotConfig()
    assert mb.enabled is False
    assert mb.command_prefix == "!pawn"
    assert mb.inviters == []
    assert mb.progress_updates is True
    assert mb.progress_reactions is True


def test_load_config_parses_matrix_bot(tmp_path: Path) -> None:
    path = tmp_path / "pawnai.yaml"
    path.write_text(
        yaml.dump(
            {
                "matrix_bot": {
                    "enabled": True,
                    "homeserver_url": "https://matrix.example.com",
                    "user_id": "@bot:example.com",
                    "user_token": "syt_token",
                    "command_prefix": "!x",
                    "inviters": ["@you:example.com"],
                    "progress_updates": False,
                    "progress_reactions": False,
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.matrix_bot.enabled is True
    assert cfg.matrix_bot.user_id == "@bot:example.com"
    assert cfg.matrix_bot.command_prefix == "!x"
    assert cfg.matrix_bot.inviters == ["@you:example.com"]
    assert cfg.matrix_bot.progress_updates is False
    assert cfg.matrix_bot.progress_reactions is False


def test_require_nio_message() -> None:
    from pawn_server.core import matrix_bot as mb

    # Smoke: validation helpers raise clear errors without a live homeserver.
    with pytest.raises(RuntimeError, match="homeserver_url"):
        mb._validate_cfg(MatrixBotConfig(enabled=True))
    with pytest.raises(RuntimeError, match="user_token or user_password"):
        mb._validate_cfg(
            MatrixBotConfig(
                enabled=True,
                homeserver_url="https://hs",
                user_id="@b:hs",
            )
        )


def test_matrix_turn_progress_finish_edits_status() -> None:
    async def _run() -> None:
        client = MagicMock()
        counter = {"n": 0}

        async def _send(*_a, **_k):
            counter["n"] += 1
            resp = MagicMock()
            resp.event_id = f"$e{counter['n']}"
            return resp

        client.room_send = AsyncMock(side_effect=_send)
        client.room_typing = AsyncMock()
        client.room_redact = AsyncMock()

        progress = MatrixTurnProgress(
            client,
            "!room:hs",
            "$user",
            loop=asyncio.get_running_loop(),
            updates=True,
            reactions=True,
            debounce_s=0.01,
            typing_interval_s=0.05,
        )
        await progress.start()
        assert progress._status_event_id == "$e2"  # react then status

        progress._tools_ran = True
        await progress._queue_status("Ran `sessions_list`…")
        await asyncio.sleep(0.05)

        # Typing keepalive should refresh at least once.
        await asyncio.sleep(0.08)
        assert client.room_typing.await_count >= 2

        await progress.finish("Final answer here.")
        assert client.room_redact.await_count >= 1
        edit_bodies = [
            call.args[2].get("m.new_content", {}).get("body")
            for call in client.room_send.await_args_list
            if isinstance(call.args[2], dict)
            and call.args[2].get("m.relates_to", {}).get("rel_type")
            == "m.replace"
        ]
        assert "Final answer here." in edit_bodies

        await progress.close()
        assert any(
            c.kwargs.get("typing_state") is False
            for c in client.room_typing.await_args_list
        )

    asyncio.run(_run())


def test_matrix_turn_progress_on_progress_schedules_edit() -> None:
    async def _run() -> None:
        client = MagicMock()
        send_resp = MagicMock()
        send_resp.event_id = "$status"
        client.room_send = AsyncMock(return_value=send_resp)
        client.room_typing = AsyncMock()

        progress = MatrixTurnProgress(
            client,
            "!room:hs",
            None,
            loop=asyncio.get_running_loop(),
            updates=True,
            reactions=False,
            debounce_s=0.01,
            typing_interval_s=60.0,
        )
        await progress.start()
        progress.on_progress(
            "tool", {"gen_ai.tool.name": "sessions_list"}
        )
        await asyncio.sleep(0.05)
        assert progress._tools_ran is True
        assert any(
            call.args[2].get("m.relates_to", {}).get("rel_type") == "m.replace"
            for call in client.room_send.await_args_list
            if len(call.args) >= 3 and isinstance(call.args[2], dict)
        )
        await progress.close()

    asyncio.run(_run())


def test_ask_sync_installs_temporary_tracer() -> None:
    """SallmChatSession bridges Tracer events into on_progress for one ask()."""
    from pawn_agent.core.sallm_session import SallmChatSession

    events: list[tuple[str, dict]] = []

    class FakeAgent:
        session_id = "conv-test"
        trace = None

        def ask(self, text: str):
            assert self.trace is not None
            self.trace.turn_start(text, [])
            self.trace.tool(
                name="sessions_list",
                command=["sessions_list"],
                observation="ok",
            )
            return {"answer": f"got:{text}", "steps": [], "metrics": {}}

    session = SallmChatSession(FakeAgent(), MagicMock())  # type: ignore[arg-type]
    result = session._ask_sync(
        "hi",
        on_progress=lambda kind, attrs: events.append((kind, attrs)),
    )
    assert result["answer"] == "got:hi"
    assert session._agent.trace is None
    kinds = [k for k, _ in events]
    assert "turn.start" in kinds
    assert "tool" in kinds
    tool_attrs = next(a for k, a in events if k == "tool")
    assert tool_attrs.get("gen_ai.tool.name") == "sessions_list"
