from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _make_cfg(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        db_dsn="postgresql+psycopg://dummy/dummy",
        strip_thinking=True,
        history_mode=mode,
        history_recent_turns=4,
        history_replay_max_tokens=8000,
        history_max_text_chars=500,
        history_sanitize_leaked_thoughts=True,
        api_model_idle_timeout_minutes=10.0,
    )


def _make_result(output: str = "ok") -> MagicMock:
    result = MagicMock()
    result.output = output
    result.new_messages.return_value = []
    return result


def test_chat_completions_uses_compact_replay_history_by_default() -> None:
    from pawn_server.core.api_server import ChatCompletionRequest, chat_completions

    req = ChatCompletionRequest(
        model="pawn_agent",
        user="sess-1",
        messages=[{"role": "user", "content": "Analyse conversation."}],
    )
    cfg = _make_cfg("compact")

    with (
        patch("pawn_agent.core.session_store.build_replay_history", return_value=["compact"]) as mock_compact,
        patch("pawn_agent.core.session_store.load_history", return_value=["raw"]) as mock_raw,
        patch("pawn_agent.core.session_store.append_turn"),
        patch("pawn_server.core.queue_listener._get_or_create_agent") as mock_get,
    ):
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=_make_result())
        mock_get.return_value = agent

        asyncio.run(chat_completions(req, cfg))

    mock_compact.assert_called_once_with(
        "sess-1",
        cfg.db_dsn,
        strip_thinking=True,
        recent_turns=4,
        replay_max_tokens=8000,
        max_text_chars=500,
        sanitize_leaked_thoughts=True,
    )
    mock_raw.assert_not_called()
    agent.run_async.assert_awaited_once_with(
        "Analyse conversation.",
        message_history=["compact"],
        model_settings=None,
    )


def test_chat_completions_can_fall_back_to_raw_history() -> None:
    from pawn_server.core.api_server import ChatCompletionRequest, chat_completions

    req = ChatCompletionRequest(
        model="pawn_agent",
        user="sess-1",
        messages=[{"role": "user", "content": "Analyse conversation."}],
    )
    cfg = _make_cfg("raw")

    with (
        patch("pawn_agent.core.session_store.build_replay_history", return_value=["compact"]) as mock_compact,
        patch("pawn_agent.core.session_store.load_history", return_value=["raw"]) as mock_raw,
        patch("pawn_agent.core.session_store.append_turn"),
        patch("pawn_server.core.queue_listener._get_or_create_agent") as mock_get,
    ):
        agent = MagicMock()
        agent.run_async = AsyncMock(return_value=_make_result())
        mock_get.return_value = agent

        asyncio.run(chat_completions(req, cfg))

    mock_raw.assert_called_once_with("sess-1", cfg.db_dsn, strip_thinking=True)
    mock_compact.assert_not_called()
    agent.run_async.assert_awaited_once_with(
        "Analyse conversation.",
        message_history=["raw"],
        model_settings=None,
    )
