"""Tests for pawn_server queue listener."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_cfg(**kwargs):
    from pawn_agent.utils.config import AgentConfig

    return AgentConfig(
        db_dsn="postgresql+psycopg://dummy/dummy",
        pydantic_model="openai:test-model",
        **kwargs,
    )


def _make_msg(payload: Dict[str, Any], msg_id: str = "msg-001") -> MagicMock:
    """Build a minimal mock pawn_queue.Message."""
    msg = MagicMock()
    msg.id = msg_id
    msg.payload = payload
    msg.ack = AsyncMock()
    msg.nack = AsyncMock()
    return msg


# ──────────────────────────────────────────────────────────────────────────────
# make_message_handler — routing / dead-letter paths
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeMessageHandlerRouting:

    def test_missing_command_key_nacks(self):
        """Messages without a 'command' key are dead-lettered."""
        cfg = _make_cfg()
        msg = _make_msg({"session_id": "abc", "prompt": "Hello"})

        from pawn_server.core.queue_listener import make_message_handler

        handler = make_message_handler(cfg)
        asyncio.run(handler(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()

    def test_unsupported_command_nacks(self):
        """Messages with an unregistered command are dead-lettered."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "transcribe-diarize", "audio_paths": ["audio.wav"]})

        from pawn_server.core.queue_listener import make_message_handler

        handler = make_message_handler(cfg)
        asyncio.run(handler(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────────────
# make_message_handler — "run" happy path
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeMessageHandlerRun:

    def _patch_run(self, reply: str = "Agent reply."):
        """Patch DB helpers and the registry handle_turn for a clean run."""
        return (
            patch("pawn_server.core.queue_listener.create_agent_run", return_value="run-1"),
            patch("pawn_server.core.queue_listener.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
                return_value=reply,
            ),
        )

    def test_well_formed_message_acked(self):
        """A complete run message is dispatched and acked."""
        cfg = _make_cfg()
        msg = _make_msg({
            "command": "run",
            "session_id": "meeting-2026-04-23",
            "prompt": "Analyse this session and save to SiYuan.",
        }, msg_id="msg-chain-001")

        p1, p2, p3 = self._patch_run()
        with p1, p2, p3 as mock_turn:
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        msg.ack.assert_awaited_once()
        msg.nack.assert_not_awaited()
        mock_turn.assert_awaited_once_with(
            "meeting-2026-04-23",
            "Analyse this session and save to SiYuan.",
            cfg,
            cfg.db_dsn,
        )

    def test_completed_status_and_response_stored(self):
        """run row progresses pending → running → completed with the reply."""
        cfg = _make_cfg()
        msg = _make_msg({
            "command": "run",
            "session_id": "s1",
            "prompt": "Summarise.",
        })

        with (
            patch(
                "pawn_server.core.queue_listener.create_agent_run",
                return_value="run-uuid",
            ) as mock_create,
            patch("pawn_server.core.queue_listener.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
                return_value="Summary text.",
            ),
        ):
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        mock_create.assert_called_once()
        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert statuses == ["running", "completed"]
        completed_kwargs = mock_update.call_args_list[1].kwargs
        assert completed_kwargs.get("response") == "Summary text."

    def test_per_message_model_override(self):
        """A per-message model is applied before calling handle_turn."""
        cfg = _make_cfg()
        msg = _make_msg({
            "command": "run",
            "session_id": "s2",
            "prompt": "Go.",
            "model": "openai:gpt-4o",
        })

        with (
            patch("pawn_server.core.queue_listener.create_agent_run", return_value="r") as mock_create,
            patch("pawn_server.core.queue_listener.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
                return_value="ok",
            ),
            patch("pawn_server.core.queue_listener._apply_model_override") as mock_override,
        ):
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        mock_override.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["model"] is not None  # recorded (whatever override produced)

    def test_missing_prompt_nacks_and_marks_failed(self):
        """A run message with no prompt is nacked; run row marked failed."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "run", "session_id": "s3"})

        with (
            patch(
                "pawn_server.core.queue_listener.create_agent_run",
                return_value="run-bad",
            ),
            patch("pawn_server.core.queue_listener.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
            ) as mock_turn,
        ):
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()
        mock_turn.assert_not_awaited()
        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses

    def test_missing_session_id_nacks_and_marks_failed(self):
        """A run message with no session_id is nacked; run row marked failed."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "run", "prompt": "Analyse."})

        with (
            patch(
                "pawn_server.core.queue_listener.create_agent_run",
                return_value="run-no-sess",
            ),
            patch("pawn_server.core.queue_listener.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
            ) as mock_turn,
        ):
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()
        mock_turn.assert_not_awaited()
        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses

    def test_agent_exception_nacks_and_marks_failed(self):
        """An error from handle_turn nacks the message and marks the run failed."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "run", "session_id": "s4", "prompt": "Do it."})

        with (
            patch("pawn_server.core.queue_listener.create_agent_run", return_value="run-err"),
            patch("pawn_server.core.queue_listener.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._registry.handle_turn",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM timeout"),
            ),
        ):
            from pawn_server.core.queue_listener import make_message_handler
            asyncio.run(make_message_handler(cfg)(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()
        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses
        failed_kwargs = next(
            c.kwargs for c in mock_update.call_args_list if c.args[2] == "failed"
        )
        assert "LLM timeout" in (failed_kwargs.get("error") or "")


# ──────────────────────────────────────────────────────────────────────────────
# dispatch unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatch:

    def test_dispatch_unsupported_command_raises(self):
        """dispatch raises ValueError for any command not in COMMAND_DEFAULTS."""
        from pawn_server.core.queue_listener import dispatch

        cfg = _make_cfg()

        with pytest.raises(ValueError, match="Unsupported command"):
            asyncio.run(dispatch("transcribe-diarize", {}, cfg))

    def test_dispatch_run_calls_run_langgraph(self):
        """dispatch('run') delegates to _run_langgraph."""
        from pawn_server.core.queue_listener import dispatch

        cfg = _make_cfg()
        params = {"prompt": "Hello.", "session_id": "sess-x", "model": None}

        with patch("pawn_server.core.queue_listener._run_langgraph", new_callable=AsyncMock) as mock_run:
            asyncio.run(dispatch("run", params, cfg, message_id="m1"))

        mock_run.assert_awaited_once_with(params, cfg, "m1")


