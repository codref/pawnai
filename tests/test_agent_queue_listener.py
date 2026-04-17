"""Tests for pawn_server queue listener.

Simulates the end-to-end flow of a message published by pawn-diarize's
chain_agent feature arriving at pawn-agent's make_message_handler.

pawn-diarize publishes this payload after a successful transcribe-diarize job:

    {
        "command": "run",
        "session_id": "<session>",
        "prompt": "Analyze this conversation and save the analysis to SiYuan.",
    }

The tests cover:
  - Happy path: message processed, agent called, acked, DB persisted
  - Per-message model override
  - No session_id (prompt passed unchanged)
  - Missing command key → nack
  - Unsupported command → nack
  - Missing prompt → nack + run marked failed
  - Agent exception → nack + run marked failed
  - dispatch / _run_prompt unit tests
  - _merge_params defaults
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

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
# make_message_handler — happy path
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeMessageHandlerHappyPath:
    """Simulate chain_agent messages arriving at pawn-agent listen."""

    def test_run_message_from_pawn_diarize(self):
        """Well-formed chain_agent message: agent runs, acked, DB persisted."""
        cfg = _make_cfg()
        # Exact payload published by pawn-diarize chain_agent
        msg = _make_msg(
            {
                "command": "run",
                "session_id": "session-abc123",
                "prompt": "Analyze this conversation and save the analysis to SiYuan.",
            },
            msg_id="msg-chain-001",
        )

        with (
            patch(
                "pawn_agent.utils.db.create_agent_run", return_value="run-uuid-1"
            ) as mock_create,
            patch("pawn_agent.utils.db.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get_agent,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "## Analysis\n\nSummary of session."
            mock_get_agent.return_value = mock_agent

            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        # Message acknowledged
        msg.ack.assert_awaited_once()
        msg.nack.assert_not_awaited()

        # AgentRun row created with correct fields
        mock_create.assert_called_once_with(
            cfg.db_dsn,
            message_id="msg-chain-001",
            command="run",
            prompt="Analyze this conversation and save the analysis to SiYuan.",
            session_id="session-abc123",
            model=cfg.pydantic_model,
        )

        # Status progression: pending → running → completed
        assert mock_update.call_args_list[0] == call(
            cfg.db_dsn, "run-uuid-1", "running"
        )
        assert mock_update.call_args_list[1] == call(
            cfg.db_dsn,
            "run-uuid-1",
            "completed",
            response="## Analysis\n\nSummary of session.",
        )

        # Agent called with session hint prepended to prompt
        expected_prompt = (
            "[Session ID: session-abc123]\n"
            "Analyze this conversation and save the analysis to SiYuan."
        )
        mock_agent.run.assert_called_once_with(expected_prompt)

    def test_run_message_without_session_id(self):
        """A run message without session_id passes the prompt unchanged."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "run", "prompt": "Summarise the latest meeting."})

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-uuid-2"),
            patch("pawn_agent.utils.db.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get_agent,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "Result."
            mock_get_agent.return_value = mock_agent

            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        msg.ack.assert_awaited_once()
        mock_agent.run.assert_called_once_with("Summarise the latest meeting.")

    def test_per_message_model_override(self):
        """A per-message model override is recorded in the DB and passed to the agent."""
        cfg = _make_cfg()
        msg = _make_msg(
            {
                "command": "run",
                "prompt": "Analyse session xyz.",
                "session_id": "xyz",
                "model": "anthropic:claude-sonnet-4-5",
            }
        )

        with (
            patch(
                "pawn_agent.utils.db.create_agent_run", return_value="run-uuid-3"
            ) as mock_create,
            patch("pawn_agent.utils.db.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get_agent,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "Done."
            mock_get_agent.return_value = mock_agent

            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        # Override model recorded in the run row
        _, kwargs = mock_create.call_args
        assert kwargs["model"] == "anthropic:claude-sonnet-4-5"

        # Agent built with the override
        mock_get_agent.assert_called_once_with(
            cfg,
            "anthropic:claude-sonnet-4-5",
            session_id="xyz",
        )

        msg.ack.assert_awaited_once()

    def test_command_key_case_insensitive(self):
        """Command matching is case-insensitive (e.g. 'RUN' == 'run')."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "RUN", "prompt": "Hello."})

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="r"),
            patch("pawn_agent.utils.db.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get_agent,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "ok"
            mock_get_agent.return_value = mock_agent

            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        msg.ack.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────────────
# make_message_handler — error / dead-letter paths
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeMessageHandlerErrorPaths:

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
        """Messages with a command that isn't 'run' are dead-lettered.

        Note: pawn-diarize commands (transcribe-diarize, diarize, etc.) are
        NOT valid in the agent queue — only 'run' is supported here.
        """
        cfg = _make_cfg()
        msg = _make_msg(
            {"command": "transcribe-diarize", "audio_paths": ["audio.wav"]}
        )

        from pawn_server.core.queue_listener import make_message_handler

        handler = make_message_handler(cfg)
        asyncio.run(handler(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()

    def test_missing_prompt_nacks_and_marks_failed(self):
        """A 'run' message with no prompt is dead-lettered; run row marked failed."""
        cfg = _make_cfg()
        msg = _make_msg({"command": "run", "session_id": "abc"})  # no prompt

        with (
            patch(
                "pawn_agent.utils.db.create_agent_run", return_value="run-uuid-4"
            ),
            patch("pawn_agent.utils.db.update_agent_run") as mock_update,
        ):
            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()

        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses

    def test_agent_exception_nacks_and_marks_failed(self):
        """An agent error dead-letters the message and marks the run failed."""
        cfg = _make_cfg()
        msg = _make_msg(
            {"command": "run", "prompt": "Do something", "session_id": "err-session"}
        )

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-uuid-5"),
            patch("pawn_agent.utils.db.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get_agent,
        ):
            mock_agent = MagicMock()
            mock_agent.run.side_effect = RuntimeError("LLM timeout")
            mock_get_agent.return_value = mock_agent

            from pawn_server.core.queue_listener import make_message_handler

            handler = make_message_handler(cfg)
            asyncio.run(handler(msg))

        msg.nack.assert_awaited_once()
        msg.ack.assert_not_awaited()

        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses

        # Error text is captured in the run row
        failed_call = next(
            c for c in mock_update.call_args_list if c.args[2] == "failed"
        )
        assert "LLM timeout" in (failed_call.kwargs.get("error") or "")


# ──────────────────────────────────────────────────────────────────────────────
# dispatch unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatch:

    def test_dispatch_run_calls_run_prompt(self):
        """dispatch('run') delegates to _run_prompt via executor."""
        from pawn_server.core.queue_listener import dispatch

        cfg = _make_cfg()
        params = {"prompt": "Hello", "session_id": None, "model": None}

        with patch("pawn_server.core.queue_listener._run_prompt") as mock_run:
            asyncio.run(dispatch("run", params, cfg, message_id="m1"))

        mock_run.assert_called_once_with(params, cfg, "m1")

    def test_dispatch_unsupported_command_raises(self):
        """dispatch raises ValueError for any command other than 'run'."""
        from pawn_server.core.queue_listener import dispatch

        cfg = _make_cfg()

        with pytest.raises(ValueError, match="Unsupported command"):
            asyncio.run(dispatch("transcribe-diarize", {}, cfg))


# ──────────────────────────────────────────────────────────────────────────────
# _run_prompt unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRunPrompt:

    def test_success_persists_completed_status(self):
        """_run_prompt: pending → running → completed, response stored."""
        from pawn_server.core.queue_listener import _run_prompt

        cfg = _make_cfg()
        params = {
            "prompt": "Summarise session test-01.",
            "session_id": "test-01",
            "model": None,
        }

        with (
            patch(
                "pawn_agent.utils.db.create_agent_run", return_value="r1"
            ) as mock_create,
            patch("pawn_agent.utils.db.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "The session was about X."
            mock_get.return_value = mock_agent

            _run_prompt(params, cfg, message_id="m-test")

        mock_create.assert_called_once()
        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert statuses == ["running", "completed"]

        completed_call = mock_update.call_args_list[1]
        assert completed_call.kwargs.get("response") == "The session was about X."

    def test_failure_persists_failed_status(self):
        """_run_prompt: re-raises the exception and marks the run failed."""
        from pawn_server.core.queue_listener import _run_prompt

        cfg = _make_cfg()
        params = {"prompt": "Test", "session_id": None, "model": None}

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="r2"),
            patch("pawn_agent.utils.db.update_agent_run") as mock_update,
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get,
        ):
            mock_agent = MagicMock()
            mock_agent.run.side_effect = ConnectionError("DB unreachable")
            mock_get.return_value = mock_agent

            with pytest.raises(ConnectionError):
                _run_prompt(params, cfg)

        statuses = [c.args[2] for c in mock_update.call_args_list]
        assert "failed" in statuses

    def test_session_hint_prepended_to_prompt(self):
        """Session ID is prepended to the effective prompt sent to the agent."""
        from pawn_server.core.queue_listener import _run_prompt

        cfg = _make_cfg()
        params = {
            "prompt": "Analyse conversation.",
            "session_id": "my-session",
            "model": None,
        }

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="r3"),
            patch("pawn_agent.utils.db.update_agent_run"),
            patch(
                "pawn_server.core.queue_listener._get_or_create_agent"
            ) as mock_get,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = "ok"
            mock_get.return_value = mock_agent

            _run_prompt(params, cfg)

        mock_get.assert_called_once_with(cfg, None, session_id="my-session")
        mock_agent.run.assert_called_once_with(
            "[Session ID: my-session]\nAnalyse conversation."
        )


# ──────────────────────────────────────────────────────────────────────────────
# _merge_params
# ──────────────────────────────────────────────────────────────────────────────


class TestMergeParams:

    def test_defaults_filled_in(self):
        from pawn_server.core.queue_listener import _merge_params

        result = _merge_params("run", {"prompt": "Hello"})
        assert result["prompt"] == "Hello"
        assert result["session_id"] is None
        assert result["model"] is None

    def test_payload_overrides_defaults(self):
        from pawn_server.core.queue_listener import _merge_params

        result = _merge_params("run", {"prompt": "Hi", "model": "openai:gpt-4o"})
        assert result["model"] == "openai:gpt-4o"

    def test_extra_fields_preserved(self):
        """Unknown payload fields pass through (not stripped)."""
        from pawn_server.core.queue_listener import _merge_params

        result = _merge_params("run", {"prompt": "Hi", "custom_field": "value"})
        assert result["custom_field"] == "value"


class TestReplayHistorySelection:

    def _make_history_cfg(self, mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            db_dsn="postgresql+psycopg://dummy/dummy",
            pydantic_model="openai:test-model",
            strip_thinking=True,
            history_mode=mode,
            history_recent_turns=4,
            history_replay_max_tokens=8000,
            history_max_text_chars=500,
            history_sanitize_leaked_thoughts=True,
        )

    def test_run_prompt_uses_compact_replay_history_by_default(self) -> None:
        from pawn_server.core.queue_listener import _run_prompt

        cfg = self._make_history_cfg("compact")
        params = {"prompt": "Analyse conversation.", "session_id": "sess-1", "model": None}
        result = MagicMock()
        result.output = "ok"
        result.new_messages.return_value = []

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-1"),
            patch("pawn_agent.utils.db.update_agent_run"),
            patch("pawn_agent.core.session_store.build_replay_history", return_value=["compact"]) as mock_compact,
            patch("pawn_agent.core.session_store.load_history", return_value=["raw"]) as mock_raw,
            patch("pawn_agent.core.session_store.append_turn"),
            patch("pawn_server.core.queue_listener._get_or_create_agent") as mock_get,
        ):
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = result
            mock_get.return_value = mock_agent

            _run_prompt(params, cfg, message_id="msg-1")

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
        mock_agent.run_sync.assert_called_once_with(
            "[Session ID: sess-1]\nAnalyse conversation.",
            message_history=["compact"],
        )

    def test_run_prompt_can_fall_back_to_raw_history(self) -> None:
        from pawn_server.core.queue_listener import _run_prompt

        cfg = self._make_history_cfg("raw")
        params = {"prompt": "Analyse conversation.", "session_id": "sess-1", "model": None}
        result = MagicMock()
        result.output = "ok"
        result.new_messages.return_value = []

        with (
            patch("pawn_agent.utils.db.create_agent_run", return_value="run-1"),
            patch("pawn_agent.utils.db.update_agent_run"),
            patch("pawn_agent.core.session_store.build_replay_history", return_value=["compact"]) as mock_compact,
            patch("pawn_agent.core.session_store.load_history", return_value=["raw"]) as mock_raw,
            patch("pawn_agent.core.session_store.append_turn"),
            patch("pawn_server.core.queue_listener._get_or_create_agent") as mock_get,
        ):
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = result
            mock_get.return_value = mock_agent

            _run_prompt(params, cfg, message_id="msg-1")

        mock_raw.assert_called_once_with("sess-1", cfg.db_dsn, strip_thinking=True)
        mock_compact.assert_not_called()
        mock_agent.run_sync.assert_called_once_with(
            "[Session ID: sess-1]\nAnalyse conversation.",
            message_history=["raw"],
        )
