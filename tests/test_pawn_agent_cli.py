from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pawn_agent.cli.commands import app


runner = CliRunner()


def test_inspect_session_command_renders_troubleshooting_report() -> None:
    report = {
        "session_id": "sess-123",
        "turn_count": 17,
        "raw_message_count": 40,
        "replay_message_count": 40,
        "raw_context_kb": 12.5,
        "raw_context_tokens": 3200,
        "replay_context_kb": 11.0,
        "replay_context_tokens": 2800,
        "session_vars": {"listen_only": True},
        "issues": ["Found 1 request message(s) containing RetryPromptPart."],
        "counts": {"retry_prompt_requests": 1},
        "replay_payload": [{"parts": [{"content": "hi", "part_kind": "user-prompt"}], "kind": "request"}],
        "recent_messages": [
            {
                "index": 40,
                "kind": "response",
                "parts": ["TextPart"],
                "excerpt": "ok",
                "issues": [],
            }
        ],
        "turns": [],
    }

    with (
        patch("pawn_agent.utils.config.load_config", return_value=SimpleNamespace(db_dsn="postgresql://dummy")),
        patch("pawn_agent.core.session_store.inspect_session_history", return_value=report) as mock_inspect,
    ):
        result = runner.invoke(app, ["inspect-session", "sess-123"])

    assert result.exit_code == 0
    assert "Session Inspection" in result.stdout
    assert "sess-123" in result.stdout
    assert "RetryPromptPart" in result.stdout
    mock_inspect.assert_called_once_with("sess-123", "postgresql://dummy", tail=12)


def test_inspect_session_command_can_dump_llm_context() -> None:
    report = {
        "session_id": "sess-123",
        "turn_count": 1,
        "raw_message_count": 2,
        "replay_message_count": 2,
        "raw_context_kb": 1.0,
        "raw_context_tokens": 100,
        "replay_context_kb": 1.0,
        "replay_context_tokens": 100,
        "session_vars": {},
        "issues": [],
        "counts": {},
        "replay_payload": [
            {"parts": [{"content": "how much is 1+1", "part_kind": "user-prompt"}], "kind": "request"},
            {"parts": [{"content": "2", "part_kind": "text"}], "kind": "response"},
        ],
        "recent_messages": [],
        "turns": [],
    }

    with (
        patch("pawn_agent.utils.config.load_config", return_value=SimpleNamespace(db_dsn="postgresql://dummy")),
        patch("pawn_agent.core.session_store.inspect_session_history", return_value=report) as mock_inspect,
    ):
        result = runner.invoke(app, ["inspect-session", "sess-123", "--dump-context"])

    assert result.exit_code == 0
    assert "LLM Context Dump" in result.stdout
    assert "how much is 1+1" in result.stdout
    assert "\"kind\": \"response\"" in result.stdout
    mock_inspect.assert_called_once_with("sess-123", "postgresql://dummy", tail=12)
