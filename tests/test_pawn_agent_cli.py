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


def test_chat_burr_mode_routes_to_burr_runner() -> None:
    captured: dict[str, object] = {}

    async def fake_run_burr_chat(**kwargs) -> None:
        captured.update(kwargs)

    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with (
        patch("pawn_agent.utils.config.load_config", return_value=cfg),
        patch("pawn_agent.core.burr_chat.run_burr_chat", side_effect=fake_run_burr_chat),
    ):
        result = runner.invoke(app, ["chat", "--burr", "--burr-graph", "graph.png"])

    assert result.exit_code == 0
    assert "mode=burr" in result.stdout
    assert captured["cfg"] is cfg
    assert callable(captured["emit"])
    assert callable(captured["on_thinking"])
    assert captured["graph_output_path"] == "graph.png"


def test_chat_burr_mode_rejects_session_flag() -> None:
    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with patch("pawn_agent.utils.config.load_config", return_value=cfg):
        result = runner.invoke(app, ["chat", "--burr", "--session", "abc"])

    assert result.exit_code == 1
    assert "does not support --session yet" in result.stdout


def test_chat_burr_graph_requires_burr_flag() -> None:
    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with patch("pawn_agent.utils.config.load_config", return_value=cfg):
        result = runner.invoke(app, ["chat", "--burr-graph", "graph.png"])

    assert result.exit_code == 1
    assert "--burr-graph requires --burr" in result.stdout


def test_chat_langgraph_mode_routes_to_langgraph_runner() -> None:
    captured: dict[str, object] = {}

    async def fake_run_langgraph_chat(**kwargs) -> None:
        captured.update(kwargs)

    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with (
        patch("pawn_agent.utils.config.load_config", return_value=cfg),
        patch("pawn_agent.core.langgraph_chat.run_langgraph_chat", side_effect=fake_run_langgraph_chat),
    ):
        result = runner.invoke(app, ["chat", "--langgraph"])

    assert result.exit_code == 0
    assert "mode=langgraph" in result.stdout
    assert captured["cfg"] is cfg
    assert callable(captured["emit"])
    assert callable(captured["on_thinking"])
    assert captured["trace_full_state"] is False


def test_chat_langgraph_mode_passes_trace_state_flag() -> None:
    captured: dict[str, object] = {}

    async def fake_run_langgraph_chat(**kwargs) -> None:
        captured.update(kwargs)

    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with (
        patch("pawn_agent.utils.config.load_config", return_value=cfg),
        patch("pawn_agent.core.langgraph_chat.run_langgraph_chat", side_effect=fake_run_langgraph_chat),
    ):
        result = runner.invoke(app, ["chat", "--langgraph", "--langgraph-trace-state"])

    assert result.exit_code == 0
    assert captured["trace_full_state"] is True


def test_chat_langgraph_mode_rejects_session_flag() -> None:
    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with patch("pawn_agent.utils.config.load_config", return_value=cfg):
        result = runner.invoke(app, ["chat", "--langgraph", "--session", "abc"])

    assert result.exit_code == 1
    assert "LangGraph mode does not support --session yet" in result.stdout


def test_chat_langgraph_trace_state_requires_langgraph_flag() -> None:
    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with patch("pawn_agent.utils.config.load_config", return_value=cfg):
        result = runner.invoke(app, ["chat", "--langgraph-trace-state"])

    assert result.exit_code == 1
    assert "--langgraph-trace-state requires --langgraph" in result.stdout


def test_chat_rejects_multiple_orchestrators() -> None:
    cfg = SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
    )

    with patch("pawn_agent.utils.config.load_config", return_value=cfg):
        result = runner.invoke(app, ["chat", "--burr", "--langgraph"])

    assert result.exit_code == 1
    assert "Choose only one orchestration mode" in result.stdout
