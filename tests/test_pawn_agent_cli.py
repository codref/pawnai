from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pawn_agent.cli.commands import app


runner = CliRunner()


def test_chat_routes_to_langgraph_runner() -> None:
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
        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "mode=langgraph" in result.stdout
    assert captured["cfg"] is cfg
    assert callable(captured["emit"])
    assert callable(captured["on_thinking"])
    assert captured["trace_full_state"] is False


def test_chat_passes_trace_state_flag() -> None:
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
        result = runner.invoke(app, ["chat", "--trace-state"])

    assert result.exit_code == 0
    assert captured["trace_full_state"] is True


