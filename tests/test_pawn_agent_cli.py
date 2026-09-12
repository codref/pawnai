"""CLI smoke tests for pawn-agent chat (sallm-backed)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pawn_agent.cli.commands import app
from pawn_agent.utils.config import SallmSection

runner = CliRunner()


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        db_dsn="postgresql://dummy",
        agent_name="Bob",
        pydantic_model="openai:gpt-4o",
        litellm_model="openai/gpt-4o",
        agent=SimpleNamespace(sallm=SallmSection()),
    )


def test_chat_routes_to_sallm_runner() -> None:
    captured: dict[str, object] = {}

    async def fake_run_sallm_chat(**kwargs) -> None:
        captured.update(kwargs)

    cfg = _cfg()

    with (
        patch("pawn_agent.utils.config.load_config", return_value=cfg),
        patch(
            "pawn_agent.core.sallm_session.run_sallm_chat",
            side_effect=fake_run_sallm_chat,
        ),
    ):
        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "mode=sallm" in result.stdout
    assert captured["cfg"] is cfg
    assert callable(captured["emit"])
    assert callable(captured["on_thinking"])
    assert captured["conversation_id"] == "cli"


def test_chat_passes_otlp_into_config() -> None:
    captured: dict[str, object] = {}

    async def fake_run_sallm_chat(**kwargs) -> None:
        captured.update(kwargs)

    cfg = _cfg()

    with (
        patch("pawn_agent.utils.config.load_config", return_value=cfg),
        patch(
            "pawn_agent.core.sallm_session.run_sallm_chat",
            side_effect=fake_run_sallm_chat,
        ),
    ):
        result = runner.invoke(app, ["chat", "--otlp", "http://localhost:4318"])

    assert result.exit_code == 0
    assert cfg.agent.sallm.otlp_endpoint == "http://localhost:4318"
    assert captured["cfg"] is cfg


def test_tools_lists_clitools() -> None:
    fake_tool = MagicMock()
    fake_tool.summary = "List sessions"
    with patch(
        "pawn_agent.core.sallm_tools.build_pawn_clitools",
        return_value={"sessions_list": fake_tool},
    ):
        result = runner.invoke(app, ["tools"])

    assert result.exit_code == 0
    assert "sessions_list" in result.stdout
