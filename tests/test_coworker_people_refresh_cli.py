"""CLI wiring for ``pawn-server coworker people-refresh``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pawn_server.cli.commands import coworker_app

runner = CliRunner()


def test_people_refresh_cli_calls_refresh_not_process():
    called: dict = {}

    async def fake_refresh(cfg, session_id, *, store=None, force=False):
        called["session_id"] = session_id
        called["force"] = force
        return {"session_id": session_id, "applied": 0, "proposed": 0}

    with (
        patch(
            "pawn_agent.utils.config.load_config",
            return_value=SimpleNamespace(coworker=SimpleNamespace(enabled=True)),
        ),
        patch(
            "pawn_agent.core.people.refresh.refresh_people_for_session",
            side_effect=fake_refresh,
        ),
        patch(
            "pawn_agent.core.coworker.pipeline.process_session",
            side_effect=AssertionError("process_session must not run"),
        ),
    ):
        result = runner.invoke(
            coworker_app,
            ["people-refresh", "--session", "insoghts-alignment-1006"],
        )

    assert result.exit_code == 0, result.output
    assert called["session_id"] == "insoghts-alignment-1006"
    assert called["force"] is True  # CLI defaults to force


def test_people_refresh_cli_no_force():
    called: dict = {}

    async def fake_refresh(cfg, session_id, *, store=None, force=False):
        called["force"] = force
        return {"skipped": "duplicate"}

    with (
        patch(
            "pawn_agent.utils.config.load_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "pawn_agent.core.people.refresh.refresh_people_for_session",
            side_effect=fake_refresh,
        ),
    ):
        result = runner.invoke(
            coworker_app,
            ["people-refresh", "--session", "s1", "--no-force"],
        )

    assert result.exit_code == 0, result.output
    assert called["force"] is False
