"""Agent turns publish a vault resync only after a successful write."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pawn_agent.core.agent_runner import run_agent_turn


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(db_dsn="sqlite://", pydantic_model="openai:gpt-4o")


def _patch_db(monkeypatch) -> None:
    monkeypatch.setattr(
        "pawn_agent.core.agent_runner.create_agent_run",
        lambda *args, **kwargs: "run-1",
    )
    monkeypatch.setattr(
        "pawn_agent.core.agent_runner.update_agent_run",
        lambda *args, **kwargs: None,
    )


def test_run_agent_turn_publishes_vault_paths(monkeypatch) -> None:
    _patch_db(monkeypatch)
    published: list[tuple] = []

    class Registry:
        async def handle_turn(
            self, *args: Any, vault_paths_out: list[str] | None = None, **kwargs: Any
        ) -> str:
            if vault_paths_out is not None:
                vault_paths_out.extend(["Pawn/Notes/A.md"])
            return "done"

    def publish(paths: list[str], *, source: str, run_id: str | None = None) -> None:
        published.append((list(paths), source, run_id))

    monkeypatch.setattr("pawn_server.core.vault_events.publish_vault_event", publish)
    result = asyncio.run(
        run_agent_turn(
            cfg=_cfg(),
            registry=Registry(),
            prompt="write a note",
            session_id="chat:1",
            source="api",
        )
    )
    assert result.run_id == "run-1"
    assert result.response == "done"
    assert published == [(["Pawn/Notes/A.md"], "api", "run-1")]


def test_run_agent_turn_skips_publish_without_writes(monkeypatch) -> None:
    _patch_db(monkeypatch)
    published: list[tuple] = []

    class Registry:
        async def handle_turn(self, *args: Any, **kwargs: Any) -> str:
            return "just chatted"

    def publish(paths: list[str], *, source: str, run_id: str | None = None) -> None:
        published.append((list(paths), source, run_id))

    monkeypatch.setattr("pawn_server.core.vault_events.publish_vault_event", publish)
    result = asyncio.run(
        run_agent_turn(
            cfg=_cfg(),
            registry=Registry(),
            prompt="hello",
            session_id="chat:1",
            source="matrix",
        )
    )
    assert result.response == "just chatted"
    assert published == []
