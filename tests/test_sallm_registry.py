"""Tests for the sallm session registry façade."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.utils.config import AgentConfig, SallmSection


def _cfg(tmp_path) -> AgentConfig:
    return AgentConfig(
        agent={
            "openai": {"model": "gpt-4o", "api_key": "test", "base_url": "http://x"},
            "sallm": {"state_dir": str(tmp_path / ".sallm"), "max_steps": 4},
        }
    )


def test_litellm_model_mapping(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.litellm_model == "openai/gpt-4o"
    assert isinstance(cfg.sallm, SallmSection)
    assert cfg.sallm.max_steps == 4


def test_registry_handle_turn_offloads_ask(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    registry = SallmSessionRegistry()
    fake_session = MagicMock()

    async def fake_handle(text: str) -> str:
        return f"echo:{text}"

    fake_session.handle_user_input = fake_handle
    fake_session.apply_config = MagicMock()

    async def fake_build(session_id, cfg_arg):
        return fake_session

    with patch.object(registry, "_build_session", side_effect=fake_build):
        reply = asyncio.run(registry.handle_turn("conv-1", "hello", cfg))

    assert reply == "echo:hello"


def test_registry_reset_clears_session(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    registry = SallmSessionRegistry()
    fake_session = MagicMock()
    fake_session.cleared = False

    async def fake_reset() -> None:
        fake_session.cleared = True

    fake_session.reset = fake_reset
    fake_session.apply_config = MagicMock()

    async def fake_build(session_id, cfg_arg):
        return fake_session

    with patch.object(registry, "_build_session", side_effect=fake_build):
        asyncio.run(registry.get_or_create("conv-1", cfg))
        assert "conv-1" in registry._sessions
        asyncio.run(registry.reset("conv-1"))
        assert "conv-1" not in registry._sessions
        assert fake_session.cleared is True


def test_registry_evict_all() -> None:
    registry = SallmSessionRegistry()
    registry._sessions["a"] = MagicMock()
    registry._locks["a"] = asyncio.Lock()
    assert registry.evict_all() == 1
    assert registry._sessions == {}
