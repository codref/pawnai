"""Tests for the queue listener message handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pawn_agent.agent.planner import ExecutionPlan, PlanStep
from pawn_agent.listener import make_message_handler


def _make_msg(payload: dict, msg_id: str = "msg-001") -> MagicMock:
    msg = MagicMock()
    msg.id = msg_id
    msg.payload = payload
    msg.ack = AsyncMock()
    msg.nack = AsyncMock()
    return msg


@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.plan.return_value = ExecutionPlan(
        steps=[PlanStep(step=1, skill="transcribe", params={"audio_path": "s3://b/a.wav"})]
    )
    return planner


@pytest.mark.asyncio
async def test_handler_acks_on_success(mock_planner, skill_runner) -> None:
    from pawn_agent.config import AgentConfig

    cfg = AgentConfig()

    with patch("pawn_agent.listener.execute_plan", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"transcript": "Hello"}
        handler = make_message_handler(cfg, mock_planner, skill_runner)
        msg = _make_msg({
            "request": "Transcribe the audio file",
            "context": {"audio_path": "s3://bucket/audio.wav"},
        })
        await handler(msg)

    msg.ack.assert_called_once()
    msg.nack.assert_not_called()
    mock_planner.plan.assert_called_once()


@pytest.mark.asyncio
async def test_handler_nacks_on_missing_request(mock_planner, skill_runner) -> None:
    from pawn_agent.config import AgentConfig

    cfg = AgentConfig()
    handler = make_message_handler(cfg, mock_planner, skill_runner)
    msg = _make_msg({"context": {}})  # no 'request' key
    await handler(msg)

    msg.nack.assert_called_once()
    msg.ack.assert_not_called()


@pytest.mark.asyncio
async def test_handler_nacks_on_empty_request(mock_planner, skill_runner) -> None:
    from pawn_agent.config import AgentConfig

    cfg = AgentConfig()
    handler = make_message_handler(cfg, mock_planner, skill_runner)
    msg = _make_msg({"request": "   "})  # whitespace only
    await handler(msg)

    msg.nack.assert_called_once()


@pytest.mark.asyncio
async def test_handler_nacks_on_execution_error(mock_planner, skill_runner) -> None:
    from pawn_agent.config import AgentConfig

    cfg = AgentConfig()

    with patch("pawn_agent.listener.execute_plan", new_callable=AsyncMock) as mock_exec:
        mock_exec.side_effect = RuntimeError("Something went wrong")
        handler = make_message_handler(cfg, mock_planner, skill_runner)
        msg = _make_msg({"request": "Do something", "context": {}})
        await handler(msg)

    msg.nack.assert_called_once()
    msg.ack.assert_not_called()


@pytest.mark.asyncio
async def test_handler_nacks_on_planner_error(mock_planner, skill_runner) -> None:
    from pawn_agent.config import AgentConfig

    cfg = AgentConfig()
    mock_planner.plan.side_effect = ValueError("Cannot produce a valid plan")
    handler = make_message_handler(cfg, mock_planner, skill_runner)
    msg = _make_msg({"request": "Something impossible", "context": {}})
    await handler(msg)

    msg.nack.assert_called_once()
    msg.ack.assert_not_called()
