"""Tests for AgentPlanner — mocking the DSPy LM to avoid real Copilot calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pawn_agent.agent.planner import AgentPlanner, ExecutionPlan, PlanStep


VALID_PLAN_JSON = json.dumps([
    {
        "step": 1,
        "skill": "transcribe",
        "params": {"audio_path": "s3://bucket/audio.wav"},
        "input_from": [],
        "description": "Transcribe the audio file",
    },
    {
        "step": 2,
        "skill": "analyze",
        "params": {},
        "input_from": [1],
        "description": "Analyze the transcript",
    },
])

FENCED_PLAN_RESPONSE = f"Here is the plan:\n```json\n{VALID_PLAN_JSON}\n```"
RAW_PLAN_RESPONSE = VALID_PLAN_JSON


def _mock_lm(response: str):
    lm = MagicMock()
    lm.return_value = [response]
    return lm


@pytest.fixture(autouse=True)
def mock_dspy(monkeypatch):
    """Patch dspy.settings.lm for all planner tests."""
    import pawn_agent.agent.planner as planner_mod

    mock_settings = MagicMock()
    monkeypatch.setattr("pawn_agent.agent.planner.dspy", MagicMock(settings=mock_settings))
    return mock_settings


def test_planner_parses_fenced_json(mock_dspy) -> None:
    mock_dspy.settings.lm = _mock_lm(FENCED_PLAN_RESPONSE)

    import dspy
    with patch("pawn_agent.agent.planner.dspy", MagicMock(settings=MagicMock(lm=_mock_lm(FENCED_PLAN_RESPONSE)))):
        planner = AgentPlanner()
        plan = planner.plan(
            user_request="Transcribe and analyze the meeting",
            available_skills="[...]",
            context={"audio_path": "s3://bucket/audio.wav"},
        )

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 2
    assert plan.steps[0].skill == "transcribe"
    assert plan.steps[1].skill == "analyze"
    assert plan.steps[1].input_from == [1]


def test_planner_parses_raw_json() -> None:
    with patch("pawn_agent.agent.planner.dspy", MagicMock(settings=MagicMock(lm=_mock_lm(RAW_PLAN_RESPONSE)))):
        planner = AgentPlanner()
        plan = planner.plan(
            user_request="Transcribe audio",
            available_skills="[...]",
        )
    assert len(plan.steps) == 2


def test_planner_retries_on_bad_json() -> None:
    good_response = VALID_PLAN_JSON
    bad_response = "This is not JSON at all."

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [bad_response]
        return [good_response]

    mock_lm = MagicMock(side_effect=side_effect)
    with patch("pawn_agent.agent.planner.dspy", MagicMock(settings=MagicMock(lm=mock_lm))):
        planner = AgentPlanner(max_retries=1)
        plan = planner.plan("Do something", "[...]")

    assert len(plan.steps) == 2
    assert call_count == 2


def test_planner_raises_after_all_retries() -> None:
    mock_lm = MagicMock(return_value=["not json at all"])
    with patch("pawn_agent.agent.planner.dspy", MagicMock(settings=MagicMock(lm=mock_lm))):
        planner = AgentPlanner(max_retries=1)
        with pytest.raises(ValueError, match="failed to produce a valid plan"):
            planner.plan("Do something", "[...]")
