"""Tests for the plan executor (execute_plan)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pawn_agent.agent.executor import _resolve_step_params, execute_plan
from pawn_agent.agent.planner import ExecutionPlan, PlanStep


def make_plan(*steps) -> ExecutionPlan:
    return ExecutionPlan(steps=list(steps))


class TestResolveStepParams:
    def test_static_params_only(self):
        step = PlanStep(step=1, skill="transcribe", params={"audio_path": "s3://a/b.wav"})
        result = _resolve_step_params(step, {})
        assert result == {"audio_path": "s3://a/b.wav"}

    def test_input_from_merges_prior_output(self):
        step = PlanStep(step=2, skill="analyze", params={}, input_from=[1])
        prior = {1: {"transcript": "Hello", "session_id": "s1"}}
        result = _resolve_step_params(step, prior)
        assert result == {"transcript": "Hello", "session_id": "s1"}

    def test_explicit_params_override_input_from(self):
        step = PlanStep(
            step=2,
            skill="analyze",
            params={"analysis_type": "summary"},
            input_from=[1],
        )
        prior = {1: {"transcript": "Hello", "analysis_type": "all"}}
        result = _resolve_step_params(step, prior)
        assert result["analysis_type"] == "summary"
        assert result["transcript"] == "Hello"

    def test_missing_input_from_raises(self):
        step = PlanStep(step=2, skill="analyze", input_from=[1])
        with pytest.raises(KeyError, match="step 1 has not yet been executed"):
            _resolve_step_params(step, {})


@pytest.mark.asyncio
async def test_execute_plan_single_step(skill_runner) -> None:
    plan = make_plan(
        PlanStep(
            step=1,
            skill="transcribe",
            params={"audio_path": "s3://bucket/a.wav"},
            description="Transcribe audio",
        )
    )
    result = await execute_plan(plan, skill_runner, context={}, cfg=None)
    assert result["transcript"] == "Hello world."
    assert result["session_id"] == "sess-001"


@pytest.mark.asyncio
async def test_execute_plan_multi_step_piping(skill_runner) -> None:
    """Two steps: transcribe then a mock analyze that receives transcript."""
    from pawn_agent.skills.executor_registry import ExecutorRegistry
    from pawn_agent.skills.models import SkillDefinition, SkillToolStep, ToolDefinition
    from pawn_agent.skills.registry import SkillRegistry
    from pawn_agent.skills.runner import SkillRunner

    # Extend the registry with an analysis tool
    analysis_calls: list = []

    analysis_tool = ToolDefinition(
        name="run_analysis", description="Analyze", function="analysis.run",
        input_schema={}, output_schema={},
    )
    analyze_skill = SkillDefinition(
        name="analyze",
        description="Analyze transcript",
        tools=[
            SkillToolStep(step=1, tool="run_analysis", params={"transcript": "{{input.transcript}}"}),
        ],
        output={"analysis": "{{steps.1.analysis}}"},
        input_schema={},
    )

    async def fake_analysis(params, cfg):
        analysis_calls.append(params)
        return {"analysis": f"Analysis of: {params['transcript']}", "session_id": None}

    exec_reg = skill_runner._executors
    exec_reg.register("analysis.run", fake_analysis)

    # Add analysis tool and skill to registry
    tools = list(skill_runner._skills._tools.values()) + [analysis_tool]
    skills = list(skill_runner._skills._skills.values()) + [analyze_skill]
    new_runner = SkillRunner(SkillRegistry(skills=skills, tools=tools), exec_reg)

    plan = make_plan(
        PlanStep(step=1, skill="transcribe", params={"audio_path": "s3://b/a.wav"}),
        PlanStep(step=2, skill="analyze", params={}, input_from=[1]),
    )

    result = await execute_plan(plan, new_runner, cfg=None)
    assert "analysis" in result
    assert "Hello world." in result["analysis"]
    # Ensure the transcript from step 1 was piped into step 2
    assert analysis_calls[0]["transcript"] == "Hello world."
