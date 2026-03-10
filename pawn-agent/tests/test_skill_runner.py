"""Tests for SkillRunner — the tool pipeline executor."""

from __future__ import annotations

import pytest

from pawn_agent.skills.runner import SkillRunner


@pytest.mark.asyncio
async def test_skill_runner_basic_pipeline(skill_runner: SkillRunner) -> None:
    """A happy-path run through the transcribe skill."""
    result = await skill_runner.run(
        skill_name="transcribe",
        input_params={"audio_path": "s3://bucket/audio.wav"},
        context={},
        cfg=None,
    )
    assert result["transcript"] == "Hello world."
    assert result["session_id"] == "sess-001"


@pytest.mark.asyncio
async def test_skill_runner_unknown_skill(skill_runner: SkillRunner) -> None:
    with pytest.raises(KeyError, match="Unknown skill"):
        await skill_runner.run(
            skill_name="nonexistent_skill",
            input_params={},
        )


@pytest.mark.asyncio
async def test_skill_runner_step_outputs_accumulate(skill_runner: SkillRunner) -> None:
    """Verify steps run in order and each step receives correct inputs."""
    call_log: list = []

    from pawn_agent.skills.executor_registry import ExecutorRegistry
    from pawn_agent.skills.models import SkillDefinition, SkillToolStep, ToolDefinition
    from pawn_agent.skills.registry import SkillRegistry

    tool_a = ToolDefinition(name="tool_a", description="A", function="a.run", input_schema={}, output_schema={})
    tool_b = ToolDefinition(name="tool_b", description="B", function="b.run", input_schema={}, output_schema={})

    async def run_a(params, cfg):
        call_log.append(("a", params))
        return {"value": 42}

    async def run_b(params, cfg):
        call_log.append(("b", params))
        return {"doubled": params["value"] * 2}

    reg = ExecutorRegistry()
    reg.register("a.run", run_a)
    reg.register("b.run", run_b)

    skill = SkillDefinition(
        name="chain",
        description="Chain two tools",
        tools=[
            SkillToolStep(step=1, tool="tool_a", params={"x": "{{input.x}}"}),
            SkillToolStep(step=2, tool="tool_b", params={"value": "{{steps.1.value}}"}),
        ],
        output={"result": "{{steps.2.doubled}}"},
        input_schema={},
    )

    runner = SkillRunner(
        SkillRegistry(skills=[skill], tools=[tool_a, tool_b]),
        reg,
    )
    result = await runner.run("chain", {"x": "ignored"}, cfg=None)

    assert result == {"result": 84}
    assert call_log[0] == ("a", {"x": "ignored"})
    assert call_log[1] == ("b", {"value": 42})
