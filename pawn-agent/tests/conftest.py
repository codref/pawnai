"""Shared pytest fixtures for pawn-agent tests."""

from __future__ import annotations

import pytest

from pawn_agent.skills.executor_registry import ExecutorRegistry
from pawn_agent.skills.models import SkillDefinition, SkillToolStep, ToolDefinition
from pawn_agent.skills.registry import SkillRegistry
from pawn_agent.skills.runner import SkillRunner


@pytest.fixture
def download_tool() -> ToolDefinition:
    return ToolDefinition(
        name="download_s3",
        description="Download from S3",
        function="s3.download",
        input_schema={"type": "object", "required": ["audio_path"]},
        output_schema={
            "type": "object",
            "properties": {"local_path": {"type": "string"}, "temp_dir": {"type": "string"}},
        },
    )


@pytest.fixture
def transcription_tool() -> ToolDefinition:
    return ToolDefinition(
        name="run_transcription",
        description="Run ASR",
        function="transcription.run",
        input_schema={"type": "object", "required": ["local_path"]},
        output_schema={
            "type": "object",
            "properties": {"transcript": {"type": "string"}, "session_id": {"type": "string"}},
        },
    )


@pytest.fixture
def cleanup_tool() -> ToolDefinition:
    return ToolDefinition(
        name="cleanup_temp",
        description="Remove temp dir",
        function="s3.cleanup",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


@pytest.fixture
def transcribe_skill(
    download_tool: ToolDefinition,
    transcription_tool: ToolDefinition,
    cleanup_tool: ToolDefinition,
) -> SkillDefinition:
    return SkillDefinition(
        name="transcribe",
        description="Transcribe audio",
        tools=[
            SkillToolStep(step=1, tool="download_s3", params={"audio_path": "{{input.audio_path}}"}),
            SkillToolStep(step=2, tool="run_transcription", params={"local_path": "{{steps.1.local_path}}"}),
            SkillToolStep(step=3, tool="cleanup_temp", params={"temp_dir": "{{steps.1.temp_dir}}"}),
        ],
        output={"transcript": "{{steps.2.transcript}}", "session_id": "{{steps.2.session_id}}"},
        input_schema={"type": "object", "required": ["audio_path"]},
    )


@pytest.fixture
def skill_registry(
    transcribe_skill: SkillDefinition,
    download_tool: ToolDefinition,
    transcription_tool: ToolDefinition,
    cleanup_tool: ToolDefinition,
) -> SkillRegistry:
    return SkillRegistry(
        skills=[transcribe_skill],
        tools=[download_tool, transcription_tool, cleanup_tool],
    )


@pytest.fixture
def executor_registry() -> ExecutorRegistry:
    registry = ExecutorRegistry()

    async def fake_download(params, cfg):
        return {"local_path": "/tmp/fake/audio.wav", "temp_dir": "/tmp/fake"}

    async def fake_transcribe(params, cfg):
        return {"transcript": "Hello world.", "session_id": "sess-001"}

    async def fake_cleanup(params, cfg):
        return {}

    registry.register("s3.download", fake_download)
    registry.register("transcription.run", fake_transcribe)
    registry.register("s3.cleanup", fake_cleanup)
    return registry


@pytest.fixture
def skill_runner(skill_registry: SkillRegistry, executor_registry: ExecutorRegistry) -> SkillRunner:
    return SkillRunner(skill_registry, executor_registry)
