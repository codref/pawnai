"""Tests for DirectorySkillLoader and DirectoryToolLoader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pawn_agent.skills.dir_loader import DirectorySkillLoader, DirectoryToolLoader
from pawn_agent.skills.models import SkillDefinition, ToolDefinition


@pytest.fixture
def tools_dir(tmp_path: Path) -> Path:
    (tmp_path / "download_s3.yaml").write_text(textwrap.dedent("""\
        name: download_s3
        description: Download from S3
        function: s3.download
        enabled: true
        input_schema:
          type: object
          required: [audio_path]
          properties:
            audio_path: {type: string}
        output_schema:
          type: object
          properties:
            local_path: {type: string}
            temp_dir: {type: string}
    """))
    (tmp_path / "disabled_tool.yaml").write_text(textwrap.dedent("""\
        name: disabled_tool
        description: This tool is disabled
        function: noop.noop
        enabled: false
        input_schema: {}
        output_schema: {}
    """))
    (tmp_path / "malformed.yaml").write_text("this: is: not: valid: yaml: {{{")
    return tmp_path


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    (d / "transcribe.yaml").write_text(textwrap.dedent("""\
        name: transcribe
        description: Transcribe audio
        enabled: true
        tools:
          - step: 1
            tool: download_s3
            params:
              audio_path: "{{input.audio_path}}"
          - step: 2
            tool: run_transcription
            params:
              local_path: "{{steps.1.local_path}}"
        output:
          transcript: "{{steps.2.transcript}}"
        input_schema:
          type: object
          required: [audio_path]
    """))
    (d / "disabled.yaml").write_text(textwrap.dedent("""\
        name: disabled_skill
        description: Disabled
        enabled: false
        tools:
          - step: 1
            tool: noop
            params: {}
        output: {}
    """))
    return d


@pytest.mark.asyncio
async def test_tool_loader_loads_enabled(tools_dir: Path) -> None:
    loader = DirectoryToolLoader(tools_dir)
    tools = await loader.load()
    assert len(tools) == 1
    assert tools[0].name == "download_s3"
    assert tools[0].function == "s3.download"


@pytest.mark.asyncio
async def test_tool_loader_skips_disabled(tools_dir: Path) -> None:
    loader = DirectoryToolLoader(tools_dir)
    tools = await loader.load()
    names = [t.name for t in tools]
    assert "disabled_tool" not in names


@pytest.mark.asyncio
async def test_tool_loader_tolerates_malformed(tools_dir: Path) -> None:
    """Malformed YAML files should be skipped without raising."""
    loader = DirectoryToolLoader(tools_dir)
    tools = await loader.load()
    # Only the valid enabled tool should be loaded despite the malformed file
    assert all(isinstance(t, ToolDefinition) for t in tools)


@pytest.mark.asyncio
async def test_tool_loader_missing_dir(tmp_path: Path) -> None:
    """Missing directory returns empty list without raising."""
    loader = DirectoryToolLoader(tmp_path / "nonexistent")
    tools = await loader.load()
    assert tools == []


@pytest.mark.asyncio
async def test_skill_loader_loads_enabled(skills_dir: Path) -> None:
    loader = DirectorySkillLoader(skills_dir)
    skills = await loader.load()
    assert len(skills) == 1
    assert skills[0].name == "transcribe"
    assert len(skills[0].tools) == 2


@pytest.mark.asyncio
async def test_skill_loader_skips_disabled(skills_dir: Path) -> None:
    loader = DirectorySkillLoader(skills_dir)
    skills = await loader.load()
    assert all(s.name != "disabled_skill" for s in skills)


@pytest.mark.asyncio
async def test_skill_step_numbers_unique_validation(tmp_path: Path) -> None:
    d = tmp_path / "skills"
    d.mkdir()
    (d / "bad.yaml").write_text(textwrap.dedent("""\
        name: bad_skill
        description: Skill with duplicate step numbers
        enabled: true
        tools:
          - step: 1
            tool: a
            params: {}
          - step: 1
            tool: b
            params: {}
        output: {}
    """))
    loader = DirectorySkillLoader(d)
    skills = await loader.load()
    # Validation error — the bad skill is skipped, not raised
    assert skills == []
