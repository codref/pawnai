"""Directory-based skill and tool loaders.

Scans a directory for ``*.yaml`` / ``*.yml`` files and parses each one into a
:class:`~pawn_agent.skills.models.SkillDefinition` or
:class:`~pawn_agent.skills.models.ToolDefinition`.

Malformed files emit a warning and are skipped (fault-tolerant).
Disabled entries (``enabled: false``) are filtered out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml
from pydantic import ValidationError

from .base_loader import BaseSkillLoader, BaseToolLoader
from .models import SkillDefinition, ToolDefinition

logger = logging.getLogger(__name__)


class DirectorySkillLoader(BaseSkillLoader):
    """Load skill definitions from ``*.yaml`` files in a directory.

    Args:
        skills_dir: Path to the directory containing skill YAML files.
    """

    def __init__(self, skills_dir: Path) -> None:
        self._dir = Path(skills_dir)

    async def load(self) -> List[SkillDefinition]:
        """Scan the directory and return all valid, enabled skill definitions."""
        skills: List[SkillDefinition] = []

        if not self._dir.exists():
            logger.warning("Skills directory %r does not exist — no skills loaded", str(self._dir))
            return skills

        for path in sorted(self._dir.glob("*.y*ml")):
            try:
                with path.open() as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    raise ValueError("YAML root must be a mapping")
                skill = SkillDefinition(**data)
                if not skill.enabled:
                    logger.debug("Skill %r is disabled — skipping", skill.name)
                    continue
                skills.append(skill)
                logger.debug("Loaded skill %r from %s", skill.name, path.name)
            except (ValidationError, ValueError, yaml.YAMLError) as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        logger.info("Loaded %d skill(s) from %s", len(skills), self._dir)
        return skills


class DirectoryToolLoader(BaseToolLoader):
    """Load tool definitions from ``*.yaml`` files in a directory.

    Args:
        tools_dir: Path to the directory containing tool YAML files.
    """

    def __init__(self, tools_dir: Path) -> None:
        self._dir = Path(tools_dir)

    async def load(self) -> List[ToolDefinition]:
        """Scan the directory and return all valid, enabled tool definitions."""
        tools: List[ToolDefinition] = []

        if not self._dir.exists():
            logger.warning("Tools directory %r does not exist — no tools loaded", str(self._dir))
            return tools

        for path in sorted(self._dir.glob("*.y*ml")):
            try:
                with path.open() as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    raise ValueError("YAML root must be a mapping")
                tool = ToolDefinition(**data)
                if not tool.enabled:
                    logger.debug("Tool %r is disabled — skipping", tool.name)
                    continue
                tools.append(tool)
                logger.debug("Loaded tool %r from %s", tool.name, path.name)
            except (ValidationError, ValueError, yaml.YAMLError) as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        logger.info("Loaded %d tool(s) from %s", len(tools), self._dir)
        return tools
