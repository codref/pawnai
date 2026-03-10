"""SkillRegistry — owns loaded skill and tool definitions.

Acts as the single source of truth for what skills and tools are available.
The DSPy planner reads skill descriptions from it; the SkillRunner reads tool
steps from it; the ExecutorRegistry resolves the actual callables.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .models import SkillDefinition, ToolDefinition

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Holds loaded :class:`SkillDefinition` and :class:`ToolDefinition` objects.

    Args:
        skills: List of loaded skill definitions.
        tools:  List of loaded tool definitions.
    """

    def __init__(
        self,
        skills: List[SkillDefinition],
        tools: List[ToolDefinition],
    ) -> None:
        self._skills: Dict[str, SkillDefinition] = {s.name: s for s in skills}
        self._tools: Dict[str, ToolDefinition] = {t.name: t for t in tools}

        if len(self._skills) != len(skills):
            logger.warning("Duplicate skill names detected — later entries overwrite earlier ones")
        if len(self._tools) != len(tools):
            logger.warning("Duplicate tool names detected — later entries overwrite earlier ones")

        logger.info(
            "SkillRegistry ready: %d skill(s), %d tool(s)",
            len(self._skills),
            len(self._tools),
        )

    # ── skills ────────────────────────────────────────────────────────────────

    def get_skill(self, name: str) -> SkillDefinition:
        """Return the skill definition for *name*.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        try:
            return self._skills[name]
        except KeyError:
            available = ", ".join(sorted(self._skills))
            raise KeyError(
                f"Unknown skill {name!r}. Available: {available or '(none)'}"
            ) from None

    def list_skills(self) -> List[SkillDefinition]:
        return list(self._skills.values())

    def describe_skills(self) -> str:
        """Return a JSON string listing all skills — fed to the DSPy planner."""
        descriptions = [s.describe() for s in self._skills.values()]
        return json.dumps(descriptions, indent=2)

    # ── tools ─────────────────────────────────────────────────────────────────

    def get_tool(self, name: str) -> ToolDefinition:
        """Return the tool definition for *name*.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools))
            raise KeyError(
                f"Unknown tool {name!r}. Available: {available or '(none)'}"
            ) from None

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    # ── misc ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SkillRegistry(skills={list(self._skills)}, tools={list(self._tools)})"
        )
