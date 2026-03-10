"""Abstract base classes for skill and tool loaders.

Implement these interfaces to support different storage backends
(directory, database, remote API, etc.) without touching the rest of the
agent infrastructure.

Example — future database loader::

    class DatabaseSkillLoader(BaseSkillLoader):
        def __init__(self, dsn: str) -> None:
            self._dsn = dsn

        async def load(self) -> list[SkillDefinition]:
            async with get_session(self._dsn) as session:
                rows = await session.execute(select(SkillRow))
                return [SkillDefinition(**row.to_dict()) for row in rows.scalars()]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import SkillDefinition, ToolDefinition


class BaseSkillLoader(ABC):
    """Load :class:`~pawn_agent.skills.models.SkillDefinition` objects."""

    @abstractmethod
    async def load(self) -> List[SkillDefinition]:
        """Return all available (enabled) skill definitions."""


class BaseToolLoader(ABC):
    """Load :class:`~pawn_agent.skills.models.ToolDefinition` objects."""

    @abstractmethod
    async def load(self) -> List[ToolDefinition]:
        """Return all available (enabled) tool definitions."""
