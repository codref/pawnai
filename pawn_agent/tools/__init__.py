"""pawn_agent.tools — auto-discovered agent tool registry (PydanticAI).

Drop any Python module into this package that exports:

    NAME: str          — tool name shown in ``pawn-agent tools``
    DESCRIPTION: str   — one-line description
    build(cfg)         — returns a pydantic_ai.Tool instance

Private modules (names starting with ``_``) are ignored.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, List, Tuple

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


def _load_tool_modules() -> List[Any]:
    """Import every non-private module in this package that has a ``build`` callable."""
    package_dir = Path(__file__).parent
    modules = []
    for _, module_name, _ in sorted(pkgutil.iter_modules([str(package_dir)])):
        if module_name.startswith("_"):
            continue
        mod = importlib.import_module(f"pawn_agent.tools.{module_name}")
        if callable(getattr(mod, "build", None)):
            modules.append(mod)
    return modules


def get_registry() -> List[Tuple[str, str]]:
    """Return ``(name, description)`` for every discovered tool."""
    return [
        (getattr(mod, "NAME", mod.__name__), getattr(mod, "DESCRIPTION", ""))
        for mod in _load_tool_modules()
    ]


def build_tools(cfg: AgentConfig) -> List[Tool]:
    """Return all discovered tools, built for the given config."""
    return [mod.build(cfg) for mod in _load_tool_modules()]
