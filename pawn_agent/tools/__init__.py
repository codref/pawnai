"""pawn_agent.tools — auto-discovered agent tool registry (PydanticAI).

Drop any Python module into this package that exports:

    NAME: str          — tool name shown in ``pawn-agent tools``
    DESCRIPTION: str   — one-line description
    build(cfg)         — returns a pydantic_ai.Tool instance

Private modules (names starting with ``_``) are ignored.
"""

from __future__ import annotations

import importlib
import inspect
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


def build_tools(cfg: AgentConfig, session_vars=None) -> List[Tool]:
    """Return all discovered tools, built for the given config.

    When *session_vars* is provided, it is passed as a keyword argument to any
    tool module whose ``build()`` function declares a ``session_vars`` parameter.
    Modules without that parameter are called as ``build(cfg)`` unchanged, so
    all existing tools remain compatible without modification.
    """
    tools = []
    for mod in _load_tool_modules():
        sig = inspect.signature(mod.build)
        if session_vars is not None and "session_vars" in sig.parameters:
            tools.append(mod.build(cfg, session_vars=session_vars))
        else:
            tools.append(mod.build(cfg))
    return tools
