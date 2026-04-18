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

    Tool modules whose ``build()`` has *required* parameters that cannot be
    satisfied (e.g. ``session_vars`` when none is provided) are silently
    skipped so the registry still loads cleanly.
    """
    tools = []
    for mod in _load_tool_modules():
        sig = inspect.signature(mod.build)
        has_sv_param = "session_vars" in sig.parameters
        sv_required = has_sv_param and (
            sig.parameters["session_vars"].default is inspect.Parameter.empty
        )
        if has_sv_param and session_vars is not None:
            tools.append(mod.build(cfg, session_vars=session_vars))
        elif sv_required:
            # Required param we can't supply — skip this tool
            continue
        else:
            tools.append(mod.build(cfg))
    return tools


def build_tools_registry(cfg: AgentConfig) -> dict[str, Any]:
    """Return ``{tool_name: callable}`` for every discovered tool.

    Extracts the underlying callable from each :class:`pydantic_ai.Tool` so
    ``tool_executor`` can call it directly with ``**arguments``.
    """
    registry: dict[str, Any] = {}
    for tool in build_tools(cfg):
        name: str = getattr(tool, "name", "") or getattr(tool, "_name", "")
        fn: Any = getattr(tool, "function", None) or getattr(tool, "_function", None)
        if name and fn is not None:
            registry[name] = fn
    return registry
