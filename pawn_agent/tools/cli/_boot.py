"""Shared bootstrap for pawn CliTool entrypoints.

CliTools run as subprocesses. They must:
- load AgentConfig via the same discovery as the parent process
- print human-readable stdout (the model reads this)
- exit non-zero only on hard failure

Async ``*_impl`` helpers are driven with ``asyncio.run`` here because
sallm's tool runner is synchronous.
"""

from __future__ import annotations

import sys
from typing import Optional

from pawn_agent.utils.config import AgentConfig, load_config


def load_agent_config(config_path: Optional[str] = None) -> AgentConfig:
    """Load config; prefer explicit --config, else cwd yaml / env defaults."""
    return load_config(config_path)


def print_out(text: str) -> None:
    """Write tool output to stdout (observation text for the ReAct loop)."""
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def fail(message: str, code: int = 1) -> int:
    """Print an error observation and return a non-zero exit code."""
    print_out(message if message.startswith("Error") else f"Error: {message}")
    return code
