"""Configuration loader for pawn-agent.

Reads an optional YAML config file (``pawnai.yaml`` shared with pawn-diarize,
or a custom path) and merges with environment variables.

Config file schema (all keys optional)::

    db_dsn: postgresql+psycopg://postgres:postgres@localhost:5432/pawn_diarize

    agent:
      model: gpt-4o          # Copilot model identifier
      name: Bob              # Display name shown before agent responses in chat
      anima: anima.md        # Path to Markdown file injected as system prompt personality

    siyuan:
      url: http://127.0.0.1:6806
      token: ""
      notebook: ""
      path_template: "/Conversations/{date}/{title}"
      daily_note_path: "/daily note/{year}/{month}/{date}"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_DEFAULT_DSN = "postgresql+psycopg://postgres:postgres@localhost:5432/pawnai"
_DEFAULT_SIYUAN_URL = "http://127.0.0.1:6806"
_DEFAULT_PATH_TEMPLATE = "/Conversations/{date}/{title}"
_DEFAULT_DAILY_TEMPLATE = "/daily note/{year}/{month}/{date}"


@dataclass
class AgentConfig:
    # Database
    db_dsn: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", _DEFAULT_DSN))

    # Copilot model
    model: str = "claude-sonnet-4.6"

    # Chat display name shown before agent responses
    agent_name: str = "Bob"

    # Path to a Markdown file that defines the agent's personality (system prompt addition).
    # If None or the file does not exist, no personality is injected.
    anima_path: Optional[str] = None

    # SiYuan
    siyuan_url: str = _DEFAULT_SIYUAN_URL
    siyuan_token: str = ""
    siyuan_notebook: str = ""
    siyuan_path_template: str = _DEFAULT_PATH_TEMPLATE
    siyuan_daily_template: str = _DEFAULT_DAILY_TEMPLATE


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """Load AgentConfig from a YAML file and environment variables.

    Priority (highest → lowest):
    1. Explicit *config_path* argument
    2. ``PAWN_AGENT_CONFIG`` environment variable
    3. ``pawnai.yaml`` / ``pawnai.yml`` in the current working directory
    4. Legacy ``.pawn-diarize.yml`` / ``.pawn-diarize.yaml`` (backward compat)
    5. Dataclass defaults / ``DATABASE_URL`` env var

    CLI overrides are applied on top by the command layer.
    """
    cfg = AgentConfig()

    path = (
        config_path
        or os.environ.get("PAWN_AGENT_CONFIG")
        or _find_default_config()
    )
    if path and yaml is not None:
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            data = {}

        if "db_dsn" in data:
            cfg.db_dsn = data["db_dsn"]

        agent_sec = data.get("agent", {}) or {}
        if "model" in agent_sec:
            cfg.model = agent_sec["model"]
        if "name" in agent_sec:
            cfg.agent_name = agent_sec["name"]
        if "anima" in agent_sec:
            cfg.anima_path = agent_sec["anima"]

        sy = data.get("siyuan", {}) or {}
        if "url" in sy:
            cfg.siyuan_url = sy["url"]
        if "token" in sy:
            cfg.siyuan_token = sy["token"]
        if "notebook" in sy:
            cfg.siyuan_notebook = sy["notebook"]
        if "path_template" in sy:
            cfg.siyuan_path_template = sy["path_template"]
        if "daily_note_path" in sy:
            cfg.siyuan_daily_template = sy["daily_note_path"]

    return cfg


def _find_default_config() -> Optional[str]:
    candidates = (
        "pawnai.yaml", "pawnai.yml",
        ".pawn-diarize.yml", ".pawn-diarize.yaml",
    )
    for name in candidates:
        candidate = Path.cwd() / name
        if candidate.exists():
            return str(candidate)
    return None
