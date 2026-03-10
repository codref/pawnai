"""Configuration management for pawn-agent.

Reads a YAML file (default: ``.pawnai.yml`` in the working directory, or the
same file used by pawnai) and exposes an :class:`AgentConfig` object that
covers queue connectivity, skill loading, and LLM settings.

Relevant YAML sections::

    # Reuses the existing s3: / queue: sections from .pawnai.yml
    s3:
      endpoint_url: https://...
      bucket_name:  audio-bucket
      access_key:   ...
      secret_key:   ...
      region:       us-east-1
      verify_ssl:   true

    queue:
      bucket_name:   pawnai-queue
      polling:
        interval_seconds: 5
        max_messages_per_poll: 1
        visibility_timeout_seconds: 300
        lease_refresh_interval_seconds: 60

    agent:
      topic:          agent-tasks        # topic this listener subscribes to
      consumer_name:  pawn-agent
      skills_dir:     skills/            # directory of skill YAML files
      tools_dir:      tools/             # directory of tool YAML files
      llm_model:      gpt-4o             # Copilot model to use for planning
      llm_temperature: 0.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_FILE = ".pawnai.yml"

DEFAULT_TOPIC = "agent-tasks"
DEFAULT_CONSUMER_NAME = "pawn-agent"
DEFAULT_SKILLS_DIR = "skills"
DEFAULT_TOOLS_DIR = "tools"
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_LLM_TEMPERATURE = 0.0


class AgentConfig:
    """Configuration for the pawn-agent service.

    Args:
        config_path: Path to a YAML config file.  When *None* the default
            ``.pawnai.yml`` in the current working directory is used.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._raw: Dict[str, Any] = {}
        self._load(config_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self, config_path: Optional[str]) -> None:
        path = Path(config_path) if config_path else Path(CONFIG_FILE)
        if path.exists():
            with path.open() as fh:
                self._raw = yaml.safe_load(fh) or {}

    def _agent_section(self) -> Dict[str, Any]:
        return self._raw.get("agent", {})

    # ── queue connectivity ────────────────────────────────────────────────────

    def get_s3_config(self) -> Dict[str, Any]:
        return self._raw.get("s3", {})

    def get_queue_config(self) -> Optional[Dict[str, Any]]:
        return self._raw.get("queue") or None

    # ── agent settings ────────────────────────────────────────────────────────

    @property
    def topic(self) -> str:
        return self._agent_section().get("topic", DEFAULT_TOPIC)

    @property
    def consumer_name(self) -> str:
        return self._agent_section().get("consumer_name", DEFAULT_CONSUMER_NAME)

    @property
    def skills_dir(self) -> Path:
        raw = self._agent_section().get("skills_dir", DEFAULT_SKILLS_DIR)
        return Path(raw)

    @property
    def tools_dir(self) -> Path:
        raw = self._agent_section().get("tools_dir", DEFAULT_TOOLS_DIR)
        return Path(raw)

    @property
    def llm_model(self) -> str:
        return self._agent_section().get("llm_model", DEFAULT_LLM_MODEL)

    @property
    def llm_temperature(self) -> float:
        return float(self._agent_section().get("llm_temperature", DEFAULT_LLM_TEMPERATURE))

    # ── raw access ────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)
