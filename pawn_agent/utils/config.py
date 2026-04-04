"""Configuration loader for pawn-agent.

Reads an optional YAML config file (``pawnai.yaml`` shared with pawn-diarize,
or a custom path) and merges with environment variables.

Config file schema (all keys optional)::

    db_dsn: postgresql+psycopg://postgres:postgres@localhost:5432/pawn_diarize

    agent:
      name: Bob              # Display name shown before agent responses in chat
      anima: anima.md        # Path to Markdown file injected as system prompt personality

      # Primary agent provider — pick one section.
      # Supported providers: openai, anthropic, google, groq, mistral
      openai:
        model: gpt-4o
        api_key: sk-...      # optional; falls back to OPENAI_API_KEY env var
        base_url: http://localhost:11434/v1  # optional; for Ollama / custom endpoints

      # anthropic:
      #   model: claude-sonnet-4-5-20251001
      #   api_key: sk-ant-...

      # Copilot sub-agent used internally by the analyze_custom tool.
      copilot:
        model: gpt-4.1
        # backend: openai        # uncomment to use an OpenAI-compatible endpoint
        # base_url: http://localhost:11434/v1
        # api_key: ollama

    siyuan:
      url: http://127.0.0.1:6806
      token: ""
      notebook: ""
      path_template: "/Conversations/{date}/{title}"
      daily_note_path: "/daily note/{year}/{month}/{date}"

    # Agent queue listener (separate from the pawn-diarize queue: section)
    agent_queue:
      topic: pawn-agent-jobs
      consumer_name: pawn-agent-listener
      bucket_name: my-bucket
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_DEFAULT_DSN = "postgresql+psycopg://postgres:postgres@localhost:5432/pawnai"
_DEFAULT_SIYUAN_URL = "http://127.0.0.1:6806"
_DEFAULT_PATH_TEMPLATE = "/Conversations/{date}/{session_id}/{title}"
_DEFAULT_DAILY_TEMPLATE = "/daily note/{year}/{month}/{date}"

# Provider key → PydanticAI model string prefix mapping.
# "openai" is also used for any OpenAI-compatible endpoint (Ollama, Azure, etc.)
_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-gla",
    "groq": "groq",
    "mistral": "mistral",
}


@dataclass
class AgentConfig:
    # Database
    db_dsn: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", _DEFAULT_DSN))

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

    # Primary agent (PydanticAI) — derived from the provider sub-section in YAML.
    # Format: "{provider_prefix}:{model_name}", e.g. "openai:gpt-4o"
    pydantic_model: str = "openai:gpt-4o"
    pydantic_api_key: Optional[str] = None
    # Custom base URL for OpenAI-compatible providers (Ollama, Azure, etc.)
    pydantic_base_url: Optional[str] = None

    # Copilot SDK sub-agent (used by analyze_custom tool for GitHub-billed calls)
    model: str = "claude-sonnet-4.6"
    backend: str = "copilot"
    openai_base_url: Optional[str] = None
    openai_api_key: str = "ollama"

    # RAG / text embedding
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_device: str = "cpu"

    # S3 and queue configuration (populated from pawnai.yaml s3:/queue: sections)
    s3_config: Optional[Dict[str, Any]] = None
    queue_config: Optional[Dict[str, Any]] = None

    # HTTP API server (pawn-agent serve)
    api_token: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_model_idle_timeout_minutes: float = 10.0

    # Optional MLflow tracing for PydanticAI runs.
    mlflow_enabled: bool = field(
        default_factory=lambda: os.environ.get("PAWN_AGENT_MLFLOW_ENABLED", "").lower()
        in {"1", "true", "yes", "on"}
    )
    mlflow_tracking_uri: Optional[str] = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI")
    )
    mlflow_experiment: Optional[str] = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "pawn-agent")
    )


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
        if "name" in agent_sec:
            cfg.agent_name = agent_sec["name"]
        if "anima" in agent_sec:
            cfg.anima_path = agent_sec["anima"]

        # Primary agent: detect provider sub-section (first match wins)
        for provider_key, prefix in _PROVIDER_PREFIX.items():
            if provider_key in agent_sec:
                p = agent_sec[provider_key] or {}
                if "model" in p:
                    cfg.pydantic_model = f"{prefix}:{p['model']}"
                cfg.pydantic_api_key = p.get("api_key")
                cfg.pydantic_base_url = p.get("base_url")
                break

        # Copilot sub-agent: nested under agent.copilot or flat (legacy)
        copilot_sec = agent_sec.get("copilot", {}) or {}
        if copilot_sec:
            if "model" in copilot_sec:
                cfg.model = copilot_sec["model"]
            if "backend" in copilot_sec:
                cfg.backend = copilot_sec["backend"]
            if "base_url" in copilot_sec:
                cfg.openai_base_url = copilot_sec["base_url"]
            if "api_key" in copilot_sec:
                cfg.openai_api_key = copilot_sec["api_key"]
        else:
            # Legacy flat keys under agent: for backward compat
            if "model" in agent_sec:
                cfg.model = agent_sec["model"]
            if "backend" in agent_sec:
                cfg.backend = agent_sec["backend"]
            if "openai_base_url" in agent_sec:
                cfg.openai_base_url = agent_sec["openai_base_url"]
            if "openai_api_key" in agent_sec:
                cfg.openai_api_key = agent_sec["openai_api_key"]

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

        rag = data.get("rag", {}) or {}
        if "embed_model" in rag:
            cfg.embed_model = rag["embed_model"]
        if "embed_dim" in rag:
            cfg.embed_dim = int(rag["embed_dim"])
        if "embed_device" in rag:
            cfg.embed_device = rag["embed_device"]

        if "s3" in data:
            cfg.s3_config = data["s3"]
        if "agent_queue" in data:
            cfg.queue_config = data["agent_queue"]

        api_sec = data.get("api", {}) or {}
        if "token" in api_sec:
            cfg.api_token = api_sec["token"]
        if "host" in api_sec:
            cfg.api_host = api_sec["host"]
        if "port" in api_sec:
            cfg.api_port = int(api_sec["port"])
        if "model_idle_timeout_minutes" in api_sec:
            cfg.api_model_idle_timeout_minutes = float(api_sec["model_idle_timeout_minutes"])

        obs = data.get("observability", {}) or {}
        mlflow_sec = obs.get("mlflow", data.get("mlflow", {})) or {}
        if "enabled" in mlflow_sec:
            cfg.mlflow_enabled = bool(mlflow_sec["enabled"])
        if "tracking_uri" in mlflow_sec:
            cfg.mlflow_tracking_uri = mlflow_sec["tracking_uri"]
        if "experiment" in mlflow_sec:
            cfg.mlflow_experiment = mlflow_sec["experiment"]

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
