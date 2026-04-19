"""Configuration loader for pawn-agent.

Delegates to :class:`pawn_core.config.PawnConfig` (pydantic-settings) and adds
agent-specific sections.  All names exported here keep the same signatures they
had before so no import sites in tools/, core/, or tests need to change.

Config file schema (all keys optional)::

    db_dsn: postgresql+psycopg://postgres:postgres@localhost:5432/pawn_diarize

    models:
      transcription_model: nvidia/parakeet-tdt-0.6b-v3
      transcription_backend: nemo  # nemo | whisper

    agent:
      name: Bob
      anima: anima.md

      openai:
        model: gpt-4o
        api_key: sk-...
        base_url: http://localhost:11434/v1

      copilot:
        model: gpt-4.1

    siyuan:
      url: http://127.0.0.1:6806
      token: ""
      notebook: ""

    rag:
      embed_model: Qwen/Qwen3-Embedding-0.6B
      embed_dim: 1024
      embed_device: cpu

    api:
      token: ""
      host: 0.0.0.0
      port: 8000
      model_idle_timeout_minutes: 10.0

    agent_queue:
      topic: pawn-agent-jobs
      consumer_name: pawn-agent-listener
      bucket_name: my-bucket
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_settings import SettingsConfigDict
import yaml

from pawn_core.config import (  # noqa: F401
    LoggingConfig,
    PawnConfig,
    RagConfig,
    S3Config,
    SiYuanConfig,
)


# ── Agent-specific section models ─────────────────────────────────────────────


class AgentProviderConfig(BaseModel):
    """LLM provider settings (one per provider key under ``agent:``)."""

    model: str = "gpt-4o"
    fast_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class CopilotConfig(BaseModel):
    """GitHub Copilot SDK sub-agent settings."""

    model: str = "claude-sonnet-4.6"
    backend: str = "copilot"
    base_url: Optional[str] = None
    api_key: str = "ollama"


class AgentSection(BaseModel):
    """Top-level ``agent:`` section."""

    name: str = "Bob"
    anima: Optional[str] = None
    strip_thinking: bool = True
    openai: Optional[AgentProviderConfig] = None
    anthropic: Optional[AgentProviderConfig] = None
    google: Optional[AgentProviderConfig] = None
    groq: Optional[AgentProviderConfig] = None
    mistral: Optional[AgentProviderConfig] = None
    copilot: CopilotConfig = Field(default_factory=CopilotConfig)


class ApiSection(BaseModel):
    """``api:`` section — HTTP server settings."""

    token: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000
    model_idle_timeout_minutes: float = 10.0


class MlflowSection(BaseModel):
    """``mlflow:`` section (top-level in pawnai.yaml)."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "PAWN_MLFLOW__ENABLED", "PAWN_AGENT_MLFLOW_ENABLED"
        ),
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "PAWN_MLFLOW__TRACKING_URI", "MLFLOW_TRACKING_URI"
        ),
    )
    experiment: str = Field(
        default="pawn-agent",
        validation_alias=AliasChoices(
            "PAWN_MLFLOW__EXPERIMENT", "MLFLOW_EXPERIMENT_NAME"
        ),
    )


class AgentQueueConfig(BaseModel):
    """``agent_queue:`` section — pawn-queue listener for pawn-agent jobs."""

    topic: str = "pawn-agent-jobs"
    consumer_name: str = "pawn-agent-listener"
    bucket_name: str = "my-bucket"


class BurrSection(BaseModel):
    """``burr:`` section — Burr chat evaluation settings."""

    enabled: bool = False
    project: str = "pawn-agent"
    backend: Literal["local", "postgres"] = "local"
    storage_dir: Optional[str] = None
    table_name: str = "burr_state"


class PhoenixSection(BaseModel):
    """``phoenix:`` section — optional Phoenix observability settings."""

    enabled: bool = False
    endpoint: str = "http://localhost:6006/v1/traces"
    project_name: str = "parakeet-langgraph"
    protocol: Literal["grpc", "http/protobuf"] = "http/protobuf"
    api_key: Optional[str] = None


# ── AgentConfig ───────────────────────────────────────────────────────────────

_PROVIDER_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-gla",
    "groq": "groq",
    "mistral": "mistral",
}


class AgentConfig(PawnConfig):
    """Full configuration for the pawn-agent application.

    Inherits all shared sections (models, device, s3, siyuan, rag, db_dsn)
    from :class:`pawn_core.config.PawnConfig` and adds agent-specific ones.

    Flat property aliases preserve the old ``cfg.api_token``, ``cfg.embed_model``,
    ``cfg.siyuan_token`` etc. so all existing tools, core modules, and tests
    continue to work without changes.
    """

    model_config = SettingsConfigDict(
        yaml_file=["pawnai.yaml", "pawnai.yml", ".pawn-diarize.yml", ".pawn-diarize.yaml"],
        yaml_file_encoding="utf-8",
        env_prefix="PAWN_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    _model_override: Optional[str] = PrivateAttr(default=None)

    agent: AgentSection = Field(default_factory=AgentSection)
    api: ApiSection = Field(default_factory=ApiSection)
    mlflow: MlflowSection = Field(default_factory=MlflowSection)
    burr: BurrSection = Field(default_factory=BurrSection)
    phoenix: PhoenixSection = Field(default_factory=PhoenixSection)
    agent_queue: Optional[AgentQueueConfig] = None

    # ── Flat property aliases (old flat-field names used throughout pawn_agent) ─

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def anima_path(self) -> Optional[str]:
        return self.agent.anima

    @property
    def strip_thinking(self) -> bool:
        return self.agent.strip_thinking

    @property
    def api_token(self) -> Optional[str]:
        return self.api.token

    @property
    def api_host(self) -> str:
        return self.api.host

    @property
    def api_port(self) -> int:
        return self.api.port

    @property
    def api_model_idle_timeout_minutes(self) -> float:
        return self.api.model_idle_timeout_minutes

    @property
    def embed_model(self) -> str:
        return self.rag.embed_model

    @property
    def embed_dim(self) -> int:
        return self.rag.embed_dim

    @property
    def embed_device(self) -> str:
        return self.rag.embed_device

    @property
    def embed_local_files_only(self) -> bool:
        return self.rag.embed_local_files_only

    @property
    def mlflow_enabled(self) -> bool:
        return self.mlflow.enabled

    @property
    def mlflow_tracking_uri(self) -> Optional[str]:
        return self.mlflow.tracking_uri

    @property
    def mlflow_experiment(self) -> Optional[str]:
        return self.mlflow.experiment

    @property
    def phoenix_enabled(self) -> bool:
        return self.phoenix.enabled

    @property
    def phoenix_endpoint(self) -> str:
        return self.phoenix.endpoint

    @property
    def phoenix_project_name(self) -> str:
        return self.phoenix.project_name

    @property
    def phoenix_protocol(self) -> str:
        return self.phoenix.protocol

    @property
    def phoenix_api_key(self) -> Optional[str]:
        return self.phoenix.api_key

    # Transcription (consumed by api_server.py transcription endpoint)
    @property
    def transcription_model(self) -> str:
        return self.models.transcription_model

    @property
    def transcription_backend(self) -> str:
        return self.models.transcription_backend

    @property
    def transcription_device(self) -> str:
        return self.device.resolved

    # TTS (consumed by api_server.py speech endpoint)
    @property
    def tts_language(self) -> str:
        return self.models.tts_language

    @property
    def tts_voice(self) -> str:
        return self.models.tts_voice

    @property
    def tts_device(self) -> str:
        return self.models.tts_device or self.device.resolved

    @property
    def tts_idle_timeout_minutes(self) -> float:
        return self.models.tts_idle_timeout_minutes

    # Primary PydanticAI provider — resolved from first non-None provider section
    @property
    def pydantic_model(self) -> str:
        if self._model_override is not None:
            return self._model_override
        for provider, prefix in _PROVIDER_PREFIXES.items():
            p = getattr(self.agent, provider)
            if p is not None:
                return f"{prefix}:{p.model}"
        return "openai:gpt-4o"

    @pydantic_model.setter
    def pydantic_model(self, value: str) -> None:
        self._model_override = value

    @property
    def pydantic_api_key(self) -> Optional[str]:
        for provider in _PROVIDER_PREFIXES:
            p = getattr(self.agent, provider)
            if p is not None:
                return p.api_key
        return None

    @property
    def pydantic_base_url(self) -> Optional[str]:
        for provider in _PROVIDER_PREFIXES:
            p = getattr(self.agent, provider)
            if p is not None:
                return p.base_url
        return None

    @property
    def langgraph_fast_model(self) -> str:
        for provider, prefix in _PROVIDER_PREFIXES.items():
            p = getattr(self.agent, provider)
            if p is not None and p.fast_model:
                return f"{prefix}:{p.fast_model}"
        return self.pydantic_model

    @property
    def langgraph_deep_model(self) -> str:
        return self.pydantic_model

    @property
    def langgraph_api_key(self) -> Optional[str]:
        return self.pydantic_api_key

    @property
    def langgraph_base_url(self) -> Optional[str]:
        return self.pydantic_base_url

    # Copilot sub-agent flat attrs
    @property
    def model(self) -> str:
        return self.agent.copilot.model

    @property
    def backend(self) -> str:
        return self.agent.copilot.backend

    @property
    def openai_base_url(self) -> Optional[str]:
        return self.agent.copilot.base_url

    @property
    def openai_api_key(self) -> str:
        return self.agent.copilot.api_key

    # S3 / queue as raw dicts (used by queue_listener)
    @property
    def s3_config(self) -> Optional[dict]:
        return self.s3.model_dump() if self.s3 else None

    @property
    def queue_config(self) -> Optional[dict]:
        return self.agent_queue.model_dump() if self.agent_queue else None

    # SiYuan flat attrs
    @property
    def siyuan_url(self) -> str:
        return self.siyuan.url

    @property
    def siyuan_token(self) -> str:
        return self.siyuan.token

    @property
    def siyuan_notebook(self) -> str:
        return self.siyuan.notebook

    @property
    def siyuan_path_template(self) -> str:
        return self.siyuan.path_template

    @property
    def siyuan_daily_template(self) -> str:
        return self.siyuan.daily_note_path


# ── Public factory (keeps load_config() signature unchanged) ──────────────────


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """Load :class:`AgentConfig` from YAML + environment variables.

    Priority (highest → lowest):
    1. ``config_path`` argument
    2. ``PAWN_AGENT_CONFIG`` environment variable
    3. ``pawnai.yaml`` / ``pawnai.yml`` / ``.pawn-diarize.yml`` in cwd
    4. Dataclass defaults / environment variables
    """
    yaml_file = config_path or os.environ.get("PAWN_AGENT_CONFIG")
    cfg: AgentConfig
    if yaml_file and Path(yaml_file).exists():
        raw = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8")) or {}
        cfg = AgentConfig(**raw)
    else:
        cfg = AgentConfig()
    logging.basicConfig(
        level=cfg.logging.level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return cfg
