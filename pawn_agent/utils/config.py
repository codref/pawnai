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

      # Durable sallm harness (SQLite + Lance memory). Chat model still comes
      # from the provider block above; this section owns memory paths + profile.
      sallm:
        state_dir: .sallm
        max_steps: 8
        # CompiledProfile YAML/JSON (budgets overlay). Default: packaged large
        # (10× token budgets). Set "" to use stock sallm ModelProfile limits.
        profile: large.yaml

      copilot:
        model: gpt-4.1

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

    diarize_queue:
      topic: audio-chunks
      consumer_name: pawn-diarize-listener
      bucket_name: my-bucket

    queue_producers:
      matrix:
        topic: matrix-jobs
        bucket_name: my-bucket

    matrix_bot:
      enabled: false
      homeserver_url: https://matrix.example.com
      user_id: "@pawn:example.com"
      user_token: "..."
      device_id: PAWNBOT01
      device_name: pawn-matrix
      store_path: .matrix-store
      command_prefix: "!pawn"
      inviters:
        - "@you:example.com"
      progress_updates: true
      progress_reactions: true
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_settings import SettingsConfigDict

from pawn_core.config import (  # noqa: F401
    LoggingConfig,
    PawnConfig,
    RagConfig,
    S3Config,
    VaultConfig,
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


class SallmSection(BaseModel):
    """``agent.sallm:`` — durable ReAct harness settings (not the chat model).

    Chat model / api_key / base_url still come from ``agent.openai`` (etc.).
    This section controls session memory files, the CompiledProfile overlay,
    and optional Tempo metrics.
    """

    # Directory for state.db + vectors/ (relative paths resolve from cwd).
    state_dir: str = ".sallm"
    max_steps: int = 8
    # sallm CompiledProfile path (YAML/JSON). Relative names resolve from cwd
    # then ``pawn_agent/profiles/``. Default ``large.yaml`` = 10× budgets.
    # Empty string disables the overlay (stock ModelProfile limits).
    profile: Optional[str] = "large.yaml"
    # Observability off by default for the server; CLI may enable.
    otlp_endpoint: Optional[str] = None
    metrics_port: int = 0
    # Embedding defaults match sallm (local Ollama). Override when needed.
    embedding_model: str = "ollama/qwen3-embedding:0.6b"
    embedding_api_base: str = "http://localhost:11434"


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
    sallm: SallmSection = Field(default_factory=SallmSection)
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
        validation_alias=AliasChoices("PAWN_MLFLOW__ENABLED", "PAWN_AGENT_MLFLOW_ENABLED"),
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("PAWN_MLFLOW__TRACKING_URI", "MLFLOW_TRACKING_URI"),
    )
    experiment: str = Field(
        default="pawn-agent",
        validation_alias=AliasChoices("PAWN_MLFLOW__EXPERIMENT", "MLFLOW_EXPERIMENT_NAME"),
    )


class AgentQueueConfig(BaseModel):
    """``agent_queue:`` section — pawn-queue listener for pawn-agent jobs."""

    model_config = ConfigDict(extra="allow")

    topic: str = "pawn-agent-jobs"
    consumer_name: str = "pawn-agent-listener"
    bucket_name: str = "my-bucket"


class AgentSchedulerConfig(BaseModel):
    """``agent_scheduler:`` section — durable pawn-agent run scheduler."""

    enabled: bool = True
    poll_interval_seconds: float = 30.0
    max_due_per_tick: int = 5
    default_timezone: str = "UTC"
    stale_fire_after_seconds: int = 3600
    allow_agent_auto_apply: bool = False


class QueueProducerConfig(BaseModel):
    """Single named producer target for the queue-publishing tool."""

    topic: str
    bucket_name: str
    producer_name: Optional[str] = None
    polling: Optional[dict] = None
    concurrency: Optional[dict] = None


class MatrixBotConfig(BaseModel):
    """``matrix_bot:`` section — inbound Matrix chatbot under ``pawn-server serve``."""

    enabled: bool = False
    homeserver_url: str = ""
    user_id: str = ""
    user_token: Optional[str] = None
    user_password: Optional[str] = None
    device_id: str = "PAWNBOT01"
    device_name: str = "pawn-matrix"
    store_path: str = ".matrix-store"
    command_prefix: str = "!pawn"
    inviters: list[str] = Field(default_factory=list)
    # Live status message edits while the agent ReAct loop runs.
    progress_updates: bool = True
    # Glanceable ⏳ / ✅ / ❌ reactions on the user's prompting message.
    progress_reactions: bool = True
    # Room for outbound ready-for-review alerts (Matrix notifier worker).
    notify_room_id: Optional[str] = None


class VaultWatcherConfig(BaseModel):
    """``vault_watcher:`` — poll vault task notes and enqueue agent runs."""

    enabled: bool = True
    poll_interval_seconds: float = 15.0
    max_claims_per_tick: int = 3
    matrix_target: str = "matrix"


# ── AgentConfig ───────────────────────────────────────────────────────────────

# PydanticAI-style prefixes (colon) → LiteLLM-style prefixes (slash).
_PROVIDER_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google-gla",
    "groq": "groq",
    "mistral": "mistral",
}

_LITELLM_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google-gla": "gemini",
    "google": "gemini",
    "groq": "groq",
    "mistral": "mistral",
}


class AgentConfig(PawnConfig):
    """Full configuration for the pawn-agent application.

    Inherits all shared sections (models, device, s3, rag, db_dsn)
    from :class:`pawn_core.config.PawnConfig` and adds agent-specific ones.

    Flat property aliases preserve the old ``cfg.api_token``, ``cfg.embed_model``,
    etc. so all existing tools, core modules, and tests continue to work.
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
    agent_scheduler: AgentSchedulerConfig = Field(default_factory=AgentSchedulerConfig)
    agent_queue: Optional[AgentQueueConfig] = None
    diarize_queue: Optional[AgentQueueConfig] = None
    queue_producers: Optional[dict[str, QueueProducerConfig]] = None
    matrix_bot: MatrixBotConfig = Field(default_factory=MatrixBotConfig)
    vault_watcher: VaultWatcherConfig = Field(default_factory=VaultWatcherConfig)

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
    def sallm(self) -> SallmSection:
        """Shortcut to ``agent.sallm`` memory / harness settings."""
        return self.agent.sallm

    @property
    def litellm_model(self) -> str:
        """Map ``openai:gpt-4o`` (PydanticAI) → ``openai/gpt-4o`` (LiteLLM/sallm)."""
        raw = self.pydantic_model
        if "/" in raw and ":" not in raw.split("/", 1)[0]:
            return raw
        if ":" not in raw:
            return f"openai/{raw}"
        prefix, rest = raw.split(":", 1)
        litellm_prefix = _LITELLM_PREFIXES.get(prefix, prefix)
        return f"{litellm_prefix}/{rest}"

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

    @property
    def vault_bucket(self) -> str:
        return self.vault.bucket

    @property
    def vault_agent_root(self) -> str:
        return self.vault.agent_root

    @property
    def vault_auto_push_transcript(self) -> bool:
        return self.vault.auto_push_transcript


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
        yaml_path = Path(yaml_file).resolve()
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

        class ExplicitAgentConfig(AgentConfig):
            model_config = {**AgentConfig.model_config, "yaml_file": [str(yaml_path)]}

        cfg = ExplicitAgentConfig(**raw)
    else:
        cfg = AgentConfig()
    logging.basicConfig(
        level=cfg.logging.level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return cfg
