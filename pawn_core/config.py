"""Shared configuration base for pawn_diarize and pawn_agent.

Both apps subclass :class:`PawnConfig` and add their own fields.
All common sections (models, device, s3, siyuan, rag) are defined here so the
same ``pawnai.yaml`` drives both without duplicated parsing code.

Source priority (highest → lowest):
    1. Constructor kwargs (``AgentConfig(db_dsn=...)`` or ``DiarizeConfig(...)``)
    2. Environment variables (``PAWN_DB_DSN``, ``DATABASE_URL``,
       ``PAWN_MODELS__HF_TOKEN``, ``HF_TOKEN``, …)
    3. ``pawnai.yaml`` / ``pawnai.yml`` / ``.pawn-diarize.yml`` in cwd
    4. Dataclass defaults

Environment variable convention
--------------------------------
The prefix ``PAWN_`` plus double-underscore nesting maps to sections::

    PAWN_DB_DSN                   → db_dsn
    PAWN_MODELS__HF_TOKEN         → models.hf_token
    PAWN_MODELS__TRANSCRIPTION_BACKEND → models.transcription_backend
    PAWN_DEVICE__TYPE             → device.type
    DATABASE_URL                  → db_dsn  (legacy alias)
    HF_TOKEN                      → models.hf_token  (legacy alias, via validator)
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


# ── Section models ─────────────────────────────────────────────────────────────


class ModelsConfig(BaseModel):
    """ML model identifiers and related settings."""

    transcription_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    transcription_backend: Literal["nemo", "whisper"] = "nemo"
    whisper_model: str = "large-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    embedding_model: str = "pyannote/embedding"
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[str] = None  # override HF_HUB_CACHE; applies to all HF model downloads
    model_idle_timeout_minutes: float = 10.0
    tts_language: str = "en"           # BCP-47 code, e.g. "en", "it", "fr"
    tts_voice: str = "af_heart"        # Kokoro voice ID or OpenAI alias
    tts_device: Optional[str] = None  # None → fall back to global device.type; use "cpu" if CUDA unavailable
    tts_idle_timeout_minutes: float = 10.0

    @model_validator(mode="after")
    def _propagate_hf_settings(self) -> "ModelsConfig":
        """Mirror hf_token/hf_cache_dir → environment variables read by HF Hub."""
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        elif not self.hf_token:
            # Fall back to env var so the field reflects what pyannote will use
            env_token = os.environ.get("HF_TOKEN")
            if env_token:
                object.__setattr__(self, "hf_token", env_token)
        if self.hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = self.hf_cache_dir
        return self


class DeviceConfig(BaseModel):
    """Compute device selection."""

    type: Literal["auto", "cuda", "cpu"] = "auto"

    @property
    def resolved(self) -> str:
        """Return the concrete device string, resolving ``"auto"`` via torch."""
        if self.type != "auto":
            return self.type
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


class S3Config(BaseModel):
    """S3 / MinIO connection settings."""

    bucket: str
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    prefix: str = ""
    verify_ssl: bool = True
    path_style: bool = True


class SiYuanConfig(BaseModel):
    """SiYuan Notes API settings."""

    url: str = "http://127.0.0.1:6806"
    token: str = ""
    notebook: str = ""
    path_template: str = "/Conversations/{date}/{session_id}/{title}"
    daily_note_path: str = "/daily note/{year}/{month}/{date}"


class RagConfig(BaseModel):
    """Text embedding / RAG pipeline settings."""

    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_device: str = "cpu"
    embed_local_files_only: bool = True


class LoggingConfig(BaseModel):
    """Console log verbosity settings.

    Set ``level`` to one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """

    level: str = "WARNING"


# ── Shared base ────────────────────────────────────────────────────────────────

_YAML_CANDIDATES = [
    "pawnai.yaml",
    "pawnai.yml",
    ".pawn-diarize.yml",
    ".pawn-diarize.yaml",
]


class PawnConfig(BaseSettings):
    """Base settings shared by pawn_diarize and pawn_agent.

    Subclass this and add app-specific fields::

        class AgentConfig(PawnConfig):
            agent: AgentSection = AgentSection()
    """

    model_config = SettingsConfigDict(
        yaml_file=_YAML_CANDIDATES,
        yaml_file_encoding="utf-8",
        env_prefix="PAWN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    db_dsn: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5433/pawnai",
        validation_alias=AliasChoices("PAWN_DB_DSN", "DATABASE_URL", "db_dsn"),
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    s3: Optional[S3Config] = None
    siyuan: SiYuanConfig = Field(default_factory=SiYuanConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Priority: init kwargs > env vars > pawnai.yaml > defaults."""
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )
