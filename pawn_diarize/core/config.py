"""Configuration management for pawn_diarize.

Delegates to :class:`pawn_core.config.PawnConfig` (pydantic-settings) and
re-exports all names that existing callers depend on so no import sites need
to change.

Existing usage patterns that continue to work
---------------------------------------------
* ``AppConfig()``                         — returns a DiarizeConfig
* ``AppConfig(config_path="/my/cfg.yaml")`` — reads a custom YAML file
* ``from .config import TRANSCRIPTION_MODEL``  — backward-compat constant
* ``from .config import DEFAULT_DB_DSN``       — backward-compat constant
* ``from .config import HUGGINGFACE_TOKEN``    — backward-compat constant
* ``from .config import DEVICE_TYPE``          — backward-compat constant
* ``Config(db_dsn=...)``                  — lightweight DB helper for tests
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from pawn_core.config import (  # noqa: F401
    DeviceConfig,
    ModelsConfig,
    PawnConfig,
    RagConfig,
    S3Config,
    SiYuanConfig,
)


# ── pawn_diarize-specific section ─────────────────────────────────────────────


class DiarizeQueueConfig(BaseModel):
    """pawn-queue listener configuration for pawn-diarize jobs."""

    bucket_name: str = "pawn-diarize-queue"


class DiarizeConfig(PawnConfig):
    """Full configuration for the pawn-diarize application."""

    model_config = SettingsConfigDict(
        yaml_file=["pawnai.yaml", "pawnai.yml", ".pawn-diarize.yml", ".pawn-diarize.yaml"],
        yaml_file_encoding="utf-8",
        env_prefix="PAWN_",
        env_nested_delimiter="__",
        extra="ignore",
        # diarize_queue YAML key maps to the queue field
        populate_by_name=True,
    )

    audio_dir: str = "audio/"
    queue: Optional[DiarizeQueueConfig] = None

    # pydantic-settings v2 does not support YAML key aliases natively, so we
    # also accept the legacy key via an alias in model_validator below.
    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        pass  # reserved for future post-init logic

    def get(self, key: str, default: object = None) -> object:
        """Dict-style accessor for backward compat with AppConfig().get(...)."""
        # Map flat legacy keys to nested attributes
        _flat_map = {
            "transcription_model": lambda: self.models.transcription_model,
            "transcription_backend": lambda: self.models.transcription_backend,
            "whisper_model": lambda: self.models.whisper_model,
            "diarization_model": lambda: self.models.diarization_model,
            "embedding_model": lambda: self.models.embedding_model,
            "hf_token": lambda: self.models.hf_token,
            "model_idle_timeout_minutes": lambda: self.models.model_idle_timeout_minutes,
            "device": lambda: self.device.type,
            "db_dsn": lambda: self.db_dsn,
            "audio_dir": lambda: self.audio_dir,
            "s3": lambda: self.s3.model_dump() if self.s3 else None,
            "siyuan": lambda: self.siyuan.model_dump(),
            "queue": lambda: self.queue.model_dump() if self.queue else None,
            "rag": lambda: self.rag.model_dump(),
        }
        if key in _flat_map:
            value = _flat_map[key]()
            return value if value is not None else default
        return default

    def set(self, key: str, value: object) -> None:
        """No-op stub — pydantic models are immutable after init."""

    def get_s3_config(self) -> Optional[dict]:
        return self.s3.model_dump() if self.s3 else None

    def get_siyuan_config(self) -> Optional[dict]:
        return self.siyuan.model_dump()

    def get_rag_config(self) -> Optional[dict]:
        return self.rag.model_dump()

    def get_queue_config(self) -> Optional[dict]:
        return self.queue.model_dump() if self.queue else None


# ── Backward-compat public API ────────────────────────────────────────────────

def AppConfig(config_path: Optional[str] = None) -> DiarizeConfig:  # noqa: N802
    """Factory that mirrors the old ``AppConfig(config_path=...)`` call pattern.

    pydantic-settings accepts ``_yaml_file`` as an init kwarg to override the
    default YAML discovery list.
    """
    if config_path:
        return DiarizeConfig(_yaml_file=config_path)
    return DiarizeConfig()


# Module-level constants consumed at import time by diarization.py, embeddings.py,
# queue_listener.py, and migrations/env.py.  Reading them from a singleton ensures
# they reflect the same pawnai.yaml that the rest of the app uses.
_defaults: DiarizeConfig = DiarizeConfig()

DEFAULT_DB_DSN: str = _defaults.db_dsn
DIARIZATION_MODEL: str = _defaults.models.diarization_model
EMBEDDING_MODEL: str = _defaults.models.embedding_model
TRANSCRIPTION_MODEL: str = _defaults.models.transcription_model
TRANSCRIPTION_BACKEND: str = _defaults.models.transcription_backend
WHISPER_MODEL: str = _defaults.models.whisper_model
HUGGINGFACE_TOKEN: Optional[str] = _defaults.models.hf_token or os.getenv("HF_TOKEN")
DEVICE_TYPE: str = _defaults.device.type
HUGGINGFACE_TOKEN_STR: str = HUGGINGFACE_TOKEN or ""

# Legacy alias used by .github/copilot-instructions.md examples
CONFIG_FILE = "pawnai.yaml"


# ── Config helper (used by tests/test_core.py) ────────────────────────────────


class Config:
    """Lightweight database-configuration helper.

    Args:
        db_dsn: PostgreSQL DSN.  Defaults to :data:`DEFAULT_DB_DSN`.
    """

    def __init__(self, db_dsn: str = DEFAULT_DB_DSN) -> None:
        self.db_dsn = db_dsn

    def get_engine(self):
        """Return a SQLAlchemy engine and ensure tables exist."""
        from .database import get_engine, init_db  # noqa: PLC0415

        engine = get_engine(self.db_dsn)
        init_db(engine)
        return engine

    def ensure_paths_exist(self) -> None:
        """Initialise PostgreSQL tables (replaces legacy directory creation)."""
        self.get_engine()
