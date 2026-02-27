"""Configuration management for PawnAI.

This module provides configuration constants and the :class:`AppConfig` class
which merges defaults with values from an optional YAML file
(``.pawnai.yml`` in the working directory or a path supplied at runtime).

Configuration file (``.pawnai.yml``)
--------------------------------------
All settings can be overridden at runtime via ``.pawnai.yml`` placed in the
project root (or by passing ``--config path/to/file.yml`` on the CLI):

.. code-block:: yaml

    models:
      hf_token: your_hf_token_here
      diarization_model: pyannote/speaker-diarization-community-1
      embedding_model: pyannote/embedding
      transcription_model: nvidia/parakeet-tdt-0.6b-v3

    paths:
      db_path: speakers_db
      audio_dir: audio/

    device:
      type: auto   # auto | cuda | cpu

    s3:
      bucket: my-audio-bucket
      endpoint_url: https://s3.amazonaws.com
      access_key: AKIAIOSFODNN7EXAMPLE
      secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
      region: us-east-1      # optional
      prefix: ""             # optional object key prefix
      verify_ssl: true       # optional, default true
      path_style: true       # optional, default true

If ``models.hf_token`` is absent the ``HF_TOKEN`` environment variable is
used as a fallback so that CI/CD pipelines that inject secrets via env vars
continue to work without a config file.

When the ``s3:`` section is present any audio path starting with ``s3://``
is transparently downloaded before processing.  See :mod:`pawnai.core.s3`
for the supported URI formats.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Module-level defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "speakers_db"  # kept for backward compat; prefer DEFAULT_DB_DSN
DEFAULT_DB_DSN: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/pawnai",
)

# Model identifiers
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "pyannote/embedding"
TRANSCRIPTION_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

CONFIG_FILE = ".pawnai.yml"


# ──────────────────────────────────────────────────────────────────────────────
# AppConfig
# ──────────────────────────────────────────────────────────────────────────────

class AppConfig:
    """Application configuration management.

    Merges hard-coded defaults with optional settings from a YAML file.
    When a ``hf_token`` value is found in the YAML it is also written into
    ``os.environ["HF_TOKEN"]`` so that downstream code that reads the
    environment variable directly continues to work.

    Args:
        config_path: Path to a YAML config file.  When *None* the default
            ``.pawnai.yml`` in the current working directory is used (if it
            exists).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config: Dict[str, Any] = {
            # models
            "hf_token": None,
            "diarization_model": DIARIZATION_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "transcription_model": TRANSCRIPTION_MODEL,
            # paths
            "db_path": DEFAULT_DB_PATH,   # legacy – prefer db_dsn
            "db_dsn": DEFAULT_DB_DSN,
            "audio_dir": "audio/",
            # device
            "device": "auto",
        }
        self._load_yaml_config(config_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _load_yaml_config(self, config_path: Optional[str] = None) -> None:
        """Load optional YAML configuration and merge it into ``_config``."""
        if config_path:
            yaml_file = Path(config_path)
            if not yaml_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            yaml_file = Path.cwd() / CONFIG_FILE
            if not yaml_file.exists():
                return

        content = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not content:
            return
        if not isinstance(content, dict):
            raise ValueError(
                f"Configuration in {yaml_file} must be a YAML mapping (dict)."
            )

        # models: section
        models = content.get("models", {})
        if isinstance(models, dict):
            for key in (
                "hf_token",
                "diarization_model",
                "embedding_model",
                "transcription_model",
            ):
                if key in models:
                    self._config[key] = models[key]

        # paths: section
        paths = content.get("paths", {})
        if isinstance(paths, dict):
            for key in ("db_path", "db_dsn", "audio_dir"):
                if key in paths:
                    self._config[key] = paths[key]

        # device: section
        device = content.get("device", {})
        if isinstance(device, dict) and "type" in device:
            self._config["device"] = device["type"]

        # s3: section – stored as a raw dict; validated on first use
        s3 = content.get("s3")
        if isinstance(s3, dict):
            self._config["s3"] = s3

        # Propagate HF token to env var so any code using os.getenv("HF_TOKEN")
        # picks it up (including module-level constants evaluated after this call).
        hf_token = self._config.get("hf_token")
        if hf_token:
            os.environ["HF_TOKEN"] = str(hf_token)

    # ── public ───────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not set.

        Args:
            key: Configuration key.
            default: Fallback value.

        Returns:
            Resolved configuration value.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Override a configuration value at runtime.

        Args:
            key: Configuration key.
            value: New value.
        """
        self._config[key] = value

    def get_s3_config(self) -> Optional[Dict[str, Any]]:
        """Return the ``s3:`` configuration mapping, or ``None`` if absent.

        Returns:
            Dictionary of S3 settings from ``.pawnai.yml``, or ``None`` when
            the ``s3:`` section is not present.
        """
        s3_config = self._config.get("s3")
        if isinstance(s3_config, dict):
            return s3_config
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Module-level default instance
# Reads .pawnai.yml from cwd at first import; also seeds os.environ["HF_TOKEN"]
# so that HUGGINGFACE_TOKEN below is populated when the YAML is present.
# ──────────────────────────────────────────────────────────────────────────────

_default_config = AppConfig()

# Back-compat constants used by other core modules (e.g. diarization.py).
# HF token: prefer YAML value; fall back to bare env var (useful in CI/CD).
HUGGINGFACE_TOKEN: Optional[str] = (
    _default_config.get("hf_token") or os.getenv("HF_TOKEN")
)

DEVICE_TYPE: str = _default_config.get("device", "auto")


# ──────────────────────────────────────────────────────────────────────────────
# Config (path helper used by CLI commands)
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """Lightweight database-configuration helper.

    Args:
        db_dsn: PostgreSQL DSN used to connect to the speaker database.
            Defaults to :data:`DEFAULT_DB_DSN` (respects the ``DATABASE_URL``
            environment variable).
    """

    def __init__(self, db_dsn: str = DEFAULT_DB_DSN) -> None:
        self.db_dsn = db_dsn

    def get_engine(self):
        """Return a SQLAlchemy engine and ensure tables exist.

        Returns:
            :class:`sqlalchemy.engine.Engine` connected to the configured DB.
        """
        from .database import get_engine, init_db

        engine = get_engine(self.db_dsn)
        init_db(engine)
        return engine

    def ensure_paths_exist(self) -> None:
        """Initialise PostgreSQL tables (replaces legacy directory creation)."""
        self.get_engine()
