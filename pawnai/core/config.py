"""Configuration management for PawnAI."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_config_from_file(config_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file.
    
    Args:
        config_path: Path to .env file. If None, uses default .env in current directory.
    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        load_dotenv(config_file)
    else:
        load_dotenv()


# Load default config on module import
load_config_from_file()

# Models and tokens
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Paths
DEFAULT_DB_PATH = "speakers_db"

# Device configuration
DEVICE_TYPE = os.getenv("DEVICE", "auto")  # auto, cuda, cpu

# Model configurations
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "pyannote/embedding"
TRANSCRIPTION_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class Config:
    """Application configuration."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize configuration.

        Args:
            db_path: Path to LanceDB database
        """
        self.db_path = Path(db_path)

    def ensure_paths_exist(self) -> None:
        """Create necessary directories if they don't exist."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
