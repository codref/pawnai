"""Tests for core diarization functionality."""

import pytest
from pawn_diarize.core.diarization import DiarizationEngine
from pawn_diarize.core.config import Config


def test_diarization_engine_initialization():
    """Test that DiarizationEngine initializes correctly."""
    engine = DiarizationEngine()
    assert engine.diarization_pipeline is None
    assert engine.embedding_model is None


def test_diarization_engine_device():
    """Test device selection."""
    # Test auto-detection
    engine_auto = DiarizationEngine(device="auto")
    assert engine_auto.device is not None
    
    # Test CPU device
    engine_cpu = DiarizationEngine(device="cpu")
    assert engine_cpu.device.type == "cpu"


def test_cluster_speakers_empty_list():
    """Test clustering with empty embeddings list."""
    engine = DiarizationEngine()
    result = engine.cluster_speakers([])
    assert result == []


def test_config_initialization():
    """Test Config class initialization."""
    config = Config(db_dsn="postgresql+psycopg://localhost/test_db")
    assert config.db_dsn == "postgresql+psycopg://localhost/test_db"
