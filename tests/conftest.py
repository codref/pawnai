"""Test configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"dummy_audio_data")
    return str(audio_file)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test_db"
    db_path.mkdir()
    return str(db_path)
