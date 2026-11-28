"""Test configuration and shared fixtures."""

from pathlib import Path

import pytest


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


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Process-level caches (planner event notes, session audio files) must not leak across tests."""
    yield
    try:
        from pawn_server.core import planner, sessions_api

        planner.clear_event_cache()
        sessions_api.invalidate_session_files()
    except Exception:  # pragma: no cover - optional server deps
        pass
