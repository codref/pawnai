"""Tests for utility functions."""

import pytest
from pawnai.utils import find_audio_files, validate_audio_file
from pathlib import Path


def test_validate_audio_file_wav(tmp_path):
    """Test validating a WAV file."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("dummy")
    assert validate_audio_file(str(audio_file)) is True


def test_validate_audio_file_nonexistent():
    """Test validating a non-existent file."""
    assert validate_audio_file("/nonexistent/file.wav") is False


def test_validate_audio_file_wrong_extension(tmp_path):
    """Test validating a file with wrong extension."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("dummy")
    assert validate_audio_file(str(text_file)) is False


def test_find_audio_files(tmp_path):
    """Test finding audio files in a directory."""
    # Create some test files
    (tmp_path / "audio1.wav").write_text("dummy")
    (tmp_path / "audio2.mp3").write_text("dummy")
    (tmp_path / "not_audio.txt").write_text("dummy")
    
    files = find_audio_files(str(tmp_path))
    assert len(files) == 2
    assert any("audio1.wav" in f for f in files)
    assert any("audio2.mp3" in f for f in files)
    assert not any(".txt" in f for f in files)


def test_find_audio_files_empty_directory(tmp_path):
    """Test finding audio files in empty directory."""
    files = find_audio_files(str(tmp_path))
    assert files == []
