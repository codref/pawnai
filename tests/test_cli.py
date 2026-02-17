"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner
from pawnai.cli.commands import app


runner = CliRunner()


def test_status_command():
    """Test the status command."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "PawnAI Status" in result.stdout or "PawnAI" in result.stdout


def test_help_command():
    """Test that help is available."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PawnAI" in result.stdout


def test_diarize_missing_file():
    """Test diarize with missing audio file."""
    result = runner.invoke(app, ["diarize", "nonexistent.wav"])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_transcribe_missing_file():
    """Test transcribe with missing audio file."""
    result = runner.invoke(app, ["transcribe", "nonexistent.wav"])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_embed_missing_speaker_id():
    """Test embed command without required speaker ID."""
    result = runner.invoke(app, ["embed", "test.wav"])
    assert result.exit_code != 0
