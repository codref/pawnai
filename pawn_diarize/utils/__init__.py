"""Utility functions for the CLI application."""

from pathlib import Path
from typing import List, Optional


def find_audio_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Find audio files in a directory.
    
    Args:
        directory: Path to search
        extensions: File extensions to look for (default: .wav, .mp3, .flac)
    
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg"]
    
    directory_path = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory_path.glob(f"*{ext}"))
        audio_files.extend(directory_path.glob(f"**/*{ext}"))
    
    return [str(f) for f in sorted(set(audio_files))]


def validate_audio_file(file_path: str) -> bool:
    """Validate that a file exists and has an audio extension.
    
    Args:
        file_path: Path to check
    
    Returns:
        True if valid audio file, False otherwise
    """
    path = Path(file_path)
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return path.exists() and path.suffix.lower() in audio_extensions


__all__ = ["find_audio_files", "validate_audio_file"]
