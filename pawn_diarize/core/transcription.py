"""Backward-compatible re-export of TranscriptionEngine from pawn_core.

All callers within pawn_diarize (combined.py, queue_listener.py, cli/commands.py,
core/__init__.py lazy loader) import TranscriptionEngine from this module by name
and continue to work unchanged.
"""

from pawn_core.transcription import (  # noqa: F401
    TranscriptionEngine,
    _maybe_quiet,
    _NEMO_LOGGERS,
    _quiet_nemo,
)

__all__ = ["TranscriptionEngine", "_quiet_nemo", "_maybe_quiet", "_NEMO_LOGGERS"]
