"""Core functionality for speaker diarization and transcription.

Heavy ML modules (DiarizationEngine, TranscriptionEngine, etc.) are loaded
lazily via ``__getattr__`` so that lightweight commands (e.g. ``s3-ls``,
``status``) do not trigger NeMo / pyannote import-time side-effects.
"""

from __future__ import annotations

from .s3 import S3Client, S3Config, is_s3_path, s3_audio_paths, expand_s3_glob
from .siyuan import SiyuanClient, SiyuanError, format_session_markdown, resolve_path_template

# Names exported through lazy __getattr__ below
_LAZY = {
    "DiarizationEngine": ".diarization",
    "TranscriptionEngine": ".transcription",
    "EmbeddingManager": ".embeddings",
    "transcribe_with_diarization": ".combined",
    "format_transcript_with_speakers": ".combined",
    "AnalysisEngine": ".analysis",
}

__all__ = [
    "DiarizationEngine",
    "TranscriptionEngine",
    "EmbeddingManager",
    "AnalysisEngine",
    "transcribe_with_diarization",
    "format_transcript_with_speakers",
    "S3Client",
    "S3Config",
    "is_s3_path",
    "s3_audio_paths",
    "expand_s3_glob",
    "SiyuanClient",
    "SiyuanError",
    "format_session_markdown",
    "resolve_path_template",
]


def __getattr__(name: str):  # noqa: ANN001, ANN201
    """Lazily import heavy ML modules on first access."""
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], package=__name__)
        obj = getattr(module, name)
        # Cache in module globals to avoid re-importing on subsequent accesses
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

