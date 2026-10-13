"""Whisper language lock plumbing on TranscriptionEngine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pawn_core.transcription import TranscriptionEngine


class _FakeWhisper:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def transcribe(self, audio_path: str, **kwargs: Any):  # noqa: ANN003
        self.kwargs = dict(kwargs)
        seg = SimpleNamespace(
            text="ciao",
            start=0.0,
            end=1.0,
            words=[SimpleNamespace(word="ciao", start=0.0, end=1.0)],
        )
        return iter([seg]), None


def test_whisper_passes_language_when_set() -> None:
    engine = TranscriptionEngine(device="cpu", backend="whisper", language="it")
    fake = _FakeWhisper()
    engine.whisper_model = fake
    result = engine._transcribe_whisper_single("x.wav", include_timestamps=True)
    assert result["text"] == "ciao"
    assert fake.kwargs is not None
    assert fake.kwargs["language"] == "it"
    assert fake.kwargs["word_timestamps"] is True


def test_whisper_omits_language_when_unset() -> None:
    engine = TranscriptionEngine(device="cpu", backend="whisper", language=None)
    fake = _FakeWhisper()
    engine.whisper_model = fake
    engine._transcribe_whisper_single("x.wav", include_timestamps=False)
    assert fake.kwargs is not None
    assert "language" not in fake.kwargs


def test_model_cache_updates_language_without_reload() -> None:
    from pawn_diarize.core.queue_listener import ModelCache

    cache = ModelCache()
    eng1 = cache.get_transcription_engine("cpu", "whisper", language="it")
    assert eng1.language == "it"
    eng2 = cache.get_transcription_engine("cpu", "whisper", language="en")
    assert eng2 is eng1
    assert eng2.language == "en"
    eng3 = cache.get_transcription_engine("cpu", "whisper", language=None)
    assert eng3.language is None
