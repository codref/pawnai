"""Text-to-speech engine using Kokoro (hexgrad/Kokoro-82M).

Each language requires its own :class:`KPipeline` instance (``lang_code`` is
set at construction time).  Pipelines are loaded lazily per language and cached
for the lifetime of the engine.  All cached pipelines are released together via
:meth:`TTSEngine.unload`.

Language IDs use BCP-47 codes (``"en"``, ``"it"``, ``"fr"`` …) and are mapped
to Kokoro single-letter codes internally.  Voices can be OpenAI names
(``"alloy"``, ``"echo"`` …) or native Kokoro IDs (``"af_heart"``, ``"am_echo"``
…).

Typical usage::

    engine = TTSEngine(device="cpu", language_id="en", voice="af_heart")
    wav_bytes = engine.synthesize("Hello world.", speed=1.0)
    # ... later, when idle ...
    engine.unload()
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 24000  # Kokoro native output rate

# BCP-47 / common aliases → Kokoro lang_code
# Authoritative set from kokoro.pipeline.LANG_CODES + ALIASES
_LANG_MAP: Dict[str, str] = {
    "en":    "a",   # default English → American
    "en-us": "a",
    "en-gb": "b",
    "es":    "e",
    "fr":    "f",
    "fr-fr": "f",
    "hi":    "h",
    "it":    "i",
    "pt":    "p",
    "pt-br": "p",
    "ja":    "j",   # requires: pip install misaki[ja]
    "zh":    "z",   # requires: pip install misaki[zh]
}

# OpenAI voice names → Kokoro voice IDs (English defaults)
_VOICE_MAP: Dict[str, str] = {
    "alloy":   "af_heart",
    "echo":    "am_echo",
    "fable":   "bm_george",
    "onyx":    "am_onyx",
    "nova":    "af_bella",
    "shimmer": "af_sarah",
}


class TTSEngine:
    """Wraps Kokoro :class:`KPipeline` for text-to-speech synthesis.

    One pipeline per language is loaded lazily and cached.  Calling
    :meth:`unload` releases all cached pipelines; the next :meth:`synthesize`
    will reload as needed.

    Parameters
    ----------
    device:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
        Kokoro requires PyTorch with matching CUDA capability — use ``"cpu"``
        if CUDA is unavailable or unsupported.
    language_id:
        Default BCP-47 language code, e.g. ``"en"``, ``"it"``, ``"fr"``.
    voice:
        Default voice ID — either an OpenAI name (``"alloy"``) or a native
        Kokoro ID (``"af_heart"``).
    """

    def __init__(
        self,
        device: str = "cpu",
        language_id: str = "en",
        voice: str = "af_heart",
    ) -> None:
        self.device = device
        self.language_id = language_id
        self.voice = voice
        self._pipelines: Dict[str, object] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return _DEFAULT_SAMPLE_RATE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_lang(self, language_id: str) -> str:
        """Map a BCP-47 code to a Kokoro lang_code; pass through if unknown."""
        return _LANG_MAP.get(language_id.lower(), language_id)

    def _resolve_voice(self, voice: str) -> str:
        """Map an OpenAI voice name to a Kokoro ID; pass through if already native."""
        return _VOICE_MAP.get(voice, voice)

    def _get_pipeline(self, lang_code: str) -> object:
        """Return a cached :class:`KPipeline` for *lang_code*, creating it if needed."""
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        with self._lock:
            if lang_code not in self._pipelines:
                from kokoro import KPipeline  # noqa: PLC0415

                logger.info(
                    "Loading Kokoro pipeline lang_code=%s on %s", lang_code, self.device
                )
                self._pipelines[lang_code] = KPipeline(
                    lang_code=lang_code, device=self.device
                )
                logger.info("Kokoro pipeline lang_code=%s ready", lang_code)
            return self._pipelines[lang_code]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        language_id: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> bytes:
        """Convert *text* to speech and return raw WAV bytes (24 000 Hz, mono).

        Parameters
        ----------
        text:
            The input text to synthesise.
        speed:
            Speaking rate multiplier (0.25 – 4.0), passed directly to Kokoro.
        language_id:
            BCP-47 language code.  Falls back to the engine's default when
            ``None``.
        voice:
            Voice identifier — OpenAI name or native Kokoro ID.  Falls back to
            the engine's default when ``None``.

        Returns
        -------
        bytes
            Uncompressed PCM WAV data suitable for returning directly or piping
            through ffmpeg for format conversion.
        """
        import soundfile as sf  # noqa: PLC0415

        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        lang_code = self._resolve_lang(language_id or self.language_id)
        voice_id = self._resolve_voice(voice or self.voice)
        pipeline = self._get_pipeline(lang_code)

        chunks = [
            result.audio
            for result in pipeline(text, voice=voice_id, speed=speed)  # type: ignore[operator]
            if result.audio is not None
        ]
        if not chunks:
            raise RuntimeError("Kokoro pipeline produced no audio output.")

        audio_np = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, audio_np, _DEFAULT_SAMPLE_RATE, format="WAV")
        return buf.getvalue()

    def unload(self) -> None:
        """Release all cached pipeline weights from GPU/CPU memory.

        Safe to call even if no pipelines have been loaded.  After this call
        the engine is fully functional — the next :meth:`synthesize` will
        reload the required pipeline.
        """
        with self._lock:
            if not self._pipelines:
                return
            count = len(self._pipelines)
            self._pipelines.clear()

        if "cuda" in self.device:
            try:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("TTS: unloaded %d Kokoro pipeline(s) from %s", count, self.device)
