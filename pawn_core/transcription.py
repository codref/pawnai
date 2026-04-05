"""Transcription engine using Nvidia Parakeet (NeMo) or faster-whisper.

Moved here from pawn_diarize/core/transcription.py so both pawn_diarize and
pawn_agent can import it without duplication.  pawn_diarize re-exports this
module for backward compatibility.
"""

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

_NEMO_LOGGERS = (
    "nemo_logger",
    "nemo",
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "torch",
)

_DEFAULT_TRANSCRIPTION_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
_DEFAULT_TRANSCRIPTION_BACKEND = "nemo"
_DEFAULT_WHISPER_MODEL = "large-v3"


@contextlib.contextmanager  # type: ignore[misc]
def _quiet_nemo():  # type: ignore[return]
    """Suppress NeMo noise: redirect stdout/stderr to devnull and set loggers to ERROR."""
    import logging
    for name in _NEMO_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()
        # Re-apply: NeMo reconfigures its loggers during import/init
        for name in _NEMO_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager  # type: ignore[misc]
def _maybe_quiet(verbose: bool):  # type: ignore[return]
    """Enter _quiet_nemo() only when verbose is False."""
    if verbose:
        yield
    else:
        with _quiet_nemo():
            yield


class TranscriptionEngine:
    """Engine for audio transcription.

    Supports two backends selectable via the *backend* parameter:

    * ``"nemo"`` (default) — Nvidia NeMo Parakeet model.
    * ``"whisper"`` — faster-whisper (CTranslate2-based, word timestamps native).
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        backend: str = _DEFAULT_TRANSCRIPTION_BACKEND,
        model_name: str = _DEFAULT_TRANSCRIPTION_MODEL,
        whisper_model_name: str = _DEFAULT_WHISPER_MODEL,
    ):
        """Initialise the transcription engine.

        Args:
            device: Device to use (``"cuda"`` or ``"cpu"``)
            verbose: Show NeMo/library log output (default: suppressed)
            backend: Transcription backend — ``"nemo"`` (Parakeet) or
                ``"whisper"`` (faster-whisper).
            model_name: NeMo ASR model identifier (ignored for whisper backend).
            whisper_model_name: faster-whisper model size or path (ignored for
                nemo backend).
        """
        self.model: Optional[Any] = None          # NeMo model
        self.whisper_model: Optional[Any] = None  # faster-whisper model
        self.device = device
        self.verbose = verbose
        self.backend = backend.lower()
        self.model_name = model_name
        self.whisper_model_name = whisper_model_name
        if self.backend not in ("nemo", "whisper"):
            raise ValueError(
                f"Unknown transcription backend: {backend!r}. Choose 'nemo' or 'whisper'."
            )

    def _initialize_model(self) -> None:
        """Load the NeMo transcription model (lazy loading)."""
        if self.model is not None:
            return

        with _maybe_quiet(self.verbose):
            import nemo.collections.asr as nemo_asr  # noqa: F401  # lazy import

        print(f"Loading transcription model: {self.model_name}")
        print(f"Using device: {self.device}")

        with _maybe_quiet(self.verbose):
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )

        if self.device == "cpu":
            self.model = self.model.cpu()
        else:
            self.model = self.model.cuda()

        print("Transcription model loaded successfully")

    # ── Whisper backend ───────────────────────────────────────────────────────

    def _initialize_whisper_model(self) -> None:
        """Lazy-load the faster-whisper model."""
        if self.whisper_model is not None:
            return

        from faster_whisper import WhisperModel  # noqa: F401  # lazy import

        print(f"Loading Whisper model: {self.whisper_model_name}")
        print(f"Using device: {self.device}")
        compute_type = "float16" if self.device != "cpu" else "int8"
        self.whisper_model = WhisperModel(
            self.whisper_model_name,
            device=self.device,
            compute_type=compute_type,
        )
        print("Whisper model loaded successfully")

    def _transcribe_whisper_single(
        self, audio_path: str, include_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Transcribe one audio file via faster-whisper and normalise to the internal schema."""
        self._initialize_whisper_model()
        segments_gen, _ = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=include_timestamps,
            beam_size=5,
        )

        segments = list(segments_gen)

        text = " ".join(seg.text.strip() for seg in segments)
        result: Dict[str, Any] = {"text": text}

        if include_timestamps:
            word_timestamps: List[Dict[str, Any]] = []
            segment_timestamps: List[Dict[str, Any]] = []
            for seg in segments:
                segment_timestamps.append(
                    {"start": seg.start, "end": seg.end, "segment": seg.text.strip()}
                )
                for w in (seg.words or []):
                    word_timestamps.append(
                        {"word": w.word.strip(), "start": w.start, "end": w.end}
                    )
            result["word_timestamps"] = word_timestamps
            result["segment_timestamps"] = segment_timestamps
            result["char_timestamps"] = []

        return result

    def _transcribe_whisper_batch(
        self,
        audio_paths: List[str],
        include_timestamps: bool = True,
    ) -> List[Dict[str, Any]]:
        """Transcribe a list of independent files with the Whisper backend."""
        self._initialize_whisper_model()
        print(f"Transcribing {len(audio_paths)} file(s) [whisper]...")
        all_results: List[Dict[str, Any]] = []
        for audio_path in audio_paths:
            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=None)
            try:
                result = self._transcribe_whisper_single(chunk_paths[0], include_timestamps)
                all_results.append(result)
            finally:
                for cp in chunk_paths:
                    if cp != audio_path:
                        try:
                            os.unlink(cp)
                        except Exception:
                            pass
        return all_results

    def _transcribe_whisper_conversation(
        self,
        audio_paths: List[str],
        include_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe multiple files as one conversation with the Whisper backend."""
        self._initialize_whisper_model()
        combined_text: List[str] = []
        combined_words: List[Dict[str, Any]] = []
        combined_segs: List[Dict[str, Any]] = []
        file_offsets: List[Dict[str, Any]] = []
        time_offset = 0.0

        for file_idx, audio_path in enumerate(audio_paths):
            print(f"Transcribing file {file_idx + 1}/{len(audio_paths)}: {audio_path} [whisper]")
            audio_data, sr = sf.read(audio_path)
            file_duration = len(audio_data) / sr
            file_offset_start = time_offset

            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=None)
            try:
                result = self._transcribe_whisper_single(chunk_paths[0], include_timestamps)
                combined_text.append(result["text"])
                if include_timestamps:
                    for w in result.get("word_timestamps", []):
                        wc = w.copy()
                        wc["start"] += time_offset
                        wc["end"] += time_offset
                        combined_words.append(wc)
                    for s in result.get("segment_timestamps", []):
                        sc = s.copy()
                        sc["start"] += time_offset
                        sc["end"] += time_offset
                        combined_segs.append(sc)
            finally:
                for cp in chunk_paths:
                    if cp != audio_path:
                        try:
                            os.unlink(cp)
                        except Exception:
                            pass

            file_offsets.append(
                {"path": audio_path, "start": file_offset_start, "end": file_offset_start + file_duration}
            )
            time_offset += file_duration

        result_dict: Dict[str, Any] = {
            "text": " ".join(combined_text),
            "file_offsets": file_offsets,
        }
        if include_timestamps:
            result_dict["word_timestamps"] = combined_words
            result_dict["segment_timestamps"] = combined_segs
            result_dict["char_timestamps"] = []
        return result_dict

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess_audio(self, audio_path: str, chunk_duration: Optional[float] = None) -> List[str]:
        """Preprocess audio file to mono and optionally split into chunks.

        Args:
            audio_path: Path to input audio file
            chunk_duration: Duration of each chunk in seconds (None for no chunking)

        Returns:
            List of paths to processed audio chunks
        """
        audio, sample_rate = sf.read(audio_path)

        if audio.ndim == 1:
            duration = len(audio) / sample_rate
        else:
            duration = len(audio) / sample_rate

        if audio.ndim > 1 and audio.shape[1] > 1:
            print(f"Converting {audio.shape[1]}-channel audio to mono...")
            audio = np.mean(audio, axis=1)

        if not chunk_duration or duration <= chunk_duration:
            if audio.ndim > 1:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav", mode="wb"
                )
                temp_file.close()
                sf.write(temp_file.name, audio, sample_rate)
                return [temp_file.name]
            return [audio_path]

        print(f"Splitting audio ({duration:.1f}s) into chunks of {chunk_duration}s...")
        chunk_paths = []
        samples_per_chunk = int(chunk_duration * sample_rate)
        num_chunks = int(np.ceil(len(audio) / samples_per_chunk))

        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = min((i + 1) * samples_per_chunk, len(audio))
            chunk = audio[start_sample:end_sample]

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_chunk_{i}.wav", mode="wb"
            )
            temp_file.close()
            sf.write(temp_file.name, chunk, sample_rate)
            chunk_paths.append(temp_file.name)
            print(f"  Chunk {i+1}/{num_chunks}: {len(chunk)/sample_rate:.1f}s")

        return chunk_paths

    def transcribe(
        self,
        audio_paths: List[str],
        include_timestamps: bool = True,
        chunk_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe audio files.

        Args:
            audio_paths: List of paths to audio files
            include_timestamps: Whether to include word-level timestamps
            chunk_duration: Duration of each chunk in seconds (splits long files
                to avoid OOM).  Ignored by the ``whisper`` backend.

        Returns:
            List of transcription results with timestamps if requested
        """
        if self.backend == "whisper":
            return self._transcribe_whisper_batch(audio_paths, include_timestamps)

        self._initialize_model()
        print(f"Transcribing {len(audio_paths)} file(s)...")

        all_results = []

        for audio_path in audio_paths:
            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)

            try:
                if len(chunk_paths) == 1:
                    with _maybe_quiet(self.verbose):
                        output = self.model.transcribe(
                            chunk_paths, timestamps=include_timestamps, batch_size=1
                        )

                    result = output[0]
                    result_dict = {
                        "text": result.text if hasattr(result, "text") else str(result),
                    }

                    if include_timestamps and hasattr(result, "timestamp"):
                        result_dict["word_timestamps"] = result.timestamp.get("word", [])
                        result_dict["segment_timestamps"] = result.timestamp.get("segment", [])
                        result_dict["char_timestamps"] = result.timestamp.get("char", [])

                    all_results.append(result_dict)
                else:
                    print(f"Processing {len(chunk_paths)} chunks...")
                    combined_text = []
                    combined_word_timestamps = []
                    combined_segment_timestamps = []
                    combined_char_timestamps = []
                    time_offset = 0.0

                    for i, chunk_path in enumerate(chunk_paths):
                        print(f"  Transcribing chunk {i+1}/{len(chunk_paths)}...")
                        with _maybe_quiet(self.verbose):
                            output = self.model.transcribe(
                                [chunk_path], timestamps=include_timestamps, batch_size=1
                            )

                        result = output[0]
                        chunk_text = result.text if hasattr(result, "text") else str(result)
                        combined_text.append(chunk_text)

                        if include_timestamps and hasattr(result, "timestamp"):
                            for word in result.timestamp.get("word", []):
                                word_copy = word.copy()
                                word_copy["start"] = word.get("start", 0) + time_offset
                                word_copy["end"] = word.get("end", 0) + time_offset
                                combined_word_timestamps.append(word_copy)

                            for seg in result.timestamp.get("segment", []):
                                seg_copy = seg.copy()
                                seg_copy["start"] = seg.get("start", 0) + time_offset
                                seg_copy["end"] = seg.get("end", 0) + time_offset
                                combined_segment_timestamps.append(seg_copy)

                            for char in result.timestamp.get("char", []):
                                char_copy = char.copy()
                                char_copy["start"] = char.get("start", 0) + time_offset
                                char_copy["end"] = char.get("end", 0) + time_offset
                                combined_char_timestamps.append(char_copy)

                        time_offset += chunk_duration

                    result_dict = {"text": " ".join(combined_text)}

                    if include_timestamps:
                        result_dict["word_timestamps"] = combined_word_timestamps
                        result_dict["segment_timestamps"] = combined_segment_timestamps
                        result_dict["char_timestamps"] = combined_char_timestamps

                    all_results.append(result_dict)

            finally:
                for chunk_path in chunk_paths:
                    if chunk_path != audio_path:
                        try:
                            os.unlink(chunk_path)
                        except Exception:
                            pass

        return all_results

    def transcribe_conversation(
        self,
        audio_paths: List[str],
        include_timestamps: bool = True,
        chunk_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transcribe multiple audio files as a single ordered conversation.

        Each file is treated as the next sequential chunk of the same recording.
        Timestamps from later files are shifted by the cumulative duration of all
        preceding files so the merged result has a continuous timeline.

        Args:
            audio_paths: Ordered list of audio file paths
            include_timestamps: Whether to include word/segment-level timestamps
            chunk_duration: Per-file chunk splitting duration (to avoid OOM).
                Ignored by the ``whisper`` backend.

        Returns:
            Single transcription result dict with keys:
                - text: full concatenated transcript
                - word_timestamps, segment_timestamps, char_timestamps (if requested)
                - file_offsets: list of {path, start, end} per file
        """
        if self.backend == "whisper":
            return self._transcribe_whisper_conversation(audio_paths, include_timestamps)

        self._initialize_model()

        combined_text: List[str] = []
        combined_words: List[Dict[str, Any]] = []
        combined_segs: List[Dict[str, Any]] = []
        combined_chars: List[Dict[str, Any]] = []
        file_offsets: List[Dict[str, Any]] = []
        time_offset = 0.0

        for file_idx, audio_path in enumerate(audio_paths):
            print(f"Transcribing file {file_idx + 1}/{len(audio_paths)}: {audio_path}")

            audio_data, sr = sf.read(audio_path)
            file_duration = len(audio_data) / sr

            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)

            try:
                file_text: List[str] = []
                file_offset_start = time_offset

                if len(chunk_paths) == 1:
                    with _maybe_quiet(self.verbose):
                        output = self.model.transcribe(
                            chunk_paths, timestamps=include_timestamps, batch_size=1
                        )
                    result = output[0]
                    file_text.append(result.text if hasattr(result, "text") else str(result))

                    if include_timestamps and hasattr(result, "timestamp"):
                        for w in result.timestamp.get("word", []):
                            wc = w.copy()
                            wc["start"] = w.get("start", 0) + time_offset
                            wc["end"] = w.get("end", 0) + time_offset
                            combined_words.append(wc)
                        for s in result.timestamp.get("segment", []):
                            sc = s.copy()
                            sc["start"] = s.get("start", 0) + time_offset
                            sc["end"] = s.get("end", 0) + time_offset
                            combined_segs.append(sc)
                        for c in result.timestamp.get("char", []):
                            cc = c.copy()
                            cc["start"] = c.get("start", 0) + time_offset
                            cc["end"] = c.get("end", 0) + time_offset
                            combined_chars.append(cc)
                else:
                    inner_offset = time_offset
                    for i, chunk_path in enumerate(chunk_paths):
                        print(f"  Chunk {i + 1}/{len(chunk_paths)}…")
                        with _maybe_quiet(self.verbose):
                            output = self.model.transcribe(
                                [chunk_path], timestamps=include_timestamps, batch_size=1
                            )
                        result = output[0]
                        file_text.append(result.text if hasattr(result, "text") else str(result))

                        if include_timestamps and hasattr(result, "timestamp"):
                            for w in result.timestamp.get("word", []):
                                wc = w.copy()
                                wc["start"] = w.get("start", 0) + inner_offset
                                wc["end"] = w.get("end", 0) + inner_offset
                                combined_words.append(wc)
                            for s in result.timestamp.get("segment", []):
                                sc = s.copy()
                                sc["start"] = s.get("start", 0) + inner_offset
                                sc["end"] = s.get("end", 0) + inner_offset
                                combined_segs.append(sc)
                            for c in result.timestamp.get("char", []):
                                cc = c.copy()
                                cc["start"] = c.get("start", 0) + inner_offset
                                cc["end"] = c.get("end", 0) + inner_offset
                                combined_chars.append(cc)

                        inner_offset += chunk_duration or file_duration
            finally:
                for chunk_path in chunk_paths:
                    if chunk_path != audio_path:
                        try:
                            os.unlink(chunk_path)
                        except Exception:
                            pass

            combined_text.append(" ".join(file_text))
            file_offsets.append(
                {"path": audio_path, "start": file_offset_start, "end": file_offset_start + file_duration}
            )
            time_offset += file_duration

        result_dict: Dict[str, Any] = {
            "text": " ".join(combined_text),
            "file_offsets": file_offsets,
        }
        if include_timestamps:
            result_dict["word_timestamps"] = combined_words
            result_dict["segment_timestamps"] = combined_segs
            result_dict["char_timestamps"] = combined_chars

        return result_dict

    def transcribe_with_segments(
        self, audio_path: str, chunk_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe audio and return segment-level results.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds (splits long files
                to avoid OOM).  Ignored by the ``whisper`` backend.

        Returns:
            List of segment dictionaries with timing and text
        """
        if self.backend == "whisper":
            result = self._transcribe_whisper_single(audio_path, include_timestamps=True)
            return result.get("segment_timestamps", [])

        self._initialize_model()

        chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)

        try:
            if len(chunk_paths) == 1:
                with _maybe_quiet(self.verbose):
                    output = self.model.transcribe([chunk_paths[0]], timestamps=True)

                if not output or not hasattr(output[0], "timestamp"):
                    return []

                return output[0].timestamp.get("segment", [])
            else:
                all_segments = []
                time_offset = 0.0

                for chunk_path in chunk_paths:
                    with _maybe_quiet(self.verbose):
                        output = self.model.transcribe([chunk_path], timestamps=True)

                    if output and hasattr(output[0], "timestamp"):
                        for seg in output[0].timestamp.get("segment", []):
                            seg_copy = seg.copy()
                            seg_copy["start"] = seg.get("start", 0) + time_offset
                            seg_copy["end"] = seg.get("end", 0) + time_offset
                            all_segments.append(seg_copy)

                    time_offset += chunk_duration

                return all_segments
        finally:
            for chunk_path in chunk_paths:
                if chunk_path != audio_path:
                    try:
                        os.unlink(chunk_path)
                    except Exception:
                        pass
