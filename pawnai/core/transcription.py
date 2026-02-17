"""Transcription engine using Nvidia Parakeet model."""

from typing import List, Dict, Any, Optional
import tempfile
from pathlib import Path
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

from .config import TRANSCRIPTION_MODEL


class TranscriptionEngine:
    """Engine for audio transcription using Nvidia Parakeet model."""

    def __init__(self, device: str = "cuda"):
        """Initialize the transcription engine.
        
        Args:
            device: Device to use ("cuda" or "cpu")
        """
        self.model: Optional[Any] = None
        self.device = device

    def _initialize_model(self) -> None:
        """Load the transcription model (lazy loading)."""
        if self.model is not None:
            return  # Already loaded

        print(f"Loading transcription model: {TRANSCRIPTION_MODEL}")
        print(f"Using device: {self.device}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=TRANSCRIPTION_MODEL
        )
        
        # Move model to specified device
        if self.device == "cpu":
            self.model = self.model.cpu()
        else:
            self.model = self.model.cuda()
            
        print("Transcription model loaded successfully")

    def _preprocess_audio(self, audio_path: str, chunk_duration: Optional[float] = None) -> List[str]:
        """Preprocess audio file to ensure it's mono and optionally split into chunks.
        
        Args:
            audio_path: Path to input audio file
            chunk_duration: Duration of each chunk in seconds (None for no chunking)
            
        Returns:
            List of paths to processed audio chunks
        """
        # Load audio file
        audio, sample_rate = sf.read(audio_path)
        
        # Get duration in seconds
        if audio.ndim == 1:
            duration = len(audio) / sample_rate
        else:
            duration = len(audio) / sample_rate
        
        # Check if audio is stereo or multi-channel
        if audio.ndim > 1 and audio.shape[1] > 1:
            print(f"Converting {audio.shape[1]}-channel audio to mono...")
            # Convert to mono by averaging all channels
            audio = np.mean(audio, axis=1)
        
        # If no chunking needed, save and return single file
        if not chunk_duration or duration <= chunk_duration:
            if audio.ndim > 1:  # Only save if we converted from multi-channel
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav', mode='wb'
                )
                temp_file.close()
                sf.write(temp_file.name, audio, sample_rate)
                return [temp_file.name]
            return [audio_path]
        
        # Split into chunks
        print(f"Splitting audio ({duration:.1f}s) into chunks of {chunk_duration}s...")
        chunk_paths = []
        samples_per_chunk = int(chunk_duration * sample_rate)
        num_chunks = int(np.ceil(len(audio) / samples_per_chunk))
        
        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = min((i + 1) * samples_per_chunk, len(audio))
            chunk = audio[start_sample:end_sample]
            
            # Save chunk to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=f'_chunk_{i}.wav', mode='wb'
            )
            temp_file.close()
            sf.write(temp_file.name, chunk, sample_rate)
            chunk_paths.append(temp_file.name)
            print(f"  Chunk {i+1}/{num_chunks}: {len(chunk)/sample_rate:.1f}s")
        
        return chunk_paths

    def transcribe(
        self, audio_paths: List[str], include_timestamps: bool = True, chunk_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe audio files.

        Args:
            audio_paths: List of paths to audio files
            include_timestamps: Whether to include word-level timestamps
            chunk_duration: Duration of each chunk in seconds (splits long files to avoid OOM)

        Returns:
            List of transcription results with timestamps if requested
        """
        import os
        
        self._initialize_model()
        print(f"Transcribing {len(audio_paths)} file(s)...")
        
        all_results = []
        
        for audio_path in audio_paths:
            # Preprocess audio file (convert stereo to mono and split into chunks if needed)
            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)
            
            try:
                if len(chunk_paths) == 1:
                    # Single file, process normally
                    output = self.model.transcribe(
                        chunk_paths, timestamps=include_timestamps, batch_size=1
                    )
                    
                    result = output[0]
                    result_dict = {
                        "text": result.text if hasattr(result, 'text') else str(result),
                    }
                    
                    if include_timestamps and hasattr(result, 'timestamp'):
                        result_dict["word_timestamps"] = result.timestamp.get("word", [])
                        result_dict["segment_timestamps"] = result.timestamp.get("segment", [])
                        result_dict["char_timestamps"] = result.timestamp.get("char", [])
                    
                    all_results.append(result_dict)
                else:
                    # Multiple chunks, process and combine
                    print(f"Processing {len(chunk_paths)} chunks...")
                    combined_text = []
                    combined_word_timestamps = []
                    combined_segment_timestamps = []
                    combined_char_timestamps = []
                    time_offset = 0.0
                    
                    for i, chunk_path in enumerate(chunk_paths):
                        print(f"  Transcribing chunk {i+1}/{len(chunk_paths)}...")
                        output = self.model.transcribe(
                            [chunk_path], timestamps=include_timestamps, batch_size=1
                        )
                        
                        result = output[0]
                        chunk_text = result.text if hasattr(result, 'text') else str(result)
                        combined_text.append(chunk_text)
                        
                        if include_timestamps and hasattr(result, 'timestamp'):
                            # Adjust timestamps by adding offset
                            for word in result.timestamp.get("word", []):
                                word_copy = word.copy()
                                word_copy['start'] = word.get('start', 0) + time_offset
                                word_copy['end'] = word.get('end', 0) + time_offset
                                combined_word_timestamps.append(word_copy)
                            
                            for seg in result.timestamp.get("segment", []):
                                seg_copy = seg.copy()
                                seg_copy['start'] = seg.get('start', 0) + time_offset
                                seg_copy['end'] = seg.get('end', 0) + time_offset
                                combined_segment_timestamps.append(seg_copy)
                            
                            for char in result.timestamp.get("char", []):
                                char_copy = char.copy()
                                char_copy['start'] = char.get('start', 0) + time_offset
                                char_copy['end'] = char.get('end', 0) + time_offset
                                combined_char_timestamps.append(char_copy)
                        
                        # Update offset for next chunk
                        time_offset += chunk_duration
                    
                    # Combine all chunks
                    result_dict = {
                        "text": " ".join(combined_text),
                    }
                    
                    if include_timestamps:
                        result_dict["word_timestamps"] = combined_word_timestamps
                        result_dict["segment_timestamps"] = combined_segment_timestamps
                        result_dict["char_timestamps"] = combined_char_timestamps
                    
                    all_results.append(result_dict)
                    
            finally:
                # Clean up temporary chunk files
                for chunk_path in chunk_paths:
                    if chunk_path != audio_path:  # Don't delete original
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
            chunk_duration: Per-file chunk splitting duration (to avoid OOM)

        Returns:
            Single transcription result dict with keys:
                - text: full concatenated transcript
                - word_timestamps, segment_timestamps, char_timestamps (if requested)
                - file_offsets: list of {path, start, end} showing each file's
                  position in the merged timeline
        """
        import soundfile as sf
        import os

        self._initialize_model()

        combined_text: List[str] = []
        combined_words: List[Dict[str, Any]] = []
        combined_segs: List[Dict[str, Any]] = []
        combined_chars: List[Dict[str, Any]] = []
        file_offsets: List[Dict[str, Any]] = []
        time_offset = 0.0

        for file_idx, audio_path in enumerate(audio_paths):
            print(f"Transcribing file {file_idx + 1}/{len(audio_paths)}: {audio_path}")

            # Measure duration for offset tracking
            audio_data, sr = sf.read(audio_path)
            file_duration = len(audio_data) / sr

            chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)

            try:
                file_text: List[str] = []
                file_offset_start = time_offset

                if len(chunk_paths) == 1:
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
            chunk_duration: Duration of each chunk in seconds (splits long files to avoid OOM)

        Returns:
            List of segment dictionaries with timing and text
        """
        import os
        
        self._initialize_model()
        
        # Preprocess audio (convert stereo to mono and split into chunks if needed)
        chunk_paths = self._preprocess_audio(audio_path, chunk_duration=chunk_duration)
        
        try:
            if len(chunk_paths) == 1:
                # Single file
                output = self.model.transcribe([chunk_paths[0]], timestamps=True)
                
                if not output or not hasattr(output[0], 'timestamp'):
                    return []
                
                segment_timestamps = output[0].timestamp.get("segment", [])
                return segment_timestamps
            else:
                # Multiple chunks, combine segments
                all_segments = []
                time_offset = 0.0
                
                for chunk_path in chunk_paths:
                    output = self.model.transcribe([chunk_path], timestamps=True)
                    
                    if output and hasattr(output[0], 'timestamp'):
                        segments = output[0].timestamp.get("segment", [])
                        for seg in segments:
                            seg_copy = seg.copy()
                            seg_copy['start'] = seg.get('start', 0) + time_offset
                            seg_copy['end'] = seg.get('end', 0) + time_offset
                            all_segments.append(seg_copy)
                    
                    time_offset += chunk_duration
                
                return all_segments
        finally:
            # Clean up temporary files if created
            for chunk_path in chunk_paths:
                if chunk_path != audio_path:
                    try:
                        os.unlink(chunk_path)
                    except Exception:
                        pass
