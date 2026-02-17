"""Combined transcription and diarization functionality."""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .transcription import TranscriptionEngine
from .diarization import DiarizationEngine


def transcribe_with_diarization(
    audio_path: Union[str, List[str]],
    db_path: Optional[str] = None,
    similarity_threshold: float = 0.7,
    store_new_speakers: bool = True,
    device: str = "cuda",
    chunk_duration: Optional[float] = None,
    cross_file_threshold: float = 0.85,
    prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
    time_cursor: float = 0.0,
) -> Dict[str, Any]:
    """Transcribe audio with speaker diarization labels.

    Supports incremental sessions: pass ``prior_speaker_embeddings`` and
    ``time_cursor`` from a previously saved session to continue processing
    new audio without re-processing old files.  Each call returns
    ``session_speaker_embeddings`` and ``new_time_cursor`` which should be
    persisted and passed back on the next call.

    Args:
        audio_path: Path to audio file, or ordered list of paths treated as
                    sequential chunks of the same conversation.
        db_path: Path to speaker database (None to skip database lookup).
        similarity_threshold: Minimum similarity to match speakers against
                              the database (0-1).
        store_new_speakers: Whether to store embeddings for unknown speakers.
        device: Device to use for transcription ("cuda" or "cpu").
        chunk_duration: Split each audio file into chunks of N seconds.
        cross_file_threshold: Cosine-similarity threshold for assigning the
                              same global speaker label across files (0-1).
        prior_speaker_embeddings: Per-speaker state from a previous session.
            Mapping of global_label → {"embedding": [...], "total_duration": float}.
        time_cursor: Seconds of audio already processed in previous calls.
            All new timestamps are shifted by this value.

    Returns:
        Dictionary containing:
            - text: Full transcribed text (current call only)
            - speakers: List of unique speaker names/labels (current call)
            - num_speakers: Total number of speakers detected
            - segments: List of dicts with speaker, start, end, text, words
            - word_timestamps: Word-level timestamps (offset by time_cursor)
            - diarization: Raw diarization results
            - matched_speakers: Dict mapping local labels to matched names
            - new_speakers: List of speaker labels that were not matched
            - file_offsets: list of {path, start, end} (when multiple files)
            - session_speaker_embeddings: Updated per-speaker embeddings for
              the next incremental call (pass as prior_speaker_embeddings).
            - new_time_cursor: Updated total duration; pass as time_cursor
              on the next call.
    """
    # Normalise to list
    audio_paths: List[str] = [audio_path] if isinstance(audio_path, str) else list(audio_path)
    multiple = len(audio_paths) > 1
    resume_note = f" (resuming from t={time_cursor:.1f}s)" if time_cursor > 0 else ""

    print(f"[1/2] Running transcription{'  (' + str(len(audio_paths)) + ' files)' if multiple else ''}{resume_note}...")
    transcription_engine = TranscriptionEngine(device=device)

    if multiple:
        transcription = transcription_engine.transcribe_conversation(
            audio_paths,
            include_timestamps=True,
            chunk_duration=chunk_duration,
        )
    else:
        transcription_results = transcription_engine.transcribe(
            audio_paths,
            include_timestamps=True,
            chunk_duration=chunk_duration,
        )
        if not transcription_results:
            raise ValueError("Transcription failed or returned no results")
        transcription = transcription_results[0]

    # Shift transcription timestamps by prior time_cursor so they sit at
    # the correct position in the global timeline.
    if time_cursor > 0:
        for key in ("word_timestamps", "segment_timestamps", "char_timestamps"):
            for entry in transcription.get(key, []):
                entry["start"] = entry.get("start", 0.0) + time_cursor
                entry["end"] = entry.get("end", 0.0) + time_cursor
        if "file_offsets" in transcription:
            for fo in transcription["file_offsets"]:
                fo["start"] = fo.get("start", 0.0) + time_cursor
                fo["end"] = fo.get("end", 0.0) + time_cursor

    print(f"[2/2] Running speaker diarization{'  (' + str(len(audio_paths)) + ' files)' if multiple else ''}{resume_note}...")
    diarization_engine = DiarizationEngine(device=device if device != "cpu" else None)
    diarization = diarization_engine.diarize(
        audio_paths if multiple else audio_paths[0],
        db_path=db_path,
        similarity_threshold=similarity_threshold,
        store_new_speakers=store_new_speakers,
        cross_file_threshold=cross_file_threshold,
        prior_speaker_embeddings=prior_speaker_embeddings,
        time_cursor=time_cursor,
    )

    print("Merging transcription with speaker labels...")
    merged_segments = _merge_transcription_with_diarization(transcription, diarization)

    # Recompute speaker list from merged segments: only count speakers that
    # actually have transcribed words (diarization may produce segments that
    # don't overlap with any word, inflating the speaker count).
    active_speakers = sorted({
        seg["speaker"]
        for seg in merged_segments
        if seg.get("text", "").strip()
    })
    if not active_speakers:
        # Fallback: keep whatever diarization reported
        active_speakers = diarization.get("speakers", [])

    result: Dict[str, Any] = {
        "text": transcription.get("text", ""),
        "speakers": active_speakers,
        "num_speakers": len(active_speakers),
        "segments": merged_segments,
        "word_timestamps": transcription.get("word_timestamps", []),
        "diarization": diarization,
        "matched_speakers": diarization.get("matched_speakers", {}),
        "new_speakers": diarization.get("new_speakers", []),
        "session_speaker_embeddings": diarization.get("session_speaker_embeddings", {}),
        "new_time_cursor": diarization.get("new_time_cursor", time_cursor),
    }

    # Surface per-file offset info when processing multiple files
    if multiple:
        result["file_offsets"] = transcription.get("file_offsets", [])
        result["chunk_offsets"] = diarization.get("chunk_offsets", [])

    return result


def _merge_transcription_with_diarization(
    transcription: Dict[str, Any],
    diarization: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Merge transcription and diarization results by matching timestamps.
    
    Args:
        transcription: Transcription results with word_timestamps
        diarization: Diarization results with speaker segments
        
    Returns:
        List of merged segments with speaker, text, and timing info
    """
    word_timestamps = transcription.get("word_timestamps", [])
    speaker_segments = diarization.get("segments", [])
    
    if not word_timestamps or not speaker_segments:
        # Fallback: return speaker segments without words
        return speaker_segments
    
    merged_segments = []
    
    # For each speaker segment, find overlapping words
    for speaker_seg in speaker_segments:
        start_time = speaker_seg["start"]
        end_time = speaker_seg["end"]
        speaker = speaker_seg["speaker"]
        
        # Find all words that overlap with this speaker's time range
        overlapping_words = []
        for word_data in word_timestamps:
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", 0)
            word = word_data.get("word", "")
            
            # Check if word overlaps with speaker segment
            # A word overlaps if its midpoint falls within the speaker segment
            word_mid = (word_start + word_end) / 2
            if start_time <= word_mid <= end_time:
                overlapping_words.append({
                    "word": word,
                    "start": word_start,
                    "end": word_end
                })
        
        # Build text from overlapping words
        segment_text = " ".join([w["word"] for w in overlapping_words])
        
        # Create merged segment
        merged_segment = {
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "text": segment_text,
            "words": overlapping_words,
            "num_words": len(overlapping_words)
        }
        
        merged_segments.append(merged_segment)
    
    return merged_segments


def format_transcript_with_speakers(
    result: Dict[str, Any],
    include_timestamps: bool = True
) -> str:
    """Format combined transcription+diarization result as readable text.
    
    Args:
        result: Result from transcribe_with_diarization()
        include_timestamps: Whether to include timestamps in output
        
    Returns:
        Formatted transcript string
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("TRANSCRIPT WITH SPEAKER DIARIZATION")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Detected {result['num_speakers']} speaker(s): {', '.join(result['speakers'])}")
    
    # Show matched speakers if any
    if result.get('matched_speakers'):
        lines.append("")
        lines.append("Matched speakers:")
        for original, matched in result['matched_speakers'].items():
            lines.append(f"  {original} → {matched}")
    
    # Show new speakers if any
    if result.get('new_speakers'):
        lines.append("")
        lines.append(f"New/Unknown speakers: {', '.join(result['new_speakers'])}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("")
    
    # Segments with speaker labels
    current_speaker = None
    for segment in result["segments"]:
        speaker = segment["speaker"]
        text = segment["text"]
        
        if not text.strip():
            continue  # Skip empty segments
        
        # Add speaker label when speaker changes
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")  # Blank line between speakers
            
            if include_timestamps:
                start_time = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
                lines.append(f"[{start_time}] {speaker}:")
            else:
                lines.append(f"{speaker}:")
            
            current_speaker = speaker
        
        # Add the text
        if include_timestamps:
            start_time = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
            end_time = f"{int(segment['end']//60):02d}:{segment['end']%60:05.2f}"
            lines.append(f"  [{start_time} → {end_time}] {text}")
        else:
            lines.append(f"  {text}")
    
    return "\n".join(lines)
