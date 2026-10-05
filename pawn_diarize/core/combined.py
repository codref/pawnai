"""Combined transcription and diarization functionality."""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .transcription import TranscriptionEngine
from .diarization import DiarizationEngine


def transcribe_with_diarization(
    audio_path: Union[str, List[str]],
    db_dsn: Optional[str] = None,
    similarity_threshold: float = 0.7,
    store_new_speakers: bool = False,
    device: str = "cuda",
    chunk_duration: Optional[float] = None,
    cross_file_threshold: float = 0.85,
    prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
    time_cursor: float = 0.0,
    verbose: bool = False,
    backend: str = "nemo",
    source_map: Optional[Dict[str, str]] = None,
    transcription_engine: Optional["TranscriptionEngine"] = None,
    diarization_engine: Optional["DiarizationEngine"] = None,
    session_id: Optional[str] = None,
    identify_margin: float = 0.05,
    diarization_backend: str = "pyannote",
    diarization_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe audio with speaker diarization labels.

    Identity matching uses the curated Speakers gallery only.  Unknown
    speakers stay as ``SPEAKER_XX``; ``store_new_speakers`` is ignored.
    """
    audio_paths: List[str] = [audio_path] if isinstance(audio_path, str) else list(audio_path)
    multiple = len(audio_paths) > 1
    resume_note = f" (resuming from t={time_cursor:.1f}s)" if time_cursor > 0 else ""

    print(f"[1/2] Running transcription{'  (' + str(len(audio_paths)) + ' files)' if multiple else ''}{resume_note}...")
    if transcription_engine is None:
        transcription_engine = TranscriptionEngine(device=device, verbose=verbose, backend=backend)

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
    if diarization_engine is None:
        diarization_engine = DiarizationEngine(
            device=device,
            diarization_backend=diarization_backend,
            diarization_model=diarization_model,
            embedding_model=embedding_model,
            hf_token=hf_token,
            identify_margin=identify_margin,
        )
    diarization = diarization_engine.diarize(
        audio_paths if multiple else audio_paths[0],
        db_dsn=db_dsn,
        similarity_threshold=similarity_threshold,
        store_new_speakers=False,
        cross_file_threshold=cross_file_threshold,
        prior_speaker_embeddings=prior_speaker_embeddings,
        time_cursor=time_cursor,
        source_map=source_map,
        session_id=session_id,
        identify_margin=identify_margin,
    )

    print("Merging transcription with speaker labels...")
    merged_segments = _merge_transcription_with_diarization(transcription, diarization)

    active_speakers = sorted({
        seg["speaker"]
        for seg in merged_segments
        if seg.get("text", "").strip()
    })
    if not active_speakers:
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
        
        # Create merged segment, preserving original pyannote label and source file
        merged_segment = {
            "speaker": speaker,
            "original_label": speaker_seg.get("original_label", speaker),
            "source_file": speaker_seg.get("source_file", ""),
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
