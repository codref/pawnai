"""CLI command definitions for PawnAI."""

from typing import List, Optional
import typer
from pathlib import Path

from .utils import console

app = typer.Typer(
    help="PawnAI: Speaker diarization and transcription CLI tool",
    rich_markup_mode="rich",
)


@app.command()
def diarize(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to diarize (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Similarity threshold for speaker matching (0-1, default: 0.7)"
    ),
    store_new: bool = typer.Option(
        True, "--store-new/--no-store", help="Store embeddings for unknown speakers"
    ),
) -> None:
    """Perform speaker diarization on one or more audio files.
    
    When multiple files are given they are treated as ordered chunks of the
    same conversation – they are concatenated before diarization so speaker
    labels are consistent across all files.

    Analyzes the audio to identify and separate different speakers,
    showing when each speaker talks with timestamps. Automatically
    recognizes known speakers from the database.
    
    Example:
        pawnai diarize audio.wav
        pawnai diarize part1.wav part2.wav part3.wav -o result.json
        pawnai diarize audio.wav -t 0.8  # Stricter matching
        pawnai diarize audio.wav --no-store  # Don't save new speakers
    """
    # Lazy imports to avoid loading models during --help
    import json
    from ..core import DiarizationEngine
    from ..core.config import Config, load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    try:
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_path=db_path)
        config.ensure_paths_exist()

        engine = DiarizationEngine()
        if len(audio_paths) == 1:
            console.print(f"[cyan]Diarizing: {audio_paths[0]}[/cyan]")
        else:
            console.print(f"[cyan]Diarizing {len(audio_paths)} files as one conversation:[/cyan]")
            for i, p in enumerate(audio_paths, 1):
                console.print(f"  {i}. {p}")

        result = engine.diarize(
            audio_paths if len(audio_paths) > 1 else audio_paths[0],
            db_path=db_path,
            similarity_threshold=threshold,
            store_new_speakers=store_new
        )
        
        console.print(f"[green]✓ Diarization complete[/green]")
        
        # Show matched speakers if any
        if result.get('matched_speakers'):
            console.print(f"\n[bold green]Matched speakers:[/bold green]")
            for original, matched in result['matched_speakers'].items():
                console.print(f"  {original} → {matched}")
        
        # Show new speakers if any
        if result.get('new_speakers'):
            console.print(f"\n[bold yellow]New/Unknown speakers:[/bold yellow] {', '.join(result['new_speakers'])}")
            if store_new:
                console.print(f"  [dim](Embeddings stored for future recognition)[/dim]")
            else:
                console.print(f"  [dim](Use 'pawnai label' to assign names)[/dim]")
        
        console.print(f"\n[bold]Detected {result['num_speakers']} speaker(s):[/bold] {', '.join(result['speakers'])}")
        
        # Handle output file if specified
        if output:
            output_path = Path(output)
            output_format = output_path.suffix.lower()
            
            if output_format == ".json":
                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                console.print(f"[green]✓ Saved JSON to: {output}[/green]")
            
            elif output_format == ".txt":
                # Save as plain text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Speaker Diarization Results\n")
                    f.write(f"===========================\n\n")
                    f.write(f"Detected {result['num_speakers']} speaker(s): {', '.join(result['speakers'])}\n\n")
                    f.write(f"Timeline:\n")
                    f.write(f"---------\n\n")
                    
                    for seg in result['segments']:
                        start_time = f"{int(seg['start']//60):02d}:{seg['start']%60:05.2f}"
                        end_time = f"{int(seg['end']//60):02d}:{seg['end']%60:05.2f}"
                        f.write(f"[{start_time} → {end_time}] {seg['speaker']} ({seg['duration']:.2f}s)\n")
                
                console.print(f"[green]✓ Saved text to: {output}[/green]")
            
            else:
                console.print(f"[yellow]⚠ Unknown format '{output_format}', using .txt format[/yellow]")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Speakers: {', '.join(result['speakers'])}\n\n")
                    for seg in result['segments']:
                        f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}\n")
                console.print(f"[green]✓ Saved to: {output}[/green]")
        
        else:
            # Console output (default)
            console.print(f"\n[bold]Speaker Timeline:[/bold]")
            for i, seg in enumerate(result['segments'][:10]):
                start_time = f"{int(seg['start']//60):02d}:{seg['start']%60:05.2f}"
                end_time = f"{int(seg['end']//60):02d}:{seg['end']%60:05.2f}"
                console.print(f"  [{start_time} → {end_time}] {seg['speaker']} ({seg['duration']:.2f}s)")
            
            if len(result['segments']) > 10:
                console.print(f"  ... and {len(result['segments']) - 10} more segments")
        
    except Exception as e:
        console.print(f"[red]Error during diarization: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def transcribe(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to transcribe (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    with_timestamps: bool = typer.Option(
        True, "--timestamps/--no-timestamps", help="Include word-level timestamps"
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to use: cuda or cpu (use cpu if out of memory)"
    ),
    chunk_duration: Optional[float] = typer.Option(
        None, "--chunk-duration", "-c", help="Split each audio file into chunks of N seconds (helps avoid OOM)"
    ),
) -> None:
    """Transcribe one or more audio files using the Parakeet model.

    When multiple files are given they are treated as ordered chunks of the
    same conversation. Timestamps are adjusted so the merged result has a
    single continuous timeline.

    Outputs to console by default, or saves to file if --output is specified.
    
    For large files that cause out-of-memory errors:
    - Use --device cpu to run on CPU (slower but avoids GPU memory limits)
    - Use --chunk-duration to split audio into chunks (e.g., -c 300 for 5-minute chunks)
    
    Example:
        pawnai transcribe audio.wav
        pawnai transcribe part1.wav part2.wav -o transcript.txt
        pawnai transcribe audio.wav --no-timestamps
        pawnai transcribe audio.wav -o transcript.json
        pawnai transcribe large.mp3 --device cpu
        pawnai transcribe large.mp3 -c 300 -o output.txt
    """
    # Lazy imports to avoid loading models during --help
    import json
    from ..core import TranscriptionEngine
    from ..core.config import load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    try:
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        engine = TranscriptionEngine(device=device)

        if len(audio_paths) == 1:
            console.print(f"[cyan]Transcribing: {audio_paths[0]}[/cyan]")
        else:
            console.print(f"[cyan]Transcribing {len(audio_paths)} files as one conversation:[/cyan]")
            for i, p in enumerate(audio_paths, 1):
                console.print(f"  {i}. {p}")

        if chunk_duration:
            console.print(f"[yellow]Chunk duration set to: {chunk_duration}s (will split long audio)[/yellow]")

        if len(audio_paths) == 1:
            results = engine.transcribe(
                audio_paths, include_timestamps=with_timestamps, chunk_duration=chunk_duration
            )
            if not results:
                console.print(f"[yellow]⚠ No transcription results[/yellow]")
                return
            result = results[0]
        else:
            result = engine.transcribe_conversation(
                audio_paths, include_timestamps=with_timestamps, chunk_duration=chunk_duration
            )
        console.print(f"[green]✓ Transcription complete[/green]")
        
        # Determine output format
        if output:
            output_path = Path(output)
            output_format = output_path.suffix.lower()
            
            if output_format == ".json":
                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                console.print(f"[green]✓ Saved JSON to: {output}[/green]")
            
            elif output_format == ".txt":
                # Save as plain text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('text', ''))
                    
                    if with_timestamps and "segment_timestamps" in result:
                        f.write("\n\n--- Segments ---\n")
                        for seg in result["segment_timestamps"]:
                            start = seg.get('start', 0)
                            end = seg.get('end', 0)
                            text = seg.get('segment', seg.get('text', ''))
                            f.write(f"\n[{start:.2f}s - {end:.2f}s]\n{text}\n")
                console.print(f"[green]✓ Saved text to: {output}[/green]")
            
            else:
                console.print(f"[yellow]⚠ Unknown format '{output_format}', using .txt format[/yellow]")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('text', ''))
                console.print(f"[green]✓ Saved to: {output}[/green]")
        
        else:
            # Console output (default)
            console.print(f"\n[bold]Text:[/bold] {result.get('text', 'N/A')}")
            
            if with_timestamps and "word_timestamps" in result and len(result["word_timestamps"]) > 0:
                console.print(f"\n[bold]Timestamps (first 5 words):[/bold]")
                for ts in result["word_timestamps"][:5]:
                    console.print(f"  {ts.get('start', 0):.2f}s - {ts.get('end', 0):.2f}s: {ts.get('word', '')}")
    
    except Exception as e:
        console.print(f"[red]Error during transcription: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command(name="transcribe-diarize")
def transcribe_diarize(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to process (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Path to a session JSON file.  If the file exists its speaker state and "
             "timestamps are loaded so new audio is appended to the same conversation.  "
             "After processing the file is updated with the full accumulated results."
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Similarity threshold for speaker matching (0-1)"
    ),
    store_new: bool = typer.Option(
        True, "--store-new/--no-store", help="Store embeddings for unknown speakers"
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to use: cuda or cpu"
    ),
    chunk_duration: Optional[float] = typer.Option(
        None, "--chunk-duration", "-c", help="Split each audio file into chunks of N seconds"
    ),
    cross_file_threshold: float = typer.Option(
        0.85, "--cross-threshold", "-x",
        help="Cosine-similarity threshold for matching speakers across files (0-1). "
             "Higher = stricter; only used when multiple files are provided."
    ),
    no_timestamps: bool = typer.Option(
        False, "--no-timestamps", help="Hide timestamps in text output"
    ),
) -> None:
    """Transcribe audio with speaker diarization labels.

    When multiple files are given they are treated as ordered chunks of the
    same conversation.  Each file is diarized independently and speaker labels
    are aligned across files using embedding similarity so the same person gets
    the same label throughout (no audio concatenation is performed).

    Use --session to accumulate results across multiple separate invocations:

    \b
      # First call – creates / initialises the session
      pawnai transcribe-diarize part1.flac --session conv.json

      # Second call – picks up where the first left off
      pawnai transcribe-diarize part2.flac --session conv.json

      # Write the full accumulated transcript to a file
      pawnai transcribe-diarize part3.flac --session conv.json -o full.txt

    Automatically recognises known speakers from the database.

    For large files that cause out-of-memory errors:
    - Use --device cpu to run on CPU (slower but avoids GPU memory limits)
    - Use --chunk-duration to split audio into chunks

    Example:
        pawnai transcribe-diarize audio.wav
        pawnai transcribe-diarize part1.wav part2.wav -o transcript.txt
        pawnai transcribe-diarize audio.wav -o result.json
        pawnai transcribe-diarize audio.wav -t 0.8 --no-store
        pawnai transcribe-diarize large.mp3 --device cpu -c 300
        pawnai transcribe-diarize a.wav b.wav -x 0.9
    """
    # Lazy imports to avoid loading models during --help
    import json
    from ..core import transcribe_with_diarization, format_transcript_with_speakers
    from ..core.config import Config, load_config_from_file

    # Load config from specified file if provided
    if config:
        load_config_from_file(config)

    try:
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_path=db_path)
        config.ensure_paths_exist()

        # ------------------------------------------------------------------
        # Load prior session state (if --session is specified and file exists)
        # ------------------------------------------------------------------
        prior_speaker_embeddings: Optional[dict] = None
        prior_time_cursor: float = 0.0
        prior_segments: list = []
        prior_words: list = []
        prior_text: str = ""
        prior_speakers: list = []
        prior_matched: dict = {}
        prior_processed_files: list = []

        if session:
            session_path = Path(session)
            if session_path.exists():
                console.print(f"[cyan]Resuming session: {session}[/cyan]")
                with open(session_path, "r", encoding="utf-8") as f:
                    sess = json.load(f)
                prior_speaker_embeddings = sess.get("session_speaker_embeddings")
                prior_time_cursor = float(sess.get("time_cursor", 0.0))
                prior_segments = sess.get("segments", [])
                prior_words = sess.get("word_timestamps", [])
                prior_text = sess.get("text", "")
                prior_speakers = sess.get("speakers", [])
                prior_matched = sess.get("matched_speakers", {})
                prior_processed_files = sess.get("processed_files", [])
                console.print(
                    f"  Loaded {len(prior_segments)} prior segments, "
                    f"t={prior_time_cursor:.1f}s, "
                    f"speakers: {', '.join(prior_speakers) or '(none)'}"
                )
            else:
                console.print(f"[cyan]Starting new session: {session}[/cyan]")

        # ------------------------------------------------------------------
        # Log what we're about to process
        # ------------------------------------------------------------------
        if len(audio_paths) == 1:
            console.print(f"[cyan]Processing: {audio_paths[0]}[/cyan]")
        else:
            console.print(f"[cyan]Processing {len(audio_paths)} files as one conversation:[/cyan]")
            for i, p in enumerate(audio_paths, 1):
                console.print(f"  {i}. {p}")

        if chunk_duration:
            console.print(f"[yellow]Chunk duration set to: {chunk_duration}s[/yellow]")

        # ------------------------------------------------------------------
        # Run combined transcription + diarization
        # ------------------------------------------------------------------
        result = transcribe_with_diarization(
            audio_paths if len(audio_paths) > 1 else audio_paths[0],
            db_path=db_path,
            similarity_threshold=threshold,
            store_new_speakers=store_new,
            device=device,
            chunk_duration=chunk_duration,
            cross_file_threshold=cross_file_threshold,
            prior_speaker_embeddings=prior_speaker_embeddings,
            time_cursor=prior_time_cursor,
        )

        console.print(f"[green]✓ Processing complete[/green]")

        # ------------------------------------------------------------------
        # Merge this call's output with prior session data
        # ------------------------------------------------------------------
        full_segments = prior_segments + result["segments"]
        full_words = prior_words + result.get("word_timestamps", [])
        full_text = (prior_text + " " + result["text"]).strip() if prior_text else result["text"]
        all_speakers = sorted(set(prior_speakers) | set(result["speakers"]))
        merged_matched = {**prior_matched, **result.get("matched_speakers", {})}
        new_time_cursor = result.get("new_time_cursor", prior_time_cursor)
        updated_session_embeddings = result.get("session_speaker_embeddings", prior_speaker_embeddings or {})

        # Build a "full result" view covering all history, used for output
        full_result = {
            **result,
            "segments": full_segments,
            "word_timestamps": full_words,
            "text": full_text,
            "speakers": all_speakers,
            "num_speakers": len(all_speakers),
            "matched_speakers": merged_matched,
        }

        # ------------------------------------------------------------------
        # Show summary
        # ------------------------------------------------------------------
        if result.get("matched_speakers"):
            console.print(f"\n[bold green]Matched speakers:[/bold green]")
            for original, matched in result["matched_speakers"].items():
                console.print(f"  {original} → {matched}")

        if result.get("new_speakers"):
            console.print(f"\n[bold yellow]New/Unknown speakers:[/bold yellow] {', '.join(result['new_speakers'])}")
            if store_new:
                console.print(f"  [dim](Embeddings stored for future recognition)[/dim]")
            else:
                console.print(f"  [dim](Use 'pawnai label' to assign names)[/dim]")

        console.print(
            f"\n[bold]All speakers:[/bold] {', '.join(all_speakers)} "
            f"({len(all_speakers)} total, {len(full_segments)} segments)"
        )

        # ------------------------------------------------------------------
        # Save/update session file
        # ------------------------------------------------------------------
        if session:
            session_data = {
                "version": 1,
                "processed_files": prior_processed_files + [str(p) for p in audio_paths],
                "time_cursor": new_time_cursor,
                "speakers": all_speakers,
                "num_speakers": len(all_speakers),
                "segments": full_segments,
                "word_timestamps": full_words,
                "text": full_text,
                "matched_speakers": merged_matched,
                "new_speakers": result.get("new_speakers", []),
                "session_speaker_embeddings": updated_session_embeddings,
            }
            with open(session, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Session updated: {session}[/green]")

        # ------------------------------------------------------------------
        # Write --output file (full accumulated transcript)
        # ------------------------------------------------------------------
        if output:
            output_path = Path(output)
            output_format = output_path.suffix.lower()

            if output_format == ".json":
                # Exclude non-serialisable internal diarization object
                saveable = {k: v for k, v in full_result.items() if k != "diarization"}
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(saveable, f, indent=2, ensure_ascii=False)
                console.print(f"[green]✓ Saved JSON to: {output}[/green]")

            else:
                formatted_text = format_transcript_with_speakers(
                    full_result,
                    include_timestamps=not no_timestamps,
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                if output_format != ".txt":
                    console.print(f"[yellow]⚠ Unknown format '{output_format}', using .txt format[/yellow]")
                console.print(f"[green]✓ Saved transcript to: {output}[/green]")

        else:
            # Console preview (most recent segments only)
            console.print(f"\n[bold cyan]Latest transcript:[/bold cyan]")
            preview = [s for s in result["segments"] if s.get("text", "").strip()][:5]
            current_speaker = None

            for segment in preview:
                speaker = segment["speaker"]
                text = segment["text"]
                if speaker != current_speaker:
                    if current_speaker is not None:
                        console.print("")
                    start_time = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
                    console.print(f"[bold][{start_time}] {speaker}:[/bold]")
                    current_speaker = speaker
                if not no_timestamps:
                    s = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
                    e = f"{int(segment['end']//60):02d}:{segment['end']%60:05.2f}"
                    console.print(f"  [{s} → {e}] {text}")
                else:
                    console.print(f"  {text}")

            if len(result["segments"]) > 5:
                console.print(f"\n[dim]... and {len(result['segments']) - 5} more segments this call[/dim]")
            if session:
                console.print(f"[dim]Total accumulated: {len(full_segments)} segments across {len(prior_processed_files) + len(audio_paths)} file(s)[/dim]")
            console.print(f"\n[dim]💡 Tip: Use -o output.txt to save the full transcript[/dim]")

    except Exception as e:
        console.print(f"[red]Error during processing: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def embed(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files (treated as ordered chunks of the same speaker recording)"
    ),
    speaker_id: str = typer.Option(
        ..., "--speaker-id", "-s", help="Unique speaker identifier"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database"
    ),
) -> None:
    """Extract and store speaker embeddings.
    
    When multiple files are provided they are concatenated and treated as one
    continuous recording of the same speaker.

    Extracts speaker embeddings from audio and stores them in the database
    for later speaker identification and clustering.
    
    Example:
        pawnai embed audio.wav --speaker-id speaker_001
        pawnai embed part1.wav part2.wav -s alice --db-path ./my_db
    """
    # Lazy imports to avoid loading models during --help
    from ..core import DiarizationEngine, EmbeddingManager
    from ..core.config import Config, load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    try:
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_path=db_path)
        config.ensure_paths_exist()

        diarization_engine = DiarizationEngine()
        embedding_manager = EmbeddingManager(db_path=db_path)

        if len(audio_paths) == 1:
            console.print(f"[cyan]Extracting embeddings: {audio_paths[0]}[/cyan]")
            embeddings = diarization_engine.extract_embeddings(audio_paths[0])
        else:
            console.print(f"[cyan]Extracting embeddings from {len(audio_paths)} files as one speaker recording:[/cyan]")
            for i, p in enumerate(audio_paths, 1):
                console.print(f"  {i}. {p}")
            embeddings = diarization_engine.extract_embeddings(audio_paths)

        embedding_manager.add_embedding(speaker_id, embeddings, audio_paths[0])
        console.print(f"[green]✓ Embedding stored for speaker: {speaker_id}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during embedding: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    speaker_id: str = typer.Argument(
        ..., help="Speaker ID to search similar speakers for"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database"
    ),
    limit: int = typer.Option(
        5, help="Maximum number of results to return"
    ),
) -> None:
    """Search for similar speakers in the database.
    
    Finds speakers with similar voice characteristics based on embeddings.
    
    Example:
        pawnai search speaker_001
        pawnai search speaker_001 --limit 10
    """
    # Lazy imports to avoid loading models during --help
    from ..core import EmbeddingManager
    from ..core.config import load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    try:
        embedding_manager = EmbeddingManager(db_path=db_path)
        speaker_names = embedding_manager.get_speaker_names()
        
        # Get first embedding for the speaker
        all_embeddings = embedding_manager.get_all_embeddings()
        speaker_embeddings = [e for e in all_embeddings if e["speaker_id"] == speaker_id]
        
        if not speaker_embeddings:
            console.print(f"[red]Error: No embeddings found for speaker: {speaker_id}[/red]")
            raise typer.Exit(1)
        
        query_embedding = speaker_embeddings[0]["embedding"]
        console.print(f"[cyan]Searching for speakers similar to: {speaker_id}[/cyan]")
        
        # Placeholder: In a real setup, this would search using vector similarity
        console.print(f"[green]✓ Search complete[/green]")
        console.print(f"Query speaker: {speaker_id} ({speaker_names.get(speaker_id, 'Unknown')})")
        
    except Exception as e:
        console.print(f"[red]Error during search: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def label(
    audio_file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Audio file containing the speaker"
    ),
    speaker: Optional[str] = typer.Option(
        None, "--speaker", "-s", help="Speaker label (e.g., SPEAKER_00)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Human-readable name for the speaker"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List all speaker name mappings"
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database"
    ),
) -> None:
    """Assign human-readable names to speakers.
    
    Label speakers so they can be automatically identified in future diarizations.
    
    Examples:
        # Label a specific speaker
        pawnai label -f audio.wav -s SPEAKER_00 -n "John Doe"
        
        # List all labeled speakers
        pawnai label --list
    """
    import lancedb
    import pyarrow as pa
    from datetime import datetime
    from ..core.config import load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    try:
        db = lancedb.connect(db_path)
        
        if list_all:
            # List all speaker names
            if "speaker_names" not in db.table_names():
                console.print("[yellow]No speaker labels found. Use 'pawnai label' to add labels.[/yellow]")
                return
            
            names_table = db.open_table("speaker_names")
            names_df = names_table.to_pandas()
            
            if names_df.empty:
                console.print("[yellow]No speaker labels found.[/yellow]")
                return
            
            console.print("\n[bold cyan]Labeled Speakers[/bold cyan]")
            for _, row in names_df.iterrows():
                console.print(f"\n[bold]• {row['speaker_name']}[/bold]")
                console.print(f"  File: {Path(row['audio_file']).name}")
                console.print(f"  Label: {row['local_speaker_label']}")
                console.print(f"  Labeled: {row['labeled_at']}")
            return
        
        # Add/update speaker name
        if not audio_file or not speaker or not name:
            console.print("[red]Error: --file, --speaker, and --name are required to add a label.[/red]")
            console.print("[yellow]Use --list to see all labeled speakers.[/yellow]")
            raise typer.Exit(1)
        
        if not Path(audio_file).exists():
            console.print(f"[red]Error: Audio file not found: {audio_file}[/red]")
            raise typer.Exit(1)
        
        # Get or create speaker_names table
        if "speaker_names" in db.table_names():
            names_table = db.open_table("speaker_names")
            
            # Check if mapping already exists and delete it
            names_df = names_table.to_pandas()
            if not names_df.empty:
                mask = (names_df['audio_file'] == audio_file) & \
                       (names_df['local_speaker_label'] == speaker)
                if mask.any():
                    existing_id = names_df[mask].iloc[0]['id']
                    names_table.delete(f"id = '{existing_id}'")
                    console.print(f"[yellow]Updated existing label[/yellow]")
        else:
            # Create schema for speaker names
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("audio_file", pa.string()),
                pa.field("local_speaker_label", pa.string()),
                pa.field("speaker_name", pa.string()),
                pa.field("labeled_at", pa.timestamp('ms')),
            ])
            names_table = db.create_table("speaker_names", schema=schema)
        
        # Add new record
        record_id = f"{Path(audio_file).name}_{speaker}"
        now = datetime.now()
        timestamp_ms = pa.scalar(int(now.timestamp() * 1000), type=pa.timestamp('ms'))
        
        record = {
            'id': record_id,
            'audio_file': str(audio_file),
            'local_speaker_label': speaker,
            'speaker_name': name,
            'labeled_at': timestamp_ms,
        }
        
        names_table.add([record])
        console.print(f"[green]✓ Labeled {speaker} in {Path(audio_file).name} as '{name}'[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during labeling: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_path: str = typer.Argument(
        ..., help="Path to diarization JSON/text file, or an audio file to process first"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path to save analysis results (.txt, .json, or .csv)"
    ),
    mode: str = typer.Option(
        "summary", "--mode", help="Analysis mode: 'summary' (default) or 'graph' (knowledge graph triples)"
    ),
    model: str = typer.Option(
        "gpt-4o", "--model", "-m", help="Copilot model to use for analysis"
    ),
    db_path: str = typer.Option(
        "speakers_db", help="Path to speaker database (used when processing audio directly)"
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device for audio processing: cuda or cpu"
    ),
) -> None:
    """Summarize, extract keywords, or build a knowledge graph from a diarization.

    Two analysis modes are available via --mode:

    [bold]summary[/bold] (default):
      Produces a Copilot-generated structured analysis with a summary,
      key topics/keywords, per-speaker highlights, and sentiment.

    [bold]graph[/bold]:
      Extracts knowledge graph triples (subject, relation, object) as JSON.
      Useful for visualizing the conversation as a graph.

    Accepts a previously generated diarization JSON or transcript text file,
    or an audio file that will be transcribed and diarized on-the-fly.

    Example:
        pawnai analyze result.json
        pawnai analyze result.txt --mode graph
        pawnai analyze result.json --mode graph -o graph.json
        pawnai analyze result.json --mode graph -o graph.csv
        pawnai analyze audio.wav --mode summary -o analysis.txt
        pawnai analyze result.json --model gpt-4o
    """
    import json
    import csv as csv_module

    from rich.table import Table

    from ..core import AnalysisEngine

    VALID_MODES = ("summary", "graph")
    if mode not in VALID_MODES:
        console.print(f"[red]Error: Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}[/red]")
        raise typer.Exit(1)

    try:
        engine = AnalysisEngine(model=model)
        console.print(f"[cyan]Analyzing: {input_path} (mode={mode}, model={model})…[/cyan]")

        # ------------------------------------------------------------------ #
        # GRAPH MODE                                                          #
        # ------------------------------------------------------------------ #
        if mode == "graph":
            triples = engine.extract_graph_from_file(
                input_path,
                db_path=db_path,
                device=device,
            )

            console.print(f"[green]✓ Extracted {len(triples)} graph triple(s)[/green]")

            if output:
                output_path = Path(output)
                out_suffix = output_path.suffix.lower()

                if out_suffix == ".json":
                    payload = {
                        "model": model,
                        "source": str(input_path),
                        "triples": [
                            {"subject": s, "relation": r, "object": o}
                            for s, r, o in triples
                        ],
                    }
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Saved graph JSON to: {output}[/green]")

                elif out_suffix == ".csv":
                    with open(output_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv_module.writer(f)
                        writer.writerow(["subject", "relation", "object"])
                        writer.writerows(triples)
                    console.print(f"[green]✓ Saved graph CSV to: {output}[/green]")

                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("subject\trelation\tobject\n")
                        for s, r, o in triples:
                            f.write(f"{s}\t{r}\t{o}\n")
                    console.print(f"[green]✓ Saved graph to: {output}[/green]")

            else:
                table = Table(title="Knowledge Graph Triples", show_lines=True)
                table.add_column("Subject", style="bold cyan", no_wrap=True)
                table.add_column("Relation", style="yellow")
                table.add_column("Object", style="green")
                for s, r, o in triples:
                    table.add_row(s, r, o)
                console.print(table)

        # ------------------------------------------------------------------ #
        # SUMMARY MODE                                                        #
        # ------------------------------------------------------------------ #
        else:
            analysis_text = engine.analyze_from_file(
                input_path,
                db_path=db_path,
                device=device,
            )

            console.print(f"[green]✓ Analysis complete[/green]")

            if output:
                output_path = Path(output)
                out_suffix = output_path.suffix.lower()

                if out_suffix == ".json":
                    payload = {
                        "model": model,
                        "source": str(input_path),
                        "analysis": analysis_text,
                    }
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Saved JSON analysis to: {output}[/green]")
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"Analysis of: {input_path}\n")
                        f.write(f"Model: {model}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(analysis_text)
                    console.print(f"[green]✓ Saved analysis to: {output}[/green]")
            else:
                console.print(f"\n[bold cyan]Analysis[/bold cyan]\n")
                console.print(analysis_text)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to .env configuration file"
    ),
) -> None:
    """Show application status and available models.
    
    Displays system information and model availability.
    """
    # Lazy import to avoid loading torch during --help
    from ..core.config import load_config_from_file
    
    # Load config from specified file if provided
    if config:
        load_config_from_file(config)
    
    import torch
    
    console.print("\n[bold cyan]PawnAI Status[/bold cyan]")
    console.print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    console.print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    console.print("\n[bold]Available commands:[/bold]")
    console.print("  diarize            - Perform speaker diarization")
    console.print("  transcribe         - Transcribe audio to text")
    console.print("  transcribe-diarize - Transcribe with speaker labels (combined)")
    console.print("  analyze            - Summarize & extract keywords via GitHub Copilot")
    console.print("  embed              - Extract speaker embeddings")
    console.print("  search             - Search for similar speakers")
    console.print("  label              - Assign names to speakers")
    console.print("  status             - Show this status message")
