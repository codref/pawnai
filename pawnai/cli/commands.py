"""CLI command definitions for PawnAI."""

import shutil
from typing import Any, List, Optional, Tuple
import typer
from pathlib import Path

from .utils import console
from ..core.config import DEFAULT_DB_DSN

app = typer.Typer(
    help="PawnAI: Speaker diarization and transcription CLI tool",
    rich_markup_mode="rich",
)


def _resolve_s3_paths(
    paths: List[str],
    app_cfg: Any,
) -> Tuple[List[str], List[str]]:
    """Download any ``s3://`` paths to temporary local files.

    Paths that are already local are passed through unchanged.  The caller
    is responsible for deleting the returned temp files when processing is
    complete (typically in a ``finally`` block).

    Args:
        paths: List of file paths, which may contain ``s3://`` URIs.
        app_cfg: Active :class:`~pawnai.core.config.AppConfig` instance used
            to read the ``s3:`` configuration section.

    Returns:
        Tuple of ``(resolved_paths, temp_files)`` where *resolved_paths* has
        ``s3://`` entries replaced by local temp file paths and *temp_files*
        is the list of temporary file paths to delete after processing.

    Raises:
        typer.Exit: If any ``s3://`` URI is present but no ``s3:`` section is
            configured in ``.pawnai.yml``.
    """
    import tempfile
    from ..core.s3 import is_s3_path, S3Client, parse_s3_uri, expand_s3_glob

    has_s3 = any(is_s3_path(p) for p in paths)
    if not has_s3:
        return paths, []

    s3_cfg = app_cfg.get_s3_config()
    if s3_cfg is None:
        console.print(
            "[red]Error: S3 paths require an 's3:' section in .pawnai.yml[/red]"
        )
        raise typer.Exit(1)

    client = S3Client.from_dict(s3_cfg)

    # Resolve base download directory (configured or OS /tmp)
    _dl_dir_cfg = s3_cfg.get("download_dir")
    if _dl_dir_cfg:
        _base_dir = Path(_dl_dir_cfg)
        _base_dir.mkdir(parents=True, exist_ok=True)
    else:
        _base_dir = None  # tempfile will use the OS default (/tmp)

    # Create a single randomly-named subdirectory for this invocation so that
    # downloaded files keep their original S3 key name (e.g. 260224183013_01.flac)
    # and the speaker ID stored in PostgreSQL is derived from the original filename.
    tmp_dir = Path(tempfile.mkdtemp(prefix="pawnai_s3_", dir=str(_base_dir) if _base_dir else None))

    # Expand any wildcard URIs before downloading
    expanded_paths: List[str] = []
    for path in paths:
        if is_s3_path(path) and any(c in path for c in ("*", "?", "[")):
            matches = expand_s3_glob(path, client)
            if not matches:
                console.print(f"[yellow]Warning: no S3 objects matched {path!r}[/yellow]")
            else:
                console.print(
                    f"[cyan]Expanded {path!r} → {len(matches)} file(s)[/cyan]"
                )
                expanded_paths.extend(matches)
        else:
            expanded_paths.append(path)

    # temps holds the tmp_dir path so the caller can delete it in finally
    temps: List[str] = [str(tmp_dir)]
    resolved: List[str] = []

    for path in expanded_paths:
        if not is_s3_path(path):
            resolved.append(path)
            continue

        bucket, key = parse_s3_uri(path, configured_bucket=client.bucket)
        # Preserve the original filename so speaker IDs in PostgreSQL are meaningful
        local_path = tmp_dir / Path(key).name
        console.print(f"[cyan]Downloading s3://{bucket}/{key} …[/cyan]")
        client.download_file(key, str(local_path), bucket=bucket)
        resolved.append(str(local_path))

    return resolved, temps


@app.command()
def diarize(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to diarize (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    from ..core.config import Config, AppConfig

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps = _resolve_s3_paths(audio_paths, app_cfg)
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_dsn=db_dsn)
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
            db_dsn=db_dsn,
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
    finally:
        for _f in _s3_temps:
            shutil.rmtree(_f, ignore_errors=True)


@app.command()
def transcribe(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to transcribe (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
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
    from ..core.config import AppConfig

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps = _resolve_s3_paths(audio_paths, app_cfg)
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
    finally:
        for _f in _s3_temps:
            shutil.rmtree(_f, ignore_errors=True)


@app.command(name="transcribe-diarize")
def transcribe_diarize(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files to process (treated as ordered chunks of one conversation)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (format inferred from extension: .txt or .json)"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Path to a session JSON file.  If the file exists its speaker state and "
             "timestamps are loaded so new audio is appended to the same conversation.  "
             "After processing the file is updated with the full accumulated results."
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show verbose output from NeMo and other libraries"
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
    from ..core.config import Config, AppConfig

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps = _resolve_s3_paths(audio_paths, app_cfg)
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_dsn=db_dsn)
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
            db_dsn=db_dsn,
            similarity_threshold=threshold,
            store_new_speakers=store_new,
            device=device,
            chunk_duration=chunk_duration,
            cross_file_threshold=cross_file_threshold,
            prior_speaker_embeddings=prior_speaker_embeddings,
            time_cursor=prior_time_cursor,
            verbose=verbose,
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
    finally:
        for _f in _s3_temps:
            shutil.rmtree(_f, ignore_errors=True)


@app.command()
def embed(
    audio_paths: List[str] = typer.Argument(
        ..., help="One or more audio files (treated as ordered chunks of the same speaker recording)"
    ),
    speaker_id: str = typer.Option(
        ..., "--speaker-id", "-s", help="Unique speaker identifier"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    from ..core.config import Config, AppConfig

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps = _resolve_s3_paths(audio_paths, app_cfg)
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        config = Config(db_dsn=db_dsn)
        config.ensure_paths_exist()

        diarization_engine = DiarizationEngine()
        embedding_manager = EmbeddingManager(db_dsn=db_dsn)

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
    finally:
        for _f in _s3_temps:
            shutil.rmtree(_f, ignore_errors=True)


@app.command()
def search(
    speaker_id: str = typer.Argument(
        ..., help="Speaker ID to search similar speakers for"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    from ..core.config import AppConfig
    
    # Load config from specified file if provided
    if config:
        AppConfig(config_path=config)
    
    try:
        embedding_manager = EmbeddingManager(db_dsn=db_dsn)
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
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List all speaker name mappings"
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    from datetime import datetime, timezone
    from ..core.config import AppConfig
    from ..core.database import SpeakerName, get_engine, get_session, init_db
    from sqlalchemy import select

    # Load config from specified file if provided
    if config:
        AppConfig(config_path=config)

    try:
        engine = get_engine(db_dsn)
        init_db(engine)

        if list_all:
            with get_session(engine) as session:
                rows = session.execute(select(SpeakerName)).scalars().all()
            if not rows:
                console.print("[yellow]No speaker labels found. Use 'pawnai label' to add labels.[/yellow]")
                return
            console.print("\n[bold cyan]Labeled Speakers[/bold cyan]")
            for row in rows:
                console.print(f"\n[bold]• {row.speaker_name}[/bold]")
                console.print(f"  File: {Path(row.audio_file).name}")
                console.print(f"  Label: {row.local_speaker_label}")
                console.print(f"  Labeled: {row.labeled_at}")
            return

        # Add/update speaker name
        if not audio_file or not speaker or not name:
            console.print("[red]Error: --file, --speaker, and --name are required to add a label.[/red]")
            console.print("[yellow]Use --list to see all labeled speakers.[/yellow]")
            raise typer.Exit(1)

        if not Path(audio_file).exists():
            console.print(f"[red]Error: Audio file not found: {audio_file}[/red]")
            raise typer.Exit(1)

        record_id = f"{Path(audio_file).name}_{speaker}"
        record = SpeakerName(
            id=record_id,
            audio_file=str(audio_file),
            local_speaker_label=speaker,
            speaker_name=name,
            labeled_at=datetime.now(timezone.utc),
        )
        with get_session(engine) as session:
            existing = session.get(SpeakerName, record_id)
            if existing:
                console.print("[yellow]Updated existing label[/yellow]")
            session.merge(record)
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
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database (used when processing audio directly)"
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
                db_dsn=db_dsn,
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
                db_dsn=db_dsn,
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
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
) -> None:
    """Show application status and available models.
    
    Displays system information and model availability.
    """
    # Lazy import to avoid loading torch during --help
    from ..core.config import AppConfig
    
    # Load config from specified file if provided
    if config:
        AppConfig(config_path=config)
    
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
    console.print("  s3-ls              - List objects in the configured S3 bucket")
    console.print("  status             - Show this status message")


@app.command(name="s3-ls")
def s3_ls(
    prefix: Optional[str] = typer.Argument(
        None, help="Optional key prefix to filter results (e.g. 'recordings/2024/')"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="List all objects recursively (no common-prefix grouping)"
    ),
    long: bool = typer.Option(
        False, "--long", "-l", help="Show size and last-modified date for each object"
    ),
) -> None:
    """List objects in the configured S3 bucket.

    Reads S3 credentials from the ``s3:`` section of ``.pawnai.yml`` (or the
    file passed via ``--config``).  Without a prefix all top-level keys are
    listed; with a prefix only keys that start with that prefix are shown.

    By default the listing uses delimiter ``/`` so that common prefixes
    (virtual directories) are grouped, similar to ``aws s3 ls``.  Pass
    ``--recursive`` to list every object under the prefix without grouping.

    Example:
        pawnai s3-ls
        pawnai s3-ls recordings/
        pawnai s3-ls recordings/2024/ --long
        pawnai s3-ls --recursive --long
    """
    from ..core.config import AppConfig
    from ..core.s3 import S3Client

    app_cfg = AppConfig(config_path=config) if config else AppConfig()

    s3_cfg = app_cfg.get_s3_config()
    if s3_cfg is None:
        console.print(
            "[red]Error: no 's3:' section found in .pawnai.yml – "
            "add S3 credentials before using s3-ls[/red]"
        )
        raise typer.Exit(1)

    client = S3Client.from_dict(s3_cfg)

    if not client.check_bucket():
        console.print(
            f"[red]Error: bucket '{client.bucket}' is not accessible – "
            "check credentials and endpoint_url[/red]"
        )
        raise typer.Exit(1)

    # ── Build list_objects_v2 kwargs ─────────────────────────────────────────
    list_kwargs: dict = {"Bucket": client.bucket}
    if prefix:
        list_kwargs["Prefix"] = prefix
    if not recursive:
        list_kwargs["Delimiter"] = "/"

    console.print(
        f"\n[bold cyan]s3://{client.bucket}"
        f"{'/' + prefix.lstrip('/') if prefix else ''}[/bold cyan]\n"
    )

    # ── Paginate ─────────────────────────────────────────────────────────────
    # Access the underlying boto3 client via the private attribute (same pattern
    # used in pawnai_recorder's S3Uploader for head_bucket / upload_file).
    boto_client = client._client  # noqa: SLF001

    paginator = boto_client.get_paginator("list_objects_v2")

    total_objects = 0
    total_bytes = 0

    try:
        for page in paginator.paginate(**list_kwargs):
            # Virtual directories (common prefixes)
            for cp in page.get("CommonPrefixes") or []:
                console.print(f"  [bold blue]{cp['Prefix']}[/bold blue]")

            # Objects
            for obj in page.get("Contents") or []:
                total_objects += 1
                total_bytes += obj["Size"]
                key = obj["Key"]

                if long:
                    size_str = _fmt_size(obj["Size"])
                    modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")
                    console.print(
                        f"  [green]{modified}[/green]  "
                        f"[yellow]{size_str:>10}[/yellow]  {key}"
                    )
                else:
                    console.print(f"  {key}")

    except Exception as e:
        console.print(f"[red]Error listing bucket: {e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n[dim]Total: {total_objects} object(s), {_fmt_size(total_bytes)}[/dim]"
    )


def _fmt_size(num_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes //= 1024
    return f"{num_bytes:.1f} PB"

