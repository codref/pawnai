"""CLI command definitions for PawnAI."""

import shutil
from typing import Any, Dict, List, Optional, Tuple
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
) -> Tuple[List[str], List[str], Dict[str, str]]:
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
    # Maps local temp path → original S3 URI (or path → path for local files)
    path_map: Dict[str, str] = {}

    for path in expanded_paths:
        if not is_s3_path(path):
            path_map[path] = path
            resolved.append(path)
            continue

        bucket, key = parse_s3_uri(path, configured_bucket=client.bucket)
        # Preserve the original filename so speaker IDs in PostgreSQL are meaningful
        local_path = tmp_dir / Path(key).name
        console.print(f"[cyan]Downloading s3://{bucket}/{key} …[/cyan]")
        client.download_file(key, str(local_path), bucket=bucket)
        # Map local temp path → original S3 URI so callers store canonical paths in DB
        path_map[str(local_path)] = path
        resolved.append(str(local_path))

    return resolved, temps, path_map


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
        audio_paths, _s3_temps, _path_map = _resolve_s3_paths(audio_paths, app_cfg)
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
            store_new_speakers=store_new,
            source_map=_path_map,
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
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Session name for grouping transcript segments in the database."
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
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
    backend: str = typer.Option(
        "nemo", "--backend", "-b",
        help="Transcription backend: 'nemo' (Parakeet/NeMo) or 'whisper' (faster-whisper large-v3)"
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
        pawnai transcribe audio.wav --backend whisper
    """
    # Lazy imports to avoid loading models during --help
    import json
    import uuid
    from ..core import TranscriptionEngine
    from ..core.config import AppConfig
    from ..core.database import get_engine, init_db, save_transcription_segments

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps, _path_map = _resolve_s3_paths(audio_paths, app_cfg)
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        session_id: str = session if session else str(uuid.uuid4())
        db_engine = get_engine(db_dsn)
        init_db(db_engine)

        engine_t = TranscriptionEngine(device=device, backend=backend)

        if len(audio_paths) == 1:
            console.print(f"[cyan]Transcribing: {audio_paths[0]}[/cyan]")
        else:
            console.print(f"[cyan]Transcribing {len(audio_paths)} files as one conversation:[/cyan]")
            for i, p in enumerate(audio_paths, 1):
                console.print(f"  {i}. {p}")

        if chunk_duration:
            console.print(f"[yellow]Chunk duration set to: {chunk_duration}s (will split long audio)[/yellow]")

        if len(audio_paths) == 1:
            results = engine_t.transcribe(
                audio_paths, include_timestamps=with_timestamps, chunk_duration=chunk_duration
            )
            if not results:
                console.print(f"[yellow]⚠ No transcription results[/yellow]")
                return
            result = results[0]
        else:
            result = engine_t.transcribe_conversation(
                audio_paths, include_timestamps=with_timestamps, chunk_duration=chunk_duration
            )
        console.print(f"[green]✓ Transcription complete[/green]")

        # Save segments to database
        segs = result.get("segment_timestamps", [])
        _canonical_first = _path_map.get(audio_paths[0], audio_paths[0]) if audio_paths else ""
        for seg in segs:
            seg.setdefault("source_file", _canonical_first if len(audio_paths) == 1 else "")
            if "segment" in seg and "text" not in seg:
                seg["text"] = seg["segment"]
        saved = save_transcription_segments(segs, session_id=session_id, engine=db_engine)
        console.print(f"[green]✓ Saved {saved} segment(s) to database [session: {session_id}][/green]")
        
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
        help="Session name for continuing a conversation across separate invocations. "
             "Speaker state and timestamps are loaded from the database and updated "
             "after processing so new audio is appended to the same conversation."
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
    backend: str = typer.Option(
        "nemo", "--backend", "-b",
        help="Transcription backend: 'nemo' (Parakeet/NeMo) or 'whisper' (faster-whisper large-v3)"
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
      pawnai transcribe-diarize part1.flac --session myconv

      # Second call – picks up where the first left off
      pawnai transcribe-diarize part2.flac --session myconv

      # Write the full accumulated transcript to a file
      pawnai transcribe-diarize part3.flac --session myconv -o full.txt

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
    import uuid
    from ..core import transcribe_with_diarization, format_transcript_with_speakers
    from ..core.config import Config, AppConfig
    from ..core.database import (
        get_engine, init_db,
        load_session_state, save_session_state, save_transcription_segments,
    )

    # Load config; capture instance so S3 config is accessible
    app_cfg = AppConfig(config_path=config) if config else AppConfig()
    _s3_temps: List[str] = []
    try:
        audio_paths, _s3_temps, _path_map = _resolve_s3_paths(audio_paths, app_cfg)
        for p in audio_paths:
            if not Path(p).exists():
                console.print(f"[red]Error: Audio file not found: {p}[/red]")
                raise typer.Exit(1)

        cfg = Config(db_dsn=db_dsn)
        cfg.ensure_paths_exist()
        db_engine = get_engine(db_dsn)
        init_db(db_engine)

        # ------------------------------------------------------------------
        # Resolve session_id and load prior state from DB when --session given
        # ------------------------------------------------------------------
        session_id: str = session if session else str(uuid.uuid4())
        prior_speaker_embeddings: Optional[dict] = None
        prior_time_cursor: float = 0.0
        prior_segments: list = []
        prior_words: list = []
        prior_text: str = ""
        prior_speakers: list = []
        prior_matched: dict = {}
        prior_processed_files: list = []
        prior_segment_count: int = 0

        if session:
            prior_speaker_embeddings, prior_time_cursor, prior_processed_files, prior_segment_count = \
                load_session_state(session_id, db_engine)
            if prior_processed_files:
                console.print(f"[cyan]Resuming session '{session_id}': "
                              f"{len(prior_processed_files)} file(s) already processed, "
                              f"t={prior_time_cursor:.1f}s, "
                              f"{prior_segment_count} segment(s) stored[/cyan]")
            else:
                console.print(f"[cyan]Starting new session '{session_id}'[/cyan]")

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
            backend=backend,
            source_map=_path_map,
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
        # Persist segments to DB (always) and session state (when --session)
        # ------------------------------------------------------------------
        saved = save_transcription_segments(
            result["segments"],
            session_id=session_id,
            engine=db_engine,
            start_index=prior_segment_count,
        )
        console.print(f"[green]✓ Saved {saved} segment(s) to database [session: {session_id}][/green]")

        if session:
            updated_processed = prior_processed_files + [str(p) for p in audio_paths]
            save_session_state(
                session_id=session_id,
                speaker_embeddings=updated_session_embeddings,
                time_cursor=new_time_cursor,
                processed_files=updated_processed,
                engine=db_engine,
            )
            console.print(f"[green]✓ Session state updated: '{session_id}'[/green]")

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
                total_segs = prior_segment_count + saved
                total_files = len(prior_processed_files) + len(audio_paths)
                console.print(f"[dim]Session '{session_id}': {total_segs} segments across {total_files} file(s) total[/dim]")
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
        audio_paths, _s3_temps, _path_map = _resolve_s3_paths(audio_paths, app_cfg)
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

        embedding_manager.add_embedding(speaker_id, embeddings, _path_map.get(audio_paths[0], audio_paths[0]))
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
        None, "--file", "-f", help="Audio file path or S3 URI (optional when --session resolves it)"
    ),
    speaker: Optional[str] = typer.Option(
        None, "--speaker", "-s", help="Speaker label (e.g., SPEAKER_00)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Human-readable name for the speaker"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="Session ID to scope listing/labelling to a specific session"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List speaker mappings (all sessions, or full session view with --session)"
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
    ),
) -> None:
    """Assign human-readable names to speakers.

    Use [bold]--session[/bold] to scope listing and labelling to a specific session,
    making it easy to identify unmapped speakers across multi-file recordings.

    \b
    Examples:
        # Show unmapped speakers in a session
        pawnai label --session my-session

        # Show all speakers in a session (mapped + unmapped)
        pawnai label --session my-session --list

        # Label a speaker from a session (auto-resolves file when unambiguous)
        pawnai label --session my-session --speaker SPEAKER_00 --name "Alice"

        # Label with explicit file (required when speaker label spans multiple files)
        pawnai label --session my-session -s SPEAKER_00 -f s3://bucket/audio.flac -n "Alice"

        # List all globally labeled speakers
        pawnai label --list

        # Label with an explicit local path
        pawnai label -f audio.wav -s SPEAKER_00 -n "John Doe"
    """
    from datetime import datetime, timezone
    from ..core.config import AppConfig
    from ..core.database import SpeakerName, TranscriptionSegment, get_engine, get_session, init_db
    from sqlalchemy import select

    if config:
        AppConfig(config_path=config)

    try:
        engine = get_engine(db_dsn)
        init_db(engine)

        # ------------------------------------------------------------------
        # Session-scoped listing (no --name means display-only mode)
        # ------------------------------------------------------------------
        if session and not name:
            with get_session(engine) as db_sess:
                # Unmapped = raw SPEAKER_XX labels not yet resolved to a human name
                unmapped_rows = db_sess.execute(
                    select(
                        TranscriptionSegment.audio_file,
                        TranscriptionSegment.original_speaker_label,
                    )
                    .where(
                        TranscriptionSegment.session_id == session,
                        TranscriptionSegment.original_speaker_label.like("SPEAKER_%"),
                    )
                    .distinct()
                ).all()

                mapped_rows: list = []
                if list_all:
                    mapped_rows = db_sess.execute(
                        select(
                            TranscriptionSegment.audio_file,
                            TranscriptionSegment.original_speaker_label,
                        )
                        .where(
                            TranscriptionSegment.session_id == session,
                            ~TranscriptionSegment.original_speaker_label.like("SPEAKER_%"),
                            TranscriptionSegment.original_speaker_label.isnot(None),
                        )
                        .distinct()
                    ).all()

            if not unmapped_rows and not mapped_rows:
                console.print(f"[yellow]No segments found for session '{session}'.[/yellow]")
                return

            console.print(f"\n[bold cyan]Session: {session}[/bold cyan]")

            if unmapped_rows:
                console.print(f"\n[bold yellow]Unmapped speakers ({len(unmapped_rows)}):[/bold yellow]")
                for seg_audio, seg_label in unmapped_rows:
                    console.print(f"\n  [bold]• {seg_label}[/bold]")
                    console.print(f"    File: {seg_audio}")
                    console.print(
                        f"    [dim]pawnai label --session {session} "
                        f"--speaker {seg_label} --name \"Name\"[/dim]"
                    )
            else:
                console.print(f"[green]All speakers in session '{session}' are labeled.[/green]")

            if mapped_rows:
                console.print(f"\n[bold green]Already resolved ({len(mapped_rows)}):[/bold green]")
                for seg_audio, seg_label in mapped_rows:
                    console.print(f"  • [bold]{seg_label}[/bold]  ({Path(seg_audio).name if seg_audio else ''})")
            return

        # ------------------------------------------------------------------
        # Global list mode (no session, --list only)
        # ------------------------------------------------------------------
        if list_all:
            with get_session(engine) as db_sess:
                rows = db_sess.execute(select(SpeakerName)).scalars().all()
            if not rows:
                console.print("[yellow]No speaker labels found. Use 'pawnai label' to add labels.[/yellow]")
                return
            console.print("\n[bold cyan]Labeled Speakers[/bold cyan]")
            for row in rows:
                console.print(f"\n[bold]• {row.speaker_name}[/bold]")
                console.print(f"  File: {row.audio_file}")
                console.print(f"  Label: {row.local_speaker_label}")
                console.print(f"  Labeled: {row.labeled_at}")
            return

        # ------------------------------------------------------------------
        # Add / update a speaker name mapping
        # ------------------------------------------------------------------
        if not speaker or not name:
            console.print("[red]Error: --speaker and --name are required to add a label.[/red]")
            console.print("[yellow]Use --list or --session <id> to see current speakers.[/yellow]")
            raise typer.Exit(1)

        # Resolve audio_file: use --file if given, otherwise auto-detect from session
        resolved_audio_file: Optional[str] = audio_file
        if not resolved_audio_file and session:
            with get_session(engine) as db_sess:
                candidates = db_sess.execute(
                    select(TranscriptionSegment.audio_file)
                    .where(
                        TranscriptionSegment.session_id == session,
                        TranscriptionSegment.original_speaker_label == speaker,
                    )
                    .distinct()
                ).scalars().all()

            if not candidates:
                console.print(
                    f"[red]Error: Speaker '{speaker}' not found in session '{session}'.[/red]"
                )
                raise typer.Exit(1)

            if len(candidates) > 1:
                console.print(
                    f"[yellow]Speaker '{speaker}' appears in {len(candidates)} files in "
                    f"session '{session}'. Use --file to specify which one:[/yellow]"
                )
                for c in candidates:
                    console.print(f"  {c}")
                raise typer.Exit(1)

            resolved_audio_file = candidates[0]

        if not resolved_audio_file:
            console.print("[red]Error: --file is required (or provide --session to auto-resolve).[/red]")
            raise typer.Exit(1)

        record_id = f"{Path(resolved_audio_file).name}_{speaker}"
        record = SpeakerName(
            id=record_id,
            audio_file=str(resolved_audio_file),
            local_speaker_label=speaker,
            speaker_name=name,
            labeled_at=datetime.now(timezone.utc),
        )
        with get_session(engine) as db_sess:
            existing = db_sess.get(SpeakerName, record_id)
            if existing:
                console.print("[yellow]Updated existing label[/yellow]")
            db_sess.merge(record)
        console.print(f"[green]✓ Labeled {speaker} in {Path(resolved_audio_file).name} as '{name}'[/green]")

    except Exception as e:
        console.print(f"[red]Error during labeling: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_path: Optional[str] = typer.Argument(
        None, help="Path to diarization JSON/text file, or an audio file to process first. Omit when using --session."
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Load transcript directly from the database by session ID instead of a file."
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
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database (used when processing audio directly or loading from DB)"
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
    an audio file that will be transcribed and diarized on-the-fly, or a
    session ID to load the transcript directly from the database.

    Example:
        pawnai analyze result.json
        pawnai analyze result.txt --mode graph
        pawnai analyze result.json --mode graph -o graph.json
        pawnai analyze result.json --mode graph -o graph.csv
        pawnai analyze audio.wav --mode summary -o analysis.txt
        pawnai analyze result.json --model gpt-4o
        pawnai analyze --session myconv
        pawnai analyze --session myconv --mode graph -o graph.json
    """
    import json
    import csv as csv_module

    from rich.table import Table

    from ..core import AnalysisEngine

    VALID_MODES = ("summary", "graph")
    if mode not in VALID_MODES:
        console.print(f"[red]Error: Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}[/red]")
        raise typer.Exit(1)

    # Validate input: exactly one of input_path or --session must be provided
    if session and input_path:
        console.print("[red]Error: provide either a file path or --session, not both.[/red]")
        raise typer.Exit(1)
    if not session and not input_path:
        console.print("[red]Error: provide a file path or --session.[/red]")
        raise typer.Exit(1)

    # Label used in output headers / log lines
    source_label = f"session:{session}" if session else input_path

    try:
        engine = AnalysisEngine(model=model)
        console.print(f"[cyan]Analyzing: {source_label} (mode={mode}, model={model})…[/cyan]")

        # ------------------------------------------------------------------ #
        # GRAPH MODE                                                          #
        # ------------------------------------------------------------------ #
        if mode == "graph":
            triples = engine.extract_graph_from_file(
                input_path or "",
                db_dsn=db_dsn,
                device=device,
                session_id=session,
            )

            console.print(f"[green]✓ Extracted {len(triples)} graph triple(s)[/green]")

            if output:
                output_path = Path(output)
                out_suffix = output_path.suffix.lower()

                if out_suffix == ".json":
                    payload = {
                        "model": model,
                        "source": source_label,
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
                input_path or "",
                db_dsn=db_dsn,
                device=device,
                session_id=session,
            )

            console.print(f"[green]✓ Analysis complete[/green]")

            if output:
                output_path = Path(output)
                out_suffix = output_path.suffix.lower()

                if out_suffix == ".json":
                    payload = {
                        "model": model,
                        "source": source_label,
                        "analysis": analysis_text,
                    }
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    console.print(f"[green]✓ Saved JSON analysis to: {output}[/green]")
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"Analysis of: {source_label}\n")
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


@app.command(name="sync-siyuan")
def sync_siyuan(
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Session ID to sync to SiYuan. Must have a completed analysis in the DB."
    ),
    all_sessions: bool = typer.Option(
        False, "--all", help="Sync every session that has a completed analysis."
    ),
    notebook: Optional[str] = typer.Option(
        None, "--notebook", "-n",
        help="Target SiYuan notebook ID. Falls back to siyuan.notebook in .pawnai.yml."
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t",
        help="SiYuan API token. Falls back to siyuan.token in .pawnai.yml."
    ),
    url: Optional[str] = typer.Option(
        None, "--url",
        help="SiYuan instance URL. Falls back to siyuan.url in .pawnai.yml, then http://127.0.0.1:6806."
    ),
    path_template: Optional[str] = typer.Option(
        None, "--path-template",
        help=(
            "Document path template. Placeholders: {session_id}, {title}, {date}, "
            "{year}, {month}, {day}. Defaults to /Conversations/{date}/{session_id}."
        ),
    ),
    daily_note: bool = typer.Option(
        True, "--daily-note/--no-daily-note",
        help="Also append a backlink in today's SiYuan daily note."
    ),
    daily_path_template: Optional[str] = typer.Option(
        None, "--daily-path-template",
        help=(
            "Daily note path template. Placeholders: {date}, {year}, {month}, {day}. "
            "Defaults to /daily note/{year}/{month}/{date}."
        ),
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)."
    ),
) -> None:
    """Push conversation analysis to a running SiYuan Note instance.

    Reads the most recent stored analysis for one or all sessions from the
    database (produced by a previous [bold]analyze[/bold] run) and creates
    a rich SiYuan document containing the structured analysis and full
    transcript.  Existing documents at the same path are overwritten.

    Configuration can be provided via [bold].pawnai.yml[/bold]:

    \\b
        siyuan:
          url: http://127.0.0.1:6806
          token: your_token
          notebook: 20210817205410-2kvfpfn
          path_template: "/Conversations/{date}/{session_id}"
          daily_note_path: "/daily note/{year}/{month}/{date}"

    Example:
        pawnai sync-siyuan --session myconv
        pawnai sync-siyuan --session myconv --notebook 20210817... --token xxx
        pawnai sync-siyuan --all
    """
    from datetime import datetime, timezone

    from ..core.config import AppConfig
    from ..core.database import get_engine, get_session_analysis, SessionAnalysis
    from ..core.siyuan import (
        SiyuanClient,
        SiyuanError,
        format_session_markdown,
        resolve_path_template,
        DEFAULT_PATH_TEMPLATE,
        DEFAULT_DAILY_PATH_TEMPLATE,
    )
    from ..core.analysis import AnalysisEngine
    from sqlalchemy import select
    from sqlalchemy.orm import Session as OrmSession

    # ── Load YAML config ───────────────────────────────────────────────────────
    app_cfg = AppConfig(config_path=config)
    sy_cfg = app_cfg.get_siyuan_config() or {}

    resolved_url = url or sy_cfg.get("url", "http://127.0.0.1:6806")
    resolved_token = token or sy_cfg.get("token", "")
    resolved_notebook = notebook or sy_cfg.get("notebook", "")
    resolved_path_tpl = path_template or sy_cfg.get("path_template", DEFAULT_PATH_TEMPLATE)
    resolved_daily_tpl = daily_path_template or sy_cfg.get("daily_note_path", DEFAULT_DAILY_PATH_TEMPLATE)

    if not resolved_notebook:
        console.print(
            "[red]Error: No SiYuan notebook ID provided. Use --notebook or set "
            "siyuan.notebook in .pawnai.yml.[/red]"
        )
        raise typer.Exit(1)

    if not resolved_token:
        console.print(
            "[yellow]Warning: No SiYuan API token provided. Requests may be "
            "rejected. Use --token or set siyuan.token in .pawnai.yml.[/yellow]"
        )

    if not session and not all_sessions:
        console.print("[red]Error: provide --session or --all.[/red]")
        raise typer.Exit(1)

    if session and all_sessions:
        console.print("[red]Error: provide either --session or --all, not both.[/red]")
        raise typer.Exit(1)

    # ── Collect analysis rows to sync ──────────────────────────────────────────
    engine = get_engine(db_dsn)

    if session:
        row = get_session_analysis(session, engine)
        if row is None:
            console.print(
                f"[red]Error: No analysis found for session {session!r}. "
                "Run 'pawnai analyze --session {session}' first.[/red]"
            )
            raise typer.Exit(1)
        rows_to_sync = [row]
    else:
        # --all: fetch the latest analysis for every distinct session_id
        from sqlalchemy import select, func as sqlfunc
        with OrmSession(engine) as db:
            # Subquery: max analyzed_at per session_id
            subq = (
                select(
                    SessionAnalysis.session_id,
                    sqlfunc.max(SessionAnalysis.analyzed_at).label("max_at"),
                )
                .where(SessionAnalysis.session_id.is_not(None))
                .group_by(SessionAnalysis.session_id)
                .subquery()
            )
            rows_to_sync = db.scalars(
                select(SessionAnalysis).join(
                    subq,
                    (SessionAnalysis.session_id == subq.c.session_id)
                    & (SessionAnalysis.analyzed_at == subq.c.max_at),
                )
            ).all()
            from sqlalchemy.orm import make_transient
            for r in rows_to_sync:
                db.expunge(r)
                make_transient(r)

    if not rows_to_sync:
        console.print("[yellow]No sessions with completed analyses found.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"[cyan]Syncing {len(rows_to_sync)} session(s) → SiYuan "
        f"{resolved_url} notebook={resolved_notebook}[/cyan]"
    )

    client = SiyuanClient(
        url=resolved_url,
        token=resolved_token,
        notebook_id=resolved_notebook,
    )

    ae = AnalysisEngine()
    now = datetime.now(timezone.utc)
    success = 0
    errors = 0

    for row in rows_to_sync:
        sid = row.session_id or row.source
        try:
            # ── Load transcript ────────────────────────────────────────────────
            try:
                transcript = ae._load_transcript(
                    "", db_dsn=db_dsn, session_id=row.session_id
                )
            except Exception:
                transcript = "_Transcript not available._"

            # ── Build Markdown ─────────────────────────────────────────────────
            md = format_session_markdown(
                title=row.title,
                summary=row.summary,
                key_topics=row.key_topics,
                speaker_highlights=row.speaker_highlights,
                sentiment=row.sentiment,
                sentiment_tags=row.sentiment_tags,
                tags=row.tags,
                session_id=row.session_id,
                source=row.source,
                analyzed_at=row.analyzed_at,
                transcript=transcript,
                model=row.model,
            )

            # ── Resolve document path ──────────────────────────────────────────
            doc_path = resolve_path_template(
                resolved_path_tpl,
                session_id=sid,
                title=row.title,
                now=row.analyzed_at or now,
            )

            # ── Build block attributes ─────────────────────────────────────────
            # 'tags' is SiYuan's native attribute read by the Tags panel.
            # Values must be comma-separated with no leading #.
            attrs: dict[str, str] = {"custom-pawnai-session": sid}
            all_tags: list[str] = list(row.tags or []) + list(row.sentiment_tags or [])
            if all_tags:
                attrs["tags"] = ",".join(all_tags)
            if row.model:
                attrs["custom-model"] = row.model

            # ── Upsert document ────────────────────────────────────────────────
            with console.status(f"[bold green]Syncing {sid!r}…"):
                doc_id = client.upsert_session_doc(
                    notebook=resolved_notebook,
                    path=doc_path,
                    markdown=md,
                    attrs=attrs,
                )

            console.print(
                f"[green]✓ {sid!r} → {doc_path} (doc_id={doc_id})[/green]"
            )

            # ── SiYuan toast notification + inbox message ─────────────────────
            try:
                client.push_msg(f"PawnAI: synced '{sid}'\n{doc_path}")
            except Exception:
                pass  # notification is best-effort
            try:
                client.create_shorthand(
                    f"**PawnAI sync** — [{row.title or sid}](siyuan://blocks/{doc_id})\n"
                    f"Session: `{sid}`  ·  Path: `{doc_path}`"
                )
            except Exception:
                pass  # inbox write is best-effort

            # ── Daily note backlink ────────────────────────────────────────────
            if daily_note and doc_id:
                daily_path = resolve_path_template(
                    resolved_daily_tpl,
                    session_id=sid,
                    title=None,
                    now=now,
                )
                try:
                    client.append_daily_note_link(
                        notebook=resolved_notebook,
                        daily_path=daily_path,
                        doc_id=doc_id,
                        title=row.title or sid,
                    )
                    console.print(
                        f"  [dim]↩ backlink added to daily note: {daily_path}[/dim]"
                    )
                except SiyuanError as e:
                    console.print(
                        f"  [yellow]Warning: could not add daily note backlink: {e}[/yellow]"
                    )
            success += 1

        except SiyuanError as e:
            console.print(f"[red]✗ {sid!r}: SiYuan API error — {e}[/red]")
            errors += 1
        except Exception as e:
            console.print(f"[red]✗ {sid!r}: {e}[/red]")
            errors += 1

    console.print(
        f"\n[bold]Done:[/bold] {success} synced, {errors} failed."
    )
    if errors:
        raise typer.Exit(1)


@app.command()
def sessions(
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Session ID to inspect. Shows head/tail of transcript, files, speakers, and timing."
    ),
    db_dsn: str = typer.Option(
        DEFAULT_DB_DSN, help="PostgreSQL DSN for speaker database"
    ),
    head: int = typer.Option(
        5, "--head", help="Number of first segments to show in detail view"
    ),
    tail: int = typer.Option(
        5, "--tail", help="Number of last segments to show in detail view"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Write the full session transcript to this file (requires --session)."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
) -> None:
    """List transcription sessions or inspect a specific one.

    Without --session prints a summary table of all sessions ordered by most
    recently updated.

    With --session prints detailed information for that session: metadata,
    audio files, speaker list, and a head/tail preview of the transcript.

    With --session and --output writes the complete transcript to a file.

    Example:
        pawnai sessions
        pawnai sessions --session myconv
        pawnai sessions --session ef11c094-0d8d-41fe-93c0-c6bf1f2e8663
        pawnai sessions --session myconv --head 10 --tail 10
        pawnai sessions --session myconv --output transcript.txt
    """
    from sqlalchemy import select
    from sqlalchemy import func as sqlfunc
    from sqlalchemy.orm import Session as OrmSession
    from rich.table import Table
    from rich.rule import Rule
    from rich.text import Text
    from ..core.database import (
        get_engine, init_db, SessionState, TranscriptionSegment, SpeakerName,
    )
    from ..core.config import AppConfig

    if config:
        AppConfig(config_path=config)

    # --output requires --session
    if output is not None and not session:
        console.print("[red]Error: --output requires --session to be specified.[/red]")
        raise typer.Exit(1)

    db_engine = get_engine(db_dsn)
    init_db(db_engine)

    # ── Detail view ──────────────────────────────────────────────────────────
    if session:
        with OrmSession(db_engine) as db:
            segs = db.execute(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session)
                .order_by(TranscriptionSegment.start_time)
            ).scalars().all()

            if not segs:
                console.print(f"[red]No segments found for session '{session}'.[/red]")
                raise typer.Exit(1)

            state = db.get(SessionState, session)

        # ── Metadata panel ────────────────────────────────────────────────
        files = sorted({s.audio_file for s in segs if s.audio_file})
        speakers = sorted({s.original_speaker_label for s in segs if s.original_speaker_label})
        duration = (segs[-1].end_time or 0.0) - (segs[0].start_time or 0.0)
        duration_str = f"{int(duration // 60)}m {duration % 60:.1f}s"
        last_up = max((s.created_at for s in segs if s.created_at), default=None)

        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="bold cyan", no_wrap=True)
        meta.add_column()
        meta.add_row("Session", session)
        meta.add_row("Segments", str(len(segs)))
        meta.add_row("Duration", duration_str)
        meta.add_row("Updated", last_up.strftime("%Y-%m-%d %H:%M:%S") if last_up else "—")
        if state:
            tc = state.time_cursor or 0.0
            meta.add_row("Time cursor", f"{int(tc // 60)}m {tc % 60:.1f}s")
        meta.add_row("Speakers", ", ".join(speakers) if speakers else "(none / transcribe-only)")
        meta.add_row("Files", "\n".join(f"• {f}" for f in files) if files else "—")

        console.print()
        console.print(Rule(f"[bold cyan]Session: {session}[/bold cyan]"))
        console.print(meta)

        # ── Segment preview helper ────────────────────────────────────────
        def _fmt_time(t: float) -> str:
            return f"{int(t // 60):02d}:{t % 60:05.2f}"

        def _print_segments(title: str, seg_list) -> None:
            tbl = Table(title=title, show_lines=True, header_style="bold")
            tbl.add_column("#", style="dim", justify="right", width=4)
            tbl.add_column("Time", no_wrap=True)
            tbl.add_column("Speaker", no_wrap=True)
            tbl.add_column("Text")
            for s in seg_list:
                time_str = f"{_fmt_time(s.start_time)} → {_fmt_time(s.end_time)}"
                lbl = s.original_speaker_label or "[dim]—[/dim]"
                text_val = (s.text or "").strip() or "[dim](empty)[/dim]"
                tbl.add_row(str(s.segment_index), time_str, lbl, text_val)
            console.print(tbl)

        # ── Head ──────────────────────────────────────────────────────────
        console.print()
        head_segs = segs[:head]
        _print_segments(f"First {len(head_segs)} segment(s)", head_segs)

        # ── Tail (only if non-overlapping) ────────────────────────────────
        if len(segs) > head:
            tail_segs = segs[max(head, len(segs) - tail):]
            omitted = len(segs) - head - len(tail_segs)
            if omitted > 0:
                console.print(f"\n[dim]  … {omitted} segment(s) omitted …[/dim]\n")
            _print_segments(f"Last {len(tail_segs)} segment(s)", tail_segs)

        # ── Write full transcript to file ─────────────────────────────────
        if output is not None:
            # Resolve human-assigned speaker names in one batch query
            with OrmSession(db_engine) as db:
                audio_files = list({s.audio_file for s in segs if s.audio_file})
                labels = list({s.original_speaker_label for s in segs if s.original_speaker_label})
                name_rows = db.execute(
                    select(SpeakerName).where(
                        SpeakerName.audio_file.in_(audio_files),
                        SpeakerName.local_speaker_label.in_(labels),
                    )
                ).scalars().all() if labels else []
            name_lookup = {
                (r.audio_file, r.local_speaker_label): r.speaker_name for r in name_rows
            }

            lines = []
            for s in segs:
                mm = int(s.start_time // 60)
                ss_val = s.start_time % 60
                if s.original_speaker_label:
                    display = name_lookup.get(
                        (s.audio_file, s.original_speaker_label),
                        s.original_speaker_label,
                    )
                else:
                    display = "Speaker"
                lines.append(f"[{mm:02d}:{ss_val:05.2f}] {display}: {(s.text or '').strip()}")

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(lines), encoding="utf-8")
            console.print(f"[green]Transcript written to {output} ({len(lines)} segments)[/green]")

        console.print()
        return

    # ── List view ─────────────────────────────────────────────────────────────
    with OrmSession(db_engine) as db:
        agg = db.execute(
            select(
                TranscriptionSegment.session_id,
                sqlfunc.count(TranscriptionSegment.id).label("segments"),
                sqlfunc.min(TranscriptionSegment.start_time).label("first_start"),
                sqlfunc.max(TranscriptionSegment.end_time).label("last_end"),
                sqlfunc.max(TranscriptionSegment.created_at).label("last_updated"),
            ).group_by(TranscriptionSegment.session_id)
            .order_by(sqlfunc.max(TranscriptionSegment.created_at).desc())
        ).all()

        if not agg:
            console.print("[yellow]No sessions found in the database.[/yellow]")
            return

        state_rows = {
            row.session_id: row
            for row in db.execute(select(SessionState)).scalars().all()
        }

        speaker_map: dict = {}
        for sid, label in db.execute(
            select(TranscriptionSegment.session_id, TranscriptionSegment.original_speaker_label).distinct()
        ).all():
            if label:
                speaker_map.setdefault(sid, set()).add(label)

        file_map: dict = {}
        for sid, af in db.execute(
            select(TranscriptionSegment.session_id, TranscriptionSegment.audio_file).distinct()
        ).all():
            if af:
                file_map.setdefault(sid, []).append(af)

    table = Table(
        title=f"Transcription Sessions ({len(agg)} total)",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Session", style="bold green", no_wrap=True)
    table.add_column("Updated", style="dim", no_wrap=True)
    table.add_column("Segs", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Speakers")
    table.add_column("Files")

    for row in agg:
        sid = row.session_id
        speakers = sorted(speaker_map.get(sid, set()))
        files = sorted(file_map.get(sid, []))
        duration = (row.last_end or 0.0) - (row.first_start or 0.0)
        duration_str = f"{int(duration // 60)}m {duration % 60:.0f}s" if duration else "—"
        updated = row.last_updated.strftime("%Y-%m-%d %H:%M") if row.last_updated else "—"
        speakers_str = "\n".join(speakers) if speakers else "[dim](none)[/dim]"
        files_str = "\n".join(files) if files else "[dim]—[/dim]"
        table.add_row(sid, updated, str(row.segments), duration_str, speakers_str, files_str)

    console.print()
    console.print(table)
    console.print()


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
    console.print("  sessions           - List or inspect transcription sessions")
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


@app.command()
def listen(
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file (.pawnai.yml)"
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-T",
        help="Topic name to subscribe to. Overrides the value in the queue: config section."
    ),
    consumer_name: Optional[str] = typer.Option(
        None, "--consumer-name", "-n",
        help="Consumer registration name. Overrides the value in the queue: config section."
    ),
) -> None:
    """Listen for commands on a pawn-queue topic and execute them.

    Connects to the S3-backed pawn-queue configured in the ``queue:`` section
    of ``.pawnai.yml`` and blocks until interrupted.  Each incoming message
    must be a JSON object with at minimum a ``command`` key:

    \b
      {
        "command": "transcribe-diarize",
        "audio_paths": ["s3://bucket/audio.flac"],
        "threshold": 0.2,
        "cross_file_threshold": 0.2,
        "session": "tom-20260305",
        "device": "cpu"
      }

    Supported commands: transcribe-diarize, transcribe, diarize, embed, analyze, sync-siyuan.

    On success the message is acked; on failure it is sent to the dead-letter
    queue so it can be inspected and replayed later.

    Stop with Ctrl-C.
    """
    import asyncio
    import logging
    from ..core.config import AppConfig
    from ..core.queue_listener import start_listener, DEFAULT_TOPIC, DEFAULT_CONSUMER_NAME

    # Configure logging so queue activity is visible in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_cfg = AppConfig(config_path=config)

    queue_cfg = app_cfg.get_queue_config()
    effective_topic = topic or (queue_cfg or {}).get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or (queue_cfg or {}).get("consumer_name", DEFAULT_CONSUMER_NAME)

    console.print(
        f"[bold green]PawnAI queue listener starting[/bold green]\n"
        f"  topic    : [cyan]{effective_topic}[/cyan]\n"
        f"  consumer : [cyan]{effective_consumer}[/cyan]"
    )
    if queue_cfg:
        s3_ep = queue_cfg.get("s3", {}).get("endpoint_url", "http://localhost:9000")
        console.print(f"  endpoint : [cyan]{s3_ep}[/cyan]")
    console.print("[dim]Press Ctrl-C to stop.[/dim]\n")

    try:
        asyncio.run(
            start_listener(
                cfg=app_cfg,
                topic_override=topic,
                consumer_name_override=consumer_name,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Listener stopped by user.[/yellow]")
    except RuntimeError as exc:
        console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Listener error: {exc}[/red]")
        raise typer.Exit(1)


def _fmt_size(num_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes //= 1024
    return f"{num_bytes:.1f} PB"

