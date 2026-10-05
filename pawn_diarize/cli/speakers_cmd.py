"""CLI for the curated Speakers gallery and reidentify/rediarize commands."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .utils import console

speakers_app = typer.Typer(
    help="Curated Speakers gallery (people + manual voiceprint enrollments).",
    rich_markup_mode="rich",
)


def _gallery(db_dsn: Optional[str], config_path: Optional[str] = None):
    from ..core.config import AppConfig
    from ..core.speaker_gallery import SpeakerGallery

    cfg = AppConfig(config_path=config_path) if config_path else AppConfig()
    dsn = db_dsn or cfg.db_dsn
    return SpeakerGallery(dsn, config=cfg.speakers), cfg


@speakers_app.command("list")
def speakers_list(
    db_dsn: Optional[str] = typer.Option(None, help="PostgreSQL DSN"),
    config: Optional[str] = typer.Option(None, "--config", help="pawnai.yaml path"),
    include_inactive: bool = typer.Option(False, "--include-inactive"),
) -> None:
    """List people in the Speakers gallery."""
    gallery, _ = _gallery(db_dsn, config)
    rows = gallery.list_speakers(include_inactive=include_inactive)
    if not rows:
        console.print("[yellow]No speakers in gallery yet.[/yellow]")
        return
    for sp in rows:
        enrollments = gallery.list_enrollments(sp.id)
        flag = "" if sp.active else " (inactive)"
        console.print(
            f"[cyan]{sp.id}[/cyan]  {sp.display_name}{flag}  "
            f"enrollments={len(enrollments)}"
        )


@speakers_app.command("show")
def speakers_show(
    speaker: str = typer.Argument(..., help="Speaker id or display name"),
    db_dsn: Optional[str] = typer.Option(None, help="PostgreSQL DSN"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Show one gallery person and their enrollments."""
    gallery, _ = _gallery(db_dsn, config)
    sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
    if sp is None:
        console.print(f"[red]Unknown speaker: {speaker}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{sp.display_name}[/bold]  id={sp.id}  active={sp.active}")
    if sp.aliases:
        console.print(f"  aliases: {', '.join(sp.aliases)}")
    if sp.notes:
        console.print(f"  notes: {sp.notes}")
    for enr in gallery.list_enrollments(sp.id):
        span = ""
        if enr.start_time is not None and enr.end_time is not None:
            span = f"  [{enr.start_time:.1f}-{enr.end_time:.1f}s]"
        console.print(
            f"  enrollment {enr.id[:8]}…  model={enr.embedding_model} "
            f"dim={enr.embedding_dim} dur={enr.duration:.1f}s "
            f"quality={enr.quality_score}{span}"
        )


@speakers_app.command("create")
def speakers_create(
    name: str = typer.Argument(..., help="Display name"),
    speaker_id: Optional[str] = typer.Option(None, "--id", help="Optional stable id/slug"),
    notes: Optional[str] = typer.Option(None, "--notes"),
    db_dsn: Optional[str] = typer.Option(None, help="PostgreSQL DSN"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Create a gallery person (no voiceprint yet)."""
    gallery, _ = _gallery(db_dsn, config)
    sp = gallery.create_speaker(name, speaker_id=speaker_id, notes=notes)
    console.print(f"[green]Created speaker[/green] {sp.display_name} (id={sp.id})")


@speakers_app.command("rename")
def speakers_rename(
    speaker: str = typer.Argument(..., help="Current id or name"),
    new_name: str = typer.Argument(..., help="New display name"),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Rename a gallery person's display name."""
    gallery, _ = _gallery(db_dsn, config)
    sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
    if sp is None:
        console.print(f"[red]Unknown speaker: {speaker}[/red]")
        raise typer.Exit(1)
    updated = gallery.rename_speaker(sp.id, new_name)
    console.print(f"[green]Renamed[/green] {sp.id} → {updated.display_name}")


@speakers_app.command("deactivate")
def speakers_deactivate(
    speaker: str = typer.Argument(...),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Soft-deactivate a gallery person (excluded from matching)."""
    gallery, _ = _gallery(db_dsn, config)
    sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
    if sp is None:
        console.print(f"[red]Unknown speaker: {speaker}[/red]")
        raise typer.Exit(1)
    gallery.deactivate_speaker(sp.id)
    console.print(f"[yellow]Deactivated[/yellow] {sp.id}")


@speakers_app.command("enroll")
def speakers_enroll(
    speaker: str = typer.Option(..., "--speaker", "-s", help="Gallery id or display name"),
    session: Optional[str] = typer.Option(None, "--session", help="Source session id"),
    from_label: Optional[str] = typer.Option(
        None, "--from", help="Local label in session (e.g. SPEAKER_00 or Davide)"
    ),
    audio: Optional[List[str]] = typer.Option(
        None, "--audio", "-a", help="Clean enrollment clip(s)"
    ),
    start: Optional[float] = typer.Option(None, "--start", help="Span start seconds"),
    end: Optional[float] = typer.Option(None, "--end", help="Span end seconds"),
    force: bool = typer.Option(False, "--force", help="Bypass quality gates"),
    notes: Optional[str] = typer.Option(None, "--notes"),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Manually approve a voiceprint for a gallery person.

    Prefer enrolling from a reviewed session label::

        pawn-diarize speakers enroll -s Davide --session my-day --from SPEAKER_00

    Or from a clean dedicated clip::

        pawn-diarize speakers enroll -s Davide --audio davide-hello.wav
    """
    from ..core.diarization import DiarizationEngine
    from ..core.database import get_engine, init_db, load_session_state
    from sqlalchemy import select
    from pawn_core.database import TranscriptionSegment
    from sqlalchemy.orm import Session as OrmSession

    gallery, cfg = _gallery(db_dsn, config)
    sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
    if sp is None:
        # Convenience: create on enroll if missing.
        sp = gallery.create_speaker(speaker)
        console.print(f"[cyan]Created speaker[/cyan] {sp.display_name} (id={sp.id})")

    engine = DiarizationEngine(
        device=device,
        diarization_backend=cfg.models.diarization_backend,
        diarization_model=cfg.models.diarization_model,
        embedding_model=cfg.models.embedding_model,
        hf_token=cfg.models.hf_token,
    )

    embedding = None
    source_audio = None
    source_session = session
    span_start, span_end = start, end

    if audio:
        embedding = engine.extract_embeddings(list(audio))
        source_audio = audio[0]
        # Whole-file duration for quality gate.
        if span_start is None or span_end is None:
            import soundfile as sf

            info = sf.info(audio[0])
            span_start = 0.0
            span_end = float(info.duration)
    elif session and from_label:
        db_engine = get_engine(cfg.db_dsn if db_dsn is None else db_dsn)
        init_db(db_engine)
        prior, _, _, _ = load_session_state(session, db_engine)
        # Prefer the session centroid for this label when present.
        info = (prior or {}).get(from_label)
        if info and info.get("embedding") is not None:
            import numpy as np

            embedding = np.asarray(info["embedding"], dtype="float32")
            # Estimate span from segments with that label.
            with OrmSession(db_engine) as db:
                rows = list(
                    db.scalars(
                        select(TranscriptionSegment)
                        .where(TranscriptionSegment.session_id == session)
                        .where(TranscriptionSegment.original_speaker_label == from_label)
                        .order_by(TranscriptionSegment.start_time)
                    )
                )
                if rows:
                    source_audio = rows[0].audio_file
                    if span_start is None:
                        span_start = float(rows[0].start_time)
                    if span_end is None:
                        span_end = float(rows[-1].end_time)
        else:
            console.print(
                "[red]No session_state embedding for that label. "
                "Pass --audio, or re-process a chunk first.[/red]"
            )
            raise typer.Exit(1)
    else:
        console.print(
            "[red]Provide either --audio FILE or --session + --from LABEL[/red]"
        )
        raise typer.Exit(1)

    try:
        result = gallery.enroll(
            sp,
            embedding,
            embedding_model=getattr(engine._extractor, "model_id", cfg.models.embedding_model),
            source_session_id=source_session,
            source_audio_file=source_audio,
            start_time=span_start,
            end_time=span_end,
            notes=notes,
            force=force,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]Enrolled[/green] {result.display_name} "
        f"({result.enrollment_id[:8]}…) model={result.embedding_model} "
        f"dim={result.embedding_dim} dur={result.duration:.1f}s "
        f"quality={result.quality_score}"
    )


@speakers_app.command("enrollments")
def speakers_enrollments(
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s"),
    remove: Optional[str] = typer.Option(None, "--remove", help="Enrollment id to delete"),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """List or remove enrollments."""
    gallery, _ = _gallery(db_dsn, config)
    if remove:
        gallery.remove_enrollment(remove)
        console.print(f"[yellow]Removed enrollment[/yellow] {remove}")
        return
    speaker_id = None
    if speaker:
        sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
        if sp is None:
            console.print(f"[red]Unknown speaker: {speaker}[/red]")
            raise typer.Exit(1)
        speaker_id = sp.id
    rows = gallery.list_enrollments(speaker_id)
    if not rows:
        console.print("[yellow]No enrollments.[/yellow]")
        return
    for enr in rows:
        console.print(
            f"{enr.id}  speaker={enr.speaker_id}  model={enr.embedding_model} "
            f"dim={enr.embedding_dim} dur={enr.duration:.1f}s"
        )


@speakers_app.command("purge-legacy-embeddings")
def speakers_purge_legacy(
    confirm: bool = typer.Option(False, "--confirm", help="Required safety flag"),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Delete the old auto-accumulated ``embeddings`` table rows."""
    if not confirm:
        console.print("[red]Refusing without --confirm[/red]")
        raise typer.Exit(1)
    gallery, _ = _gallery(db_dsn, config)
    n = gallery.purge_legacy_embeddings()
    console.print(f"[green]Purged {n} legacy embedding row(s)[/green]")


@speakers_app.command("reembed")
def speakers_reembed(
    speaker: Optional[str] = typer.Option(
        None, "--speaker", "-s", help="Limit to one speaker; default = all"
    ),
    db_dsn: Optional[str] = typer.Option(None),
    config: Optional[str] = typer.Option(None, "--config"),
    device: str = typer.Option("auto", "--device", "-d"),
    force: bool = typer.Option(True, "--force/--no-force"),
) -> None:
    """Re-extract enrollment embeddings with the configured embedding model.

    Requires each enrollment to still have a readable ``source_audio_file``
    (and ideally start/end times).  Used after switching to TitaNet.
    """
    from ..core.diarization import DiarizationEngine, _load_audio, _resample
    import numpy as np
    import uuid
    from datetime import datetime, timezone
    from pawn_core.database import SpeakerEnrollment

    gallery, cfg = _gallery(db_dsn, config)
    engine = DiarizationEngine(
        device=device,
        embedding_model=cfg.models.embedding_model,
        hf_token=cfg.models.hf_token,
    )
    engine._initialize_models()
    assert engine._extractor is not None

    speakers = []
    if speaker:
        sp = gallery.get_speaker(speaker) or gallery.find_speaker_by_name(speaker)
        if sp is None:
            console.print(f"[red]Unknown speaker: {speaker}[/red]")
            raise typer.Exit(1)
        speakers = [sp]
    else:
        speakers = gallery.list_speakers()

    rebuilt = 0
    for sp in speakers:
        old = gallery.list_enrollments(sp.id)
        for enr in old:
            if not enr.source_audio_file or not Path(enr.source_audio_file).exists():
                console.print(
                    f"[yellow]Skip {enr.id[:8]}… — missing audio "
                    f"{enr.source_audio_file!r}[/yellow]"
                )
                continue
            waveform, sr = _load_audio(enr.source_audio_file)
            if sr != 16000:
                waveform = _resample(waveform, sr, 16000)
                sr = 16000
            start = float(enr.start_time or 0.0)
            end = float(enr.end_time or (waveform.shape[1] / sr))
            emb = engine._embed_crop(waveform, sr, start, end)
            if emb is None:
                console.print(f"[yellow]Skip {enr.id[:8]}… — embed failed[/yellow]")
                continue
            gallery.remove_enrollment(enr.id)
            gallery.enroll(
                sp,
                emb,
                embedding_model=engine._extractor.model_id,
                source_session_id=enr.source_session_id,
                source_audio_file=enr.source_audio_file,
                start_time=start,
                end_time=end,
                notes=enr.notes,
                force=force,
            )
            rebuilt += 1
            console.print(f"[green]Re-embedded[/green] {sp.display_name} {enr.id[:8]}…")
    console.print(f"Done. Rebuilt {rebuilt} enrollment(s).")
