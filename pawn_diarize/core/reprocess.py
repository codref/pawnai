"""Shared session reprocess helpers (wipe, retranscribe, audio discovery).

Used by CLI ``rediarize`` / ``retranscribe`` and the diarize queue listener so
HTTP/Obsidian can enqueue the same work without duplicating overwrite logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from pawn_core.database import SessionAnalysis, SessionSpeakerMap, TranscriptionSegment
from pawn_diarize.core.database import SessionState, get_engine, init_db
from pawn_diarize.core.path_repair import repair_session_audio_paths
from pawn_diarize.core.s3 import is_s3_path

logger = logging.getLogger(__name__)


def ordered_session_audio_files(engine: Any, session_id: str) -> List[str]:
    """Return distinct ``audio_file`` paths for *session_id*, oldest-first.

    Falls back to ``session_state.processed_files`` when segment rows have no
    paths (should be rare after a successful ingest).
    """
    with OrmSession(engine) as db:
        ordered = list(
            db.execute(
                select(
                    TranscriptionSegment.audio_file,
                    func.min(TranscriptionSegment.start_time),
                )
                .where(TranscriptionSegment.session_id == session_id)
                .where(TranscriptionSegment.audio_file.is_not(None))
                .group_by(TranscriptionSegment.audio_file)
                .order_by(func.min(TranscriptionSegment.start_time))
            ).all()
        )
        audio_files = [row[0] for row in ordered if row[0]]
        if audio_files:
            return audio_files
        state = db.get(SessionState, session_id)
        if state and state.processed_files:
            return [str(p) for p in state.processed_files if p]
    return []


def wipe_session_asr_state(engine: Any, session_id: str) -> Tuple[int, int]:
    """Delete segments, analyses, session_state, and speaker_map for *session_id*.

    Returns ``(segments_deleted, speaker_map_deleted)``.
    """
    with OrmSession(engine) as db:
        seg_del = db.execute(
            sql_delete(TranscriptionSegment).where(TranscriptionSegment.session_id == session_id)
        )
        db.execute(sql_delete(SessionAnalysis).where(SessionAnalysis.session_id == session_id))
        db.execute(sql_delete(SessionState).where(SessionState.session_id == session_id))
        map_del = db.execute(
            sql_delete(SessionSpeakerMap).where(SessionSpeakerMap.session_id == session_id)
        )
        db.commit()
    return int(seg_del.rowcount or 0), int(map_del.rowcount or 0)


def prepare_session_audio(
    session_id: str,
    db_dsn: str,
    app_cfg: Any,
) -> Tuple[List[str], Any]:
    """Repair stale paths and return ordered audio URIs for *session_id*.

    Raises:
        ValueError: When the session has no recoverable audio.
    """
    engine = get_engine(db_dsn)
    init_db(engine)
    repair = repair_session_audio_paths(session_id, db_dsn, app_cfg)
    if repair.remaps:
        logger.info("%s", repair.summary())

    audio_files = ordered_session_audio_files(engine, session_id)
    if not audio_files:
        raise ValueError(f"No audio files for session {session_id!r}")

    missing = [p for p in audio_files if not is_s3_path(p) and not Path(p).exists()]
    if missing:
        raise ValueError(
            "Audio paths are missing locally and could not be recovered as S3 URIs: "
            + ", ".join(Path(p).name for p in missing)
        )
    return audio_files, engine


def maybe_force_push_vault(
    session_id: str,
    *,
    db_dsn: str,
    app_cfg: Any,
    push_vault: bool,
) -> None:
    """Force a vault transcript refresh when *push_vault* is true."""
    if not push_vault:
        return
    from pawn_diarize.core.vault_transcript import push_session_transcript

    try:
        status = push_session_transcript(session_id, db_dsn=db_dsn, cfg=app_cfg)
        logger.info("vault transcript refreshed for %r: %s", session_id, status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault refresh skipped for %r: %s", session_id, exc)


def retranscribe_session(
    session_id: str,
    db_dsn: str,
    app_cfg: Any,
    *,
    backend: str = "nemo",
    language: Optional[str] = None,
    threshold: float = 0.7,
    cross_file_threshold: float = 0.55,
    device: str = "cuda",
    chunk_duration: Optional[float] = None,
    verbose: bool = False,
    push_vault: bool = False,
    resolve_audio: Any = None,
    cleanup_temps: Any = None,
    transcription_engine: Any = None,
    diarization_engine: Any = None,
) -> dict[str, Any]:
    """Wipe session ASR/diarization state and re-run full transcribe-diarize.

    *resolve_audio* should be ``(paths, cfg) -> (local_paths, temps, path_map)``.
    *cleanup_temps* cleans temp dirs from that resolver.
    """
    from pawn_diarize.core.combined import (
        format_transcript_with_speakers,
        transcribe_with_diarization,
    )
    from pawn_diarize.core.database import save_session_state, save_transcription_segments
    from pawn_diarize.core.s3 import to_canonical_paths

    audio_files, engine = prepare_session_audio(session_id, db_dsn, app_cfg)
    seg_n, map_n = wipe_session_asr_state(engine, session_id)
    logger.info(
        "retranscribe wipe %r: %d segment(s), %d speaker-map row(s)",
        session_id,
        seg_n,
        map_n,
    )

    if resolve_audio is None:
        raise ValueError("retranscribe_session requires resolve_audio")
    resolved, temps, path_map = resolve_audio(list(audio_files), app_cfg)
    try:
        result = transcribe_with_diarization(
            audio_path=resolved if len(resolved) > 1 else resolved[0],
            db_dsn=db_dsn,
            similarity_threshold=threshold,
            store_new_speakers=False,
            device=device,
            chunk_duration=chunk_duration,
            cross_file_threshold=cross_file_threshold,
            verbose=verbose,
            backend=backend,
            language=language,
            prior_speaker_embeddings=None,
            time_cursor=0.0,
            source_map=path_map,
            session_id=session_id,
            identify_margin=float(
                getattr(getattr(app_cfg, "speakers", None), "identify_margin", 0.05)
            ),
            diarization_backend=str(
                getattr(getattr(app_cfg, "models", None), "diarization_backend", "pyannote")
            ),
            diarization_model=getattr(
                getattr(app_cfg, "models", None), "diarization_model", None
            ),
            embedding_model=getattr(
                getattr(app_cfg, "models", None), "embedding_model", None
            ),
            hf_token=getattr(getattr(app_cfg, "models", None), "hf_token", None),
            transcription_engine=transcription_engine,
            diarization_engine=diarization_engine,
        )
        if result:
            new_processed = to_canonical_paths(resolved, path_map)
            save_session_state(
                session_id,
                result.get("session_speaker_embeddings") or {},
                result.get("new_time_cursor", 0.0),
                new_processed,
                engine,
            )
            if result.get("segments"):
                save_transcription_segments(result["segments"], session_id, engine, start_index=0)
        text = format_transcript_with_speakers(result, include_timestamps=True)
        logger.info("Retranscribe complete for %r (%d chars)", session_id, len(text))
    finally:
        if cleanup_temps is not None:
            cleanup_temps(temps)

    maybe_force_push_vault(session_id, db_dsn=db_dsn, app_cfg=app_cfg, push_vault=push_vault)
    return result


def rediarize_session_job(
    session_id: str,
    db_dsn: str,
    app_cfg: Any,
    *,
    threshold: float = 0.7,
    cross_file_threshold: float = 0.55,
    margin: Optional[float] = None,
    device: str = "auto",
    push_vault: bool = False,
    resolve_audio: Any = None,
    cleanup_temps: Any = None,
) -> Any:
    """Repair paths, re-run diarization+identify, optional vault push."""
    from pawn_diarize.core.reidentify import rediarize_session

    audio_files, _engine = prepare_session_audio(session_id, db_dsn, app_cfg)
    if resolve_audio is None:
        raise ValueError("rediarize_session_job requires resolve_audio")
    resolved, temps, path_map = resolve_audio(list(audio_files), app_cfg)
    try:
        identify_margin = (
            float(margin)
            if margin is not None
            else float(getattr(getattr(app_cfg, "speakers", None), "identify_margin", 0.05))
        )
        result = rediarize_session(
            session_id,
            db_dsn,
            audio_paths=resolved,
            device=device,
            threshold=threshold,
            cross_file_threshold=cross_file_threshold,
            margin=identify_margin,
            diarization_backend=str(
                getattr(getattr(app_cfg, "models", None), "diarization_backend", "pyannote")
            ),
            diarization_model=getattr(getattr(app_cfg, "models", None), "diarization_model", None),
            embedding_model=getattr(getattr(app_cfg, "models", None), "embedding_model", None),
            hf_token=getattr(getattr(app_cfg, "models", None), "hf_token", None),
            source_map=path_map,
        )
    finally:
        if cleanup_temps is not None:
            cleanup_temps(temps)

    maybe_force_push_vault(session_id, db_dsn=db_dsn, app_cfg=app_cfg, push_vault=push_vault)
    return result
