"""Repair stale /tmp audio paths stored on diarization sessions.

Queue/CLI historically persisted ephemeral download paths into
``transcription_segments.audio_file`` and ``session_state.processed_files``.
This module recovers canonical ``s3://`` URIs (by unique filename match in
the configured bucket) and rewrites those rows — no ML required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session as OrmSession

from pawn_core.database import SpeakerName, TranscriptionSegment
from pawn_diarize.core.database import SessionState, get_engine, get_session, init_db
from pawn_diarize.core.s3 import S3Client, canonicalize_audio_paths, is_s3_path


@dataclass(frozen=True)
class PathRepairResult:
    """Receipt for a session audio-path repair."""

    session_id: str
    remaps: Dict[str, str] = field(default_factory=dict)
    segments_updated: int = 0
    speaker_names_updated: int = 0
    processed_files_updated: bool = False

    def summary(self) -> str:
        if not self.remaps:
            return f"Session '{self.session_id}': no paths needed repair"
        return (
            f"Session '{self.session_id}': remapped {len(self.remaps)} path(s), "
            f"{self.segments_updated} segment(s), "
            f"{self.speaker_names_updated} speaker_names; "
            f"processed_files={'yes' if self.processed_files_updated else 'no'}"
        )


def _s3_client_from_cfg(app_cfg: Any) -> Optional[S3Client]:
    s3_cfg = None
    if hasattr(app_cfg, "get_s3_config"):
        s3_cfg = app_cfg.get_s3_config()
    elif isinstance(app_cfg, dict):
        s3_cfg = app_cfg.get("s3")
    if not s3_cfg or not s3_cfg.get("bucket"):
        return None
    try:
        return S3Client.from_dict(s3_cfg)
    except Exception:
        return None


def repair_session_audio_paths(
    session_id: str,
    db_dsn: str,
    app_cfg: Any,
    *,
    s3_client: Optional[S3Client] = None,
) -> PathRepairResult:
    """Rewrite stale local/tmp audio paths to canonical ``s3://`` URIs.

    Updates:
    - ``transcription_segments.audio_file`` for the session
    - ``session_state.processed_files`` when present
    - ``speaker_names.audio_file`` rows that still point at remapped locals
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise ValueError("session id must not be empty")

    engine = get_engine(db_dsn)
    init_db(engine)
    client = s3_client if s3_client is not None else _s3_client_from_cfg(app_cfg)

    with OrmSession(engine) as db:
        segment_files = list(
            db.scalars(
                select(TranscriptionSegment.audio_file)
                .where(TranscriptionSegment.session_id == session_id)
                .distinct()
            )
        )
        state = db.get(SessionState, session_id)
        processed = list(state.processed_files or []) if state else []

    if not segment_files and not processed:
        raise ValueError(f"No audio paths found for session '{session_id}'")

    # Union so processed_files-only leftovers are also considered.
    candidates: List[str] = []
    seen = set()
    for p in list(segment_files) + processed:
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    _, remaps = canonicalize_audio_paths(candidates, client)
    if not remaps:
        return PathRepairResult(session_id=session_id)

    segments_updated = 0
    speaker_names_updated = 0
    processed_updated = False

    with get_session(engine) as db:
        for old, new in remaps.items():
            result = db.execute(
                update(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .where(TranscriptionSegment.audio_file == old)
                .values(audio_file=new)
            )
            segments_updated += int(result.rowcount or 0)

            # speaker_names are keyed by audio_file; remap matching rows.
            sn_result = db.execute(
                update(SpeakerName)
                .where(SpeakerName.audio_file == old)
                .values(audio_file=new)
            )
            speaker_names_updated += int(sn_result.rowcount or 0)

        state = db.get(SessionState, session_id)
        if state and state.processed_files:
            new_processed = [remaps.get(p, p) for p in list(state.processed_files)]
            if new_processed != list(state.processed_files):
                state.processed_files = new_processed
                processed_updated = True

    return PathRepairResult(
        session_id=session_id,
        remaps=dict(remaps),
        segments_updated=segments_updated,
        speaker_names_updated=speaker_names_updated,
        processed_files_updated=processed_updated,
    )


def list_sessions_with_audio(db_dsn: str) -> List[str]:
    """Return distinct session ids that have at least one transcription segment."""
    engine = get_engine(db_dsn)
    init_db(engine)
    with OrmSession(engine) as db:
        rows = list(
            db.scalars(select(TranscriptionSegment.session_id).distinct())
        )
    return sorted(sid for sid in rows if sid)


def unrepaired_local_paths(paths: Sequence[str]) -> List[str]:
    """Return paths that are still non-s3 and look like missing locals."""
    from pathlib import Path

    out: List[str] = []
    for p in paths:
        if is_s3_path(p):
            continue
        if not Path(p).exists():
            out.append(p)
    return out
