"""Delete a diarization session (``session_delete`` CliTool)."""

from __future__ import annotations

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import GraphTriple, SessionAnalysis, TranscriptionSegment
from pawn_diarize.core.database import SessionState


def delete_session_impl(cfg: AgentConfig, session_id: str, confirm: str) -> str:
    """Permanently delete PostgreSQL diarization data for *session_id*.

    Requires *confirm* to equal *session_id* exactly (in-chat confirmation gate).
    Deletes segments, analyses, session_state, and graph triples in one
    transaction. Does not touch sallm chat memory, agent_runs, schedules,
    speaker_names, or embeddings.
    """
    session_id_clean = session_id.strip()
    confirm_clean = confirm.strip()
    if not session_id_clean:
        raise ValueError("session id must not be empty")
    if confirm_clean != session_id_clean:
        raise ValueError(
            f"confirmation mismatch: --confirm must exactly equal --session-id "
            f"({session_id_clean!r}); got {confirm_clean!r}"
        )

    engine = create_engine(cfg.db_dsn)
    try:
        with Session(engine) as db:
            seg_result = db.execute(
                delete(TranscriptionSegment).where(
                    TranscriptionSegment.session_id == session_id_clean
                )
            )
            analysis_result = db.execute(
                delete(SessionAnalysis).where(
                    SessionAnalysis.session_id == session_id_clean
                )
            )
            state_result = db.execute(
                delete(SessionState).where(SessionState.session_id == session_id_clean)
            )
            triples_result = db.execute(
                delete(GraphTriple).where(GraphTriple.session_id == session_id_clean)
            )
            db.commit()

            segments = int(seg_result.rowcount or 0)
            analyses = int(analysis_result.rowcount or 0)
            states = int(state_result.rowcount or 0)
            triples = int(triples_result.rowcount or 0)
    finally:
        if hasattr(engine, "dispose"):
            engine.dispose()

    total = segments + analyses + states + triples
    if total == 0:
        return (
            f"No matching rows for session '{session_id_clean}' "
            "(already gone or never stored)."
        )
    return (
        f"Deleted session '{session_id_clean}': "
        f"{segments} segment(s), {analyses} analysis row(s), "
        f"{states} session_state row(s), {triples} graph triple(s)."
    )
