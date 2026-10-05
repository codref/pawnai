"""PostgreSQL helpers for ``vault_notes`` mappings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session, make_transient

from pawn_core.database import VaultNote, _get_session, get_engine


def get_vault_note(dsn: str, session_id: str) -> Optional[VaultNote]:
    """Return a detached vault note row or None."""
    with Session(get_engine(dsn)) as db:
        row = db.get(VaultNote, session_id)
        if row is None:
            return None
        db.expunge(row)
        make_transient(row)
        return row


def upsert_vault_note(
    dsn: str,
    *,
    session_id: str,
    key: str,
    content_hash: str = "",
) -> None:
    """Insert or update the S3 key mapping for a diarization session."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(VaultNote, session_id)
        if row is None:
            db.add(
                VaultNote(
                    session_id=session_id,
                    key=key,
                    content_hash=content_hash,
                    updated_at=now,
                )
            )
        else:
            row.key = key
            row.content_hash = content_hash
            row.updated_at = now
