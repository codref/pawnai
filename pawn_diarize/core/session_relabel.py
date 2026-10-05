"""Bulk-rename a speaker across one diarization session.

Used by ``pawn-diarize session-relabel`` and the agent ``session_relabel``
CliTool. Updates transcript segments, ``speaker_names`` mappings (so future
embedding matches resolve to the new display name), and ``session_state``
prior-speaker keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Set

from sqlalchemy import select, update
from sqlalchemy.orm import Session as OrmSession

from .database import (
    Embedding,
    SessionState,
    SpeakerName,
    TranscriptionSegment,
    get_engine,
    init_db,
)


@dataclass(frozen=True)
class RelabelResult:
    """Receipt for a successful session speaker relabel."""

    session_id: str
    from_label: str
    to_label: str
    aliases: tuple[str, ...]
    segments_updated: int
    speaker_names_updated: int
    speaker_names_created: int
    session_state_updated: bool

    def summary(self) -> str:
        alias_note = ""
        if len(self.aliases) > 1:
            alias_note = f" (aliases: {', '.join(self.aliases)})"
        parts = [
            f"Relabeled '{self.from_label}' → '{self.to_label}' in session "
            f"'{self.session_id}'{alias_note}: "
            f"{self.segments_updated} segment(s)"
        ]
        if self.speaker_names_updated:
            parts.append(
                f"{self.speaker_names_updated} speaker_names update(s)"
            )
        if self.speaker_names_created:
            parts.append(
                f"{self.speaker_names_created} speaker_names created"
            )
        if self.session_state_updated:
            parts.append("session_state embeddings key updated")
        return ", ".join(parts) + "."


def _expand_aliases(
    db: OrmSession,
    session_files: list[str],
    from_label: str,
) -> Set[str]:
    """Expand *from_label* to raw SPEAKER_XX labels and current display names.

    Dirty sessions often store display names on segments while embeddings keep
    ``SPEAKER_XX``. Expanding both directions lets ``SPEAKER_00 → Davide`` and
    ``WrongName → Davide`` hit the same underlying voice identity.
    """
    aliases: Set[str] = {from_label}
    if not session_files:
        return aliases

    rows = db.execute(
        select(
            SpeakerName.local_speaker_label,
            SpeakerName.speaker_name,
        ).where(SpeakerName.audio_file.in_(session_files))
    ).all()
    for local_label, speaker_name in rows:
        if local_label == from_label or speaker_name == from_label:
            aliases.add(local_label)
            aliases.add(speaker_name)

    emb_labels = (
        db.execute(
            select(Embedding.local_speaker_label)
            .where(
                Embedding.audio_file.in_(session_files),
                Embedding.local_speaker_label == from_label,
            )
            .distinct()
        )
        .scalars()
        .all()
    )
    aliases.update(emb_labels)
    return aliases


def relabel_session_speaker(
    db_dsn: str,
    session_id: str,
    from_label: str,
    to_label: str,
) -> RelabelResult:
    """Apply a speaker rename across one session.

    Raises:
        ValueError: when inputs are empty, identical, or nothing matches.
    """
    session_id = (session_id or "").strip()
    from_label = (from_label or "").strip()
    to_label = (to_label or "").strip()
    if not session_id:
        raise ValueError("session id must not be empty")
    if not from_label:
        raise ValueError("--from speaker label must not be empty")
    if not to_label:
        raise ValueError("--to speaker name must not be empty")
    if from_label == to_label:
        raise ValueError(f"--from and --to are the same ({from_label!r})")

    engine = get_engine(db_dsn)
    init_db(engine)

    with OrmSession(engine) as db:
        session_files = list(
            db.execute(
                select(TranscriptionSegment.audio_file)
                .where(TranscriptionSegment.session_id == session_id)
                .distinct()
            )
            .scalars()
            .all()
        )
        if not session_files:
            raise ValueError(f"No segments found for session '{session_id}'")

        aliases = _expand_aliases(db, session_files, from_label)
        # Never treat the destination as something to rewrite away from.
        aliases.discard(to_label)

        affected_segs = (
            db.execute(
                select(TranscriptionSegment)
                .where(
                    TranscriptionSegment.session_id == session_id,
                    TranscriptionSegment.original_speaker_label.in_(aliases),
                )
                .order_by(TranscriptionSegment.start_time)
            )
            .scalars()
            .all()
        )

        affected_names = (
            db.execute(
                select(SpeakerName).where(
                    SpeakerName.audio_file.in_(session_files),
                    (
                        SpeakerName.speaker_name.in_(aliases)
                        | SpeakerName.local_speaker_label.in_(aliases)
                    ),
                )
            )
            .scalars()
            .all()
        )

        existing_sn_keys = {
            (r.audio_file, r.local_speaker_label) for r in affected_names
        }
        # Also load every SpeakerName key for these files so we do not recreate
        # rows that already exist under a non-alias local label.
        for af, lbl in db.execute(
            select(
                SpeakerName.audio_file,
                SpeakerName.local_speaker_label,
            ).where(SpeakerName.audio_file.in_(session_files))
        ).all():
            existing_sn_keys.add((af, lbl))

        emb_label_rows = db.execute(
            select(Embedding.audio_file, Embedding.local_speaker_label)
            .where(
                Embedding.audio_file.in_(session_files),
                Embedding.local_speaker_label.in_(aliases),
            )
            .distinct()
        ).all()
        missing_sn_pairs = [
            (af, lbl)
            for af, lbl in emb_label_rows
            if (af, lbl) not in existing_sn_keys
        ]

        # Prefer creating SpeakerName rows keyed by the raw embedding label
        # (SPEAKER_XX). When the only alias is a display name with no
        # embedding rows, fall back to that display name so future label
        # lookups still work.
        if not affected_segs and not affected_names and not missing_sn_pairs:
            raise ValueError(
                f"No segments, speaker_names, or embeddings in session "
                f"'{session_id}' match speaker '{from_label}'"
            )

        # ── Apply ─────────────────────────────────────────────────────────
        seg_count = 0
        if affected_segs:
            result = db.execute(
                update(TranscriptionSegment)
                .where(
                    TranscriptionSegment.session_id == session_id,
                    TranscriptionSegment.original_speaker_label.in_(aliases),
                )
                .values(original_speaker_label=to_label)
            )
            seg_count = int(getattr(result, "rowcount", 0) or 0)

        name_update_count = 0
        if affected_names:
            # Keep local_speaker_label as the embedding key (usually
            # SPEAKER_XX) so cosine matches keep resolving.
            result = db.execute(
                update(SpeakerName)
                .where(
                    SpeakerName.audio_file.in_(session_files),
                    (
                        SpeakerName.speaker_name.in_(aliases)
                        | SpeakerName.local_speaker_label.in_(aliases)
                    ),
                )
                .values(
                    speaker_name=to_label,
                    labeled_at=datetime.now(timezone.utc),
                )
            )
            name_update_count = int(getattr(result, "rowcount", 0) or 0)

        created = 0
        now = datetime.now(timezone.utc)
        for emb_af, emb_label in missing_sn_pairs:
            db.merge(
                SpeakerName(
                    id=f"{os.path.basename(emb_af)}_{emb_label}",
                    audio_file=emb_af,
                    local_speaker_label=emb_label,
                    speaker_name=to_label,
                    labeled_at=now,
                )
            )
            created += 1

        state_updated = False
        state = db.get(SessionState, session_id)
        if state and state.speaker_embeddings:
            new_embs = dict(state.speaker_embeddings)
            moved = False
            for alias in aliases:
                if alias in new_embs:
                    payload = new_embs.pop(alias)
                    # Prefer keeping an existing destination entry if both
                    # keys were present.
                    if to_label not in new_embs:
                        new_embs[to_label] = payload
                    else:
                        # Merge durations when both keys were present.
                        prev = new_embs[to_label]
                        try:
                            prev_dur = float(
                                prev.get("total_duration", 0) or 0
                            )
                            cur_dur = float(
                                payload.get("total_duration", 0) or 0
                            )
                            if cur_dur > prev_dur:
                                new_embs[to_label] = payload
                            else:
                                prev["total_duration"] = prev_dur + cur_dur
                                new_embs[to_label] = prev
                        except Exception:
                            new_embs[to_label] = payload
                    moved = True
            if moved:
                state.speaker_embeddings = new_embs
                state_updated = True

        db.commit()

    return RelabelResult(
        session_id=session_id,
        from_label=from_label,
        to_label=to_label,
        aliases=tuple(sorted(aliases)),
        segments_updated=seg_count,
        speaker_names_updated=name_update_count,
        speaker_names_created=created,
        session_state_updated=state_updated,
    )


def preview_session_relabel(
    db_dsn: str,
    session_id: str,
    from_label: str,
    to_label: str,
) -> tuple[list, list, list[tuple[str, str]], Set[str]]:
    """Return preview rows for the CLI (no DB writes).

    Does not mutate the database. Raises the same validation errors as
    :func:`relabel_session_speaker` when the session or labels are empty.
    """
    session_id = (session_id or "").strip()
    from_label = (from_label or "").strip()
    to_label = (to_label or "").strip()
    if not session_id:
        raise ValueError("session id must not be empty")
    if not from_label or not to_label:
        raise ValueError("--from and --to are required")
    if from_label == to_label:
        raise ValueError(f"--from and --to are the same ({from_label!r})")

    engine = get_engine(db_dsn)
    init_db(engine)

    with OrmSession(engine) as db:
        session_files = list(
            db.execute(
                select(TranscriptionSegment.audio_file)
                .where(TranscriptionSegment.session_id == session_id)
                .distinct()
            )
            .scalars()
            .all()
        )
        if not session_files:
            return [], [], [], {from_label}

        aliases = _expand_aliases(db, session_files, from_label)
        aliases.discard(to_label)

        affected_segs = (
            db.execute(
                select(TranscriptionSegment)
                .where(
                    TranscriptionSegment.session_id == session_id,
                    TranscriptionSegment.original_speaker_label.in_(aliases),
                )
                .order_by(TranscriptionSegment.start_time)
            )
            .scalars()
            .all()
        )

        affected_names = (
            db.execute(
                select(SpeakerName).where(
                    SpeakerName.audio_file.in_(session_files),
                    (
                        SpeakerName.speaker_name.in_(aliases)
                        | SpeakerName.local_speaker_label.in_(aliases)
                    ),
                )
            )
            .scalars()
            .all()
        )

        existing_sn_keys = set(
            db.execute(
                select(
                    SpeakerName.audio_file,
                    SpeakerName.local_speaker_label,
                ).where(SpeakerName.audio_file.in_(session_files))
            ).all()
        )
        emb_label_rows = db.execute(
            select(Embedding.audio_file, Embedding.local_speaker_label)
            .where(
                Embedding.audio_file.in_(session_files),
                Embedding.local_speaker_label.in_(aliases),
            )
            .distinct()
        ).all()
        missing_sn_pairs = [
            (af, lbl)
            for af, lbl in emb_label_rows
            if (af, lbl) not in existing_sn_keys
        ]
        return (
            list(affected_segs),
            list(affected_names),
            missing_sn_pairs,
            aliases,
        )
