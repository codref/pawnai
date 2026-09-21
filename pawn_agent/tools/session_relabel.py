"""session_relabel — rename a speaker across one diarization session."""

from __future__ import annotations

from pawn_agent.utils.config import AgentConfig
from pawn_diarize.core.session_relabel import relabel_session_speaker
from pawn_diarize.core.siyuan_transcript import (
    refresh_transcript_after_relabel,
)


def session_relabel_impl(
    cfg: AgentConfig,
    *,
    session_id: str,
    from_speaker: str,
    to_speaker: str,
    push_siyuan: bool = False,
) -> str:
    """Bulk-rename a speaker in *session_id* and propagate the display name.

    Updates transcript segments, ``speaker_names`` (so embedding matches
    resolve to *to_speaker*), and ``session_state`` prior-speaker keys.
    Accepts either a raw ``SPEAKER_XX`` label or a current display name as
    *from_speaker*.

    After the DB update, refreshes the SiYuan diary transcript when a
    ``siyuan_session_docs`` mapping already exists. Pass *push_siyuan* to
    force create/update even without a prior mapping.
    """
    result = relabel_session_speaker(
        db_dsn=cfg.db_dsn,
        session_id=session_id,
        from_label=from_speaker,
        to_label=to_speaker,
    )
    parts = [result.summary()]

    sy_status = refresh_transcript_after_relabel(
        session_id,
        db_dsn=cfg.db_dsn,
        url=cfg.siyuan_url,
        token=cfg.siyuan_token,
        notebook=cfg.siyuan_notebook,
        path_template=cfg.siyuan_path_template,
        daily_path_template=cfg.siyuan_daily_template,
        force=push_siyuan,
    )
    if sy_status:
        parts.append(f"SiYuan: {sy_status}")
    elif push_siyuan:
        parts.append("SiYuan: skipped (notebook not configured)")

    return " ".join(parts)
