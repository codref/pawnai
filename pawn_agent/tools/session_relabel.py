"""session_relabel — rename a speaker across one diarization session."""

from __future__ import annotations

from pawn_agent.utils.config import AgentConfig
from pawn_diarize.core.session_relabel import relabel_session_speaker


def session_relabel_impl(
    cfg: AgentConfig,
    *,
    session_id: str,
    from_speaker: str,
    to_speaker: str,
    push_vault: bool = False,
) -> str:
    """Bulk-rename a speaker in *session_id* and propagate the display name.

    Updates transcript segments, ``speaker_names`` (so embedding matches
    resolve to *to_speaker*), and ``session_state`` prior-speaker keys.
    Accepts either a raw ``SPEAKER_XX`` label or a current display name as
    *from_speaker*.
    """
    result = relabel_session_speaker(
        db_dsn=cfg.db_dsn,
        session_id=session_id,
        from_label=from_speaker,
        to_label=to_speaker,
    )
    summary = result.summary()
    from pawn_diarize.core.vault_transcript import refresh_transcript_after_relabel

    vault_status = refresh_transcript_after_relabel(
        session_id,
        db_dsn=cfg.db_dsn,
        cfg=cfg,
        force=push_vault,
    )
    if vault_status:
        return f"{summary}\n{vault_status}"
    return summary
