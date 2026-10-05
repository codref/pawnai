"""Phone audio dropped in the vault becomes a diarization session."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_AUDIO = {".m4a", ".webm", ".wav", ".mp3", ".ogg", ".flac"}


def is_audio_key(key: str) -> bool:
    return Path(key).suffix.lower() in _AUDIO


async def capture_audio(cfg: AgentConfig, store: Any, key: str) -> str:
    """Copy *key* to the upload bucket and enqueue transcription."""
    from pawn_agent.tools.push_queue_message import push_queue_message_impl  # noqa: PLC0415
    from pawn_server.core.jobs import _upload_audio_to_s3  # noqa: PLC0415

    data = store.read_bytes(key)
    stem = Path(key).stem
    day = datetime.now(timezone.utc).date().isoformat()
    session = f"capture-{day}-{stem}"[:80]
    prefix = str(getattr(cfg.api, "upload_s3_prefix", "uploads/obsidian")).strip("/")
    uri = _upload_audio_to_s3(
        cfg, f"{prefix}/capture/{session}{Path(key).suffix}", data, "application/octet-stream"
    )
    target = getattr(cfg.api, "upload_audio_target", "diarize")
    receipt = await push_queue_message_impl(
        cfg,
        target=target,
        command="transcribe-diarize",
        payload={
            "audio_paths": [uri],
            "session": session,
            "chain_agent": {"command": "session_completed"},
        },
    )
    stub = (
        f"---\npawn: capture\nsession: {session}\n---\n"
        f"# {session}\n\nRecording: [[{key}]]\n\n{receipt}\n"
    )
    try:
        store.write(f"{cfg.coworker.capture_dir.strip('/')}/{session}.md", stub)
    except Exception as exc:
        logger.warning("capture stub skipped: %s", exc)
    itemdb.upsert_note_state(
        cfg.db_dsn,
        key,
        content_hash="audio",
        last_processed_hash="audio",
        last_processed_at=datetime.now(timezone.utc),
    )
    return receipt
