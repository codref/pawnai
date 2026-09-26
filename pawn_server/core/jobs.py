"""Background jobs submitted by the Obsidian plugin (``/v1/jobs``).

Every job is accepted immediately and runs as an asyncio task in the server
process. Rows live in ``vault_tasks`` (``kind`` = ``ask`` | ``push_note`` |
``upload``). ``ask`` jobs also get a ``Pawn/Tasks/{id}.md`` note on S3 so
devices that are offline or on mobile see status and result through Sync
Engine, and the vault watcher can resume them.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from pathlib import PurePosixPath
from typing import Any, Optional, Sequence

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.tools.push_queue_message import push_queue_message_impl
from pawn_agent.utils.db import (
    VaultTask,
    claim_vault_task,
    get_vault_task,
    list_vault_tasks,
    update_vault_task,
    upsert_vault_task,
)
from pawn_core.vault import VaultError, VaultWriteDenied, normalize_vault_key
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.job_events import publish_job_event
from pawn_server.core.vault_protocol import instruction_hash, render_task_note
from pawn_server.core.vault_tasks import (
    effective_conversation,
    execute_vault_task,
    get_background_task,
    task_key_for,
    track_background_task,
)

logger = logging.getLogger(__name__)

JOB_KINDS = ("ask", "push_note", "upload")
TERMINAL_STATUSES = frozenset({"review", "done", "blocked"})

_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".aac", ".mp4"}
)
_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".markdown", ".csv", ".json"})


class JobError(Exception):
    """Job cannot be accepted or changed (maps to a 4xx response)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _vault_store_or_none(cfg: Any) -> Any:
    try:
        return vault_store_from_config(cfg)
    except (ValueError, VaultError):
        return None


def _agent_root(cfg: Any) -> str:
    return normalize_vault_key(cfg.vault.agent_root).rstrip("/") or "Pawn"


def _job_key(cfg: Any, job_id: str) -> str:
    return f"{_agent_root(cfg)}/Jobs/{job_id}"


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value is not None else None


def serialize_job(row: VaultTask) -> dict[str, Any]:
    """JSON shape returned by the jobs API and SSE stream."""
    payload = dict(row.payload or {})
    instruction = row.instruction_text or ""
    return {
        "id": row.id,
        "kind": row.kind or "ask",
        "status": row.status,
        "title": (instruction.strip().splitlines() or [row.kind or "job"])[0][:120],
        "instruction": instruction,
        "conversation": row.conversation_id,
        "note_path": row.note_path,
        "task_key": row.key if (row.kind or "ask") == "ask" else None,
        "result": row.result_text,
        "error_code": row.error_code,
        "approved": row.indexed_at is not None,
        "agent_run_id": row.agent_run_id,
        "via": row.via,
        "payload": {k: v for k, v in payload.items() if k != "context"},
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def build_job_context(
    *,
    selection: Optional[str] = None,
    context_paths: Sequence[str] = (),
    context: Optional[str] = None,
) -> str:
    """Render the ``## Context`` section for an ask job."""
    parts: list[str] = []
    if selection and selection.strip():
        parts.append("Selected text:\n\n" + selection.strip())
    links = [p for p in (normalize_vault_key(x).removesuffix(".md") for x in context_paths) if p]
    if links:
        parts.append("Context notes:\n" + "\n".join(f"- [[{p}]]" for p in links))
    if context and context.strip():
        parts.append(context.strip())
    return "\n\n".join(parts)


def _spawn(job_id: str, coro: Any) -> None:
    task = asyncio.create_task(coro, name=f"pawn-job-{job_id}")
    track_background_task(job_id, task)


# ── ask ────────────────────────────────────────────────────────────────────


async def create_ask_job(
    cfg: Any,
    *,
    registry: SallmSessionRegistry,
    job_id: str,
    instruction: str,
    note_path: Optional[str] = None,
    selection: Optional[str] = None,
    context_paths: Sequence[str] = (),
    context: Optional[str] = None,
    conversation: Optional[str] = None,
) -> VaultTask:
    """Accept an agent job and start it in the background."""
    if not instruction.strip():
        raise JobError("instruction is required", status_code=422)
    key = task_key_for(cfg, task_id=job_id)
    conv = effective_conversation(conversation, note_path, key)
    ctx = build_job_context(selection=selection, context_paths=context_paths, context=context)
    effective_id = upsert_vault_task(
        cfg.db_dsn,
        task_id=job_id,
        key=key,
        instruction_hash=instruction_hash(instruction),
        conversation_id=conv,
        instruction_text=instruction,
        note_path=note_path,
        via="http",
        status="queued",
        kind="ask",
        payload={"context": ctx, "context_paths": list(context_paths)},
    )
    if not claim_vault_task(cfg.db_dsn, effective_id):
        row = get_vault_task(cfg.db_dsn, effective_id)
        if row is None:
            raise JobError("job disappeared", status_code=500)
        return row

    store = _vault_store_or_none(cfg)
    if store is not None:
        note = render_task_note(
            task_id=effective_id,
            status="running",
            instruction=instruction,
            context=ctx,
            conversation=conv,
            note_path=note_path,
        )
        try:
            st = await asyncio.to_thread(store.write, key, note)
            update_vault_task(cfg.db_dsn, effective_id, etag=st.etag)
        except Exception as exc:
            logger.warning("Could not write task note %s: %s", key, exc)
            store = None

    publish_job_event(effective_id, "claimed", conversation=conv)
    _spawn(
        effective_id,
        execute_vault_task(
            cfg,
            effective_id,
            registry=registry,
            store=store,
            write_result_to_vault=store is not None,
        ),
    )
    row = get_vault_task(cfg.db_dsn, effective_id)
    assert row is not None
    return row


# ── push_note ──────────────────────────────────────────────────────────────


async def _run_push_note(cfg: Any, job_id: str, path: str, content: str, mode: str) -> None:
    store = _vault_store_or_none(cfg)
    row = get_vault_task(cfg.db_dsn, job_id)
    conv = row.conversation_id if row else None
    update_vault_task(cfg.db_dsn, job_id, status="running")
    publish_job_event(job_id, "running", kind="push_note", conversation=conv)
    try:
        if store is None:
            raise VaultError("vault is not configured on the server")
        if mode == "append":
            await asyncio.to_thread(store.append, path, content)
        else:
            await asyncio.to_thread(store.write, path, content)
    except VaultWriteDenied as exc:
        _finish(cfg, job_id, "blocked", "push_note", conv, f"Write denied: {exc}", "write_denied")
        return
    except Exception as exc:
        logger.error("push_note %s failed: %s", job_id, exc, exc_info=True)
        _finish(cfg, job_id, "blocked", "push_note", conv, str(exc)[:2000], "write_failed")
        return
    verb = "Appended to" if mode == "append" else "Wrote"
    _finish(cfg, job_id, "done", "push_note", conv, f"{verb} [[{path.removesuffix('.md')}]]")


def _finish(
    cfg: Any,
    job_id: str,
    status: str,
    kind: str,
    conversation: Optional[str],
    result: str,
    error_code: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    update_vault_task(
        cfg.db_dsn,
        job_id,
        status=status,
        result_text=result,
        error_code=error_code,
        payload=payload,
    )
    publish_job_event(job_id, status, kind=kind, conversation=conversation, error_code=error_code)


def _new_side_job(
    cfg: Any,
    *,
    job_id: str,
    kind: str,
    title: str,
    conversation: Optional[str],
    note_path: Optional[str],
    payload: dict,
) -> str:
    key = _job_key(cfg, job_id)
    effective_id = upsert_vault_task(
        cfg.db_dsn,
        task_id=job_id,
        key=key,
        instruction_hash=instruction_hash(f"{kind}:{job_id}"),
        conversation_id=conversation or f"job:{job_id}",
        instruction_text=title,
        note_path=note_path,
        via="http",
        status="queued",
        kind=kind,
        payload=payload,
    )
    publish_job_event(effective_id, "queued", kind=kind, conversation=conversation)
    return effective_id


async def create_push_note_job(
    cfg: Any,
    *,
    job_id: str,
    path: str,
    content: str,
    mode: str = "replace",
    conversation: Optional[str] = None,
) -> VaultTask:
    """Accept a note write (vault guards apply) and run it in the background."""
    key = normalize_vault_key(path)
    if not key:
        raise JobError("path is required", status_code=422)
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    if mode not in {"replace", "append"}:
        raise JobError("mode must be 'replace' or 'append'", status_code=422)
    effective_id = _new_side_job(
        cfg,
        job_id=job_id,
        kind="push_note",
        title=f"{'Append to' if mode == 'append' else 'Write'} {key}",
        conversation=conversation,
        note_path=key,
        payload={"path": key, "mode": mode},
    )
    _spawn(effective_id, _run_push_note(cfg, effective_id, key, content, mode))
    row = get_vault_task(cfg.db_dsn, effective_id)
    assert row is not None
    return row


# ── upload ─────────────────────────────────────────────────────────────────


def _safe_filename(name: str) -> str:
    base = PurePosixPath((name or "").replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ]+", "_", base).strip(" .")
    return base or "upload.bin"


def is_audio(filename: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.startswith("audio/"):
        return True
    return PurePosixPath(filename).suffix.lower() in _AUDIO_EXTENSIONS


def _upload_audio_to_s3(cfg: Any, key: str, data: bytes, content_type: str) -> str:
    import boto3  # noqa: PLC0415
    from botocore.config import Config as BotoConfig  # noqa: PLC0415

    s3 = getattr(cfg, "s3", None)
    if s3 is None or not getattr(s3, "bucket", None):
        raise VaultError("no s3: section configured for audio uploads")
    session_kwargs: dict[str, Any] = {}
    if s3.access_key and s3.secret_key:
        session_kwargs["aws_access_key_id"] = s3.access_key
        session_kwargs["aws_secret_access_key"] = s3.secret_key
    if s3.region:
        session_kwargs["region_name"] = s3.region
    client = boto3.session.Session(**session_kwargs).client(
        "s3",
        endpoint_url=s3.endpoint_url,
        verify=s3.verify_ssl,
        config=BotoConfig(s3={"addressing_style": "path" if s3.path_style else "virtual"}),
    )
    prefix = (s3.prefix or "").strip("/")
    full_key = f"{prefix}/{key}" if prefix else key
    client.put_object(Bucket=s3.bucket, Key=full_key, Body=data, ContentType=content_type)
    return f"s3://{s3.bucket}/{full_key}"


async def _run_upload(
    cfg: Any,
    job_id: str,
    *,
    filename: str,
    data: bytes,
    content_type: str,
    index: bool,
    registry: SallmSessionRegistry,
) -> None:
    row = get_vault_task(cfg.db_dsn, job_id)
    conv = row.conversation_id if row else None
    payload = dict((row.payload if row else None) or {})
    update_vault_task(cfg.db_dsn, job_id, status="running")
    publish_job_event(job_id, "running", kind="upload", conversation=conv)
    try:
        if is_audio(filename, content_type):
            prefix = str(getattr(cfg.api, "upload_s3_prefix", "uploads/obsidian")).strip("/")
            uri = await asyncio.to_thread(
                _upload_audio_to_s3, cfg, f"{prefix}/{job_id}/{filename}", data, content_type
            )
            session = PurePosixPath(filename).stem
            target = getattr(cfg.api, "upload_audio_target", "diarize")
            receipt = await push_queue_message_impl(
                cfg,
                target=target,
                command="transcribe-diarize",
                payload={"audio_paths": [uri], "session": session},
            )
            if receipt.startswith("Error"):
                raise VaultError(f"{receipt} (stored at {uri})")
            payload.update({"s3_uri": uri, "session": session, "target": target})
            _finish(
                cfg,
                job_id,
                "done",
                "upload",
                conv,
                f"Queued `{filename}` for transcription as session `{session}`.",
                payload=payload,
            )
            return

        store = vault_store_from_config(cfg)
        key = f"{_agent_root(cfg)}/Inbox/{filename}"
        suffix = PurePosixPath(filename).suffix.lower()
        text: Optional[str] = None
        if suffix in _TEXT_EXTENSIONS:
            text = data.decode("utf-8", errors="replace")
            await asyncio.to_thread(store.write, key, text, skip_guards=False)
        else:
            await asyncio.to_thread(store.write_bytes, key, data, content_type=content_type)
        payload["vault_key"] = key
        message = f"Saved to [[{key}]]."
        if index and text and text.strip() and conv:
            session = await registry.get_or_create(conv, cfg, cfg.db_dsn)
            await asyncio.to_thread(
                session._agent.remember,  # noqa: SLF001 — intentional index path
                f"Uploaded document {key}:\n\n{text[:20000]}",
                source=f"upload:{job_id}",
                index_raw=False,
            )
            payload["indexed"] = True
            message += " Indexed into Pawn memory."
        _finish(cfg, job_id, "done", "upload", conv, message, payload=payload)
    except Exception as exc:
        logger.error("upload %s failed: %s", job_id, exc, exc_info=True)
        _finish(cfg, job_id, "blocked", "upload", conv, str(exc)[:2000], "upload_failed")


async def create_upload_job(
    cfg: Any,
    *,
    registry: SallmSessionRegistry,
    job_id: str,
    filename: str,
    data: bytes,
    content_type: Optional[str] = None,
    note_path: Optional[str] = None,
    conversation: Optional[str] = None,
    index: bool = False,
) -> VaultTask:
    """Accept an upload: audio goes to transcription, other files to ``Pawn/Inbox/``."""
    if not data:
        raise JobError("empty upload", status_code=422)
    name = _safe_filename(filename)
    ctype = content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    effective_id = _new_side_job(
        cfg,
        job_id=job_id,
        kind="upload",
        title=f"Upload {name}",
        conversation=conversation,
        note_path=note_path,
        payload={
            "filename": name,
            "content_type": ctype,
            "size": len(data),
            "audio": is_audio(name, ctype),
        },
    )
    _spawn(
        effective_id,
        _run_upload(
            cfg,
            effective_id,
            filename=name,
            data=data,
            content_type=ctype,
            index=index,
            registry=registry,
        ),
    )
    row = get_vault_task(cfg.db_dsn, effective_id)
    assert row is not None
    return row


# ── queries / control ──────────────────────────────────────────────────────


def list_jobs(
    cfg: Any,
    *,
    conversation: Optional[str] = None,
    statuses: Optional[Sequence[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = list_vault_tasks(
        cfg.db_dsn,
        statuses=list(statuses) if statuses else None,
        conversation_id=conversation,
        newest_first=True,
        limit=max(1, min(int(limit), 200)),
    )
    return [serialize_job(r) for r in rows]


def get_job(cfg: Any, job_id: str) -> Optional[dict[str, Any]]:
    row = get_vault_task(cfg.db_dsn, job_id)
    return serialize_job(row) if row else None


async def cancel_job(cfg: Any, job_id: str) -> dict[str, Any]:
    row = get_vault_task(cfg.db_dsn, job_id)
    if row is None:
        raise JobError("job not found", status_code=404)
    if row.status in TERMINAL_STATUSES:
        raise JobError(f"job is already {row.status}", status_code=409)
    task = get_background_task(job_id)
    if task is not None:
        task.cancel()
    _finish(
        cfg, job_id, "blocked", row.kind or "ask", row.conversation_id, "Cancelled.", "cancelled"
    )
    if (row.kind or "ask") == "ask":
        store = _vault_store_or_none(cfg)
        if store is not None:
            try:
                note = render_task_note(
                    task_id=job_id,
                    status="blocked",
                    instruction=row.instruction_text or "",
                    context=str((row.payload or {}).get("context") or ""),
                    result="Cancelled.",
                    conversation=row.conversation_id,
                    note_path=row.note_path,
                )
                await asyncio.to_thread(store.write, row.key, note)
            except Exception as exc:
                logger.debug("Could not mark task note cancelled: %s", exc)
    out = get_job(cfg, job_id)
    assert out is not None
    return out
