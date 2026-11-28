"""HTTP routes for the phone bridge (``/v1/devices/*``).

Auth: registration and listing use the shared API token. Per-device routes
accept that device's own token (issued once at registration) or the shared
token, so a phone can be revoked without rotating the server token.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from pawn_server.core import device_commands, devices

_bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    name: str = ""
    # phone | desktop (see docs/ASSISTANTS.md)
    kind: str = "phone"
    platform: Optional[str] = None


class PresenceRequest(BaseModel):
    state: str
    headset: Optional[bool] = None
    on_call: Optional[bool] = None
    in_session: Optional[bool] = None
    last_wake_at: Optional[Any] = None


class UpdateRequest(BaseModel):
    name: Optional[str] = None
    unifiedpush_endpoint: Optional[str] = None
    enabled_categories: Optional[list[str]] = None
    capabilities: Optional[list[str]] = None


class EventsRequest(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)


class CommandRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_s: Optional[float] = None


class ResultRequest(BaseModel):
    status: str
    result: Optional[dict[str, Any]] = None
    message: str = ""


def _raise(exc: devices.DeviceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc))


def build_router(get_cfg: Callable[[], Any], require_token: Callable[..., None]) -> APIRouter:
    """Build the router; dependencies come from api_server to avoid a cycle."""
    router = APIRouter(prefix="/v1/devices", tags=["devices"])

    async def require_device(
        device_id: str,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        cfg: Any = Depends(get_cfg),
    ) -> str:
        token = credentials.credentials if credentials else None
        shared = cfg.api_token
        if shared and token == shared:
            return device_id
        if await asyncio.to_thread(devices.authenticate, cfg.db_dsn, device_id, token):
            return device_id
        raise HTTPException(status_code=401, detail="Invalid or missing device token")

    @router.post("/register", dependencies=[Depends(require_token)])
    async def register(body: RegisterRequest, cfg: Any = Depends(get_cfg)) -> dict:
        try:
            return await asyncio.to_thread(
                devices.register_device,
                cfg.db_dsn,
                body.name,
                kind=body.kind,
                platform=body.platform,
            )
        except devices.DeviceError as exc:
            _raise(exc)
            raise

    @router.get("", dependencies=[Depends(require_token)])
    async def list_all(cfg: Any = Depends(get_cfg)) -> dict:
        return {"devices": await asyncio.to_thread(devices.list_devices, cfg.db_dsn)}

    @router.get("/{device_id}")
    async def get_one(device_id: str = Depends(require_device), cfg: Any = Depends(get_cfg)) -> dict:
        try:
            return await asyncio.to_thread(devices.get_device, cfg.db_dsn, device_id)
        except devices.DeviceError as exc:
            _raise(exc)
            raise

    @router.put("/{device_id}")
    async def update(
        body: UpdateRequest,
        device_id: str = Depends(require_device),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        try:
            view, disabled = await asyncio.to_thread(
                devices.update_device,
                cfg.db_dsn,
                device_id,
                name=body.name,
                unifiedpush_endpoint=body.unifiedpush_endpoint,
                enabled_categories=body.enabled_categories,
                capabilities=body.capabilities,
            )
        except devices.DeviceError as exc:
            _raise(exc)
            raise
        return {**view, "purged_categories": disabled}

    @router.delete("/{device_id}")
    async def remove(device_id: str = Depends(require_device), cfg: Any = Depends(get_cfg)) -> dict:
        try:
            removed = await asyncio.to_thread(devices.delete_device, cfg.db_dsn, device_id)
        except devices.DeviceError as exc:
            _raise(exc)
            raise
        return {"deleted": device_id, "events_removed": removed}

    @router.post("/{device_id}/events")
    async def ingest(
        body: EventsRequest,
        background: BackgroundTasks,
        device_id: str = Depends(require_device),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        settings = getattr(cfg, "devices", None)
        max_batch = int(getattr(settings, "max_batch", 500))
        if len(body.events) > max_batch:
            raise HTTPException(status_code=413, detail=f"At most {max_batch} events per request")
        try:
            result = await asyncio.to_thread(
                devices.ingest_events,
                cfg.db_dsn,
                device_id,
                body.events,
                index_kinds=list(getattr(settings, "index_kinds", []) or []),
                retention_days=int(getattr(settings, "retention_days", 0) or 0),
                coworker_kinds=list(getattr(settings, "coworker_kinds", []) or []),
            )
        except devices.DeviceError as exc:
            _raise(exc)
            raise
        if result.to_index:
            background.add_task(asyncio.to_thread, devices.index_events, cfg, result.to_index)
        if result.to_coworker:
            background.add_task(devices.forward_to_coworker, cfg, result.to_coworker)
        return {
            "accepted": result.accepted,
            "duplicates": result.duplicates,
            "rejected": result.rejected,
        }

    @router.post("/{device_id}/presence")
    async def presence(
        body: PresenceRequest,
        device_id: str = Depends(require_device),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        try:
            return await asyncio.to_thread(
                devices.update_presence, cfg.db_dsn, device_id, body.model_dump(exclude_none=True)
            )
        except devices.DeviceError as exc:
            _raise(exc)
            raise

    # ── Live tools (command queue) ────────────────────────────────────────────

    @router.post("/{device_id}/commands", dependencies=[Depends(require_token)])
    async def run(device_id: str, body: CommandRequest, cfg: Any = Depends(get_cfg)) -> dict:
        """Queue a live tool and wait for the phone (shared token: agent / admin)."""
        settings = getattr(cfg, "devices", None)
        budget = float(getattr(settings, "command_timeout_s", 45.0))
        timeout = min(budget, body.timeout_s) if body.timeout_s else budget
        try:
            view = await asyncio.to_thread(
                device_commands.run_command,
                cfg.db_dsn,
                body.tool,
                body.args,
                device=device_id,
                timeout_s=timeout,
                fresh_s=float(getattr(settings, "presence_fresh_s", 600.0)),
            )
        except device_commands.CommandError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {**view, "text": device_commands.describe(view)}

    @router.get("/{device_id}/commands", dependencies=[Depends(require_token)])
    async def history(
        device_id: str, limit: int = Query(20, ge=1, le=200), cfg: Any = Depends(get_cfg)
    ) -> dict:
        return {
            "commands": await asyncio.to_thread(
                device_commands.list_commands, cfg.db_dsn, device_id, limit=limit
            )
        }

    @router.get("/{device_id}/commands/stream")
    async def stream(
        device_id: str = Depends(require_device),
        cfg: Any = Depends(get_cfg),
        poll: float = Query(1.0, ge=0.05, le=10.0),
        max_seconds: float = Query(0.0, ge=0.0),
    ) -> StreamingResponse:
        """SSE command channel for assistants without UnifiedPush (desktops).

        Emits ``event: cmd`` with ``{"commands": [...]}`` (rows become
        ``delivered``, exactly like ``…/pending``) and ``: keep-alive``
        comments. The DB is polled because CliTools queue commands from other
        processes. ``max_seconds`` > 0 closes the stream (tests, proxies).
        """
        await asyncio.to_thread(devices.get_device, cfg.db_dsn, device_id)
        keepalive = float(getattr(cfg.api, "stream_keepalive_seconds", 10.0) or 10.0)

        async def _gen() -> AsyncIterator[str]:
            yield "event: ready\ndata: " + json.dumps({"device_id": device_id}) + "\n\n"
            started = last_beat = time.monotonic()
            while True:
                commands = await asyncio.to_thread(
                    device_commands.take_pending, cfg.db_dsn, device_id
                )
                if commands:
                    yield "event: cmd\ndata: " + json.dumps({"commands": commands}) + "\n\n"
                    last_beat = time.monotonic()
                now = time.monotonic()
                if now - last_beat >= keepalive:
                    yield ": keep-alive\n\n"
                    last_beat = now
                if max_seconds and now - started >= max_seconds:
                    return
                await asyncio.sleep(poll)

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/{device_id}/commands/pending")
    async def pending(device_id: str = Depends(require_device), cfg: Any = Depends(get_cfg)) -> dict:
        return {"commands": await asyncio.to_thread(device_commands.take_pending, cfg.db_dsn, device_id)}

    @router.post("/{device_id}/commands/{command_id}/result")
    async def result(
        command_id: str,
        body: ResultRequest,
        device_id: str = Depends(require_device),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        try:
            return await asyncio.to_thread(
                device_commands.post_result,
                cfg.db_dsn,
                device_id,
                command_id,
                status=body.status,
                result=body.result,
                message=body.message,
            )
        except device_commands.CommandError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return router
