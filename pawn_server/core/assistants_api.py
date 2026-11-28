"""HTTP routes for voice assistants beyond the phone bridge.

* ``/v1/assistants/wake/*`` — shared wake-word training (clips in, models out).
  Clip upload and model download accept a device token (``device_id`` form
  field / query) or the shared token; listing, deleting and publishing need
  the shared token.
* ``/v1/portal/assistants`` — the portal's Assistants tab: one call with every
  device, its presence, voice thread, recent commands, access requests and
  wake-clip counts.

See docs/ASSISTANTS.md.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pawn_server.core import device_commands, devices, voice_threads, wake_store

_bearer = HTTPBearer(auto_error=False)


def _http(exc: Exception) -> HTTPException:
    return HTTPException(status_code=getattr(exc, "status_code", 422), detail=str(exc))


def _parse_dt(raw: Optional[str], field: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return devices.parse_ts(int(raw) if raw.isdigit() else raw)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be epoch or ISO-8601") from exc


def build_router(get_cfg: Callable[[], Any], require_token: Callable[..., None]) -> APIRouter:
    router = APIRouter(tags=["assistants"])

    async def _device_or_shared(
        device_id: Optional[str], credentials: Optional[HTTPAuthorizationCredentials], cfg: Any
    ) -> Optional[str]:
        """Return the authenticated device id (None for the shared token)."""
        token = credentials.credentials if credentials else None
        shared = cfg.api_token
        if not shared or token == shared:
            return device_id or None
        if device_id and await asyncio.to_thread(devices.authenticate, cfg.db_dsn, device_id, token):
            return device_id
        raise HTTPException(status_code=401, detail="Invalid or missing device token")

    # ── Wake clips ────────────────────────────────────────────────────────────

    @router.post("/v1/assistants/wake/clips")
    async def upload_clip(
        file: UploadFile = File(...),
        label: str = Form(...),
        device_id: Optional[str] = Form(None),
        score: Optional[float] = Form(None),
        ts: Optional[str] = Form(None),
        meta: Optional[str] = Form(None),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        device = await _device_or_shared(device_id, credentials, cfg)
        data = await file.read()
        try:
            extra = json.loads(meta) if meta else {}
            if not isinstance(extra, dict):
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="meta must be a JSON object") from exc
        try:
            view = await asyncio.to_thread(
                wake_store.add_clip,
                cfg,
                data,
                label=label,
                device_id=device,
                score=score,
                captured_at=_parse_dt(ts, "ts"),
                meta=extra,
            )
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc
        # Retention runs on upload, like phone events on ingest.
        await asyncio.to_thread(wake_store.purge_clips, cfg)
        return view

    @router.get("/v1/assistants/wake/clips", dependencies=[Depends(require_token)])
    async def list_clips(
        since: Optional[str] = None,
        label: Optional[str] = None,
        device_id: Optional[str] = None,
        limit: int = Query(500, ge=1, le=5000),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        clips = await asyncio.to_thread(
            wake_store.list_clips,
            cfg.db_dsn,
            since=_parse_dt(since, "since"),
            label=label,
            device_id=device_id,
            limit=limit,
        )
        return {"clips": clips}

    @router.get("/v1/assistants/wake/clips/{clip_id}", dependencies=[Depends(require_token)])
    async def get_clip(clip_id: str, cfg: Any = Depends(get_cfg)) -> FileResponse:
        try:
            path = await asyncio.to_thread(wake_store.clip_path, cfg, clip_id)
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @router.delete("/v1/assistants/wake/clips/{clip_id}", dependencies=[Depends(require_token)])
    async def delete_clip(clip_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        try:
            await asyncio.to_thread(wake_store.delete_clip, cfg, clip_id)
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc
        return {"deleted": clip_id}

    # ── Wake models ───────────────────────────────────────────────────────────

    @router.post("/v1/assistants/wake/models", dependencies=[Depends(require_token)])
    async def publish_model(
        file: UploadFile = File(...),
        version: Optional[str] = Form(None),
        name: str = Form("hey_pawn"),
        notes: Optional[str] = Form(None),
        thresholds: Optional[str] = Form(None),
        activate: bool = Form(True),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        data = await file.read()
        try:
            levels = json.loads(thresholds) if thresholds else {}
            if not isinstance(levels, dict):
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="thresholds must be a JSON object") from exc
        try:
            return await asyncio.to_thread(
                wake_store.publish_model,
                cfg,
                data,
                version=version,
                name=name,
                notes=notes,
                thresholds=levels,
                activate=activate,
            )
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc

    @router.get("/v1/assistants/wake/models", dependencies=[Depends(require_token)])
    async def list_models(cfg: Any = Depends(get_cfg)) -> dict:
        return {"models": await asyncio.to_thread(wake_store.list_models, cfg.db_dsn)}

    @router.get("/v1/assistants/wake/models/current")
    async def current_model(
        device_id: Optional[str] = None,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        await _device_or_shared(device_id, credentials, cfg)
        return {"model": await asyncio.to_thread(wake_store.current_model, cfg.db_dsn)}

    @router.post(
        "/v1/assistants/wake/models/{version}/activate", dependencies=[Depends(require_token)]
    )
    async def activate_model(version: str, cfg: Any = Depends(get_cfg)) -> dict:
        try:
            return await asyncio.to_thread(wake_store.activate_model, cfg.db_dsn, version)
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc

    @router.get("/v1/assistants/wake/models/{version}")
    async def download_model(
        version: str,
        device_id: Optional[str] = None,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        cfg: Any = Depends(get_cfg),
    ) -> FileResponse:
        await _device_or_shared(device_id, credentials, cfg)
        try:
            path = await asyncio.to_thread(wake_store.model_path, cfg, version)
        except wake_store.WakeStoreError as exc:
            raise _http(exc) from exc
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    # ── Portal overview ───────────────────────────────────────────────────────

    def _overview(cfg: Any) -> dict[str, Any]:
        dsn = cfg.db_dsn
        window = float(getattr(getattr(cfg, "devices", None), "handoff_window_s", 600.0))
        threads = voice_threads.list_threads(dsn, limit=50)
        requests = voice_threads.list_access_requests(dsn, limit=50)
        out = []
        for view in devices.list_devices(dsn):
            device_id = view["device_id"]
            mine = [t for t in threads if t["device_id"] == device_id]
            active = next((t for t in mine if t["state"] == "active"), None)
            out.append(
                {
                    **view,
                    "thread": active,
                    "recent_threads": mine[:5],
                    "commands": device_commands.list_commands(dsn, device_id, limit=10),
                    "access_requests": [
                        r
                        for r in requests
                        if device_id in (r["requester_device_id"], r["approver_device_id"])
                    ][:10],
                    "wake_clips": wake_store.clip_counts(dsn, device_id=device_id),
                }
            )
        return {
            "assistants": out,
            "wake": {
                "model": wake_store.current_model(dsn),
                "models": wake_store.list_models(dsn)[:10],
                "clips": wake_store.clip_counts(dsn),
            },
            "handoff_window_s": window,
            "presence_fresh_s": float(
                getattr(getattr(cfg, "devices", None), "presence_fresh_s", 600.0)
            ),
            "tools": {
                name: {"action": spec.action, "summary": spec.summary, "args": [*spec.required, *spec.optional]}
                for name, spec in device_commands.TOOLS.items()
            },
        }

    @router.get("/v1/portal/assistants", dependencies=[Depends(require_token)])
    async def overview(cfg: Any = Depends(get_cfg)) -> dict:
        return await asyncio.to_thread(_overview, cfg)

    @router.get("/v1/portal/assistants/{device_id}/commands", dependencies=[Depends(require_token)])
    async def commands(
        device_id: str, limit: int = Query(50, ge=1, le=200), cfg: Any = Depends(get_cfg)
    ) -> dict:
        return {
            "commands": await asyncio.to_thread(
                device_commands.list_commands, cfg.db_dsn, device_id, limit=limit
            )
        }

    return router
