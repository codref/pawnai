"""Portal settings routes (``/v1/portal/settings``). See ``settings_registry``.

Changes go to ``{state_dir}/settings.override.yaml`` (``pawnai.yaml`` is never
rewritten) and ``load_config`` merges that file over yaml and env. ``live`` and
``new_chats`` fields are also patched into the running config in place;
``restart`` fields show as pending until the server restarts.

Every value in a response is the *configured* one (yaml + env + overrides,
rebuilt per request); ``running`` is what the live process currently uses.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Body, Depends, HTTPException

from pawn_server.core import settings_registry as reg

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(Exception):
    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


# ── value helpers ─────────────────────────────────────────────────────────────


def _config_mod() -> Any:
    from pawn_agent.utils import config  # noqa: PLC0415

    return config


def _catalog(cfg: Any) -> dict[str, Any]:
    try:
        from pawn_agent.utils.model_catalog import public_catalog  # noqa: PLC0415

        return public_catalog(cfg)
    except Exception as exc:  # noqa: BLE001 - legacy configs without a catalog
        logger.debug("settings catalog failed: %s", exc)
        return {"default": None, "background": None, "models": []}


def _source(path: str, overrides: dict, raw: dict, explicit: bool) -> str:
    if reg.dict_get(overrides, path)[0]:
        return "override"
    in_yaml = reg.dict_get(raw, path)[0]
    if explicit and in_yaml:
        return "yaml"
    if reg.env_name(path) in os.environ:
        return "env"
    return "yaml" if in_yaml else "default"


def _next_fires(expr: str, tz: str, n: int = 3) -> list[str]:
    from croniter import croniter  # noqa: PLC0415

    try:
        zone = ZoneInfo(tz or "UTC")
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    if not expr or not croniter.is_valid(expr):
        return []
    it = croniter(expr, datetime.now(zone))
    return [it.get_next(datetime).isoformat() for _ in range(n)]


def _check(cfg: Any, f: reg.SettingField, value: Any) -> Any:
    """Validate and coerce one value through its section's pydantic model."""
    from croniter import croniter  # noqa: PLC0415

    if f.path == reg.BACKGROUND_MODEL or f.type == "model":
        text = str(value or "").strip()
        if not text and f.nullable:
            return ""
        from pawn_agent.utils.model_catalog import is_catalog_id  # noqa: PLC0415

        if not is_catalog_id(cfg, text):
            raise ValueError(f"Unknown model {text!r}")
        return text
    if f.type == "cron":
        if not croniter.is_valid(str(value or "")):
            raise ValueError(f"Not a valid cron expression: {value!r}")
    if f.type == "tz":
        try:
            ZoneInfo(str(value or ""))
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone {value!r}") from exc
    if f.type == "enum" and f.choices and value not in f.choices:
        raise ValueError(f"Choose one of: {', '.join(f.choices)}")
    if f.type == "list":
        if isinstance(value, str):
            value = [p.strip() for p in value.split(",")]
        if not isinstance(value, list):
            raise ValueError("Expected a list")
        value = [str(v).strip() for v in value if str(v).strip()]
    parent_path, leaf = f.path.rsplit(".", 1)
    parent = reg.get_path(cfg, parent_path)
    model = type(parent)
    if f.nullable and value in ("", None):
        info = model.model_fields.get(leaf)
        value = None if info is not None and info.default is None else ""
    try:
        checked = model.model_validate({**parent.model_dump(), leaf: value})
    except Exception as exc:  # pydantic ValidationError
        errors = getattr(exc, "errors", None)
        msg = errors()[0]["msg"] if callable(errors) and errors() else str(exc)
        raise ValueError(msg) from exc
    out = getattr(checked, leaf)
    if f.min is not None and isinstance(out, (int, float)) and out < f.min:
        raise ValueError(f"Must be at least {f.min:g}")
    return out


def _field_row(
    f: reg.SettingField,
    configured: Any,
    running: Any,
    overrides: dict,
    raw: dict,
    explicit: bool,
) -> dict[str, Any]:
    row = asdict(f)
    row.pop("extra", None)
    row.pop("min", None)
    if f.path == reg.BACKGROUND_MODEL:
        from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
            background_model_path,
            get_background_model,
        )

        value = get_background_model(running)
        has_yaml, yaml_value = reg.dict_get(raw, "agent.default")
        source = (
            "override"
            if background_model_path(running).is_file()
            else ("yaml" if has_yaml else ("env" if reg.env_name(f.path) in os.environ else "default"))
        )
        row.update(value=value, running=value, source=source, yaml_value=yaml_value if has_yaml else None)
        row["pending_restart"] = False
        return row
    value = reg.get_path(configured, f.path)
    live = reg.get_path(running, f.path)
    has_yaml, yaml_value = reg.dict_get(raw, f.path)
    row["source"] = _source(f.path, overrides, raw, explicit)
    row["min"] = f.min
    if f.secret:
        row.update(value=None, running=None, set=bool(value), yaml_value=None)
        row["pending_restart"] = False
        return row
    row.update(value=value, running=live, yaml_value=yaml_value if has_yaml else None)
    row["pending_restart"] = f.apply == "restart" and value != live
    if f.type == "cron":
        tz = reg.get_path(configured, f.tz_path) if f.tz_path else "UTC"
        row["next"] = _next_fires(str(value or ""), str(tz or "UTC"))
    return row


def settings_payload(cfg: Any) -> dict[str, Any]:
    """GET body: groups, fields with value/source, catalog, override file."""
    mod = _config_mod()
    overrides = mod.read_overrides(cfg)
    try:
        configured = mod.build_config(overrides)
    except Exception as exc:  # noqa: BLE001 - a hand-broken override file
        logger.warning("settings: configured rebuild failed, showing running values: %s", exc)
        configured = cfg
    raw = mod.raw_yaml()
    explicit = mod.loaded_config_explicit()
    fields = [_field_row(f, configured, cfg, overrides, raw, explicit) for f in reg.FIELDS]
    return {
        "groups": [{"id": gid, "label": label} for gid, label in reg.GROUPS],
        "fields": fields,
        "catalog": _catalog(cfg),
        "override_file": str(mod.override_path(cfg)),
        "config_file": mod.loaded_config_file(),
        "restart_required": [r["path"] for r in fields if r.get("pending_restart")],
    }


def apply_patch(cfg: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Validate all *values*, write the override file, patch the running config."""
    mod = _config_mod()
    errors: dict[str, str] = {}
    checked: dict[str, Any] = {}
    for path, value in values.items():
        f = reg.BY_PATH.get(path)
        if f is None:
            errors[path] = "Not an editable setting"
            continue
        if f.secret and value in ("", None):
            continue  # write-only: empty keeps the current secret
        try:
            checked[path] = _check(cfg, f, value)
        except ValueError as exc:
            errors[path] = str(exc)
    if errors:
        raise SettingsError(errors)

    overrides = mod.read_overrides(cfg)
    for path, value in checked.items():
        if path != reg.BACKGROUND_MODEL:
            overrides = mod.deep_merge(overrides, reg.nested(path, value))
    try:
        mod.build_config(overrides)
    except Exception as exc:  # noqa: BLE001 - cross-field pydantic failure
        raise SettingsError({"_": str(exc)}) from exc
    mod.write_overrides(cfg, overrides)

    for path, value in checked.items():
        f = reg.BY_PATH[path]
        if path == reg.BACKGROUND_MODEL:
            from pawn_agent.utils.model_catalog import set_background_model  # noqa: PLC0415

            set_background_model(cfg, value)
        elif f.apply != "restart":
            parent_path, leaf = path.rsplit(".", 1)
            setattr(reg.get_path(cfg, parent_path), leaf, value)
    return settings_payload(cfg)


def reset_field(cfg: Any, path: str) -> dict[str, Any]:
    """Drop one override and fall back to the env / yaml / default value."""
    mod = _config_mod()
    f = reg.BY_PATH.get(path)
    if f is None:
        raise SettingsError({path: "Not an editable setting"})
    if path == reg.BACKGROUND_MODEL:
        from pawn_agent.utils.model_catalog import reset_background_model  # noqa: PLC0415

        reset_background_model(cfg)
        return settings_payload(cfg)
    overrides = mod.read_overrides(cfg)
    if reg.dict_pop(overrides, path):
        mod.write_overrides(cfg, overrides)
    if f.apply != "restart":
        baseline = mod.build_config(overrides)
        parent_path, leaf = path.rsplit(".", 1)
        setattr(reg.get_path(cfg, parent_path), leaf, reg.get_path(baseline, path))
    return settings_payload(cfg)


# ── status ────────────────────────────────────────────────────────────────────


def _db_status(cfg: Any) -> dict[str, Any]:
    from sqlalchemy import text  # noqa: PLC0415

    from pawn_core.database import get_engine  # noqa: PLC0415

    out: dict[str, Any] = {"reachable": False, "revision": None, "head": None}
    try:
        from alembic.script import ScriptDirectory  # noqa: PLC0415

        out["head"] = ScriptDirectory(str(_REPO_ROOT / "migrations")).get_current_head()
    except Exception as exc:  # noqa: BLE001
        logger.debug("alembic head lookup failed: %s", exc)
    try:
        with get_engine(cfg.db_dsn).connect() as conn:
            out["reachable"] = True
            try:
                out["revision"] = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
            except Exception:  # noqa: BLE001 - no alembic table (tests, fresh DB)
                out["revision"] = None
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


async def status_payload(cfg: Any) -> dict[str, Any]:
    mod = _config_mod()
    try:
        from pawn_diarize import __version__ as version  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        version = None
    out: dict[str, Any] = {
        "version": version,
        "config_file": mod.loaded_config_file(),
        "override_file": str(mod.override_path(cfg)),
        "state_dir": str(mod.state_dir_of(cfg)),
        "db": await asyncio.to_thread(_db_status, cfg),
        "queues": [],
    }
    try:
        from pawn_server.core.queue_admin import queue_stats  # noqa: PLC0415

        stats = await asyncio.wait_for(queue_stats(cfg), timeout=5)
        out["queues"] = [asdict(s) for s in stats]
    except Exception as exc:  # noqa: BLE001 - no S3 / no queues configured
        out["queues_error"] = str(exc) or type(exc).__name__
    return out


# ── router ────────────────────────────────────────────────────────────────────


def build_router(get_cfg: Callable[[], Any], require_token: Callable[..., None]) -> APIRouter:
    router = APIRouter(
        prefix="/v1/portal/settings", tags=["portal"], dependencies=[Depends(require_token)]
    )

    def _http(exc: SettingsError) -> HTTPException:
        return HTTPException(status_code=422, detail={"errors": exc.errors})

    @router.get("")
    async def settings_get(cfg: Any = Depends(get_cfg)) -> dict:
        return await asyncio.to_thread(settings_payload, cfg)

    @router.patch("")
    async def settings_patch(
        values: dict[str, Any] = Body(...), cfg: Any = Depends(get_cfg)
    ) -> dict:
        try:
            return await asyncio.to_thread(apply_patch, cfg, values)
        except SettingsError as exc:
            raise _http(exc) from exc

    @router.get("/status")
    async def settings_status(cfg: Any = Depends(get_cfg)) -> dict:
        return await status_payload(cfg)

    @router.delete("/{path:path}")
    async def settings_reset(path: str, cfg: Any = Depends(get_cfg)) -> dict:
        try:
            return await asyncio.to_thread(reset_field, cfg, path)
        except SettingsError as exc:
            raise _http(exc) from exc

    return router

