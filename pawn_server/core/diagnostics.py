"""Portal Diagnostics views: scheduler activity and the effective config.

``scheduler_payload`` joins three sources:

- the database (``agent_schedules``, ``agent_schedule_fires``,
  ``agent_schedule_proposals``, ``agent_runs``), which every process writes;
- the coworker crons (briefing, weekly review, emergence night shift), whose
  next fire times come from the configured cron expressions;
- :mod:`pawn_agent.core.worker_status`, the in-process heartbeat of the
  scheduler and coworker loops. A loop running in another process
  (``--scheduler-only``) has no heartbeat here, only its rows.

``config_payload`` is the running ``AgentConfig`` with credentials masked.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pawn_server.core import settings_registry as reg

logger = logging.getLogger(__name__)

# Background sources shown in the run history (interactive chat is left out).
BACKGROUND_SOURCES = ("schedule", "emergence", "coworker", "vault", "planner", "capture", "queue")

_SECRET_KEY = re.compile(
    r"(^|_)(token|api_key|apikey|secret|secret_key|access_key|password|passwd|"
    r"private_key|credentials?|user_token|hf_token)$",
    re.IGNORECASE,
)
_URL_CREDS = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")
MASK = "••••••"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _clip(text: Optional[str], n: int = 300) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# ── config ────────────────────────────────────────────────────────────────────


def redact(value: Any, key: str = "") -> Any:
    """Mask credential-looking keys and passwords inside URLs, recursively."""
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if key and _SECRET_KEY.search(key):
        return MASK if value not in (None, "") else value
    if isinstance(value, str):
        return _URL_CREDS.sub(rf"\g<1>{MASK}@", value)
    return value


def config_payload(cfg: Any) -> dict[str, Any]:
    from pawn_agent.utils import config as mod  # noqa: PLC0415

    try:
        tree = cfg.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - a field that does not serialise
        logger.warning("config dump failed: %s", exc)
        tree = {}
    try:
        overrides = mod.read_overrides(cfg)
    except Exception:  # noqa: BLE001
        overrides = {}
    return {
        "config": redact(tree),
        "overrides": redact(overrides),
        "editable": {f.path: f.group for f in reg.FIELDS},
        "config_file": mod.loaded_config_file(),
        "override_file": str(mod.override_path(cfg)),
    }


# ── scheduler ─────────────────────────────────────────────────────────────────


def _crons(cfg: Any, heartbeat: dict[str, Any]) -> list[dict[str, Any]]:
    from pawn_server.core.settings_api import _next_fires  # noqa: PLC0415

    coworker = cfg.coworker
    emergence = getattr(cfg, "emergence", None)
    tz = str(getattr(coworker, "timezone", None) or "UTC")
    beat = heartbeat.get("coworker") or {}
    last = beat.get("last") or {}
    running_next = beat.get("next") or {}
    rows = [
        ("briefing", "Morning briefing", coworker.briefing_cron, coworker.enabled, "coworker.briefing_cron", "coworker.enabled"),
        ("weekly", "Weekly review", coworker.weekly_cron, coworker.enabled, "coworker.weekly_cron", "coworker.enabled"),
        (
            "night",
            "Emergence night shift",
            str(getattr(emergence, "night_cron", "") or ""),
            bool(coworker.enabled and getattr(emergence, "enabled", False)),
            "emergence.night_cron",
            "emergence.enabled" if coworker.enabled else "coworker.enabled",
        ),
    ]
    out = []
    for job, label, expr, enabled, path, gate in rows:
        out.append(
            {
                "id": job,
                "label": label,
                "cron": expr,
                "timezone": tz,
                "enabled": enabled,
                "path": path,
                "gate": gate,
                "next": _next_fires(expr, tz) if enabled else [],
                "running_next": running_next.get(job),
                "last": last.get(job),
            }
        )
    return out


def _db_activity(cfg: Any, *, fires_limit: int, runs_limit: int) -> dict[str, Any]:
    import sqlalchemy as sa  # noqa: PLC0415

    from pawn_agent.core.scheduler import serialize_proposal, serialize_schedule  # noqa: PLC0415
    from pawn_agent.utils.db import (  # noqa: PLC0415
        AgentRun,
        AgentSchedule,
        AgentScheduleFire,
        AgentScheduleProposal,
        _get_session,
    )

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    with _get_session(cfg.db_dsn) as db:
        live = db.scalars(
            sa.select(AgentSchedule)
            .where(AgentSchedule.status.in_(["active", "paused"]))
            .order_by(AgentSchedule.next_run_at.asc().nulls_last())
        ).all()
        done = db.scalars(
            sa.select(AgentSchedule)
            .where(AgentSchedule.status.not_in(["active", "paused"]))
            .order_by(AgentSchedule.updated_at.desc().nulls_last())
            .limit(10)
        ).all()
        schedules = []
        for row in [*live, *done]:
            item = serialize_schedule(row)
            item["prompt"] = _clip(item.get("prompt"), 240)
            schedules.append(item)

        fire_rows = db.execute(
            sa.select(AgentScheduleFire, AgentSchedule.name, AgentRun.status, AgentRun.error)
            .join(AgentSchedule, AgentSchedule.id == AgentScheduleFire.schedule_id, isouter=True)
            .join(AgentRun, AgentRun.id == AgentScheduleFire.agent_run_id, isouter=True)
            .order_by(AgentScheduleFire.created_at.desc().nulls_last())
            .limit(fires_limit)
        ).all()
        fires = []
        for fire, name, run_status, run_error in fire_rows:
            duration = None
            if fire.started_at and fire.completed_at:
                duration = round((fire.completed_at - fire.started_at).total_seconds(), 1)
            fires.append(
                {
                    "id": fire.id,
                    "schedule_id": fire.schedule_id,
                    "schedule_name": name,
                    "scheduled_for": _iso(fire.scheduled_for),
                    "status": fire.status,
                    "attempt": fire.attempt,
                    "claimed_at": _iso(fire.claimed_at),
                    "started_at": _iso(fire.started_at),
                    "completed_at": _iso(fire.completed_at),
                    "duration_s": duration,
                    "error": _clip(fire.error or run_error),
                    "agent_run_id": fire.agent_run_id,
                    "run_status": run_status,
                }
            )

        counts = dict(
            db.execute(
                sa.select(AgentScheduleFire.status, sa.func.count())
                .where(AgentScheduleFire.created_at >= since)
                .group_by(AgentScheduleFire.status)
            ).all()
        )

        proposals = [
            serialize_proposal(row)
            for row in db.scalars(
                sa.select(AgentScheduleProposal)
                .where(AgentScheduleProposal.status == "proposed")
                .order_by(AgentScheduleProposal.created_at.desc())
            ).all()
        ]

        run_rows = db.scalars(
            sa.select(AgentRun)
            .where(AgentRun.source.in_(BACKGROUND_SOURCES))
            .order_by(AgentRun.created_at.desc().nulls_last())
            .limit(runs_limit)
        ).all()
        runs = []
        for run in run_rows:
            duration = None
            if run.started_at and run.completed_at:
                duration = round((run.completed_at - run.started_at).total_seconds(), 1)
            runs.append(
                {
                    "id": run.id,
                    "source": run.source,
                    "command": run.command,
                    "session_id": run.session_id,
                    "schedule_id": run.schedule_id,
                    "model": run.model,
                    "status": run.status,
                    "error": _clip(run.error),
                    "prompt": _clip(run.prompt, 160),
                    "created_at": _iso(run.created_at),
                    "started_at": _iso(run.started_at),
                    "completed_at": _iso(run.completed_at),
                    "duration_s": duration,
                }
            )
        last_night = db.scalars(
            sa.select(AgentRun)
            .where(AgentRun.source == "emergence", AgentRun.command == "night")
            .order_by(AgentRun.created_at.desc().nulls_last())
            .limit(1)
        ).first()
    return {
        "schedules": schedules,
        "fires": fires,
        "fire_counts_24h": {str(k): int(v) for k, v in counts.items()},
        "proposals": proposals,
        "runs": runs,
        "last_night_run": (
            {
                "id": last_night.id,
                "status": last_night.status,
                "created_at": _iso(last_night.created_at),
                "completed_at": _iso(last_night.completed_at),
                "error": _clip(last_night.error),
            }
            if last_night is not None
            else None
        ),
    }


def scheduler_payload(cfg: Any, *, fires_limit: int = 50, runs_limit: int = 40) -> dict[str, Any]:
    from pawn_agent.core import worker_status  # noqa: PLC0415

    heartbeat = worker_status.snapshot()
    out: dict[str, Any] = {
        "now": datetime.now(timezone.utc).isoformat(),
        "config": cfg.agent_scheduler.model_dump(mode="json"),
        "workers": heartbeat,
        "crons": _crons(cfg, heartbeat),
        "schedules": [],
        "fires": [],
        "fire_counts_24h": {},
        "proposals": [],
        "runs": [],
        "last_night_run": None,
    }
    try:
        out.update(_db_activity(cfg, fires_limit=fires_limit, runs_limit=runs_limit))
    except Exception as exc:  # noqa: BLE001 - DB down or not migrated
        logger.warning("scheduler diagnostics: %s", exc)
        out["db_error"] = str(exc) or type(exc).__name__
    return out
