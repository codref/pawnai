"""The night shift: slow jobs over subjects, run once a night under a budget.

Jobs run in priority order (``emergence.night_jobs``); each catches its own
errors, and the shared :class:`Budget` stops model work when the call or
time cap is reached. Results are recorded as an ``agent_runs`` row.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Awaitable, Callable, Optional

from pawn_agent.core.emergence.llm import Budget, BudgetExhausted

logger = logging.getLogger(__name__)

JOBS = (
    "sync",
    "boards",
    "clustering",
    "synthesis",
    "assumptions",
    "commitments",
    "predictions",
    "contradictions",
    "people",
    "dreaming",
    "council",
    "retros",
    "provocations",
    "prompt_search",
)


async def _job(name: str, cfg: Any, store: Any, budget: Budget, today: date) -> Any:
    from pawn_agent.core.emergence import (  # noqa: PLC0415
        dream,
        ledger,
        provocations,
        reflect,
        search,
        threads,
    )

    if name == "sync":
        return {"goals": threads.sync_goals(cfg, store), "notes": threads.sync_notes(cfg, store)}
    if name == "boards":
        from pawn_agent.core.emergence import board_night  # noqa: PLC0415

        return await board_night.work_boards(cfg, store, budget=budget)
    if name == "clustering":
        return len(await threads.emergent_threads(cfg, store, budget=budget))
    if name == "synthesis":
        return await threads.synthesize(cfg, store, budget=budget)
    if name == "assumptions":
        return await ledger.watch_assumptions(cfg, store, budget=budget)
    if name == "commitments":
        return ledger.age_commitments(cfg, store, today=today)
    if name == "predictions":
        return await ledger.resolve_predictions(cfg, store, budget=budget, today=today)
    if name == "contradictions":
        return await reflect.scan_contradictions(cfg, budget=budget)
    if name == "people":
        return reflect.refresh_people(cfg, store)
    if name == "dreaming":
        return await dream.dream(cfg, store, budget=budget)
    if name == "council":
        return await dream.council_due(cfg, store, budget=budget)
    if name == "retros":
        return await reflect.retros_due(cfg, store, today=today, budget=budget)
    if name == "provocations":
        return len(provocations.list_recent(cfg, days=1))
    if name == "prompt_search":
        results = {}
        for type_name in ("decision", "commitment", "block"):
            if budget.exhausted:
                break
            results[type_name] = (await search.prompt_search(cfg, store, type_name, budget=budget))[
                "status"
            ]
        return results
    raise ValueError(f"unknown night job {name!r}")


async def run_night(
    cfg: Any,
    *,
    store: Any = None,
    jobs: Optional[list[str]] = None,
    budget: Optional[Budget] = None,
    today: Optional[date] = None,
    on_job: Optional[Callable[[str, Any], Awaitable[None] | None]] = None,
) -> dict[str, Any]:
    """Run the night jobs; returns ``{job: result | "error: …" | "skipped: budget"}``."""
    if store is None:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
    budget = budget or Budget.from_cfg(cfg)
    today = today or date.today()
    chosen = jobs or list(getattr(getattr(cfg, "emergence", None), "night_jobs", None) or JOBS)
    results: dict[str, Any] = {}
    run_id = None
    try:
        from pawn_agent.utils.db import create_agent_run, update_agent_run  # noqa: PLC0415

        run_id = create_agent_run(
            cfg.db_dsn,
            source="emergence",
            command="night",
            prompt=",".join(chosen),
            session_id="emergence:night",
            model=getattr(cfg, "chat_model_id", None) or "emergence",
        )
        update_agent_run(cfg.db_dsn, run_id, "running")
    except Exception as exc:
        logger.debug("night run not recorded: %s", exc)
    started = time.monotonic()
    for name in chosen:
        if budget.exhausted and name not in {"sync", "commitments", "people", "provocations"}:
            results[name] = "skipped: budget"
            continue
        t0 = time.monotonic()
        try:
            results[name] = await _job(name, cfg, store, budget, today)
        except BudgetExhausted:
            results[name] = "stopped: budget"
        except Exception as exc:
            logger.exception("night job %s failed", name)
            results[name] = f"error: {exc}"
        logger.info("night job %s: %s (%.0fs)", name, results[name], time.monotonic() - t0)
        if on_job is not None:
            maybe = on_job(name, results[name])
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
    results["_calls"] = budget.calls
    results["_minutes"] = round((time.monotonic() - started) / 60, 1)
    if run_id is not None:
        try:
            from pawn_agent.utils.db import update_agent_run  # noqa: PLC0415

            update_agent_run(cfg.db_dsn, run_id, "completed", response=str(results)[:4000])
        except Exception:
            pass
    return results
