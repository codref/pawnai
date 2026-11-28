"""Budgeted JSON model calls for the night shift and other subject jobs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pawn_agent.core.coworker.jsonutil import parse_json_value
from pawn_agent.core.emergence import prompts
from pawn_agent.core.llm_sub import run as llm_run

logger = logging.getLogger(__name__)

_RETRY = (
    "\n\nYour previous reply was not valid JSON. Reply again with only the "
    "JSON value, double-quoted keys, no trailing commas."
)


class BudgetExhausted(RuntimeError):
    """The night budget (calls or minutes) is spent."""


@dataclass
class Budget:
    """Caps model calls and wall time for one night (or one job)."""

    max_calls: int = 400
    max_seconds: float = 5 * 3600
    calls: int = 0
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def from_cfg(cls, cfg: Any) -> "Budget":
        em = getattr(cfg, "emergence", None)
        return cls(
            max_calls=int(getattr(em, "night_max_calls", 400) or 400),
            max_seconds=float(getattr(em, "night_max_minutes", 300) or 300) * 60,
        )

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls or time.monotonic() - self.started >= self.max_seconds

    def spend(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(f"{self.calls} calls")
        self.calls += 1


async def ask_json(
    cfg: Any,
    prompt: str,
    *,
    think: bool = False,
    budget: Optional[Budget] = None,
    system: str = prompts.SYSTEM,
) -> Any:
    """One model call returning parsed JSON, retried once on a bad reply.

    Returns None when both replies are unusable or the call fails.
    Raises :class:`BudgetExhausted` when the budget is spent.
    """
    for attempt in range(2):
        if budget is not None:
            budget.spend()
        try:
            raw = await llm_run(
                cfg, prompt if attempt == 0 else prompt + _RETRY, system_prompt=system, think=think
            )
        except Exception as exc:
            logger.error("emergence model call failed: %s", exc)
            return None
        try:
            return parse_json_value(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("emergence reply unparseable (attempt %d)", attempt + 1)
    return None
