"""Deterministic plan executor.

Takes an :class:`~pawn_agent.agent.planner.ExecutionPlan` and runs each step
in order by invoking the corresponding skill via
:class:`~pawn_agent.skills.runner.SkillRunner`.

Key behaviour:
- Step outputs are collected into a ``step_results`` dict keyed by step number.
- Before invoking each step, any ``input_from`` step numbers are merged into
  the step's ``params`` (later entries win; explicit ``params`` always override).
- The final return value is the merged dict of all step outputs.
- On failure, the exception propagates immediately (caller is responsible for
  nacking the queue message and dead-lettering).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pawn_agent.agent.planner import ExecutionPlan, PlanStep
from pawn_agent.skills.runner import SkillRunner

logger = logging.getLogger(__name__)


async def execute_plan(
    plan: ExecutionPlan,
    runner: SkillRunner,
    context: Optional[Dict[str, Any]] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    """Execute *plan* step by step using *runner*.

    Args:
        plan:    The validated plan returned by :class:`~pawn_agent.agent.planner.AgentPlanner`.
        runner:  A :class:`~pawn_agent.skills.runner.SkillRunner` instance
                 initialised with the current skill and executor registries.
        context: Message-level context dict (forwarded to each skill run).
        cfg:     :class:`~pawn_agent.config.AgentConfig` instance.

    Returns:
        Merged dict of all step outputs (later steps override earlier ones
        for keys that collide).

    Raises:
        Any exception raised by a skill propagates immediately.
    """
    step_results: Dict[int, Dict[str, Any]] = {}
    context = context or {}

    logger.info(
        "Starting plan execution — %d step(s):\n%s",
        len(plan.steps),
        _format_plan(plan),
    )

    for plan_step in sorted(plan.steps, key=lambda s: s.step):
        params = _resolve_step_params(plan_step, step_results)

        logger.info(
            "Step %d/%d — skill=%r  params=%s",
            plan_step.step,
            len(plan.steps),
            plan_step.skill,
            list(params),
        )

        result = await runner.run(
            skill_name=plan_step.skill,
            input_params=params,
            context=context,
            cfg=cfg,
        )

        step_results[plan_step.step] = result
        logger.info("Step %d completed — output keys: %s", plan_step.step, list(result))

    # Merge all step outputs into a single final dict (last step wins on collision)
    final: Dict[str, Any] = {}
    for step_num in sorted(step_results):
        final.update(step_results[step_num])

    logger.info("Plan execution complete — final output keys: %s", list(final))
    return final


def _resolve_step_params(
    step: PlanStep,
    step_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge prior step outputs into the step's static params.

    Precedence (highest wins):
    1. Explicit ``step.params`` (from the plan)
    2. Earlier ``input_from`` step outputs (steps listed first win over later)
    """
    merged: Dict[str, Any] = {}

    # Merge input_from outputs in order (earlier step number → lower priority)
    for from_step in sorted(step.input_from):
        if from_step not in step_results:
            raise KeyError(
                f"Plan step {step.step} references input_from={from_step} "
                f"but step {from_step} has not yet been executed"
            )
        merged.update(step_results[from_step])

    # Explicit params always override
    merged.update(step.params)
    return merged


def _format_plan(plan: ExecutionPlan) -> str:
    """Return a human-readable multi-line plan summary for logging."""
    lines = []
    for s in sorted(plan.steps, key=lambda x: x.step):
        input_from_str = f" ← steps {s.input_from}" if s.input_from else ""
        lines.append(
            f"  Step {s.step}: [{s.skill}]{input_from_str} — {s.description}"
        )
    return "\n".join(lines)
