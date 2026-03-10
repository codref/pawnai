"""SkillRunner — executes a skill's tool pipeline for given input params.

The runner:
1. Iterates over a skill's ``tools`` steps in order.
2. Resolves ``{{template}}`` expressions against the current context
   (``input``, ``steps``, ``context``).
3. Calls the resolved tool via :class:`~pawn_agent.skills.executor_registry.ExecutorRegistry`.
4. Collects each step's output into the context under ``steps.<n>``.
5. Maps the skill's ``output`` mapping (also template-resolved) to the final
   return dict.

Usage::

    runner = SkillRunner(registry, executor_registry)
    result = await runner.run(
        skill_name="transcribe",
        input_params={"audio_path": "s3://bucket/audio.wav", "timestamps": True},
        context={"session_id": "abc123"},
        cfg=agent_cfg,
    )
    # result: {"transcript": "...", "session_id": "..."}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .executor_registry import ExecutorRegistry
from .registry import SkillRegistry
from .template import TemplateResolutionError, resolve_params, resolve_value

logger = logging.getLogger(__name__)


class SkillRunner:
    """Executes a named skill by running its tool pipeline.

    Args:
        skill_registry:    The loaded skill and tool definitions.
        executor_registry: Maps tool ``function`` keys to async callables.
    """

    def __init__(self, skill_registry: SkillRegistry, executor_registry: ExecutorRegistry) -> None:
        self._skills = skill_registry
        self._executors = executor_registry

    async def run(
        self,
        skill_name: str,
        input_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        cfg: Any = None,
    ) -> Dict[str, Any]:
        """Execute *skill_name* with *input_params*.

        Args:
            skill_name:   Name of the skill to execute.
            input_params: Caller-supplied parameters (matches the skill's ``input_schema``).
            context:      Optional message-level context (e.g. session_id from the queue message).
            cfg:          :class:`~pawn_agent.config.AgentConfig` instance.

        Returns:
            The resolved ``output`` mapping for the skill.

        Raises:
            KeyError: If the skill or any referenced tool is not found.
            TemplateResolutionError: If a template expression cannot be resolved.
        """
        skill = self._skills.get_skill(skill_name)
        context = context or {}

        # Resolution context accumulates as steps complete
        resolution_ctx: Dict[str, Any] = {
            "input": input_params,
            "steps": {},
            "context": context,
        }

        logger.info(
            "Running skill %r — %d step(s)", skill_name, len(skill.tools)
        )

        for step in sorted(skill.tools, key=lambda s: s.step):
            tool_def = self._skills.get_tool(step.tool)
            resolved_params = resolve_params(step.params, resolution_ctx)

            logger.debug(
                "Skill %r step %d: tool=%r params=%s",
                skill_name,
                step.step,
                step.tool,
                list(resolved_params),
            )

            step_result = await self._executors.call(tool_def.function, resolved_params, cfg)

            # Store step output so later steps can reference it
            resolution_ctx["steps"][step.step] = step_result
            logger.debug("Step %d output keys: %s", step.step, list(step_result))

        # Resolve the skill's output mapping
        output: Dict[str, Any] = {}
        for out_key, out_template in skill.output.items():
            try:
                output[out_key] = resolve_value(out_template, resolution_ctx)
            except TemplateResolutionError as exc:
                logger.error("Skill %r output key %r: %s", skill_name, out_key, exc)
                raise

        logger.info("Skill %r completed — output keys: %s", skill_name, list(output))
        return output
