"""DSPy-powered plan-then-execute planner.

Uses ``dspy.ChainOfThought`` to convert a free-text user request into an
explicit, ordered execution plan.  The plan is a list of
:class:`PlanStep` objects that the :mod:`pawn_agent.agent.executor` then
runs deterministically.

The plan is logged in full *before* any skill execution begins — this is
the key property of the plan-then-execute pattern: full transparency and
debuggability, with no adaptive loop.

Planner signature::

    Inputs:
        user_request     — the natural-language request from the queue message
        available_skills — JSON description of loaded skills (name + description + input_schema)
        context          — JSON string of message-level context (e.g. audio_path, session_id)

    Output:
        plan             — JSON array of plan steps

Plan step schema::

    {
      "step": 1,
      "skill": "transcribe",
      "params": {"audio_path": "s3://bucket/audio.wav", "timestamps": true},
      "input_from": [],          // prior step numbers whose outputs are merged into params
      "description": "Transcribe the audio file to text"
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Plan data model
# ──────────────────────────────────────────────────────────────────────────────

class PlanStep(BaseModel):
    """A single step in an agent execution plan."""

    step: int = Field(..., description="1-based step number, unique within the plan")
    skill: str = Field(..., description="Skill name to invoke (must exist in SkillRegistry)")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Static parameters to pass to the skill",
    )
    input_from: List[int] = Field(
        default_factory=list,
        description=(
            "List of prior step numbers whose output dicts are merged into "
            "params (later steps override earlier ones; explicit params override all)"
        ),
    )
    description: str = Field(
        default="",
        description="Human-readable explanation of what this step does",
    )


class ExecutionPlan(BaseModel):
    """A complete, ordered execution plan produced by the planner."""

    steps: List[PlanStep] = Field(..., description="Ordered list of plan steps")

    @model_validator(mode="after")
    def _unique_step_numbers(self) -> "ExecutionPlan":
        numbers = [s.step for s in self.steps]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Duplicate step numbers in plan")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# DSPy signature and module
# ──────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
You are an AI agent planner. Given a user request and a list of available skills, \
produce a JSON execution plan.

Rules:
- Return ONLY a valid JSON array of plan steps, wrapped in ```json ... ```.
- Each step must have: step (int), skill (string from available skills), \
params (object), input_from (array of prior step ints), description (string).
- Use input_from when a step needs output from a prior step \
(e.g. a transcription result fed into an analysis step).
- Only use skills listed in available_skills.
- Use context values (like audio_path) to populate params.
- Keep the plan minimal — only include steps needed to fulfil the request.

Example output:
```json
[
  {
    "step": 1,
    "skill": "transcribe",
    "params": {"audio_path": "s3://bucket/audio.wav"},
    "input_from": [],
    "description": "Transcribe the audio file"
  },
  {
    "step": 2,
    "skill": "analyze",
    "params": {},
    "input_from": [1],
    "description": "Analyze the transcript from step 1"
  }
]
```"""


def _extract_json(text: str) -> str:
    """Extract a JSON block from a markdown-fenced or raw response."""
    # Try ```json ... ``` block first
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        return fenced.group(1).strip()
    # Try bare JSON array
    array = re.search(r"(\[[\s\S]+\])", text)
    if array:
        return array.group(1).strip()
    return text.strip()


class AgentPlanner:
    """Produces an :class:`ExecutionPlan` from a free-text user request.

    Uses the configured DSPy LM (set via ``dspy.configure(lm=...)``) to
    generate a JSON plan, then validates it against :class:`ExecutionPlan`.

    Args:
        max_retries: Number of times to retry if plan parsing fails.
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries

    def plan(
        self,
        user_request: str,
        available_skills: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Generate an execution plan for *user_request*.

        Args:
            user_request:     The natural-language request from the queue message.
            available_skills: JSON string from ``SkillRegistry.describe_skills()``.
            context:          Optional message-level context dict.

        Returns:
            A validated :class:`ExecutionPlan`.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid plan
                        after all retries are exhausted.
        """
        import dspy  # lazy import — avoids import-time DSPy initialisation

        context_str = json.dumps(context or {})
        system_prompt = _build_system_prompt()

        user_prompt = (
            f"User request: {user_request}\n\n"
            f"Available skills:\n{available_skills}\n\n"
            f"Context: {context_str}\n\n"
            "Produce the execution plan as a JSON array."
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                # Use DSPy's configured LM directly for maximum control
                lm = dspy.settings.lm
                responses: List[str] = lm(prompt=full_prompt)
                raw_response = responses[0] if responses else ""

                json_text = _extract_json(raw_response)
                steps_data = json.loads(json_text)

                if not isinstance(steps_data, list):
                    raise ValueError("Expected a JSON array of plan steps")

                steps = [PlanStep(**s) for s in steps_data]
                plan = ExecutionPlan(steps=steps)

                logger.info(
                    "Planner produced %d step(s) on attempt %d",
                    len(plan.steps),
                    attempt,
                )
                return plan

            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                logger.warning("Plan parsing failed (attempt %d/%d): %s", attempt, self._max_retries + 1, exc)

        raise ValueError(
            f"AgentPlanner failed to produce a valid plan after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
