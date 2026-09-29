"""Single-turn OpenAI-compatible completion on the active model selection.

Used by structured analysis and coworker extract/score. The conversational
agent is sallm; this helper is a one-shot LiteLLM call on the same catalog
selection (background default unless the caller already overrode ``cfg``).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from pawn_agent.utils.config import AgentConfig


async def run(
    cfg: AgentConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Run a single-turn completion and return the response string."""
    from sallm.llm import complete  # noqa: PLC0415
    from sallm.messages import system, user  # noqa: PLC0415

    selection = cfg.model_selection
    messages = []
    if system_prompt:
        messages.append(system(system_prompt))
    messages.append(user(prompt))

    def _call() -> str:
        extra: dict = {}
        if selection.api_key:
            extra["api_key"] = selection.api_key
        result = complete(
            model=selection.litellm_model,
            messages=messages,
            api_base=selection.api_base,
            **extra,
        )
        return str(result.get("content") or "")

    return await asyncio.to_thread(_call)
