"""Single-turn completion via the primary OpenAI-compatible LLM.

Mirrors the interface of :mod:`pawn_agent.core.copilot_sub` but routes
through the PydanticAI model configured in ``agent.openai`` (or equivalent
provider) rather than the Copilot SDK.  Use this for analysis tools that
should bill against the same model as the main agent.
"""

from __future__ import annotations

from typing import Optional

from pawn_agent.utils.config import AgentConfig


async def run(
    cfg: AgentConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Run a single-turn completion and return the response string.

    Args:
        cfg: Agent configuration.  Uses ``cfg.pydantic_model``,
            ``cfg.pydantic_api_key``, and ``cfg.pydantic_base_url``.
        prompt: User prompt to send.
        system_prompt: Optional system prompt.

    Returns:
        The model's response as a plain string.
    """
    import os
    from pydantic_ai import Agent

    model_str = cfg.pydantic_model
    api_key = cfg.pydantic_api_key
    base_url = cfg.pydantic_base_url

    if base_url and model_str.startswith("openai:"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model_name = model_str[len("openai:"):]
        provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")
        model = OpenAIChatModel(model_name, provider=provider)
    else:
        if api_key:
            if model_str.startswith("openai:"):
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif model_str.startswith("anthropic:"):
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        model = model_str

    agent: Agent = Agent(model, system_prompt=system_prompt or ())
    result = await agent.run(prompt)
    return result.output
