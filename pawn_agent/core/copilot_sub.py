"""Thin async wrapper around the GitHub Copilot SDK for use as a sub-agent.

Provides a single coroutine :func:`run` that manages the full Copilot client
lifecycle (start / create session / send / disconnect / stop) so that
PydanticAI tools can delegate LLM reasoning tasks to the Copilot backend
without owning the session.
"""

from __future__ import annotations

from typing import Optional

from copilot import CopilotClient, MessageOptions, PermissionHandler

from pawn_agent.utils.config import AgentConfig


async def run(
    cfg: AgentConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Run a single-turn completion via the Copilot SDK and return the response.

    Args:
        cfg: Agent configuration (uses ``cfg.model`` for the model selection and
            ``cfg.backend`` / ``cfg.openai_*`` for provider overrides).
        prompt: User-facing prompt to send.
        system_prompt: Optional system prompt.  When provided the session's
            system message is *replaced* with this value so the sub-agent stays
            focused on the delegated task.

    Returns:
        The model's response content as a plain string.
    """
    session_cfg: dict = {
        "model": cfg.model,
        "on_permission_request": PermissionHandler.approve_all,
    }
    if system_prompt:
        session_cfg["system_message"] = {"mode": "replace", "content": system_prompt}
    if cfg.backend == "openai" and cfg.openai_base_url:
        session_cfg["provider"] = {
            "type": "openai",
            "base_url": cfg.openai_base_url,
            "api_key": cfg.openai_api_key,
        }

    client = CopilotClient()
    await client.start()
    try:
        session = await client.create_session(session_cfg)
        try:
            response = await session.send_and_wait(
                MessageOptions(prompt=prompt),
                timeout=120,
            )
            return response.data.content if (response is not None and response.data.content is not None) else ""
        finally:
            await session.disconnect()
    finally:
        await client.stop()
