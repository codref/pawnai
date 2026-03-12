"""Core agent module for pawn-agent.

:class:`ConversationAgent` wraps the GitHub Copilot SDK, managing a
:class:`~copilot.CopilotClient` and :class:`~copilot.CopilotSession` to
fulfil user prompts via native tool-calling.

Usage::

    from pawn_agent.core.agent import ConversationAgent
    from pawn_agent.utils.config import load_config

    cfg = load_config()
    agent = ConversationAgent(cfg=cfg)
    print(agent.run("Summarise session abc123 and save it to SiYuan"))
"""

from __future__ import annotations

import asyncio

from copilot import CopilotClient, MessageOptions, PermissionHandler

from pawn_agent.core.tools import build_tools
from pawn_agent.utils.config import AgentConfig


class ConversationAgent:
    """Copilot SDK agent that selects and invokes tools to fulfil a user prompt.

    Manages the full :class:`~copilot.CopilotClient` lifecycle (start / stop)
    and registers the custom tools on each session so the model can call them.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg

    def run(self, user_prompt: str) -> str:
        """Run the agent on a user prompt and return its response.

        Args:
            user_prompt: Free-text instruction from the user.

        Returns:
            Agent's final response string.
        """
        return asyncio.run(self._async_run(user_prompt))

    async def _async_run(self, user_prompt: str) -> str:
        client = CopilotClient()
        await client.start()
        try:
            tools = build_tools(self.cfg, client)
            session = await client.create_session(
                {
                    "model": self.cfg.model,
                    "tools": tools,
                    "on_permission_request": PermissionHandler.approve_all,
                }
            )
            try:
                response = await session.send_and_wait(
                    MessageOptions(prompt=user_prompt),
                    timeout=120,
                )
                return response.data.content if response is not None else ""
            finally:
                await session.disconnect()
        finally:
            await client.stop()
