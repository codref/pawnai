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
from pathlib import Path
from typing import Callable, Optional

from copilot import CopilotClient, CopilotSession, MessageOptions, PermissionHandler
from copilot.types import SessionConfig, SystemMessageAppendConfig

from pawn_agent.tools import build_tools
from pawn_agent.utils.config import AgentConfig


class ConversationAgent:
    """Copilot SDK agent that selects and invokes tools to fulfil a user prompt.

    Manages the full :class:`~copilot.CopilotClient` lifecycle (start / stop)
    and registers the custom tools on each session so the model can call them.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.
        emit: Optional callback for outputting agent responses. Defaults to
            :func:`print`. Used by the chat REPL to render with Rich.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = cfg
        self._emit_fn = emit or print
        self._on_thinking = on_thinking

    def run(self, user_prompt: str) -> str:
        """Run the agent on a user prompt and return its response.

        Args:
            user_prompt: Free-text instruction from the user.

        Returns:
            Agent's final response string.
        """
        return asyncio.run(self._async_run(user_prompt))

    def _emit(self, text: str) -> None:
        self._emit_fn(text)

    def _get_input(self, prompt: str = "") -> str:
        return input(prompt)

    def chat(self, first_message: str | None = None) -> None:
        """Start an interactive multi-turn chat session.

        Keeps a single :class:`~copilot.CopilotSession` alive across turns so
        the model retains conversation context. Exits on ``/exit``, ``/quit``,
        Ctrl-D, or Ctrl-C.

        Args:
            first_message: Optional pre-built first turn (e.g. with a session
                hint already prepended). If ``None`` the REPL prompts the user
                immediately.
        """
        asyncio.run(self._async_chat(first_message))

    async def _async_chat(self, first_message: str | None = None) -> None:
        client = CopilotClient()
        await client.start()
        try:
            tools = build_tools(self.cfg, client)
            session = await client.create_session(self._session_config(tools))
            try:
                await self._repl_loop(session, first_message)
            finally:
                await session.disconnect()
        finally:
            await client.stop()

    async def _repl_loop(
        self, session: CopilotSession, first_message: str | None
    ) -> None:
        pending = first_message
        while True:
            if pending is None:
                try:
                    pending = self._get_input("You: ")
                except (EOFError, KeyboardInterrupt):
                    return
            text = pending.strip()
            pending = None
            if not text:
                continue
            if text.lower() in {"/exit", "/quit"}:
                return
            if self._on_thinking is not None:
                self._on_thinking()
            response = await session.send_and_wait(
                MessageOptions(prompt=text),
                timeout=120,
            )
            self._emit(response.data.content or "" if response is not None else "")

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        p = Path(self.cfg.anima_path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None

    def _session_config(self, tools: list) -> SessionConfig:
        cfg = SessionConfig(
            model=self.cfg.model,
            tools=tools,
            on_permission_request=PermissionHandler.approve_all,
        )
        anima = self._load_anima()
        if anima:
            cfg["system_message"] = SystemMessageAppendConfig(mode="append", content=anima)
        return cfg

    async def _async_run(self, user_prompt: str) -> str:
        client = CopilotClient()
        await client.start()
        try:
            tools = build_tools(self.cfg, client)
            session = await client.create_session(self._session_config(tools))
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
