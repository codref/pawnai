"""One durable sallm conversation bound to a conversation key.

Owns: wrapping sync ``Agent.ask`` / ``Agent.clear`` for async callers.
Does not own: pooling multiple conversations (see sallm_registry.py).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from sallm import Agent

from pawn_agent.core.sallm_factory import build_sallm_agent, rebuild_agent_for_config
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)


class SallmChatSession:
    """Async façade around one sallm.Agent instance.

    ``Agent.ask`` is synchronous (LiteLLM + subprocess tools). We offload it
    with ``asyncio.to_thread`` so FastAPI / queue handlers stay responsive.
    """

    def __init__(self, agent: Agent, cfg: AgentConfig) -> None:
        self._agent = agent
        self._cfg = cfg
        self.conversation_id = agent.session_id

    @classmethod
    def create(
        cls,
        cfg: AgentConfig,
        *,
        conversation_id: str,
        trace: Any = None,
    ) -> "SallmChatSession":
        """Build a session (sync). Safe to call from a worker thread."""
        agent = build_sallm_agent(
            cfg,
            conversation_id=conversation_id,
            trace=trace,
        )
        return cls(agent, cfg)

    def apply_config(self, cfg: AgentConfig) -> None:
        """Replace the underlying Agent when model/settings change mid-session."""
        if cfg.litellm_model == self._cfg.litellm_model and cfg.sallm == self._cfg.sallm:
            self._cfg = cfg
            return
        self._agent = rebuild_agent_for_config(self._agent, cfg)
        self._cfg = cfg

    async def handle_user_input(self, text: str) -> str:
        """Run one user turn; return the assistant answer string.

        When tools ran, prefix a short dim trail so the CLI shows whether a
        `` ```run `` block actually executed (vs the model printing argv as prose).
        """
        # Offload: ask() blocks on LLM + CliTool subprocesses.
        result = await asyncio.to_thread(self._agent.ask, text)
        if not isinstance(result, dict):
            return str(result or "")

        answer = str(result.get("answer") or "")
        tool_lines: list[str] = []
        for step in result.get("steps") or []:
            if not isinstance(step, dict) or step.get("kind") != "action":
                continue
            for tc in step.get("tool_calls") or []:
                name = tc.get("action") or "?"
                obs = str(tc.get("observation") or "").replace("\n", " ")
                if len(obs) > 120:
                    obs = obs[:117] + "..."
                tool_lines.append(f"[tool] {name} → {obs}")
        if tool_lines:
            return "\n".join(tool_lines) + "\n\n" + answer
        return answer

    async def reset(self) -> None:
        """Wipe durable memory for this conversation id (SQLite + vectors)."""
        await asyncio.to_thread(self._agent.clear)


async def run_sallm_chat(
    cfg: AgentConfig,
    emit: Callable[[str], None] = print,
    on_thinking: Optional[Callable[[], None]] = None,
    *,
    conversation_id: str = "cli",
) -> None:
    """Interactive REPL for ``pawn-agent chat`` (sallm-backed)."""
    from prompt_toolkit import PromptSession  # noqa: PLC0415

    session = SallmChatSession.create(cfg, conversation_id=conversation_id)
    prompt_session: Any = PromptSession()
    while True:
        try:
            raw = await prompt_session.prompt_async("You: ")
        except (EOFError, KeyboardInterrupt):
            return
        text = raw.strip()
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            return
        if text.lower() == "/reset":
            await session.reset()
            emit("Session cleared.")
            continue
        if text.startswith("/"):
            emit("Supported slash commands: /reset, /exit, /quit.")
            continue
        if on_thinking is not None:
            on_thinking()
        try:
            reply = await session.handle_user_input(text)
        except Exception as exc:
            logger.exception("sallm chat turn failed")
            emit(f"Error: {exc}")
            continue
        emit(reply)
