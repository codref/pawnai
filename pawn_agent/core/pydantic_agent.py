"""PydanticAI-based agent for pawn-agent.

:class:`PydanticAgent` is the primary agent implementation.  It uses
``pydantic_ai.Agent`` as the LLM runner and registers all auto-discovered
tools from :mod:`pawn_agent.tools`.  The GitHub Copilot SDK is available as a
sub-agent for tools that need it (e.g. :mod:`pawn_agent.tools.analyze_custom`).

Usage::

    from pawn_agent.core.pydantic_agent import PydanticAgent
    from pawn_agent.utils.config import load_config

    cfg = load_config()
    agent = PydanticAgent(cfg=cfg)
    print(agent.run("Summarise session abc123 and save it to SiYuan"))
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable, List, Optional

from prompt_toolkit import PromptSession
from pydantic_ai import Agent

from pawn_agent.core.session_store import context_size
from pawn_agent.tools import build_tools
from pawn_agent.utils.config import AgentConfig


class PydanticAgent:
    """PydanticAI agent that selects and invokes tools to fulfil a user prompt.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.
        emit: Optional callback for outputting agent responses. Defaults to
            :func:`print`. Used by the chat REPL to render with Rich.
        on_thinking: Optional callback invoked just before the model is called.
            Used by the chat REPL to show a "thinking…" spinner.
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
        self._agent = self._build_agent()

    def _build_agent(self) -> Agent:
        model = self._resolve_model()
        tools = build_tools(self.cfg)
        system_prompts: list[str] = []
        anima = self._load_anima()
        if anima:
            system_prompts.append(anima)
        return Agent(
            model,
            tools=tools,
            system_prompt=tuple(system_prompts),
        )

    def _resolve_model(self):
        """Return a PydanticAI model object or model string for the configured provider.

        When a custom ``base_url`` is set (Ollama, Azure, etc.) the OpenAI provider
        is configured with a custom ``AsyncOpenAI`` client so the endpoint is honoured.
        Otherwise returns the plain model string (e.g. ``"openai:gpt-4o"``).
        """
        model_str = self.cfg.pydantic_model
        api_key = self.cfg.pydantic_api_key
        base_url = self.cfg.pydantic_base_url

        if base_url and model_str.startswith("openai:"):
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            model_name = model_str[len("openai:"):]
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")
            return OpenAIChatModel(model_name, provider=provider)

        # No custom base URL — set API key via env var if provided, then return string
        if api_key:
            if model_str.startswith("openai:"):
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif model_str.startswith("anthropic:"):
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif model_str.startswith("google"):
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            elif model_str.startswith("groq:"):
                os.environ.setdefault("GROQ_API_KEY", api_key)
            elif model_str.startswith("mistral:"):
                os.environ.setdefault("MISTRAL_API_KEY", api_key)

        return model_str

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        p = Path(self.cfg.anima_path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None

    def run(self, user_prompt: str) -> str:
        """Run the agent on a user prompt and return its response.

        Args:
            user_prompt: Free-text instruction from the user.

        Returns:
            Agent's final response string.
        """
        if self._on_thinking is not None:
            self._on_thinking()
        result = self._agent.run_sync(user_prompt)
        return result.output

    def chat(
        self,
        first_message: str | None = None,
        initial_history: List | None = None,
        on_turn_complete: Callable[[list], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        """Start an interactive multi-turn chat session.

        Threads ``message_history`` across turns so the model retains context.
        Exits on ``/exit``, ``/quit``, Ctrl-D, or Ctrl-C.

        Args:
            first_message: Optional pre-built first turn (e.g. with a session
                hint already prepended). If ``None`` the REPL prompts the user
                immediately.
            initial_history: Optional message history to seed the session with
                (e.g. loaded from the session store). Defaults to empty.
            on_turn_complete: Optional callback invoked with ``result.new_messages()``
                after each turn completes. Used to persist turns to the session store.
            on_reset: Optional callback invoked when the user runs ``/reset``.
                Called after in-memory history is cleared; use it to also wipe
                persistent storage.
        """
        asyncio.run(self._repl_loop(first_message, initial_history, on_turn_complete, on_reset))

    async def _repl_loop(
        self,
        first_message: str | None,
        initial_history: List | None = None,
        on_turn_complete: Callable[[list], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        message_history: list = list(initial_history or [])
        pending = first_message
        pt_session: PromptSession = PromptSession()
        while True:
            if pending is None:
                try:
                    kb, tok = context_size(message_history)
                    tok_str = f"{tok // 1000}k" if tok >= 1000 else str(tok)
                    pending = await pt_session.prompt_async(f"You [{kb:.1f} KB · ~{tok_str} tok]: ")
                except (EOFError, KeyboardInterrupt):
                    return
            text = pending.strip()
            pending = None
            if not text:
                continue
            if text.lower() in {"/exit", "/quit"}:
                return
            if text.lower() == "/reset":
                message_history = []
                if on_reset is not None:
                    on_reset()
                continue
            if self._on_thinking is not None:
                self._on_thinking()
            result = await self._agent.run(text, message_history=message_history)
            message_history = list(result.all_messages())
            if on_turn_complete is not None:
                on_turn_complete(list(result.new_messages()))
            self._emit_fn(result.output)
