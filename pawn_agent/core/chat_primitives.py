"""Shared chat primitives used by the evaluation chat paths."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pawn_agent.utils.config import AgentConfig


def apply_user_message(state: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Return a copy of *state* updated with the next user message."""
    history = list(state.get("chat_history", []))
    history.append({"role": "user", "content": prompt})
    return {
        **state,
        "chat_history": history,
        "latest_user_message": prompt,
        "turn_count": int(state.get("turn_count", 0)) + 1,
    }


def apply_assistant_message(state: dict[str, Any], reply: str) -> dict[str, Any]:
    """Return a copy of *state* updated with the next assistant message."""
    history = list(state.get("chat_history", []))
    history.append({"role": "assistant", "content": reply})
    return {
        **state,
        "chat_history": history,
        "latest_assistant_message": reply,
    }


def normalize_output(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _import_pydantic_agent_cls():
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Evaluation chat requires the optional 'pydantic-ai' dependency to be installed."
        ) from exc
    return Agent


def _import_pydantic_messages():
    try:
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Evaluation chat requires the optional 'pydantic-ai' dependency to be installed."
        ) from exc
    return ModelRequest, ModelResponse, TextPart, UserPromptPart


def _import_openai_model_classes():
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Evaluation chat requires the optional 'pydantic-ai' OpenAI provider support."
        ) from exc
    return OpenAIChatModel, OpenAIProvider


class PlainPydanticChatAgent:
    """Tool-less PydanticAI chat agent for lightweight evaluation flows."""

    def __init__(
        self,
        cfg: AgentConfig,
        on_thinking: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_thinking = on_thinking
        self._agent = self._build_agent()

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        path = Path(self.cfg.anima_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _resolve_model(self):
        model_str = self.cfg.pydantic_model
        api_key = self.cfg.pydantic_api_key
        base_url = self.cfg.pydantic_base_url

        if base_url and model_str.startswith("openai:"):
            OpenAIChatModel, OpenAIProvider = _import_openai_model_classes()

            model_name = model_str[len("openai:"):]
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")

            class _CompatModel(OpenAIChatModel):
                def _map_model_response(self, message):
                    param = super()._map_model_response(message)
                    if param.get("content") is None:
                        param["content"] = ""
                    return param

            return _CompatModel(model_name, provider=provider)

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

    def _build_agent(self):
        Agent = _import_pydantic_agent_cls()
        system_prompts: list[str] = []
        anima = self._load_anima()
        if anima:
            system_prompts.append(anima)
        return Agent(
            self._resolve_model(),
            system_prompt=tuple(system_prompts),
            output_retries=2,
        )

    def _history_to_messages(self, chat_history: list[dict[str, str]]) -> list:
        ModelRequest, ModelResponse, TextPart, UserPromptPart = _import_pydantic_messages()

        messages: list = []
        for item in chat_history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                messages.append(
                    ModelResponse(
                        parts=[TextPart(content=content)],
                        model_name=self.cfg.pydantic_model,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
        return messages

    async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        if self._on_thinking is not None:
            self._on_thinking()

        prior_history = list(chat_history)
        if (
            prior_history
            and prior_history[-1].get("role") == "user"
            and prior_history[-1].get("content") == user_prompt
        ):
            prior_history = prior_history[:-1]

        result = await self._agent.run(
            user_prompt,
            message_history=self._history_to_messages(prior_history),
        )
        return normalize_output(getattr(result, "output", ""))
