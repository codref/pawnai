"""Runtime helpers for the minimal LangGraph chat path."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Callable

from pawn_agent.core.burr_chat import _normalize_output
from pawn_agent.utils.config import AgentConfig


def _import_smolagents_litellm_model():
    try:
        from smolagents import LiteLLMModel
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat requires the optional 'smolagents' dependency. "
            "Install dependencies with 'pip install smolagents litellm' or reinstall the project extras."
        ) from exc
    return LiteLLMModel


def _import_phoenix_register():
    try:
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat Phoenix tracing requires the optional 'arize-phoenix-otel' dependency."
        ) from exc
    return register


def _import_trace_api():
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat Phoenix tracing requires OpenTelemetry support to be installed."
        ) from exc
    return trace


def _normalize_model_prefix(model_str: str) -> tuple[str, str]:
    prefix, sep, model_name = model_str.partition(":")
    if not sep or not prefix or not model_name:
        raise RuntimeError(
            f"LangGraph chat expects an 'openai:<model>' model string, got {model_str!r}."
        )
    return prefix, model_name


def resolve_smolagents_model_config(
    model_str: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, dict[str, str]]:
    prefix, model_name = _normalize_model_prefix(model_str)
    if prefix != "openai":
        raise RuntimeError(
            f"LangGraph chat currently supports only openai:<model> configurations via smolagents; got {prefix!r}."
        )

    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key

    if base_url:
        kwargs["api_base"] = base_url.rstrip("/")
    return f"openai/{model_name}", kwargs


def _extract_text_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_extract_text_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return _extract_text_content(value.get("text"))
        for key in ("content", "text", "value"):
            if key in value:
                return _extract_text_content(value.get(key))
        return ""
    for attr in ("content", "text", "value"):
        if hasattr(value, attr):
            return _extract_text_content(getattr(value, attr))
    return str(value)


def _normalize_model_reply(reply: object) -> str:
    return _normalize_output(_extract_text_content(reply)).strip()


def _make_text_message(role: str, text: str) -> dict[str, object]:
    return {
        "role": role,
        "content": [{"type": "text", "text": _normalize_output(text)}],
    }


class PlainSmolagentsChatAgent:
    """Tool-less smolagents chat wrapper used by the LangGraph evaluation path."""

    def __init__(
        self,
        cfg: AgentConfig,
        on_thinking: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_thinking = on_thinking
        self._system_prompt = self._load_anima()
        self.model_name, model_kwargs = resolve_smolagents_model_config(
            cfg.pydantic_model,
            cfg.pydantic_base_url,
            cfg.pydantic_api_key,
        )
        LiteLLMModel = _import_smolagents_litellm_model()
        self._model = LiteLLMModel(model_id=self.model_name, **model_kwargs)

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        path = Path(self.cfg.anima_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _history_to_messages(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if self._system_prompt:
            messages.append(_make_text_message("system", self._system_prompt))

        prior_history = list(chat_history)
        if (
            prior_history
            and prior_history[-1].get("role") == "user"
            and prior_history[-1].get("content") == user_prompt
        ):
            prior_history = prior_history[:-1]

        for item in prior_history:
            role = item.get("role")
            if role not in {"user", "assistant", "system"}:
                continue
            messages.append(_make_text_message(role, _normalize_output(item.get("content", ""))))

        messages.append(_make_text_message("user", user_prompt))
        return messages

    async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        if self._on_thinking is not None:
            self._on_thinking()

        messages = self._history_to_messages(user_prompt, chat_history)
        result = await asyncio.to_thread(self._model, messages)
        if inspect.isawaitable(result):
            result = await result
        return _normalize_model_reply(result)


def build_phoenix_tracer(cfg: AgentConfig):
    """Build a Phoenix-backed tracer when tracing is enabled."""
    if not cfg.phoenix_enabled:
        return None

    register = _import_phoenix_register()
    trace = _import_trace_api()
    kwargs: dict[str, object] = {}
    if cfg.phoenix_api_key:
        kwargs["headers"] = {"authorization": f"Bearer {cfg.phoenix_api_key}"}
    register(
        project_name=cfg.phoenix_project_name,
        endpoint=cfg.phoenix_endpoint,
        protocol=cfg.phoenix_protocol,
        auto_instrument=False,
        batch=True,
        **kwargs,
    )
    return trace.get_tracer(__name__)
